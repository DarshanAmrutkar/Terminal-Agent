"""ReAct Agent Loop — the heart of Terminal Agent.

Orchestrates the think-act-observe cycle:
1. Gather context (system prompt, repo structure, history)
2. Send to LLM and stream the response
3. Execute any tool calls from the response
4. Feed tool results back to the LLM
5. Repeat until the LLM emits a final text response
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path
from typing import AsyncGenerator

from terminal_agent.core.config import AgentConfig
from terminal_agent.core.session import Session
from terminal_agent.llm.base import LLMProvider
from terminal_agent.llm.anthropic import AnthropicProvider
from terminal_agent.llm.message import (
    Message,
    ToolCall,
    ToolResultContent,
    StreamEvent,
    LLMResponse,
)
from terminal_agent.repo.analyzer import RepoAnalyzer
from terminal_agent.repo.map_builder import MapBuilder
from terminal_agent.repo.git import GitRepo
from terminal_agent.tools.registry import ToolRegistry
from terminal_agent.tools.base import ToolResult
from terminal_agent.utils.cost import CostTracker
from terminal_agent.utils.display import (
    console,
    display_tool_call,
    display_tool_result,
    display_streaming_token,
    display_approval_prompt,
    display_agent_message,
)
from terminal_agent.utils.permissions import SafetyLevel, classify_command

logger = logging.getLogger(__name__)

# Providers that support native function calling (tools parameter in API)
NATIVE_TOOL_PROVIDERS = {"anthropic", "openai"}


class Agent:
    """The main agent that runs the ReAct loop.
    
    Connects the LLM provider, tool system, and session management
    into a coherent agentic workflow.
    """

    def __init__(self, config: AgentConfig, working_directory: str | None = None):
        self.config = config
        self.working_dir = working_directory or str(Path.cwd())
        
        # Validate API key
        config.validate_api_key()
        
        # Initialize LLM provider
        self.provider = self._create_provider()
        
        # Determine if provider supports native tool calling
        self.native_tools = config.provider in NATIVE_TOOL_PROVIDERS
        
        # Initialize tool registry — import tool modules to trigger registration
        self._register_tools()
        self.registry = ToolRegistry()
        
        # Initialize session
        self.session = Session(working_directory=self.working_dir)
        
        # Initialize cost tracker
        self.cost_tracker = CostTracker(model_name=config.model_name)
        
        # Load and set system prompt
        system_prompt = self._load_system_prompt()
        self.session.set_system_message(system_prompt)

    def _create_provider(self) -> LLMProvider:
        """Create the appropriate LLM provider based on config."""
        if self.config.provider == "anthropic":
            return AnthropicProvider(
                model=self.config.model_name,
                api_key=self.config.get_api_key(),
                max_tokens=self.config.max_tokens,
            )
        elif self.config.provider == "openai":
            from terminal_agent.llm.openai import OpenAIProvider
            return OpenAIProvider(
                model=self.config.model_name,
                api_key=self.config.get_api_key(),
                max_tokens=self.config.max_tokens,
            )
        elif self.config.provider == "nvidia":
            from terminal_agent.llm.openai import OpenAIProvider
            return OpenAIProvider(
                model=self.config.model_name,
                api_key=self.config.get_api_key(),
                base_url=self.config.nvidia_base_url,
                max_tokens=self.config.max_tokens,
            )
        else:
            raise ValueError(
                f"Unsupported provider: {self.config.provider}. "
                f"Currently supported: anthropic, openai, nvidia"
            )

    def _register_tools(self) -> None:
        """Import tool modules to trigger @register_tool decorators."""
        # Each import triggers the @register_tool decorator
        import terminal_agent.tools.read_file       # noqa: F401
        import terminal_agent.tools.write_file      # noqa: F401
        import terminal_agent.tools.search_replace  # noqa: F401
        import terminal_agent.tools.grep_search     # noqa: F401
        import terminal_agent.tools.list_directory  # noqa: F401
        import terminal_agent.tools.run_command     # noqa: F401
        import terminal_agent.tools.git_operations  # noqa: F401

    def _load_system_prompt(self) -> str:
        """Load the system prompt template and fill in placeholders."""
        prompt_path = Path(__file__).parent.parent.parent.parent / "prompts" / "system.md"
        
        if prompt_path.exists():
            template = prompt_path.read_text(encoding="utf-8")
        else:
            template = (
                "You are Terminal Agent, an expert AI coding assistant. "
                "You help users understand, modify, and debug code in their repositories. "
                "Current directory: {cwd}\n\n{repo_map}\n\n{git_context}"
            )
        
        # Build repo map using tree-sitter (falls back to file listing)
        repo_map = self._build_repo_map()
        
        # Build git context
        git_context = self._build_git_context()
        
        prompt = template.format(
            cwd=self.working_dir,
            repo_map=repo_map,
            git_context=git_context if "{git_context}" in template else "",
        )
        
        # For providers without native tool support, inject tool
        # definitions directly into the system prompt
        if not self.native_tools:
            prompt += "\n\n" + self._build_tool_prompt()
        
        return prompt

    def _build_tool_prompt(self) -> str:
        """Build a text-based tool definition prompt for models without native tool calling.
        
        Instructs the model to use a specific JSON format to invoke tools.
        """
        tool_defs = self.registry.get_tool_definitions()
        
        lines = [
            "## Tool Use Instructions",
            "",
            "You have access to the following tools. To use a tool, respond with a JSON block "
            "wrapped in ```tool_call``` markers. You can include thinking/explanation text "
            "before and after the tool call.",
            "",
            "Format:",
            "```tool_call",
            '{"name": "tool_name", "arguments": {"param1": "value1"}}',
            "```",
            "",
            "After a tool executes, you will receive the result in the next message. "
            "You can then make additional tool calls or provide your final answer.",
            "",
            "Available tools:",
            "",
        ]
        
        for tool_def in tool_defs:
            name = tool_def["name"]
            desc = tool_def.get("description", "")
            params = tool_def.get("input_schema", {})
            props = params.get("properties", {})
            required = params.get("required", [])
            
            lines.append(f"### {name}")
            lines.append(f"{desc}")
            lines.append("Parameters:")
            for prop_name, prop_info in props.items():
                req = " (required)" if prop_name in required else " (optional)"
                prop_type = prop_info.get("type", "any")
                prop_desc = prop_info.get("description", "")
                lines.append(f"  - {prop_name}: {prop_type}{req} — {prop_desc}")
            lines.append("")
        
        return "\n".join(lines)

    def _parse_tool_calls_from_text(self, text: str) -> tuple[str, list[ToolCall]]:
        """Parse tool calls from text output for non-native-tool providers.
        
        Looks for ```tool_call ... ``` blocks in the text, extracts them as
        ToolCall objects, and returns the cleaned text (without tool call blocks).
        
        Returns:
            (cleaned_text, list_of_tool_calls)
        """
        tool_calls: list[ToolCall] = []
        
        # Match ```tool_call\n{...}\n``` blocks
        pattern = r'```tool_call\s*\n?\s*(\{.*?\})\s*\n?\s*```'
        matches = list(re.finditer(pattern, text, re.DOTALL))
        
        for match in matches:
            try:
                data = json.loads(match.group(1))
                name = data.get("name", "")
                arguments = data.get("arguments", {})
                
                if name:  # Only create a tool call if we have a name
                    tool_calls.append(ToolCall(
                        id=f"tc_{uuid.uuid4().hex[:12]}",
                        name=name,
                        arguments=arguments,
                    ))
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse tool call from text: {e}")
                continue
        
        # Remove the tool_call blocks from the text to get clean content
        cleaned = re.sub(pattern, '', text, flags=re.DOTALL).strip()
        
        return cleaned, tool_calls

    def _build_repo_map(self) -> str:
        """Build a structural repository map using tree-sitter.
        
        Falls back to a simple file listing if tree-sitter parsing fails.
        """
        try:
            analyzer = RepoAnalyzer(self.working_dir)
            symbols = analyzer.analyze()
            
            if symbols:
                builder = MapBuilder(max_tokens=4000)
                repo_map = builder.build(symbols)
                if repo_map.strip():
                    return repo_map
            
            # Fallback to simple file listing
            return self._build_simple_file_listing()
        except Exception as e:
            logger.warning(f"Tree-sitter repo map failed, using file listing: {e}")
            return self._build_simple_file_listing()

    def _build_simple_file_listing(self) -> str:
        """Build a simple file listing as fallback for repo map."""
        try:
            repo_path = Path(self.working_dir)
            ignore_dirs = {
                ".git", "__pycache__", "node_modules", ".venv", "venv",
                ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist",
                "build", ".egg-info", ".tox",
            }
            
            files = []
            for p in sorted(repo_path.rglob("*")):
                if any(part in ignore_dirs for part in p.parts):
                    continue
                if p.is_file():
                    rel = p.relative_to(repo_path)
                    files.append(str(rel))
            
            if not files:
                return "(empty repository)"
            
            if len(files) > 200:
                listing = "\n".join(f"  {f}" for f in files[:200])
                listing += f"\n  ... and {len(files) - 200} more files"
            else:
                listing = "\n".join(f"  {f}" for f in files)
            
            return listing
        except Exception as e:
            logger.warning(f"Failed to build file listing: {e}")
            return "(could not read repository)"

    def _build_git_context(self) -> str:
        """Build git context info for the system prompt."""
        try:
            git = GitRepo(self.working_dir)
            if not git.is_git_repo():
                return ""
            
            parts = []
            branch = git.current_branch()
            if branch:
                parts.append(f"Git branch: {branch}")
            
            status = git.status()
            if status:
                parts.append(f"Git status:\n{status}")
            
            return "\n".join(parts) if parts else ""
        except Exception as e:
            logger.debug(f"Git context unavailable: {e}")
            return ""

    async def process_message(self, user_input: str) -> None:
        """Process a user message through the full ReAct loop.
        
        This is the main entry point for each user turn. It:
        1. Adds the user message to history
        2. Enters the agent loop
        3. Streams LLM responses and executes tool calls
        4. Continues until the LLM emits a final text response
        """
        # Add user message
        self.session.add_user_message(user_input)
        self.session.reset_iteration_count()
        
        # Enter the agent loop
        await self._agent_loop()

    async def _agent_loop(self) -> None:
        """The core ReAct loop: Think → Act → Observe → Repeat."""
        while True:
            iteration = self.session.increment_iteration()
            
            # Guard against infinite loops
            if iteration > self.config.max_iterations:
                console.print(
                    f"\n[bold yellow]⚠ Reached maximum iterations ({self.config.max_iterations}). "
                    f"Stopping to prevent infinite loop.[/bold yellow]"
                )
                break
            
            # Stream the LLM response
            response = await self._stream_response()
            
            # Track token usage
            self.session.update_token_usage(response.input_tokens, response.output_tokens)
            self.cost_tracker.add_usage(response.input_tokens, response.output_tokens)
            
            # For non-native-tool providers, parse tool calls from text
            if not self.native_tools and response.message.content:
                cleaned_text, parsed_tool_calls = self._parse_tool_calls_from_text(
                    response.message.content
                )
                if parsed_tool_calls:
                    # Update the message with parsed tool calls and cleaned text
                    response = LLMResponse(
                        message=Message.assistant(
                            content=cleaned_text if cleaned_text else None,
                            tool_calls=parsed_tool_calls,
                        ),
                        input_tokens=response.input_tokens,
                        output_tokens=response.output_tokens,
                        stop_reason="tool_use",
                    )
            
            # Add assistant message to history
            self.session.add_assistant_message(response.message)
            
            # Check if we need to execute tool calls
            if response.message.tool_calls:
                # Execute all tool calls and collect results
                tool_results = await self._execute_tool_calls(response.message.tool_calls)
                
                # Add tool results to history
                # For non-native providers, format results as a user message
                if self.native_tools:
                    result_msg = Message.tool_result(tool_results)
                    self.session.add_tool_result_message(result_msg)
                else:
                    # Pack tool results as a user message since the provider
                    # doesn't understand tool_result role
                    result_lines = []
                    for tr in tool_results:
                        status = "ERROR" if tr.is_error else "OK"
                        result_lines.append(
                            f"Tool result ({status}):\n```\n{tr.output}\n```"
                        )
                    result_text = "\n\n".join(result_lines)
                    self.session.add_user_message(result_text)
                
                # Continue the loop — LLM needs to process tool results
                continue
            else:
                # No tool calls — this is the final response
                # The text was already streamed to the console
                console.print()  # Final newline after streaming
                
                # Auto-save session after each completed turn
                try:
                    self.session.auto_save()
                except Exception as e:
                    logger.debug(f"Auto-save failed: {e}")
                
                break

    async def _stream_response(self) -> LLMResponse:
        """Stream a response from the LLM, displaying tokens in real-time.
        
        Returns the complete LLMResponse after the stream ends.
        """
        messages = self.session.get_messages()
        
        # Only pass tool definitions for providers with native tool support
        tool_defs = self.registry.get_tool_definitions() if self.native_tools else None
        
        # Accumulate the full response from stream events
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        current_tool_call: ToolCall | None = None
        input_tokens = 0
        output_tokens = 0
        stop_reason = None
        
        started_text = False
        console.print("\n[bold blue]🤖 Agent:[/bold blue] [dim]Thinking...[/dim]", end="\r")
        
        try:
            async for event in self.provider.stream(messages, tools=tool_defs):
                if event.type == "text_delta" and event.text:
                    if not started_text:
                        console.print("\r[bold blue]🤖 Agent:[/bold blue] ", end="")
                        started_text = True
                    display_streaming_token(event.text)
                    text_parts.append(event.text)
                    
                elif event.type == "tool_call_start":
                    if started_text:
                        console.print()  # Newline after text before tool call
                        started_text = False
                    current_tool_call = event.tool_call
                    
                elif event.type == "tool_call_delta":
                    pass  # JSON accumulation handled by provider
                    
                elif event.type == "tool_call_end":
                    if event.tool_call:
                        tool_calls.append(event.tool_call)
                        current_tool_call = None
                        
                elif event.type == "message_end":
                    break
        except Exception as e:
            if started_text:
                console.print()
            console.print(f"\n[bold red]Stream error:[/bold red] {e}")
            logger.error(f"Stream error: {e}", exc_info=True)
            # Return whatever we accumulated so far
        
        if started_text:
            console.print()  # Final newline after streamed text
        
        # Build the complete response
        content = "".join(text_parts) if text_parts else None
        message = Message.assistant(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
        )
        
        # Get usage from the provider's stream (approximate if not available)
        try:
            total_text = content or ""
            for tc in tool_calls:
                total_text += json.dumps(tc.arguments)
            output_tokens = self.provider.count_tokens(total_text)
            
            # Estimate input tokens from messages
            input_text = ""
            for msg in messages:
                if msg.content:
                    input_text += msg.content
            input_tokens = self.provider.count_tokens(input_text)
        except Exception:
            pass
        
        return LLMResponse(
            message=message,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason="tool_use" if tool_calls else "end_turn",
        )

    async def _execute_tool_calls(
        self, tool_calls: list[ToolCall]
    ) -> list[ToolResultContent]:
        """Execute a list of tool calls and return their results."""
        results: list[ToolResultContent] = []
        
        for tc in tool_calls:
            result = await self._execute_single_tool(tc)
            results.append(
                ToolResultContent(
                    tool_call_id=tc.id,
                    output=result.output,
                    is_error=result.is_error,
                )
            )
        
        return results

    async def _execute_single_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call with approval flow if needed."""
        try:
            tool = self.registry.get(tool_call.name)
        except KeyError:
            return ToolResult(
                output=f"Error: Unknown tool '{tool_call.name}'",
                is_error=True,
            )
        
        # Display the tool call
        display_tool_call(tool_call.name, tool_call.arguments)
        
        # Check if approval is needed
        if tool.requires_approval and self.config.permission_mode != "yolo":
            # For run_command, also check safety classification
            if tool_call.name == "run_command":
                command = tool_call.arguments.get("command", "")
                safety = classify_command(command, self.config)
                
                if safety == SafetyLevel.BLOCKED:
                    result = ToolResult(
                        output=f"Command blocked for safety: {command}",
                        is_error=True,
                    )
                    display_tool_result(tool_call.name, result.output, is_error=True)
                    return result
                    
                if safety == SafetyLevel.SAFE:
                    # Auto-approved
                    pass
                else:
                    # Needs approval
                    if not display_approval_prompt(command):
                        result = ToolResult(
                            output="User declined to execute command.",
                            is_error=True,
                        )
                        display_tool_result(tool_call.name, result.output, is_error=True)
                        return result
            else:
                # Non-command tools that require approval
                desc = f"{tool_call.name}({json.dumps(tool_call.arguments, indent=2)})"
                if not display_approval_prompt(desc):
                    result = ToolResult(
                        output=f"User declined {tool_call.name} operation.",
                        is_error=True,
                    )
                    display_tool_result(tool_call.name, result.output, is_error=True)
                    return result
        
        # Execute the tool
        try:
            result = await tool.execute(**tool_call.arguments)
            
            # Truncate output if too long
            if len(result.output) > self.config.max_output_per_tool:
                truncated = result.output[:self.config.max_output_per_tool]
                truncated += f"\n[...truncated {len(result.output) - self.config.max_output_per_tool} chars...]"
                result = ToolResult(
                    output=truncated,
                    is_error=result.is_error,
                    metadata=result.metadata,
                )
            
            # Track files that were read or modified
            path_arg = tool_call.arguments.get("path")
            if path_arg and tool_call.name in ("read_file", "write_file", "search_replace"):
                self.session.track_file(path_arg)
            
        except Exception as e:
            logger.error(f"Tool execution error: {e}", exc_info=True)
            result = ToolResult(
                output=f"Error executing {tool_call.name}: {str(e)}",
                is_error=True,
            )
        
        # Display the result
        display_tool_result(
            tool_call.name,
            result.output[:500] + ("..." if len(result.output) > 500 else ""),
            is_error=result.is_error,
        )
        
        return result
