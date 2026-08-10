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
        else:
            raise ValueError(
                f"Unsupported provider: {self.config.provider}. "
                f"Currently supported: anthropic"
            )

    def _register_tools(self) -> None:
        """Import tool modules to trigger @register_tool decorators."""
        # Each import triggers the @register_tool decorator
        import terminal_agent.tools.read_file      # noqa: F401
        import terminal_agent.tools.write_file     # noqa: F401
        import terminal_agent.tools.search_replace # noqa: F401
        import terminal_agent.tools.grep_search    # noqa: F401
        import terminal_agent.tools.list_directory # noqa: F401
        import terminal_agent.tools.run_command    # noqa: F401

    def _load_system_prompt(self) -> str:
        """Load the system prompt template and fill in placeholders."""
        prompt_path = Path(__file__).parent.parent.parent.parent / "prompts" / "system.md"
        
        if prompt_path.exists():
            template = prompt_path.read_text(encoding="utf-8")
        else:
            template = (
                "You are Terminal Agent, an expert AI coding assistant. "
                "You help users understand, modify, and debug code in their repositories. "
                "Current directory: {cwd}\n\n{repo_map}"
            )
        
        # Build a basic repo map (file listing)
        repo_map = self._build_repo_map()
        
        return template.format(
            cwd=self.working_dir,
            repo_map=repo_map,
        )

    def _build_repo_map(self) -> str:
        """Build a simple repository file listing for context."""
        try:
            repo_path = Path(self.working_dir)
            ignore_dirs = {
                ".git", "__pycache__", "node_modules", ".venv", "venv",
                ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist",
                "build", ".egg-info", ".tox",
            }
            
            files = []
            for p in sorted(repo_path.rglob("*")):
                # Skip ignored directories
                if any(part in ignore_dirs for part in p.parts):
                    continue
                if p.is_file():
                    rel = p.relative_to(repo_path)
                    files.append(str(rel))
            
            if not files:
                return "(empty repository)"
            
            # Limit to 200 files to avoid huge prompts
            if len(files) > 200:
                listing = "\n".join(f"  {f}" for f in files[:200])
                listing += f"\n  ... and {len(files) - 200} more files"
            else:
                listing = "\n".join(f"  {f}" for f in files)
            
            return listing
        except Exception as e:
            logger.warning(f"Failed to build repo map: {e}")
            return "(could not read repository)"

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
            
            # Add assistant message to history
            self.session.add_assistant_message(response.message)
            
            # Check if we need to execute tool calls
            if response.message.tool_calls:
                # Execute all tool calls and collect results
                tool_results = await self._execute_tool_calls(response.message.tool_calls)
                
                # Add tool results to history
                result_msg = Message.tool_result(tool_results)
                self.session.add_tool_result_message(result_msg)
                
                # Continue the loop — LLM needs to process tool results
                continue
            else:
                # No tool calls — this is the final response
                # The text was already streamed to the console
                console.print()  # Final newline after streaming
                break

    async def _stream_response(self) -> LLMResponse:
        """Stream a response from the LLM, displaying tokens in real-time.
        
        Returns the complete LLMResponse after the stream ends.
        """
        messages = self.session.get_messages()
        tool_defs = self.registry.get_tool_definitions()
        
        # Accumulate the full response from stream events
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        current_tool_call: ToolCall | None = None
        input_tokens = 0
        output_tokens = 0
        stop_reason = None
        
        started_text = False
        
        async for event in self.provider.stream(messages, tools=tool_defs):
            if event.type == "text_delta" and event.text:
                if not started_text:
                    console.print("\n[bold blue]🤖 Agent:[/bold blue] ", end="")
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
        
        if started_text:
            console.print()  # Final newline after streamed text
        
        # Build the complete response
        content = "".join(text_parts) if text_parts else None
        message = Message.assistant(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
        )
        
        # Get usage from the provider's stream (approximate if not available)
        # The actual usage comes from the stream's final message event
        # For now, estimate based on text length
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
