"""ReAct Agent Loop — the heart of Terminal Agent.

Orchestrates the think-act-observe cycle:
1. Gather context (system prompt, repo structure, history)
2. Send to LLM and stream the response
3. Execute any tool calls from the response
4. Feed tool results back to the LLM
5. Repeat until the LLM emits a final text response
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import AsyncGenerator

from terminal_agent.core.config import AgentConfig
from terminal_agent.core.session import Session
from terminal_agent.llm.base import LLMProvider
from terminal_agent.llm.anthropic import AnthropicProvider
from terminal_agent.llm.openai_compatible import OpenAICompatibleProvider
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
from terminal_agent.utils.permissions import PermissionChecker, PermissionOutcome

logger = logging.getLogger(__name__)


class Agent:
    """The main agent that runs the ReAct loop.
    
    Connects the LLM provider, tool system, and session management
    into a coherent agentic workflow.
    """

    def __init__(self, config: AgentConfig, working_directory: str | None = None):
        self.config = config
        self.working_dir = working_directory or str(Path.cwd())
        
        # Validate API key early — fail fast before any other setup.
        config.validate_api_key()
        
        # Initialize LLM provider
        self.provider = self._create_provider()
        
        # Initialize tool registry.
        # Import tool modules first — this triggers @register_tool decorators,
        # which append tool classes to the module-level _TOOL_CLASSES list.
        # Then create a fresh registry instance and load those classes into it.
        # This is dependency injection: Agent owns its registry. Tests can
        # construct a ToolRegistry with only the tools they need.
        self._import_tool_modules()
        self.registry = ToolRegistry()
        self.registry.load_defaults()

        # Initialize permission checker.
        # PermissionChecker owns all policy decisions about whether a tool call
        # can proceed. Agent delegates to it without knowing tool-specific rules.
        self.permission_checker = PermissionChecker(
            permission_mode=config.permission_mode.value,
            safe_commands=list(config.safe_commands),
            blocked_patterns=list(config.blocked_patterns),
        )
        
        # Initialize session
        self.session = Session(working_directory=self.working_dir)
        
        # Initialize cost tracker
        self.cost_tracker = CostTracker(model_name=config.model_name)
        
        # Token tracking for real-time status bar updates
        self._last_input_tokens = 0
        self._last_output_tokens = 0
        
        # Load and set system prompt
        system_prompt = self._load_system_prompt()
        self.session.set_system_message(system_prompt)

    def _create_provider(self) -> LLMProvider:
        """Factory method: create the right LLM provider for the configured provider name.

        Design: Why a factory method instead of instantiating in __init__?
        The factory is easily overridable in tests (subclass Agent and override
        _create_provider to return a mock). It also keeps __init__ clean and
        makes the provider creation logic easy to read in one place.

        Provider routing:
          anthropic  → AnthropicProvider (uses Anthropic SDK directly)
          nvidia     → OpenAICompatibleProvider (NVIDIA NIM is OpenAI-compatible)
          openai     → OpenAICompatibleProvider (real OpenAI API)

        Why is nvidia routed to OpenAICompatibleProvider?
        NVIDIA NIM implements the OpenAI Chat Completions API spec exactly.
        There is no meaningful difference at the HTTP level; only the base_url
        and api_key differ. Routing both to the same provider class avoids
        duplication and demonstrates that the abstraction is correct.
        """
        provider = self.config.provider
        api_key = self.config.get_api_key()

        if provider == "anthropic":
            return AnthropicProvider(
                model=self.config.model_name,
                api_key=api_key,
                max_tokens=self.config.max_tokens,
            )

        if provider in ("nvidia", "openai"):
            base_url = self.config.get_base_url()
            return OpenAICompatibleProvider(
                model=self.config.model_name,
                api_key=api_key,
                base_url=base_url,
                max_tokens=self.config.max_tokens,
            )

        raise ValueError(
            f"Unsupported provider: {provider!r}. "
            f"Supported providers: anthropic, nvidia, openai"
        )

    def _import_tool_modules(self) -> None:
        """Import tool modules to trigger @register_tool decorators.

        Each import appends the decorated class to the module-level _TOOL_CLASSES
        list in registry.py. After all imports, self.registry.load_defaults()
        instantiates those classes and registers them in this agent's registry.

        Why import here instead of at the top of the file?
        We want Agent to be explicit about which tools it uses. A future agent
        variant (e.g., a read-only agent) could skip importing write_file or
        run_command and get a narrower tool set without any other changes.
        """
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
            
            # Track current token usage for real-time display
            current_input_tokens = 0
            current_output_tokens = 0
            
            def on_usage(input_tokens: int, output_tokens: int) -> None:
                """Callback to update token usage in real-time."""
                nonlocal current_input_tokens, current_output_tokens
                current_input_tokens = input_tokens
                current_output_tokens = output_tokens
                # Update session token counts with delta
                self.session.total_input_tokens = self.session.total_input_tokens - self._last_input_tokens + input_tokens
                self.session.total_output_tokens = self.session.total_output_tokens - self._last_output_tokens + output_tokens
                # Store for next delta calculation
                self._last_input_tokens = input_tokens
                self._last_output_tokens = output_tokens
            
            # Initialize tracking for this iteration
            self._last_input_tokens = 0
            self._last_output_tokens = 0
            
            # Stream the LLM response with real-time usage callback
            response = await self._stream_response(on_usage=on_usage)
            
            # Fallback update to session if on_usage was never triggered
            if self._last_input_tokens == 0 and self._last_output_tokens == 0:
                self.session.update_token_usage(response.input_tokens, response.output_tokens)
            
            # Track token usage in cost tracker
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

    async def _stream_response(self, on_usage: callable = None) -> LLMResponse:
        """Stream a response from the LLM, displaying tokens in real-time.
        
        Args:
            on_usage: Optional callback invoked when token usage is received.
                      Called with (input_tokens, output_tokens).
        
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

            elif event.type == "usage":
                # Real token counts from the API — authoritative,
                # replaces any client-side estimation.
                if event.usage:
                    input_tokens = event.usage.get("input_tokens", 0)
                    output_tokens = event.usage.get("output_tokens", 0)
                    # Call the callback immediately so UI can update in real-time
                    if on_usage:
                        on_usage(input_tokens, output_tokens)

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
        
        # Fallback estimation if the provider did not return usage in stream
        if input_tokens == 0 and output_tokens == 0:
            input_text = " ".join(m.content or "" for m in messages)
            output_text = content or ""
            input_tokens = self.provider.count_tokens(input_text)
            output_tokens = self.provider.count_tokens(output_text)
            if on_usage:
                on_usage(input_tokens, output_tokens)

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
        """Execute a single tool call with the full approval/safety flow.

        Flow:
          1. Resolve the tool from the registry.
          2. Display the tool call to the user.
          3. Ask PermissionChecker whether to allow, deny, or prompt for approval.
          4. Execute the tool (if allowed).
          5. Truncate output if over budget.
          6. Track any file paths that were accessed.
          7. Display the result.
        """
        try:
            tool = self.registry.get(tool_call.name)
        except KeyError:
            return ToolResult(
                output=f"Error: Unknown tool '{tool_call.name}'. "
                       f"Available tools: {[t.name for t in self.registry.get_all()]}",
                is_error=True,
            )

        # Display the tool call
        display_tool_call(tool_call.name, tool_call.arguments)

        # --- Permission check ---
        # PermissionChecker decides; Agent acts. Agent never references tool names.
        decision = self.permission_checker.check(tool, tool_call)

        if decision.outcome == PermissionOutcome.DENY:
            result = ToolResult(output=decision.reason, is_error=True)
            display_tool_result(tool_call.name, result.output, is_error=True)
            return result

        if decision.outcome == PermissionOutcome.REQUIRE_APPROVAL:
            # Show the approval prompt to the user.
            # The prompt text comes from the decision reason, not from Agent.
            approved = display_approval_prompt(decision.reason)
            if not approved:
                result = ToolResult(
                    output=f"User declined to run: {decision.reason}",
                    is_error=True,
                )
                display_tool_result(tool_call.name, result.output, is_error=True)
                return result

        # --- Execute ---
        try:
            result = await tool.execute(**tool_call.arguments)

            # Truncate output if too long to avoid context explosion.
            if len(result.output) > self.config.max_output_per_tool:
                truncated = result.output[:self.config.max_output_per_tool]
                truncated += (
                    f"\n[...truncated "
                    f"{len(result.output) - self.config.max_output_per_tool} chars...]"
                )
                result = ToolResult(
                    output=truncated,
                    is_error=result.is_error,
                    metadata=result.metadata,
                )

            # Track files that were accessed (for session context awareness).
            path_arg = tool_call.arguments.get("path")
            if path_arg:
                self.session.track_file(path_arg)

        except Exception as e:
            logger.error(f"Tool execution error: {e}", exc_info=True)
            result = ToolResult(
                output=f"Error executing {tool_call.name}: {str(e)}",
                is_error=True,
            )

        # Display the result (truncated for readability in the terminal).
        display_tool_result(
            tool_call.name,
            result.output[:500] + ("..." if len(result.output) > 500 else ""),
            is_error=result.is_error,
        )

        return result
