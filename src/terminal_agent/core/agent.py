"""ReAct Agent Loop — the heart of Terminal Agent.

Orchestrates the think-act-observe cycle:
1. Gather context (system prompt, repo structure, history)
2. Send to LLM and stream the response
3. Execute any tool calls from the response
4. Feed tool results back to the LLM
5. Repeat until the LLM emits a final text response
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from terminal_agent.core.config import AgentConfig
from terminal_agent.core.events import (
    AgentEvent,
    AgentEventListener,
    RichConsoleListener,
    TextDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from terminal_agent.core.reflexion import ReflexionMemory
from terminal_agent.core.session import Session
from terminal_agent.core.session_store import SessionStore
from terminal_agent.guardrails.domain import DomainGuardrail
from terminal_agent.llm.base import LLMProvider
from terminal_agent.llm.message import (
    LLMResponse,
    Message,
    ToolCall,
    ToolResultContent,
)
from terminal_agent.llm.profiles import ModelProfile, profile_registry
from terminal_agent.llm.registry import ProviderRegistry, default_provider_registry
from terminal_agent.tools.base import ToolResult
from terminal_agent.tools.registry import ToolRegistry
from terminal_agent.utils.approval import (
    ApprovalHandler,
    CLIApprovalHandler,
)
from terminal_agent.utils.cost import CostTracker
from terminal_agent.utils.display import (
    console,
)
from terminal_agent.utils.permissions import PermissionChecker, PermissionOutcome

logger = logging.getLogger(__name__)


class Agent:
    """The main agent that runs the ReAct loop.
    
    Connects the LLM provider, tool system, and session management
    into a coherent agentic workflow.
    """

    def __init__(
        self, 
        config: AgentConfig, 
        working_directory: str | None = None,
        listeners: list[AgentEventListener] | None = None,
        approval_handler: ApprovalHandler | None = None,
        provider_registry: ProviderRegistry | None = None,
        session: Session | None = None,
        session_store: SessionStore | None = None,
    ):
        self.config = config
        self.working_dir = working_directory or str(Path.cwd())
        self.listeners: list[AgentEventListener] = list(listeners) if listeners is not None else [RichConsoleListener()]
        self.approval_handler = approval_handler or CLIApprovalHandler()
        self.provider_registry = provider_registry or default_provider_registry
        self.session_store = session_store if session_store is not None else SessionStore()
        
        # Validate API key early — fail fast before any other setup.
        config.validate_api_key()
        
        # Initialize LLM provider via registry (Open-Closed Principle)
        self.provider = self._create_provider()
        
        # Initialize tool registry via discovery (Dependency Inversion Principle)
        self._import_tool_modules()
        self.registry = ToolRegistry()
        self.registry.load_defaults()

        # Initialize execution sandbox
        from terminal_agent.sandbox import SandboxPolicy, create_sandbox
        sandbox_policy = SandboxPolicy(
            working_dir=Path(self.working_dir),
            allow_network=getattr(config, "sandbox_allow_network", True),
            timeout_seconds=getattr(config, "timeout", 120),
        )
        self.sandbox = create_sandbox(
            mode=getattr(config, "sandbox_mode", "local"),
            working_dir=self.working_dir,
            policy=sandbox_policy,
            docker_image=getattr(config, "sandbox_docker_image", "python:3.12-slim"),
        )
        if "run_command" in self.registry:
            run_cmd = self.registry.get("run_command")
            if hasattr(run_cmd, "sandbox"):
                run_cmd.sandbox = self.sandbox
                run_cmd.working_dir = Path(self.working_dir)

        # Initialize permission checker.
        self.permission_checker = PermissionChecker(
            permission_mode=config.permission_mode.value,
            safe_commands=list(config.safe_commands),
            blocked_patterns=list(config.blocked_patterns),
        )
        
        # Initialize session (resumed session or fresh instance)
        self.session = session if session is not None else Session(working_directory=self.working_dir)
        
        # Initialize cost tracker
        self.cost_tracker = CostTracker(model_name=config.model_name)
        
        # Initialize Reflexion episodic failure memory buffer
        self.reflexion_memory = ReflexionMemory(max_episodes=5)
        
        # Initialize domain boundary guardrail
        self.domain_guardrail = DomainGuardrail()
        
        # Token tracking for real-time status bar updates
        self._last_input_tokens = 0
        self._last_output_tokens = 0
        
        # Load and set system prompt
        system_prompt = self._load_system_prompt()
        self.session.set_system_message(system_prompt)

    def add_listener(self, listener: AgentEventListener) -> None:
        """Register an observer for agent execution events."""
        if listener not in self.listeners:
            self.listeners.append(listener)

    def remove_listener(self, listener: AgentEventListener) -> None:
        """Remove an observer."""
        if listener in self.listeners:
            self.listeners.remove(listener)

    def emit(self, event: AgentEvent) -> None:
        """Dispatch domain events to registered observers."""
        for listener in self.listeners:
            try:
                if isinstance(event, TurnStartEvent):
                    listener.on_turn_start(event)
                elif isinstance(event, TurnEndEvent):
                    listener.on_turn_end(event)
                elif isinstance(event, TextDeltaEvent):
                    listener.on_text_delta(event)
                elif isinstance(event, ToolCallStartEvent):
                    listener.on_tool_call_start(event)
                elif isinstance(event, ToolCallEndEvent):
                    listener.on_tool_call_end(event)
            except Exception as e:
                logger.warning(f"Error in event listener {listener}: {e}")

    def _create_provider(self) -> LLMProvider:
        """Factory method: create LLM provider via the extensible ProviderRegistry (Open-Closed Principle)."""
        primary = self.provider_registry.create(self.config.provider, self.config)
        if getattr(self.config, "enable_failover", False) and getattr(self.config, "fallback_provider", None):
            try:
                fallback_config = self.config.model_copy(
                    update={
                        "provider": self.config.fallback_provider,
                        "model_name": self.config.fallback_model or self.config.model_name,
                    }
                )
                fallback_p = self.provider_registry.create(
                    self.config.fallback_provider, fallback_config
                )
                from terminal_agent.llm.fallback import FallbackProvider
                return FallbackProvider(primary=primary, fallbacks=[fallback_p])
            except Exception as e:
                logger.warning(f"Could not initialize fallback provider: {e}")
        return primary

    def switch_model_profile(self, profile_name_or_obj: str | ModelProfile) -> ModelProfile:
        """Switch the active LLM provider and model profile in-session.

        Args:
            profile_name_or_obj: Name of a registered profile alias or a ModelProfile instance.

        Returns:
            The activated ModelProfile.

        Raises:
            KeyError: If profile_name string is not found in the profile registry.
            ValueError: If the required API key for the new provider is not configured.
        """
        if isinstance(profile_name_or_obj, str):
            profile = profile_registry.get(profile_name_or_obj)
        elif isinstance(profile_name_or_obj, ModelProfile):
            profile = profile_name_or_obj
        else:
            raise TypeError("Expected profile name (str) or ModelProfile instance")

        # Validate that the API key for the target provider is present
        self.config.validate_api_key(profile.provider)

        # Update config fields
        self.config.profile = profile.name
        self.config.provider = profile.provider
        self.config.model_name = profile.model_name
        self.config.max_tokens = profile.max_tokens
        self.config.max_context_tokens = profile.context_window
        if profile.base_url:
            if profile.provider == "nvidia":
                self.config.nvidia_base_url = profile.base_url
            elif profile.provider == "openai":
                self.config.openai_base_url = profile.base_url
            elif profile.provider == "openrouter":
                self.config.openrouter_base_url = profile.base_url

        # Recreate provider & update cost tracker
        self.provider = self._create_provider()
        self.cost_tracker.model_name = profile.model_name

        return profile

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
        import terminal_agent.tools.grep_search
        import terminal_agent.tools.list_directory
        import terminal_agent.tools.read_file
        import terminal_agent.tools.repo_map
        import terminal_agent.tools.run_command
        import terminal_agent.tools.search_replace
        import terminal_agent.tools.write_file  # noqa: F401

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
        
        # Build an AST-driven repository symbol map
        repo_map = self._build_repo_map()
        
        return template.format(
            cwd=self.working_dir,
            repo_map=repo_map,
        )

    def _build_repo_map(self) -> str:
        """Build an architectural symbol skeleton and dependency map for context."""
        try:
            from terminal_agent.repo.map_builder import RepoMapBuilder
            builder = RepoMapBuilder(working_directory=self.working_dir)
            return builder.build_map(max_tokens=1500)
        except Exception as e:
            logger.warning(f"Failed to build repo map: {e}")
            return f"(error generating repository map: {e})"

    async def solve_task(self, task_description: str, test_command: str | None = None) -> dict[str, Any]:
        """High-level task solver implementing the Two-Tier architecture (Agentless vs ReAct).
        
        Tier 1: Agentless Fast Path (Localize -> Patch -> Validate) for focused bug-fixing.
        Tier 2: Autonomous ReAct loop with full tool access and Reflexion episodic memory.
        """
        from terminal_agent.core.config import ExecutionMode
        from terminal_agent.core.fast_path import AgentlessFastPath

        if self.config.execution_mode in (ExecutionMode.FAST_PATH, ExecutionMode.AUTO):
            fast_path = AgentlessFastPath(
                provider=self.provider,
                working_directory=self.working_dir
            )
            fp_result = await fast_path.execute(task_description, test_command=test_command)
            if fp_result.success:
                console.print(f"[bold green]⚡ Fast-Path Success:[/bold green] {fp_result.explanation}")
                return {
                    "mode": "fast_path",
                    "success": True,
                    "result": fp_result,
                }
            elif self.config.execution_mode == ExecutionMode.FAST_PATH:
                return {
                    "mode": "fast_path",
                    "success": False,
                    "result": fp_result,
                }
            logger.info(f"Fast-path did not resolve task ({fp_result.explanation}). Escalating to Tier 2 ReAct.")

        # Fallback to Tier 2: Autonomous ReAct loop
        await self.process_message(task_description)
        return {
            "mode": "react",
            "success": True,
            "session": self.session,
        }

    async def process_message(self, user_input: str) -> None:
        """Process a user message through the full ReAct loop.
        
        This is the main entry point for each user turn. It:
        1. Adds the user message to history
        2. Enters the agent loop
        3. Streams LLM responses and executes tool calls
        4. Continues until the LLM emits a final text response
        """
        # 1. Add user message to session
        self.session.add_user_message(user_input)
        self.session.reset_iteration_count()

        try:
            # 2. Check domain boundary guardrails if enabled
            if getattr(self.config, "enable_domain_guardrail", True):
                is_in_domain, refusal_msg = await self.domain_guardrail.check(user_input, provider=self.provider)
                if not is_in_domain and refusal_msg:
                    self.emit(TurnStartEvent(iteration=1))
                    self.emit(TextDeltaEvent(text=refusal_msg))
                    refusal_assistant_msg = Message.assistant(content=refusal_msg)
                    self.session.add_assistant_message(refusal_assistant_msg)
                    self.emit(
                        TurnEndEvent(
                            iteration=1,
                            response=LLMResponse(
                                message=refusal_assistant_msg,
                                input_tokens=0,
                                output_tokens=0,
                                stop_reason="end_turn",
                            ),
                        )
                    )
                    if self.session_store:
                        try:
                            self.session_store.save_session(self.session)
                        except Exception as e:
                            logger.warning(f"Failed to auto-save session: {e}")
                    return

            # 3. Enter the agent loop
            await self._agent_loop()
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.warning("Turn interrupted by user (Ctrl+C). Terminating running sandbox operations.")
            await self.sandbox.terminate()
            if self.session_store:
                try:
                    self.session_store.save_session(self.session)
                except Exception as e:
                    logger.warning(f"Failed to save session during interrupt: {e}")
            raise
        except Exception:
            if self.session_store:
                try:
                    self.session_store.save_session(self.session)
                except Exception as e:
                    logger.warning(f"Failed to save session during error: {e}")
            raise

        if self.session_store:
            try:
                self.session_store.save_session(self.session)
            except Exception as e:
                logger.warning(f"Failed to auto-save session: {e}")

    async def _agent_loop(self) -> None:
        """The core ReAct loop: Think → Act → Observe → Repeat."""
        while True:
            iteration = self.session.increment_iteration()
            self.emit(TurnStartEvent(iteration=iteration))
            
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
                # Encapsulated token arithmetic inside Session (Information Expert)
                self.session.apply_stream_usage(
                    input_tokens, 
                    output_tokens, 
                    self._last_input_tokens, 
                    self._last_output_tokens
                )
                self._last_input_tokens = input_tokens
                self._last_output_tokens = output_tokens
            
            # Initialize tracking for this iteration
            self._last_input_tokens = 0
            self._last_output_tokens = 0
            
            # Context compaction check
            if self.session.needs_compaction(self.config.max_context_tokens, self.config.summarization_threshold):
                saved = self.session.compact()
                if saved > 0:
                    logger.info(f"Compacted conversation context, saved ~{saved} characters")

            # Stream the LLM response with real-time usage callback
            response = await self._stream_response(on_usage=on_usage)
            
            # Fallback update to session if on_usage was never triggered
            if self._last_input_tokens == 0 and self._last_output_tokens == 0:
                self.session.update_token_usage(response.input_tokens, response.output_tokens)
            
            # Track token usage in cost tracker
            self.cost_tracker.add_usage(
                response.input_tokens, 
                response.output_tokens, 
                cache_creation_tokens=response.cache_creation_input_tokens,
                cache_read_tokens=response.cache_read_input_tokens,
                model_name=self.config.model_name
            )
            
            # Add assistant message to history
            self.session.add_assistant_message(response.message)
            self.emit(TurnEndEvent(iteration=iteration, response=response))
            
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
        cache_creation_tokens = 0
        cache_read_tokens = 0
        stop_reason = None
        
        started_text = False
        
        async for event in self.provider.stream(messages, tools=tool_defs):
            if event.type == "text_delta" and event.text:
                self.emit(TextDeltaEvent(text=event.text))
                text_parts.append(event.text)
                
            elif event.type == "tool_call_start":
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
                    cache_creation_tokens = event.usage.get("cache_creation_input_tokens", 0)
                    cache_read_tokens = event.usage.get("cache_read_input_tokens", 0)
                    # Call the callback immediately so UI can update in real-time
                    if on_usage:
                        on_usage(input_tokens, output_tokens)

            elif event.type == "message_end":
                break
        
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
            cache_creation_input_tokens=cache_creation_tokens,
            cache_read_input_tokens=cache_read_tokens,
            stop_reason="tool_use" if tool_calls else "end_turn",
        )

    async def _execute_tool_calls(
        self, tool_calls: list[ToolCall]
    ) -> list[ToolResultContent]:
        """Execute a list of tool calls and return their results.
        Runs read-only tools that do not require approval concurrently.
        """
        can_run_concurrently = len(tool_calls) > 1 and all(
            tc.name in self.registry and not self.registry.get(tc.name).requires_approval
            for tc in tool_calls
        )

        if can_run_concurrently:
            results_list = await asyncio.gather(
                *(self._execute_single_tool(tc) for tc in tool_calls)
            )
            return [
                ToolResultContent(
                    tool_call_id=tc.id,
                    output=res.output,
                    is_error=res.is_error,
                )
                for tc, res in zip(tool_calls, results_list)
            ]

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
        """Execute a single tool call with the full approval/safety flow."""
        try:
            tool = self.registry.get(tool_call.name)
        except KeyError:
            return ToolResult(
                output=f"Error: Unknown tool '{tool_call.name}'. "
                       f"Available tools: {[t.name for t in self.registry.get_all()]}",
                is_error=True,
            )

        # Notify observers that tool execution started
        self.emit(ToolCallStartEvent(tool_name=tool_call.name, arguments=tool_call.arguments))

        # PermissionChecker decides; Agent delegates.
        decision = self.permission_checker.check(tool, tool_call)

        if decision.outcome == PermissionOutcome.DENY:
            result = ToolResult(output=decision.reason, is_error=True)
            self.emit(
                ToolCallEndEvent(
                    tool_name=tool_call.name,
                    is_error=True,
                    output=result.output,
                )
            )
            return result

        if decision.outcome == PermissionOutcome.REQUIRE_APPROVAL:
            # Delegate to injectable ApprovalHandler strategy
            approved = await self.approval_handler.request_approval(decision.reason)
            if not approved:
                result = ToolResult(
                    output=f"User declined to run: {decision.reason}",
                    is_error=True,
                )
                self.emit(
                    ToolCallEndEvent(
                        tool_name=tool_call.name,
                        is_error=True,
                        output=result.output,
                    )
                )
                return result

        # Execute
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
                output=f"Error executing {tool_call.name}: {e!s}",
                is_error=True,
            )

        # Reflexion failure recording & verbal self-reflection reinforcement
        if result.is_error:
            episode = self.reflexion_memory.record_failure(
                tool_name=tool_call.name,
                args=tool_call.arguments,
                error_output=result.output,
            )
            result = ToolResult(
                output=f"{result.output}\n\n[Reflexion Self-Correction Advice]: {episode.reflection}",
                is_error=True,
                metadata=result.metadata,
            )
        else:
            self.reflexion_memory.mark_resolved(tool_call.name)

        # Notify observers of completion
        preview = result.output[:500] + ("..." if len(result.output) > 500 else "")
        self.emit(
            ToolCallEndEvent(
                tool_name=tool_call.name,
                is_error=result.is_error,
                output=preview,
            )
        )

        return result
