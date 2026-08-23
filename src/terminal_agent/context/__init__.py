"""Context package — intentionally minimal.

This package previously contained a ContextManager class that maintained a
parallel copy of the conversation history as raw dicts. It was never wired up
to the agent loop and was deleted (see core/session.py for the design rationale).

Context budget awareness now lives on Session via:
  - Session.token_usage_ratio(max_context_tokens) -> float
  - Session.needs_compaction(max_context_tokens, threshold) -> bool

Future additions to this package (if needed):
  - context/summarizer.py  — LLM-based message summarization
  - context/compressor.py  — Tool-result compression strategies
"""
