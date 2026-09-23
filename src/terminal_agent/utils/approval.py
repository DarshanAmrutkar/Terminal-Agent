"""Approval Handler Strategy Pattern.

Decouples permission approval interaction from the core Agent loop.
Allows seamless switching between interactive CLI prompts, automated evaluation approvals,
or webhook-based remote confirmations.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod


class ApprovalHandler(ABC):
    """Abstract strategy for approving potentially dangerous tool actions."""

    @abstractmethod
    async def request_approval(self, reason: str) -> bool:
        """Request permission to execute an action."""


class CLIApprovalHandler(ApprovalHandler):
    """Interactive terminal confirmation using Rich."""

    async def request_approval(self, reason: str) -> bool:
        from terminal_agent.utils.display import display_approval_prompt
        # Run blocking CLI prompt in worker thread to prevent freezing async event loop
        return await asyncio.to_thread(display_approval_prompt, reason)


class AutoApprovalHandler(ApprovalHandler):
    """Deterministic automated approval strategy for tests, CI, and evaluation harnesses."""

    def __init__(self, approve: bool = True) -> None:
        self.approve = approve

    async def request_approval(self, reason: str) -> bool:
        return self.approve
