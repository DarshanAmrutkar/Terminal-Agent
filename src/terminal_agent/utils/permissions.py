from enum import Enum
import re
from terminal_agent.core.config import AgentConfig

class SafetyLevel(Enum):
    SAFE = "SAFE"
    NEEDS_APPROVAL = "NEEDS_APPROVAL"
    BLOCKED = "BLOCKED"

def is_read_only(command: str) -> bool:
    """Helper to check if a command is generally read-only."""
    read_only_starts = ("ls", "cat", "echo", "pwd", "grep", "find", "head", "tail", "less", "more")
    cmd_start = command.strip().split()[0] if command.strip() else ""
    return cmd_start in read_only_starts

def classify_command(command: str, config: AgentConfig) -> SafetyLevel:
    """
    Classify a shell command's safety level based on configuration.
    """
    cmd = command.strip()
    
    # Check blocked patterns first
    for pattern in config.blocked_patterns:
        if pattern in cmd:
            return SafetyLevel.BLOCKED

    # Check safe commands
    cmd_base = cmd.split()[0] if cmd else ""
    if cmd_base in config.safe_commands or cmd in config.safe_commands:
        return SafetyLevel.SAFE
        
    if is_read_only(cmd):
        return SafetyLevel.SAFE

    # Depending on permission mode, we might auto-approve tests
    if config.permission_mode == "auto-test" and ("pytest" in cmd or "npm test" in cmd):
        return SafetyLevel.SAFE
        
    if config.permission_mode == "yolo":
        return SafetyLevel.SAFE

    return SafetyLevel.NEEDS_APPROVAL
