from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

class Role(str, Enum):
    SYSTEM = 'system'
    USER = 'user'
    ASSISTANT = 'assistant'
    TOOL_RESULT = 'tool_result'

@dataclass
class ToolCall:
    id: str              # unique ID for the tool call
    name: str            # tool name
    arguments: dict[str, Any]      # parsed arguments

@dataclass
class ToolResultContent:
    tool_call_id: str    # matches ToolCall.id
    output: str          # tool execution output
    is_error: bool = False

@dataclass 
class Message:
    role: Role
    content: str | None = None           # text content
    tool_calls: list[ToolCall] | None = None  # for assistant messages with tool use
    tool_results: list[ToolResultContent] | None = None  # for tool result messages
    
    @classmethod
    def system(cls, content: str) -> 'Message':
        return cls(role=Role.SYSTEM, content=content)
    
    @classmethod  
    def user(cls, content: str) -> 'Message':
        return cls(role=Role.USER, content=content)
    
    @classmethod
    def assistant(cls, content: str | None = None, tool_calls: list[ToolCall] | None = None) -> 'Message':
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)
    
    @classmethod
    def tool_result(cls, results: list[ToolResultContent]) -> 'Message':
        return cls(role=Role.TOOL_RESULT, tool_results=results)

@dataclass
class StreamEvent:
    """Events emitted during streaming.

    Event types:
      - 'text_delta'      : a chunk of text from the assistant
      - 'tool_call_start' : the LLM started a tool call (carries initial ToolCall)
      - 'tool_call_delta' : partial JSON being accumulated for tool arguments
      - 'tool_call_end'   : tool call is complete (carries finalized ToolCall with arguments)
      - 'usage'           : real token counts from the API (input_tokens, output_tokens)
      - 'message_end'     : stream has finished
    """
    type: str
    text: str | None = None
    tool_call: ToolCall | None = None
    usage: dict | None = None  # e.g. {'input_tokens': 512, 'output_tokens': 128}

@dataclass
class LLMResponse:
    """Complete response from the LLM."""
    message: Message
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None  # 'end_turn', 'tool_use', 'max_tokens'
