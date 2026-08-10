import tiktoken
from typing import List, Dict, Any

def count_tokens(text: str, model: str = 'cl100k_base') -> int:
    """Count tokens in a string using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Graceful fallback
        encoding = tiktoken.get_encoding('cl100k_base')
    return len(encoding.encode(text))

def count_messages_tokens(messages: List[Dict[str, Any]], model: str = 'cl100k_base') -> int:
    """Estimate token count for a message list."""
    # A rough estimation for chat formatting overhead
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            if isinstance(value, str):
                num_tokens += count_tokens(value, model)
            elif isinstance(value, list):
                # Handle tool calls/results if stored as lists of dicts
                num_tokens += count_tokens(str(value), model)
                
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens
