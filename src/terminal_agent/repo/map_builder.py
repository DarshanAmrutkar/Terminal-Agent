from __future__ import annotations
from typing import Dict, List, Any

try:
    from terminal_agent.context.tokenizer import count_tokens
except ImportError:
    # Fallback if tokenizer is not available
    def count_tokens(text: str) -> int:
        return len(text) // 4  # Rough approximation

from terminal_agent.repo.analyzer import Symbol

class MapBuilder:
    """Builds a token-efficient string representation of a repository map."""

    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens

    def build(self, symbols: Dict[str, List[Symbol]]) -> str:
        """Builds the repository map string, fitting within the token budget."""
        
        # Sort files by number of symbols (descending) to prioritize important files
        sorted_files = sorted(symbols.items(), key=lambda item: len(item[1]), reverse=True)
        
        output_lines: List[str] = []
        current_tokens = 0
        files_included = 0
        
        for file_path, file_symbols in sorted_files:
            file_block: List[str] = [f"{file_path}:"]
            
            # Group symbols by parent to properly indent methods
            parents: Dict[str, List[Symbol]] = {'None': []}
            for sym in file_symbols:
                parent_key = sym.parent if sym.parent else 'None'
                if parent_key not in parents:
                    parents[parent_key] = []
                parents[parent_key].append(sym)
                
            # Render root level symbols (classes, standalone functions)
            for sym in parents.get('None', []):
                file_block.append(f"  {sym.signature}")
                # Render children (methods)
                if sym.kind == 'class' and sym.name in parents:
                    for child in parents[sym.name]:
                        # Make sure to indent
                        indent = "    "
                        file_block.append(f"{indent}{child.signature.strip()}")
            
            file_block.append("") # Empty line between files
            
            block_text = "\n".join(file_block)
            block_tokens = count_tokens(block_text)
            
            if current_tokens + block_tokens > self.max_tokens:
                break
                
            output_lines.extend(file_block)
            current_tokens += block_tokens
            files_included += 1
            
        remaining_files = len(sorted_files) - files_included
        if remaining_files > 0:
            output_lines.append(f"... and {remaining_files} more files")
            
        return "\n".join(output_lines).strip()
