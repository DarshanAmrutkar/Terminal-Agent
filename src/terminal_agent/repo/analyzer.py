from __future__ import annotations
import os
import re
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

try:
    import tree_sitter  # type: ignore
    import tree_sitter_languages  # type: ignore
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

@dataclass
class Symbol:
    name: str
    kind: str
    signature: str
    line: int
    parent: Optional[str] = None

class RepoAnalyzer:
    """Analyzes a repository to extract structural symbols."""
    
    IGNORE_DIRS = {'.git', 'node_modules', '__pycache__', '.venv', 'venv', 'dist', 'build'}
    MAX_FILES = 500
    
    SUPPORTED_EXTENSIONS = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.go': 'go',
        '.rs': 'rust',
        '.java': 'java',
        '.c': 'c',
        '.cpp': 'cpp',
        '.h': 'cpp',
    }

    def __init__(self, root_path: str):
        self.root_path = root_path
        self._cache: Dict[str, List[Symbol]] = {}

    def _get_file_hash(self, filepath: str) -> str:
        """Returns the MD5 hash of the file content."""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def analyze(self) -> Dict[str, List[Symbol]]:
        """Analyzes the repository and returns a mapping of file path to symbols."""
        results: Dict[str, List[Symbol]] = {}
        files_processed = 0

        for root, dirs, files in os.walk(self.root_path):
            # Prune ignored directories
            dirs[:] = [d for d in dirs if d not in self.IGNORE_DIRS]

            for file in files:
                if files_processed >= self.MAX_FILES:
                    return results

                ext = os.path.splitext(file)[1]
                if ext not in self.SUPPORTED_EXTENSIONS:
                    continue

                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, self.root_path)
                rel_path = rel_path.replace(os.sep, '/')

                try:
                    symbols = self._parse_file(filepath, ext)
                    if symbols:
                        results[rel_path] = symbols
                    files_processed += 1
                except Exception as e:
                    print(f"Error parsing {filepath}: {e}")

        return results

    def _parse_file(self, filepath: str, ext: str) -> List[Symbol]:
        """Parses a single file and extracts symbols."""
        file_hash = self._get_file_hash(filepath)
        cache_key = f"{filepath}_{file_hash}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        language = self.SUPPORTED_EXTENSIONS[ext]
        
        if TREE_SITTER_AVAILABLE:
            symbols = self._parse_with_tree_sitter(content, language)
        else:
            symbols = self._parse_with_regex(content, language)

        self._cache[cache_key] = symbols
        return symbols

    def _parse_with_tree_sitter(self, content: str, language_name: str) -> List[Symbol]:
        """Extracts symbols using tree-sitter."""
        try:
            language = tree_sitter_languages.get_language(language_name)
            parser = tree_sitter.Parser()
            parser.set_language(language)
            tree = parser.parse(bytes(content, 'utf8'))
            
            # Simple implementation that can be expanded with proper queries per language
            # For brevity in this exercise, we will use a naive approach or simplified queries
            symbols = []
            
            # In a real implementation, you'd use language-specific queries
            # Example queries for Python
            if language_name == 'python':
                query_str = """
                (class_definition name: (identifier) @class.name) @class
                (function_definition name: (identifier) @func.name) @func
                """
                query = language.query(query_str)
                captures = query.captures(tree.root_node)
                
                # Simplified parsing logic
                for node, capture_name in captures:
                    if capture_name == 'class.name':
                        symbols.append(Symbol(
                            name=node.text.decode('utf8'),
                            kind='class',
                            signature=f"class {node.text.decode('utf8')}",
                            line=node.start_point[0] + 1
                        ))
                    elif capture_name == 'func.name':
                        symbols.append(Symbol(
                            name=node.text.decode('utf8'),
                            kind='function',
                            signature=f"def {node.text.decode('utf8')}",
                            line=node.start_point[0] + 1
                        ))
            
            return symbols
        except Exception:
            # Fallback to regex if tree-sitter fails for this language
            return self._parse_with_regex(content, language_name)

    def _parse_with_regex(self, content: str, language: str) -> List[Symbol]:
        """Extracts symbols using regex patterns."""
        symbols = []
        lines = content.splitlines()
        
        # Current class context for parent mapping
        current_class = None
        class_indent = -1
        
        for i, line in enumerate(lines):
            line_num = i + 1
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            
            if current_class and indent <= class_indent:
                current_class = None
                class_indent = -1
                
            if language == 'python':
                class_match = re.match(r'^class\s+([A-Za-z0-9_]+)', stripped)
                if class_match:
                    name = class_match.group(1)
                    current_class = name
                    class_indent = indent
                    symbols.append(Symbol(name=name, kind='class', signature=stripped, line=line_num))
                    continue
                    
                func_match = re.match(r'^(?:async\s+)?def\s+([A-Za-z0-9_]+)\s*\(', stripped)
                if func_match:
                    name = func_match.group(1)
                    kind = 'method' if current_class else 'function'
                    symbols.append(Symbol(name=name, kind=kind, signature=stripped, line=line_num, parent=current_class))
                    
            elif language in ('javascript', 'typescript'):
                class_match = re.match(r'^(?:export\s+)?(?:default\s+)?class\s+([A-Za-z0-9_]+)', stripped)
                if class_match:
                    name = class_match.group(1)
                    symbols.append(Symbol(name=name, kind='class', signature=stripped, line=line_num))
                    continue
                    
                func_match = re.match(r'^(?:export\s+)?(?:async\s+)?function\s+([A-Za-z0-9_]+)\s*\(', stripped)
                if func_match:
                    name = func_match.group(1)
                    symbols.append(Symbol(name=name, kind='function', signature=stripped, line=line_num))
                    continue
                    
                arrow_match = re.match(r'^(?:export\s+)?(?:const|let|var)\s+([A-Za-z0-9_]+)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z0-9_]+)\s*=>', stripped)
                if arrow_match:
                    name = arrow_match.group(1)
                    symbols.append(Symbol(name=name, kind='function', signature=stripped, line=line_num))
                    
            else:
                # Generic fallback
                class_match = re.match(r'^(?:public\s+|private\s+|protected\s+)?class\s+([A-Za-z0-9_]+)', stripped)
                if class_match:
                    name = class_match.group(1)
                    symbols.append(Symbol(name=name, kind='class', signature=stripped, line=line_num))
                    continue
                    
                func_match = re.match(r'^(?:public\s+|private\s+|protected\s+)?(?:static\s+)?(?:[A-Za-z0-9_<>\[\]]+\s+)?([A-Za-z0-9_]+)\s*\(', stripped)
                if func_match and not stripped.endswith(';'):
                    name = func_match.group(1)
                    # Ignore common control structures
                    if name not in ('if', 'for', 'while', 'switch', 'catch'):
                        symbols.append(Symbol(name=name, kind='function', signature=stripped, line=line_num))

        return symbols
