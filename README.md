# Terminal Agent

A CLI-based AI coding agent that understands repositories, inspects files, modifies code, executes commands/tests, and iteratively fixes problems.

## Quick Start

```bash
# Install
pip install -e .

# Set your API key
export ANTHROPIC_API_KEY=your-key-here

# Run
agent
```

## Features

- **ReAct Agent Loop** — Autonomous reasoning and action cycle
- **Smart Tools** — File read/write, search/replace, grep, shell execution
- **Streaming Output** — Real-time LLM response display
- **Safety First** — Command approval flow with configurable trust levels
- **Rich Terminal UI** — Syntax highlighting, markdown rendering, colored diffs
