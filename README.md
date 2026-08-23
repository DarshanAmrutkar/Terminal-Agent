# Terminal Agent

A CLI-based AI coding agent that understands repositories, inspects files, modifies code, executes commands, and iteratively solves problems — similar in concept to Claude Code or Codex CLI.

---

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
  - [NVIDIA NIM (recommended)](#nvidia-nim-recommended)
  - [Anthropic (Claude)](#anthropic-claude)
  - [OpenAI](#openai)
- [Running the Agent](#running-the-agent)
- [CLI Options](#cli-options)
- [Slash Commands](#slash-commands)
- [Permission Modes](#permission-modes)
- [Project Structure](#project-structure)

---

## Features

- **ReAct Agent Loop** — Autonomous Think → Act → Observe cycle
- **Multi-Provider** — NVIDIA NIM, Anthropic Claude, OpenAI (plug in any OpenAI-compatible endpoint)
- **Smart Tools** — File read/write, search/replace, grep, directory listing, shell execution
- **Streaming Output** — Real-time token-by-token response display
- **Safety First** — Command approval flow with configurable trust levels (`safe`, `auto-test`, `yolo`)
- **Cost Tracking** — Live token usage and estimated API cost per session
- **Rich Terminal UI** — Syntax highlighting, markdown rendering, status bar

---

## Prerequisites

- Python **3.11** or higher
- An API key for at least one supported provider (see [Configuration](#configuration))

---

## Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd "Terminal Agent"

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install the package in editable mode
pip install -e .
```

> After installation, the `agent` command is available anywhere inside the virtual environment.

---

## Configuration

All configuration is done via environment variables or a `.env` file in the project root. Copy the example first:

```bash
cp .env.example .env
```

Then edit `.env` for your chosen provider.

### NVIDIA NIM (recommended)

Get a free API key at [build.nvidia.com](https://build.nvidia.com).

```env
# .env
NVIDIA_API_KEY=nvapi-...

AGENT_PROVIDER=nvidia
AGENT_MODEL_NAME=nvidia/nemotron-3-ultra-550b-a55b
AGENT_NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
AGENT_MAX_TOKENS=15000
```

### Anthropic (Claude)

```env
# .env
ANTHROPIC_API_KEY=sk-ant-...

AGENT_PROVIDER=anthropic
AGENT_MODEL_NAME=claude-sonnet-4-20250514
AGENT_MAX_TOKENS=8192
```

### OpenAI

```env
# .env
OPENAI_API_KEY=sk-...

AGENT_PROVIDER=openai
AGENT_MODEL_NAME=gpt-4o
AGENT_MAX_TOKENS=8192
```

### Full `.env` Reference

| Variable | Default | Description |
|---|---|---|
| `NVIDIA_API_KEY` | — | NVIDIA NIM API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `AGENT_PROVIDER` | `anthropic` | Active provider: `nvidia`, `anthropic`, `openai` |
| `AGENT_MODEL_NAME` | `claude-sonnet-4-20250514` | Model identifier |
| `AGENT_MAX_TOKENS` | `8192` | Max output tokens per response |
| `AGENT_NVIDIA_BASE_URL` | `https://integrate.api.nvidia.com/v1` | NVIDIA NIM endpoint |
| `AGENT_MAX_ITERATIONS` | `25` | Max tool-use loops per user turn |
| `AGENT_PERMISSION_MODE` | `safe` | Command trust level (see below) |
| `AGENT_MAX_CONTEXT_TOKENS` | `100000` | Context window budget |

---

## Running the Agent

### Interactive mode (default)

```bash
agent
```

Starts an interactive session in the current directory. The agent can read files, run commands, and edit code in this directory.

### Point the agent at a specific project

```bash
cd /path/to/your/project
agent
```

The agent uses the current working directory as its root. Always `cd` into your project before running.

### Override provider or model at runtime

```bash
# Use NVIDIA NIM
agent --provider nvidia --model nvidia/nemotron-3-ultra-550b-a55b

# Use Anthropic
agent --provider anthropic --model claude-sonnet-4-20250514

# Use OpenAI
agent --provider openai --model gpt-4o
```

### Enable auto-approval for all commands (use with care)

```bash
agent --yolo
```

### Full launch example

```bash
cd ~/my-python-project
agent --provider nvidia --permission auto-test --max-iterations 30
```

---

## CLI Options

```
Usage: agent [OPTIONS]

Options:
  -p, --provider TEXT          LLM provider: anthropic, nvidia, openai
  -m, --model TEXT             Model name (e.g. nvidia/nemotron-3-ultra-550b-a55b)
      --max-tokens INTEGER     Max output tokens per response
      --max-iterations INTEGER Max tool-use loops per user turn
      --permission TEXT        Permission mode: safe, auto-test, yolo
      --yolo                   Auto-approve all commands (equivalent to --permission yolo)
      --help                   Show this message and exit
```

---

## Slash Commands

Once the agent is running, type these in the prompt:

| Command | Description |
|---|---|
| `/help` | Show available commands |
| `/status` | Show session info (working dir, message count, tokens used) |
| `/cost` | Show token usage and estimated API cost |
| `/model` | Show current provider, model, and context settings |
| `/clear` | Clear conversation history (start fresh without restarting) |
| `/exit` or `/quit` | End the session |

---

## Permission Modes

The agent can execute shell commands. Three trust levels control how much it can do without asking:

| Mode | Behaviour |
|---|---|
| `safe` (default) | Read-only commands (`ls`, `cat`, `grep`, `git status`, etc.) run automatically. Anything else requires your approval via a `y/n` prompt. |
| `auto-test` | Same as `safe`, but test runners (`pytest`, `npm test`, etc.) are also auto-approved. |
| `yolo` | All commands run without prompting. **Use only in throwaway environments.** Note: commands matching the blocked-patterns list (e.g. `rm -rf /`) are always rejected regardless of mode. |

Set the mode in `.env`:

```env
AGENT_PERMISSION_MODE=safe
```

Or override at launch:

```bash
agent --permission auto-test
agent --yolo
```

---

## Project Structure

```
Terminal Agent/
├── src/terminal_agent/
│   ├── core/
│   │   ├── agent.py        # ReAct agent loop (Think → Act → Observe)
│   │   ├── config.py       # Configuration (pydantic-settings, env vars)
│   │   └── session.py      # Conversation history and session state
│   ├── llm/
│   │   ├── base.py                 # LLMProvider abstract base class
│   │   ├── anthropic.py            # Anthropic Claude provider
│   │   ├── openai_compatible.py    # NVIDIA NIM / OpenAI / any compatible endpoint
│   │   └── message.py              # Message, ToolCall, StreamEvent data models
│   ├── tools/
│   │   ├── base.py         # Tool abstract base class
│   │   ├── registry.py     # Injectable tool registry
│   │   ├── read_file.py
│   │   ├── write_file.py
│   │   ├── search_replace.py
│   │   ├── grep_search.py
│   │   ├── list_directory.py
│   │   └── run_command.py
│   ├── utils/
│   │   ├── permissions.py  # PermissionChecker, SafetyLevel, classify_command
│   │   ├── cost.py         # Token usage and cost tracking
│   │   └── display.py      # Rich terminal output helpers
│   ├── cli.py              # Interactive prompt loop, slash commands
│   └── main.py             # CLI entry point (typer)
├── prompts/
│   └── system.md           # System prompt template
├── tests/
│   ├── test_registry.py
│   └── test_permissions.py
├── .env                    # Your local config (not committed)
├── .env.example            # Config template
└── pyproject.toml
```

---

## Running Tests

```bash
# Install dev dependencies (first time only)
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=terminal_agent --cov-report=term-missing
```
