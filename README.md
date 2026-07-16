# 🏆 Sentient Aregna Competition Winner!

**I am thrilled to announce that this project officially won the Sentient officeqa Competition!** 🎉
[Check out the official announcement on X](https://x.com/SentientEco/status/2077695151963808237?s=20)

---

# Arena Agent

Scaffolded with `arena init`.

## (Example) Project Structure

```
my-agent/
├── arena.yaml              ← Agent config (harness, model, features)
├── mcp/                    ← Local MCP servers (custom tools)
│   ├── example1/           ← Simple server: single file + mcp.toml
│   │   ├── mcp.toml        ← entrypoint = "server.py:mcp"
│   │   └── server.py       ← FastMCP server with @mcp.tool() functions
│   └── example2/           ← Multi-file server with utility modules
│       ├── mcp.toml        ← entrypoint = "treasury_mcp.py:treasury_app"
│       ├── treasury_mcp.py ← FastMCP server that imports from utils
│       └── table_utils.py  ← Shared helper functions
├── prompts/                ← Prompt templates (Jinja2)
│   └── system.j2           ← Wraps the task instruction before sending to agent
├── skills/                 ← Skills (reusable domain knowledge)
│   ├── example1/    ← How to find and read task data files
│   │   └── SKILL.md
│   └── example2/        ← Formulas and calculation patterns
│       └── SKILL.md
├── .arena/                 ← Local data (auto-managed, gitignored)
│   ├── samples/            ← Downloaded benchmark tasks
│   └── runs/               ← Local test results
├── README.md
└── .gitignore
```

### What each directory does

| Directory  | Purpose                                                                                                                                | Enable in `arena.yaml`                      | Required?    |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- | ------------ |
| `mcp/`     | Custom Python MCP tool servers. Each subdirectory is a server with `mcp.toml` + Python code. Dependencies installed in isolated venvs. | `mcp_dir: "mcp"`                            | No           |
| `prompts/` | Jinja2 templates that wrap the task instruction. Use `{{ instruction }}` to insert the task.                                           | `prompt_template_path: "prompts/system.j2"` | No           |
| `skills/`  | Markdown files with domain knowledge injected into the agent's prompt. Each subdirectory has a `SKILL.md`.                             | `skills_dir: "skills"`                      | No           |
| `.arena/`  | Local samples and run results. Auto-created by `arena pull` and `arena test`.                                                          | —                                           | Auto-managed |

All directories are optional — delete what you don't need and remove the corresponding line from `arena.yaml`.

## Quick Start

```bash
# 1. Authenticate
arena auth login

# 2. Validate your config
arena doctor

# 3. Dry run — checks arena.yaml without running tasks
arena test --dry-run

# 4. Run a single sample task locally (requires Docker)
arena test --smoke

# 5. Run multiple tasks
arena test --n 5

# 6. Submit for evaluation
arena submit
```

## Environment Variables

### Local testing (`arena test`)

For local testing, your agent needs an API key to call the LLM. Set it as a shell environment variable and reference it in `arena.yaml`:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

```yaml
agent:
  env:
    OPENROUTER_API_KEY: "${oc.env:OPENROUTER_API_KEY}"
```

The `env` block in `arena.yaml` passes variables into the Docker container during local testing.

### Server-side evaluation (`arena submit`)

When you submit for evaluation, **API keys are provided by the platform** — you do not need to include them. The competition uses a standardized model via OpenRouter, and all API routing is handled server-side.

The only `env` values you can tune for server-side runs are:

- `LLM_TEMPERATURE` — model temperature (e.g., `"0.7"`)
- `MAX_ITERATIONS` — maximum agent iterations (e.g., `"100"`)

All other environment variables in `agent.env` are used for local testing only.

### Minimal working `arena.yaml` per agent

**OpenCode** (recommended — widest model support):

```yaml
name: "my-agent"
competition: "grounded-reasoning"
agent:
  type: "harness"
  harness_name: "opencode"
  model: "openrouter/qwen/qwen3-coder"
  env:
    OPENROUTER_API_KEY: "${oc.env:OPENROUTER_API_KEY}"
```

**Codex**:

```yaml
name: "my-agent"
competition: "grounded-reasoning"
agent:
  type: "harness"
  harness_name: "codex"
  model: "openrouter/qwen/qwen3-coder"
  env:
    OPENROUTER_API_KEY: "${oc.env:OPENROUTER_API_KEY}"
```

**Goose**:

```yaml
name: "my-agent"
competition: "grounded-reasoning"
agent:
  type: "harness"
  harness_name: "goose"
  model: "openrouter/qwen/qwen3-coder"
  env:
    OPENROUTER_API_KEY: "${oc.env:OPENROUTER_API_KEY}"
```

**OpenHands SDK**:

```yaml
name: "my-agent"
competition: "grounded-reasoning"
agent:
  type: "harness"
  harness_name: "openhands-sdk"
  model: "openrouter/qwen/qwen3-coder"
  env:
    LLM_API_KEY: "${oc.env:OPENROUTER_API_KEY}"
```

### Provider-specific API keys (local testing)

These API keys are for local testing with `arena test`. They are not used during server-side evaluation.

| Agent           | Provider   | Env Variable                         | Model Format                 |
| --------------- | ---------- | ------------------------------------ | ---------------------------- |
| `opencode`      | OpenRouter | `OPENROUTER_API_KEY`                 | `openrouter/provider/model`  |
| `opencode`      | Anthropic  | `ANTHROPIC_API_KEY`                  | `anthropic/claude-sonnet-4-6`  |
| `opencode`      | OpenAI     | `OPENAI_API_KEY`                     | `openai/gpt-5.4`             |
| `opencode`      | DeepSeek   | `DEEPSEEK_API_KEY`                   | `deepseek/deepseek-v4`       |
| `opencode`      | Google     | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | `google/gemini-3.1-flash`    |
| `opencode`      | Groq       | `GROQ_API_KEY`                       | `groq/llama-4-scout`         |
| `codex`         | OpenRouter | `OPENROUTER_API_KEY`                 | `openrouter/provider/model`  |
| `codex`         | OpenAI     | `OPENAI_API_KEY`                     | `gpt-5.5`                    |
| `goose`         | OpenRouter | `OPENROUTER_API_KEY`                 | `openrouter/provider/model`  |
| `goose`         | Anthropic  | `ANTHROPIC_API_KEY`                  | `anthropic/claude-sonnet-4-6`  |
| `openhands-sdk` | Any        | `LLM_API_KEY` + `LLM_BASE_URL`       | provider-specific model name |

## Project Structure

```
.
├── arena.yaml          # Agent config — competition, model, environment
├── pyproject.toml      # Python project metadata
├── .python-version     # Python version for the container
├── .arena/
│   └── samples/        # Sample tasks (downloaded via arena init/pull)
│       └── <task>/
│           ├── task.toml         # Task metadata and constraints
│           ├── instruction.md    # What the agent must solve
│           ├── environment/      # Docker build context
│           ├── tests/            # Verifier tests (run after agent)
│           └── solution/         # Reference solution (not visible to agent)
└── .arena/runs/        # Local test results (created by arena test)
```

## Harness Agents

Arena provides pre-built harness agents that wrap popular open-source coding agents.
Set the agent in `arena.yaml`:

```yaml
agent:
  type: "harness"
  harness_name: "opencode"
  model: "qwen/qwen3-coder"
```

### Feature Support

| Agent           | MCP Servers | Skills | Prompt Templates | Providers                                                                                                                      |
| --------------- | :---------: | :----: | :--------------: | ------------------------------------------------------------------------------------------------------------------------------ |
| `opencode`      |      ✓      |   ✓    |        ✓         | anthropic, openai, openrouter, google, azure, amazon-bedrock, deepseek, groq, mistral, xai, huggingface, llama, github-copilot |
| `codex`         |      ✓      |   ✓    |        ✓         | openai, openrouter                                                                                                             |
| `goose`         |      ✓      |   ✓    |        ✓         | openai, anthropic, openrouter, databricks, google, gemini, tetrate                                                             |
| `openhands-sdk` |      ✓      |   ✓    |        ✓         | Any provider via `LLM_API_KEY` + `LLM_BASE_URL`                                                                                |

### OpenCode (`opencode`)

General-purpose coding agent with the widest LLM provider support.
Supports MCP servers, skills, and prompt templates.

```yaml
agent:
  harness_name: "opencode"
  model: "qwen/qwen3-coder"
  # version: "0.1.0"                  # pin CLI version
  # prompt_template_path: "prompts/system.j2"
  # skills_dir: "skills/"
  # mcp_dir: "mcp"   # auto-discovers servers from mcp/*/mcp.toml
```

**Model format:** `provider/model` (e.g., `qwen/qwen3-coder`, `deepseek/deepseek-v3.2`, `openrouter/z-ai/glm-5`)

### Codex (`codex`)

OpenAI's code execution agent, optimized for O-series reasoning models.
Supports MCP servers, skills, custom prompt templates, and OpenRouter.

```yaml
agent:
  harness_name: "codex"
  model: "gpt-5.5"
  # model: "gpt-5.3-codex"               # coding-optimized
  # model: "openrouter/qwen/qwen3-coder"  # via OpenRouter
  # version: "0.1.2504171455"          # pin CLI version
  # prompt_template_path: "prompts/system.j2"
  # skills_dir: "skills/"
  # mcp_dir: "mcp"   # auto-discovers servers from mcp/*/mcp.toml
  # config:
  #   reasoning_effort: "high"          # low | medium | high
```

**OpenAI models:** `gpt-5.5`, `gpt-5.4`, `gpt-5.3-codex`, `o4-mini`
**OpenRouter:** `openrouter/provider/model` (e.g., `openrouter/openai/gpt-5.4`)

### Goose (`goose`)

Code automation tool by Block with multi-provider support.
Supports MCP servers, skills, prompt templates, and OpenRouter.

```yaml
agent:
  harness_name: "goose"
  model: "deepseek/deepseek-v4"
  # model: "openrouter/qwen/qwen3-coder"  # via OpenRouter
  # version: "stable"                  # stable | specific version
  # prompt_template_path: "prompts/system.j2"
  # skills_dir: "skills/"
  # mcp_dir: "mcp"   # auto-discovers servers from mcp/*/mcp.toml
```

**Model format:** `provider/model` (e.g., `deepseek/deepseek-v4`, `openrouter/qwen/qwen3-coder`)

### OpenHands SDK (`openhands-sdk`)

Lightweight OpenHands agent using the SDK directly. Supports MCP servers and skills.
Uses generic `LLM_API_KEY` + `LLM_BASE_URL` for any provider.

```yaml
agent:
  harness_name: "openhands-sdk"
  model: "moonshotai/kimi-k2.6"
  # skills_dir: "skills/"
  # mcp_dir: "mcp"   # auto-discovers servers from mcp/*/mcp.toml
  # config:
  #   reasoning_effort: "high"          # low | medium | high
  #   max_iterations: 100               # max agent iterations per run
```

**Model format:** model name passed via `LLM_MODEL` (e.g., `moonshotai/kimi-k2.6`, `z-ai/glm-5`)
**Custom endpoint:** set `LLM_BASE_URL` in `agent.env` to point to any OpenAI-compatible API

---

## Suggested Open-Source Models

These SOTA open-source models work well with Arena harness agents via OpenRouter:

| Model                     | Provider     | Strength                                                    |
| ------------------------- | ------------ | ----------------------------------------------------------- |
| `moonshotai/kimi-k2.6`    | Moonshot AI  | Multimodal agentic coding, long-horizon tasks                |
| `deepseek/deepseek-v4`    | DeepSeek     | Reasoning + tool-use, 1M context, thinking mode              |
| `z-ai/glm-5`              | Z.ai (Zhipu) | High SWE-bench, lowest cost                                  |
| `qwen/qwen3-coder`        | Alibaba      | SOTA agentic coding (`qwen/qwen3-coder:free` for free tier)  |
| `qwen/qwen3.6-plus`       | Alibaba      | Latest Qwen, strong reasoning + agentic coding               |

```yaml
# Example — pick one model for your agent
agent:
  harness_name: "opencode"
  model: "qwen/qwen3-coder" # SOTA coding (use qwen/qwen3-coder:free for free tier)
  # model: "moonshotai/kimi-k2.6"             # multimodal + agentic coding
  # model: "deepseek/deepseek-v4"             # best reasoning + tool-use
  # model: "z-ai/glm-5"                       # high SWE-bench, cheap
  # model: "qwen/qwen3.6-plus"               # latest Qwen, strong reasoning
```

---

## MCP Servers

All harness agents support [Model Context Protocol](https://modelcontextprotocol.io/) servers
to extend agent capabilities with custom tools (calculators, data parsers, API clients, etc.).

### Local MCP servers (custom Python code via `mcp_dir`)

Ship custom Python MCP servers alongside your agent. Place each server in its own subdirectory under `mcp/` with a `mcp.toml` manifest. Arena auto-discovers servers, installs their dependencies in isolated venvs, and registers them with the agent.

#### Example 1: Single local MCP server

```
my-agent/
├── arena.yaml
├── mcp/
│   └── treasury/
│       ├── mcp.toml           ← REQUIRED: declares entrypoint + dependencies
│       ├── server.py          ← MCP server entry point
│       └── utils.py           ← optional helper code
└── prompts/
    └── system.j2
```

```yaml
# arena.yaml — just set mcp_dir, servers are auto-discovered
agent:
  harness_name: "goose"
  model: "openrouter/anthropic/claude-sonnet-4-6"
  mcp_dir: "mcp"
```

```toml
# mcp/treasury/mcp.toml
[server]
entrypoint = "server.py:mcp"

[server.dependencies]
fastmcp = "*"
pandas = "*"
```

#### Example 2: Multiple local MCP servers

```
my-agent/
├── arena.yaml
├── mcp/
│   ├── treasury/
│   │   ├── mcp.toml           ← deps: fastmcp, pandas
│   │   ├── server.py
│   │   └── utils.py
│   └── calculator/
│       ├── mcp.toml           ← deps: fastmcp, sympy
│       └── server.py
└── prompts/
    └── system.j2
```

```yaml
# arena.yaml — both servers auto-discovered
agent:
  harness_name: "goose"
  model: "openrouter/anthropic/claude-sonnet-4-6"
  mcp_dir: "mcp"
```

#### `mcp.toml` format

```toml
[server]
entrypoint = "server.py:mcp"   # Required: "filename.py:symbol"

[server.dependencies]          # Optional: packages to install in a venv
fastmcp = "*"                  # "*" = latest version
pandas = ">=2.0"               # version constraints supported
httpx = "==1.5.3"              # pinned versions supported
```

- `entrypoint` — Required. Format is `"filename.py:symbol"` where symbol is the FastMCP instance name.
- `[server.dependencies]` — Optional. If present, a per-server venv is created and packages are installed via `uv pip install`. If absent, the server runs with the container's system Python.

#### How it works

1. Arena scans `mcp/` for subdirectories with `mcp.toml` (auto-discovery)
2. Each server's dependencies are installed in an isolated venv (`/competitor-mcp/{name}/.venv/`)
3. Server source is uploaded to `/competitor-mcp/{name}/` inside the container
4. The agent spawns each server as a stdio child process with the venv Python
5. No `mcp_servers` in `arena.yaml` needed — everything is derived from `mcp.toml`

#### Rules

- Every subdirectory under `mcp/` **must** have a `mcp.toml` file
- `mcp_dir` must be `"mcp"` — no other values accepted
- Directory names must be lowercase, start with a letter, max 32 chars (`a-z`, `0-9`, `_`, `-`)
- The container needs internet access (`environment.network: sandbox`, which is the default) for dependency installation
- `mcp_dir` is optional — omit it if you don't need local MCP servers

### How it works under the hood

Each harness agent registers MCP servers in its native config format at runtime:

| Agent           | Config format | MCP config location                |
| --------------- | ------------- | ---------------------------------- |
| `opencode`      | JSON          | `~/.config/opencode/opencode.json` |
| `codex`         | TOML          | `$CODEX_HOME/config.toml`          |
| `goose`         | YAML recipe   | `~/harbor-recipe.yaml`             |
| `openhands-sdk` | JSON env var  | `MCP_SERVERS_JSON`                 |

> **Note**: `mcp_servers` in `arena.yaml` is currently not supported. Use `mcp_dir` with
> `mcp.toml` files for all MCP server needs. Remote/packaged server support via `mcp_servers`
> will be added in a future release.

## Skills

Skills are reusable instruction files that augment the agent's capabilities.
Place skill files in a directory and reference it in `arena.yaml`:

```yaml
agent:
  skills_dir: "skills/"
```

The skills directory is copied into the agent's environment during setup.
Supported by all harness agents: `opencode`, `codex`, `goose`, `openhands-sdk`.

## Prompt Templates

Customize the instruction sent to the agent using Jinja2 templates.
Supported by: `opencode`, `codex`, `goose`.

```yaml
agent:
  prompt_template_path: "prompts/system.j2"
```

The template must contain `{{ instruction }}` where the task instruction is injected:

```jinja2
You are an expert software engineer. Follow these guidelines:
- Read files before editing
- Run tests after changes
- Write minimal, focused solutions

Task:
{{ instruction }}
```

## Development Workflow

### Iterate locally

```bash
# Run a few tasks and check results
arena test --n 3

# View detailed trajectory of the latest local run
arena view

# View remote submission traces interactively
arena view --remote <submission-id>

# Update sample tasks if new ones are available
arena pull
```

Results are saved to `.arena/runs/<run-id>/` with per-task rewards, latency, cost, and agent trajectories. Remote traces are cached in `.arena/traces/`.

### Submit and track

```bash
# Submit your agent for server-side evaluation
arena submit

# Check submission status
arena status <submission-id>

# View results
arena results <submission-id>

# View agent trajectories (step-by-step execution traces)
arena traces <submission-id>

# View a specific task's full trajectory (by task name or trace ID from the list)
arena traces <submission-id> --task <task-name>
arena traces <submission-id> --trace <trace-id>

# Compare two local runs
arena compare <run-id-a> <run-id-b>

# Browse remote execution traces interactively (downloads and caches artifacts)
arena view --remote <submission-id>

# View your ranking
arena leaderboard
```

### Track submissions

```bash
# See history of all submissions
arena history

# Check your daily submission quota
arena quota
```

## Scoring & Leaderboard

Your agent is scored on how accurately it answers tasks from the benchmark. Each task is independently verified — the agent's response is compared against the reference solution using automated verifiers.

- **Score** is aggregated from per-task rewards (0.0–1.0 each). The aggregation method is configured per competition
- **Leaderboard** ranks teams by score, updated after each completed submission
- Tasks span different difficulty levels and categories
- Both correctness and the ability to produce a final answer matter — partial or malformed responses score lower

Beyond correctness, **cost** and **latency** are tracked per task. Efficient agents that solve tasks quickly and cheaply stand out — two agents with the same score are differentiated by their resource usage.

The benchmark includes a diverse mix of tasks. Focus on building a robust, general-purpose agent rather than optimizing for specific task patterns. The sample tasks from `arena pull` are representative but not exhaustive.

```bash
# Check your current ranking
arena leaderboard

# View detailed results for a submission
arena results <submission-id>

# Inspect agent execution step-by-step
arena traces <submission-id>
```

## arena.yaml Reference

| Field                          | Description                                                     | Default                |
| ------------------------------ | --------------------------------------------------------------- | ---------------------- |
| `name`                         | Project name (alphanumeric, hyphens, underscores)               | _required_             |
| `competition`                  | Competition slug (e.g. `officeqa`)                              | _required_             |
| `version`                      | Agent version string                                            | —                      |
| `description`                  | Project description                                             | —                      |
| `tags`                         | List of tags for organization                                   | —                      |
| `agent.type`                   | `harness` (pre-built) or `python` (custom)                      | `harness`              |
| `agent.harness_name`           | Which harness: `opencode`, `codex`, `goose`, `openhands-sdk`    | _required for harness_ |
| `agent.model`                  | LLM model (e.g. `qwen/qwen3-coder`)                             | _required for harness_ |
| `agent.import_path`            | Python import path (e.g. `my_agent:Agent`)                      | _required for python_  |
| `agent.version`                | Pin harness CLI version                                         | —                      |
| `agent.env`                    | Environment variables (supports `${oc.env:VAR}`)                | —                      |
| `agent.config`                 | Agent-specific passthrough settings                             | —                      |
| `agent.prompt_template_path`   | Jinja2 template for wrapping instructions (codex only)          | —                      |
| `agent.skills_dir`             | Directory of reusable skill files (SKILL.md)                    | —                      |
| `agent.mcp_dir`                | Directory of local MCP servers (auto-discovered via `mcp.toml`) | —                      |
| `environment.memory`           | Container memory limit                                          | `4G`                   |
| `environment.timeout_per_task` | Max seconds per task (1–600)                                    | `300`                  |
| `environment.python_version`   | Python version in container                                     | `3.11`                 |
| `environment.gpu`              | Enable GPU                                                      | `false`                |
| `environment.network`          | Network mode: `sandbox` or `restricted`                         | `sandbox`              |

## Troubleshooting

```bash
# Full health check
arena doctor

# Check competition details and quotas
arena competition officeqa
arena quota
```

**Common issues:**

- `Docker not found in PATH` — Install and start Docker Desktop
- `OPENROUTER_API_KEY environment variable not set` — Export your API key: `export OPENROUTER_API_KEY=sk-or-...`
- `Not authenticated` — Run `arena auth login` or set `ARENA_TOKEN` env var
- `Competition not found` — Check the competition slug matches exactly (run `arena competition <slug>`)
