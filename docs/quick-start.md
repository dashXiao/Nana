# Setup and quick start

## Prerequisites

### Python 3.11 or later

Check your Python version:

```bash
python3 --version
```

### uv

Install on macOS or Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install on Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### DashScope API key

Create an API key in the [Alibaba Cloud Model Studio console](https://bailian.console.aliyun.com/).

## Run the demo

```bash
git clone https://github.com/dashXiao/Nana.git
cd Nana

# Copy the configuration template.
cp .env.example .env
```

Edit `.env` and replace `sk-your-key-here` with your DashScope API key. The template also configures the API endpoint, main model, lightweight model, and local database path. Keep credentials out of version control.

### Model configuration

The default pair preserves the project's main/lightweight model split:

| Setting | Default | Purpose |
|---|---|---|
| `LLM_MODEL` | `qwen3.7-plus` | Main dialogue, tool decisions, and evaluation judging |
| `LLM_LIGHT_MODEL` | `qwen3.8-flash` | Intent classification and short-/long-term summaries |
| `LLM_ENABLE_THINKING` | `false` | Explicitly select non-thinking responses for the synchronous agent loop |

Both models support native function calling through the OpenAI-compatible Chat Completions endpoint. The client passes the thinking setting through `extra_body` and preserves the existing tool schemas and message history. Model capabilities are documented in the [official model list](https://help.aliyun.com/zh/model-studio/text-generation-model/) and [function-calling guide](https://help.aliyun.com/en/model-studio/qwen-function-calling).

Use an API key and endpoint from the same service region. The pricing values in `.env.example` are approximate Beijing/global text rates for the base context tier, excluding promotions and caching discounts. Update them for your deployment using the [official pricing page](https://docs.bailian.console.aliyun.com/zh/model-studio/model-pricing). Existing README benchmark figures describe the original model configuration and are not results for this newer pair.

### Start the interface

```bash
# Install dependencies.
uv sync

# Start the Web UI at http://127.0.0.1:7860.
uv run python scripts/webui.py
```

The Web UI and CLI reset the local demo database on startup. Session summaries are stored in `memory/sessions/` and persist across sessions.

### Command-line interface

```bash
uv run python scripts/chat.py
```

Use `/help` to list commands, `/user U002` to switch demo users, `/reset` to start a new conversation, and `/quit` to exit. Ending or switching a session saves a summary when it meets the minimum turn count.

### Example requests

- "How much are the wireless Bluetooth earbuds?"
- "What is the status of my recent orders?"
- "Please refund my most recent order."
- "What is your return policy?"
- "Talk to a human."

## Update the demo data

| Content | File |
|---|---|
| Prices, inventory, orders, and users | `scripts/seed_data.py` |
| Product descriptions | `knowledge/products/P00X.md` |
| Store policies | `knowledge/policies.md` |
| Product name-to-ID mapping used by the router | `_CATALOG_SNIPPET` in `agent/router.py` |

When adding or renaming a product, update both the seed data and the router catalog. Restart a demo interface to reseed the database. Prices remain in CNY, and the sample policies retain their original China-based shipping and payment terms.

## Verify the installation

```bash
# Automated tests use mocked model responses.
uv run pytest -q

# Live smoke check; requires API credentials.
uv run python scripts/smoke_llm.py
```
