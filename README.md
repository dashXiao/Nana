# NanaShop Agent

电商客服多轮对话 Agent · 自研 ReAct 循环 + 双模型分层 + 5 工具 + 两层记忆 + 多层防御。

## Quickstart

### 配置要求

- Python ≥ 3.11
- [uv](https://github.com/astral-sh/uv)
- 阿里云百炼（DashScope）API Key — [申请入口](https://bailian.console.aliyun.com/)

### 四步跑起来

```bash
# 1. 拉代码
git clone <repo-url> nana && cd nana

# 2. 配置 API Key
cp .env.example .env

# 3. 装依赖
uv sync

# 4. 启动 Web UI（http://127.0.0.1:7860）
uv run python scripts/webui.py
```

### 命令行交互

```bash
uv run python scripts/chat.py
```
