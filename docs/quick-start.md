### 环境准备
#### Python ≥ 3.11  
查看Python版本

  ```bash
  python3 --version
  ```

#### uv
安装命令  
macOS / Linux

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

Windows (PowerShell)

  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

#### 阿里云百炼（DashScope）API Key
[申请地址](https://bailian.console.aliyun.com/)

### 快速开始

```bash
git clone <repo-url> nana && cd nana

# 配置 API Key
cp .env.example .env

# 装依赖
uv sync

# 启动 Web UI（http://127.0.0.1:7860）
uv run python scripts/webui.py
```

#### 命令行交互

```bash
uv run python scripts/chat.py
```

### 更改商品信息

#### 价格 / 库存 / 字段 / 订单 / 用户
`scripts/seed_data.py`

#### 商品详情文案
`knowledge/products/P00X.md`

#### 路由层商品 ID 映射
新增或重命名商品时，同步 `agent/router.py` 里的 `_CATALOG_SNIPPET`