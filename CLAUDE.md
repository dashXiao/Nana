# CLAUDE.md

本文件给 Claude Code 读。记录项目目标、开发原则与协作约定。

## 1. 项目概览

**NanaShop Agent** —— 电商客服多轮对话 Agent，作者 xyh 的求职作品。

- **单一事实来源是 `PLAN.md`**（根目录）——所有架构决策、模块清单、进度、决策日志都在那里
- **技术栈**：qwen 系列（DashScope OpenAI 兼容端点；具体型号见 `.env`，代码不写死）/ 自研 ReAct 循环 / SQLite + 文件系统 / 无向量库 / 无运行期 trace
- **工具协议**：所有工具遵循 OpenAI function calling schema；返回统一 `{"ok": bool, "data" | "error_code" + "message"}`
- **投入分配**：Agent 能力 60% / 评估体系 30% / 文档 10%

## 2. 开发原则

### 2.1 极简至上

- **不添加非必要功能，不做多余设计**
  - 有现成方案就用现成；能硬编码就不抽象
  - 三条相似代码好过一个过早的抽象
  - 砍掉"看起来标配"的技术（RAG、向量记忆、运行期 trace），并在 PLAN.md §9 记录理由——这些决策本身就是简历亮点
- **质疑每一次"再加一个"**
  - 多一个参数 / 字段 / 工具 / 模块前，先问：真实场景需要吗？不加会出什么问题？

### 2.2 PLAN.md 是单一事实来源

- **开工前必读**：§7 当前进展 + §10 开放问题 / 下一步动作
- **收工前必更**：§4/§5 checkbox + §7 当前进展；有新决策时 §9 追加一行
- 进度 / 架构 / 决策三件事的变更都要在 PLAN.md 里留痕；口头共识不算数

## 3. 协作流程

### 3.1 Git 工作流（commit 由用户主导）

- **不要主动 commit**。一段工作完成后不要替用户执行 `git commit`，等用户统一检阅暂存区内容后自己决定何时提交。
- **新建文件后立刻 `git add <path>`**：写完一个新文件（`Write` 一个此前不存在的路径）就把它加入暂存区，让它在 `git status` / IDE diff 里跟改动一起显示。
- **修改已有文件不需要主动 `git add`**：`Edit` 已有文件会自然出现在 `git status` 的 modified 里，用户自己决定是否 stage。
- **只在用户明确说"commit / 提交"时才提交**，此时仍遵循：
  - 提交信息用**中文**，突出"做了什么 / 为什么"；首行 ≤ 70 字
  - 涉及 PLAN.md 的更新与代码放同一个 commit，不要拆

### 3.2 测试纪律

- 每个工具都要有对应 pytest 用例；覆盖正常路径 + 关键错误路径 + 关键架构不变式
- 交给用户检阅前跑 `uv run pytest -v` 确保全绿
- 涉及 LLM 调用的冒烟脚本放 `scripts/smoke_*.py`，不入 pytest（成本与稳定性）

## 4. 常用命令

```bash
uv sync                              # 安装/同步依赖
uv run python scripts/seed.py        # 初始化/重置开发库（db/nana.db）
uv run pytest -v                     # 跑全部测试
uv run python scripts/smoke_llm.py   # 主模型 tool-use 冒烟测试
```

## 5. 关键约束（与 PLAN.md §2 保持一致）

- LLM 只用 qwen 系列（具体型号见 `.env`），不做跨供应商路由；主 + 轻量两档同一供应商分层（PLAN §9 #11）
- **模型名字面量只允许出现在 `.env` / `.env.example`**——代码、注释、docs、测试 fixture 都不写死，便于切换 qwen 子型号不引发混乱
- 存储只有 SQLite（仅交易数据） + 文件系统（md）；**无向量库、无 trace 表**
- 知识分层：交易→DB / 商品→`knowledge/products/P*.md` / 政策→`knowledge/policies.md` / 记忆→`memory/sessions/{user_id}/*.md`
- 种子数据全量写死：2 用户 / 3 商品 / 10 订单（含 3 条异常）
- 工具预算 5 个：`query_orders` / `initiate_refund` / `query_product` / `query_policy` / `escalate_to_human`
