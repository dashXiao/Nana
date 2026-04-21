# CLAUDE.md

本文件给 Claude Code 读。记录项目目标、开发原则与协作约定。

## 1. 项目概览

**NanaShop Agent** —— 电商客服多轮对话 Agent，作者 xyh 的求职作品。

- **单一事实来源是 `PLAN.md`**（根目录）——所有架构决策、模块清单、进度、决策日志都在那里
- **技术栈**：qwen-plus（经 DashScope OpenAI 兼容端点）/ 自研 ReAct 循环 / SQLite + 文件系统 / 无向量库 / 无运行期 trace
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

### 3.1 Commit 规范

- **一次代码改动 = 一个 commit**，方便用户逐个 review
- 提交信息用**中文**，突出"做了什么 / 为什么"；首行 ≤ 70 字
- 涉及 PLAN.md 的更新与代码放同一个 commit，不要拆

### 3.2 Feedback 循环（重要）

当用户对已提交的 commit 提出修改意见时：

- **不要堆一个"修复"commit 在上面**
- **回退原 commit，用修正后的内容替代**，保持 git log 是一条"已达成共识"的干净历史
- 标准操作（针对最新一个 commit）：
  ```bash
  git reset --soft HEAD~1    # 撤回 HEAD，改动留在暂存区
  # 应用修正
  git commit -m "新的说明"    # 重新提交
  ```
- 如果 bad commit 不在 HEAD（上面已有后续 commit），先和用户确认处理方式，再动手

### 3.3 测试纪律

- 每个工具都要有对应 pytest 用例；覆盖正常路径 + 关键错误路径 + 关键架构不变式
- 提交前跑 `uv run pytest -v` 确保全绿
- 涉及 LLM 调用的冒烟脚本放 `scripts/smoke_*.py`，不入 pytest（成本与稳定性）

## 4. 常用命令

```bash
uv sync                              # 安装/同步依赖
uv run python scripts/seed.py        # 初始化/重置开发库（db/nana.db）
uv run pytest -v                     # 跑全部测试
uv run python scripts/smoke_llm.py   # qwen-plus tool-use 冒烟测试
```

## 5. 关键约束（与 PLAN.md §2 保持一致）

- LLM 只用 qwen-plus，不做多模型路由
- 存储只有 SQLite（仅交易数据） + 文件系统（md）；**无向量库、无 trace 表**
- 知识分层：交易→DB / 商品→`knowledge/products/P*.md` / 政策→`knowledge/policies.md` / 记忆→`memory/sessions/{user_id}/*.md`
- 种子数据全量写死：2 用户 / 3 商品 / 10 订单（含 3 条异常）
- 工具预算 6 个：`query_orders` / `track_logistics` / `initiate_refund` / `query_product` / `query_policy` / `escalate_to_human`
