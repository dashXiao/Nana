# NanaShop Agent · 开发计划与进度

> **本文件是项目的单一事实来源（Single Source of Truth）**
> 每次开始工作前先读它；每次有新决策/进展立刻更新它。

- **Last updated**: 2026-04-21
- **Current phase**: W1 进行中（数据层已落地）
- **Changelog**:
  - 2026-04-20 初稿
  - 2026-04-20 订单模型调整：加 `shipped_at` 字段；状态合并 `refunding/refunded → refunded`
  - 2026-04-20 去除 `recommend_products` 工具，工具总数 5 → 4
  - 2026-04-20 知识分层定稿：交易数据→DB，商品描述→每商品一 md，政策→统一 `policies.md`；加回 `query_product` 工具（总数 → 5）
  - 2026-04-20 检索策略重构：去 Chroma 向量检索；商品用"catalog 索引 + `query_product(ids)` 全量加载"；政策用 `query_policy` 整文档按需加载；工具 → 6
  - 2026-04-20 记忆系统简化：去掉语义记忆层；长期记忆用 md 文件存 per-session 摘要（`memory/sessions/{user_id}/*.md`），不再进 DB；Chroma 彻底移除
  - 2026-04-20 去除状态机模块；退货流程改为 Prompt 引导 + 工具层鉴权；模块总数 10 → 9
  - 2026-04-20 `query_policy` 简化为无参整文件返回；控制 policies.md 体积在可接受范围
  - 2026-04-21 工具 schema 统一遵循 **OpenAI function calling** 规范（入参/出参/错误形态）
  - 2026-04-21 **移除观测性与 trace 系统**（原 M8 / M9）：不落 trace 表、不做 Cost/Latency 运行期观测；评测阶段在 eval harness 现场记录即可。模块总数 9 → 7
  - 2026-04-21 种子数据极简化：商品 10 → **3**、用户 5–8 → **2**、订单 20 → **10**；seed.py 全量写死、无随机
- **Owner**: xyh (lostin0320@gmail.com)

---

## 1. 项目定位

自研一个面向电商客服场景的多轮对话 Agent，用于**求职 Agent 开发岗位**，目标是：
- **完整但不过度复杂** —— 覆盖 Agent 开发的关键能力
- **有亮点** —— 记忆、评估、可观测性是核心卖点
- **符合真实场景** —— 电商客服的工具与流程
- **可量化的工作价值** —— 系统化评测与消融实验

**投入分配**：Agent 能力 55% / 评估体系 25% / 可观测性+文档 10% / Domain 数据 10%

---

## 2. 技术栈（已定稿）

| 维度 | 选型 | 备注 |
|---|---|---|
| LLM | **qwen-plus**（OpenAI 兼容接口） | 单模型，不做路由切换 |
| 工具协议 | **OpenAI function calling** | 所有工具 schema 与错误返回统一此规范，便于跨 LLM 复用 |
| 框架 | 自研 ReAct + Tool-use 循环 | 不用 LangGraph / LangChain |
| 结构化存储 | SQLite（仅交易数据） | 无 Redis；不落运行期 trace |
| 非结构化存储 | 文件系统（md 文件） | 商品描述、政策、记忆摘要都用 md |
| 向量存储 | ❌ 不使用 | 经过评估：数据规模 < 15K tokens + 记忆需求任务导向，向量检索过度工程化 |
| 后端 | Python + FastAPI | |
| 观测 | ❌ 不做运行期 trace | 评测阶段由 eval harness 现场统计 latency / token / cost |
| 前端 | 暂无，W6 后补 Gradio | |

---

## 3. Domain 数据模型（极简，分层存储）

**存储原则**：
- **DB = "要算的、要变的、要 join 的"** → 交易数据
- **文件 = "要读的、要人看的、要版本化的"** → 商品描述 / 政策 / 记忆摘要

所有"个性化"来自 Agent 记忆，不来自预设标签。

### 3.1 交易层（SQLite）
| 表 | 数量 | 字段 |
|---|---|---|
| users | 2 | `user_id`, `name` |
| products | 3 | `product_id`, `name`, `price`, `stock` |
| orders | 10 | `order_id`, `user_id`, `product_id`, `status`, `created_at`, `shipped_at` |

- 订单状态枚举：`pending / shipped / delivered / refunded`（`refunded` 即退款完成态；退货确认过程在对话层完成，不落独立过渡态）
- `shipped_at`：nullable timestamp；`status >= shipped` 才有值。工具返回时实时算 `days_since_shipped`
- 种子数据**全量写死**（无随机）；`seed.py` 直接维护三个商品的价格/库存，md 仅作商品身份/描述的唯一事实源（小规模下不强制同步）
- 时间基于 `NOW - N days` 的固定偏移
- 故意埋 3 条异常订单（O008 疑似丢件 / O009 状态不一致 / O010 久未发货），压测 Agent 鲁棒性

### 3.2 商品描述层（每商品一 md）
路径：`knowledge/products/P001.md`、`P002.md`... 文件名即 `product_id`。

```markdown
---
product_id: P001
name: 无线蓝牙耳机 X1
---

## 产品描述
...
## 主要功能
...
## 规格参数
...
```

- Frontmatter 的 `product_id`、`name` 供 seed 脚本同步到 DB
- Body 由 `query_product(product_ids)` 工具全量读取返回（不做 chunk / 向量）
- 3 个 md 文件（P001 耳机 / P002 手表 / P003 办公椅），覆盖典型数码+家具品类即可

### 3.3 政策层（单文件）
路径：`knowledge/policies.md`，**全站共享**，不绑定商品。

内容：退换货政策、配送规则、售后 FAQ 等。`query_policy()` 工具返回**整个文件**。

### 3.4 记忆层（per-user md 文件）
路径：`memory/sessions/{user_id}/YYYYMMDD_HHMM_{session_id}.md`

```markdown
---
session_id: s_abc123
user_id: U001
started_at: 2026-04-18T10:23:00
ended_at: 2026-04-18T10:35:00
turn_count: 12
---

## 对话摘要
用户咨询订单 O012 物流，Agent 告知已发货 2 天...

## 提及的订单/商品
- O012 / P003

## 待跟进
- 用户表示若周五前未到会再次联系
```

- 会话结束时 LLM 生成摘要 → 写入 md 文件
- 新会话启动时 `listdir` 按文件名（时间前缀）排序取最近 3–5 条 → 注入 system prompt
- 文件名前缀 `YYYYMMDD_HHMM_` 保证按字典序 = 按时间序

---

## 4. Agent 核心能力（7 个模块）

进度标记：`[ ]` 未开始 / `[~]` 进行中 / `[x]` 完成

- [ ] **M1 · ReAct 循环**：自研，内存中维护 Thought/Action/Observation 链（不持久化）
- [ ] **M2 · 工具层（6 个）**：**遵循 OpenAI function calling 规范**
  - 每个工具提供标准 `{"type":"function","function":{"name","description","parameters":<JSON Schema>}}` 定义
  - 返回体统一为 JSON-serializable dict；错误用 `{"ok": false, "error_code", "message"}` 形态回传给 LLM
  - 用 Pydantic 校验入参，但对 LLM 暴露的 schema 以 OpenAI function calling 为准
  - [ ] `query_order` 订单查询
  - [ ] `track_logistics` 物流追踪（基于 status + shipped_at 计算 days_since_shipped）
  - [ ] `initiate_refund` 退换货发起
  - [ ] `query_product(product_ids)` 商品信息查询：DB 交易数据 + md 完整描述，支持多 ID 天然适配对比场景
  - [ ] `query_policy()` 政策查询：无参，返回整个 `policies.md` 文件
  - [ ] `escalate_to_human` 转人工
- [ ] **M3 · 两层记忆**（简化自原三层）
  - [ ] 短期：滚动窗口 + 超阈值时触发上下文压缩（LLM 摘要近期对话）
  - [ ] 长期：per-session 结构化摘要，`memory/sessions/{user_id}/*.md`；新会话载入最近 3–5 条
- [ ] **M4 · 知识检索策略**（非 RAG）
  - [ ] Catalog 注入：system prompt 含全部 3 个商品的 ID / 名称 / 品类 / 简介
  - [ ] 商品：LLM 从 catalog 挑 product_ids → `query_product(ids)` 全量加载完整 md
  - [ ] 政策：`query_policy()` 按需返回整个 `policies.md`（不切分 section）
  - [ ] 护栏约束：政策类问题必须走 `query_policy`，禁止 LLM 臆造
- [ ] **M5 · 意图路由**：轻量分类 → 派发
- [ ] **M6 · 护栏（分层防御）**
  - [ ] 工具层鉴权：`query_order` / `initiate_refund` 内部校验 `user_id` 与订单归属；无论对话路径如何都硬拦截
  - [ ] Prompt 注入过滤
  - [ ] 关键信息（价格/库存/政策）禁止 LLM 臆造，必须走工具
  - [ ] 退货流程安全点不靠 FSM，由 system prompt 引导（必须确认才执行）+ 工具层鉴权双层兜底
- [ ] **M7 · 情绪感知 + 升级**：检测负面情绪 → 自动 `escalate_to_human`

---

## 5. 评估体系（最大亮点）

- [ ] **E1 · 评测集**：80–100 条样本，覆盖 5 个 Fixture 剧本
  - [ ] 愤怒的催单客户（情绪识别 + 升级）
  - [ ] 退货流程（多工具编排：查订单 → 确认 → 执行）
  - [ ] 政策咨询新客（`query_policy`，"几天内可以退"）
  - [ ] 商品咨询（`query_product` 多源融合，"X1 耳机续航多久，还有货吗"）
  - [ ] 老客复购（长期记忆 + 跨会话召回，"我上次买的那款现在发货没"）
  - [ ] 恶意用户（护栏：越权查/退订单、Prompt 注入）
  - 每条样本含：输入对话、期望工具调用序列、期望信息点、禁止行为
- [ ] **E2 · 自动指标**（由 eval harness 现场统计，不依赖运行期 trace）
  - [ ] Task Success Rate
  - [ ] Tool Call Accuracy（顺序 + 参数）
  - [ ] Faithfulness（政策/商品类回答是否基于文档）
  - [ ] Latency P50 / P95
  - [ ] Tokens / Cost per session
- [ ] **E3 · LLM-as-Judge**：qwen-plus 自打分，三维度（helpfulness/tone/policy），多次采样取均值
- [ ] **E4 · 消融实验**
  - [ ] 记忆策略对比：无记忆 / 结构化摘要（本方案）/ 全量历史塞入
  - [ ] 知识检索策略对比：catalog+全量加载（本方案）/ 向量检索（实验对照）/ 全塞入
  - [ ] 组件消融：去长期记忆 / 去上下文压缩 分别跑
  - [ ] 全部出对比表，用数据支撑架构决策
- [ ] **E5 · 红队测试**：Prompt 注入、越权查订单、诱导泄露系统提示、角色劫持，记录拦截率

---

## 6. 项目目录结构

```
nana-agent/
├── agent/
│   ├── core.py              # ReAct 循环
│   ├── router.py            # 意图路由
│   ├── tools/               # 6 个工具，一工具一文件；OpenAI function calling schema
│   ├── memory/
│   │   ├── short_term.py    # 滚动窗口 + 上下文压缩
│   │   └── long_term.py     # per-session 摘要读写
│   └── guardrails/          # 工具层鉴权 / 注入过滤 / 情绪
├── knowledge/
│   ├── policies.md          # 全站统一政策（退换货/配送/FAQ）
│   └── products/            # 每商品一 md，文件名即 product_id
│       ├── P001.md
│       └── ...
├── memory/
│   └── sessions/            # per-user 会话摘要
│       └── {user_id}/
│           └── YYYYMMDD_HHMM_{session_id}.md
├── data/                    # 静态 seed 源
├── db/                      # SQLite schema + 迁移
├── api/                     # FastAPI 入口
├── eval/
│   ├── datasets/            # 评测样本
│   ├── metrics/             # 自动指标
│   ├── judge/               # LLM Judge
│   ├── red_team/            # 攻击用例
│   └── reports/             # 报告
├── scripts/                 # seed、reset、run_eval
├── tests/                   # pytest
└── README.md
```

---

## 7. 里程碑与进度

| 周 | 目标 | 状态 |
|---|---|---|
| **W1** | DB Schema + seed 脚本 + ReAct 循环 + 2 个工具（OpenAI function calling schema） | [ ] 未开始 |
| **W2** | 补齐 6 个工具 + 两层记忆（短期滚动窗口 + 长期 md 摘要） | [ ] |
| **W3** | 知识检索（catalog 注入 + `query_product` / `query_policy`） + 意图路由 + qwen-plus 接入 | [ ] |
| **W4** | 护栏（工具层鉴权 + 注入过滤）+ 情绪感知 | [ ] |
| **W5** | 评估体系（评测集 + 自动指标 + Judge + 消融 + 红队） | [ ] |
| **W6** | Gradio Demo + Docker + README + 技术博客 | [ ] |

### 当前进展

- **2026-04-20** · 完成规划，定稿技术栈与数据模型；创建 PLAN.md
- **2026-04-20** · 检索与记忆架构重构：去 Chroma、去 RAG、去语义记忆；改为 "catalog + 全量加载" + per-session md 摘要；技术栈简化为"SQLite + 文件系统"
- **2026-04-20** · 去除状态机模块；退货流程改为 Prompt 引导 + 工具层鉴权；整体架构从"10 模块"收敛到"9 模块"
- **2026-04-21** · LLM 改为 **qwen-plus**；工具 schema 统一 **OpenAI function calling**；**移除运行期观测 & trace**（M8/M9 去除），架构收敛到 **7 模块**
- **2026-04-21** · W1 数据层落地：uv 项目骨架 + SQLite DDL（users/products/orders）+ 3 商品 md + `policies.md` + `scripts/seed.py`；seed 跑通（**2 用户 / 3 商品 / 10 订单含 3 异常**，全量写死无随机）

---

## 8. 交付物

- [ ] GitHub 仓库（含架构图、设计决策、eval 报告、成本分析）
- [ ] Docker 一键部署 + Gradio Demo
- [ ] 技术博客：重点写「评估体系设计」和「记忆模块架构」
- [ ] 简历要点：
  - 自研 ReAct Agent 循环（Thought/Action/Observation），工具层遵循 **OpenAI function calling** 规范，跨 LLM 可插拔
  - **按领域特点裁剪架构**：经评估主动去除向量 RAG 与语义记忆（详述于设计文档），改用 catalog 索引 + 结构化摘要，消融实验数据支撑
  - 两层记忆架构（短期滚动窗口+压缩 / 长期 per-session md 摘要），跨会话个性化
  - **分层防御**：工具层鉴权（无论调用路径都拦截越权）+ Prompt 层约束（必须确认才执行敏感动作）
  - 80+ 样本评测体系（自动指标 + LLM Judge + 红队 + 四路消融）
  - 评测报告：Task Success Rate XX%，单 session 成本 $X.XX，P95 延迟 X.Xs

---

## 9. 决策日志（Decision Log）

记录"为什么这样选"，避免后续反复纠结。

| 日期 | 决策 | 理由 |
|---|---|---|
| 2026-04-20 | LLM 只用 qwen-plus，不做多模型路由 | 聚焦 Agent 本身；单模型时改写简历为"Prompt 分层 + Tool schema 精简降 token" |
| 2026-04-20 | 框架自研，不用 LangGraph / LangChain | 体现对原理的理解，面试可深入讲 |
| 2026-04-20 | 去 Redis 缓存层 | 性能非目标；全量落库反而支持对话回放（变成亮点） |
| 2026-04-20 | 去优惠券、VIP、用户分类 | 保持 domain 极简；所有"个性化"来自 Agent 记忆 |
| 2026-04-20 | 前端留到 W6 后 | 先把 Agent 能力做深 |
| 2026-04-20 | orders 加 `shipped_at`，状态合并为 `pending/shipped/delivered/refunded` | 存时间戳而非"X 天"避免过期；退货"处理中"短暂态由 FSM 内部管，不污染 DB schema |
| 2026-04-20 | 去掉 `recommend_products` 工具 | 推荐与电商客服核心诉求关联弱；聚焦查询/物流/退款/升级四类真实客服问题 |
| 2026-04-20 | 知识三层分离：交易→DB / 商品描述→per-product md / 政策→统一单文件 | 按"变化频率 + 适用范围"切分，贴合真实系统；md 是商品唯一事实源，seed 脚本同步到 DB |
| 2026-04-20 | 加回 `query_product` 工具 | 显式工具 + 多源融合（DB 交易数据 + md 描述）是结构化/非结构化知识融合的核心亮点 |
| 2026-04-20 | 商品检索改为 "catalog + 全量加载"，不用向量 RAG | 小规模（少量商品 × ~1K tokens/条）下向量检索过度工程化；catalog 方案天然支持多商品对比（"A 和 B 哪个好"），trace 更清晰 |
| 2026-04-20 | 政策改为整文档按需加载，不用向量 | 单文件小，chunk 反而破坏语义完整性 |
| 2026-04-20 | 记忆系统：去语义记忆层，改两层（短期+长期摘要） | 电商客服任务导向、短 horizon，语义检索无用武之地；用结构化摘要更贴合场景 |
| 2026-04-20 | 记忆摘要存为 md 文件，不进 DB | 内容是 LLM 生成的非结构化文本；md 可读、可版本化、与知识文件风格一致；规模下 `listdir` 排序足够当"查询" |
| 2026-04-20 | 彻底移除 Chroma 依赖 | 无向量检索需求后，技术栈简化为 "SQLite + 文件系统"；架构决策本身成为简历亮点 |
| 2026-04-20 | 加 `query_policy` 工具（总数 → 6） | 政策查询显式化，trace 观测性一致 |
| 2026-04-20 | 去除状态机模块，退货流程不强制 FSM | 资源有限，接受少量边界/恶意场景不覆盖；核心安全点（越权退款）改为**工具层鉴权**守住，防御能力不降反升（无论对话路径都拦得住） |
| 2026-04-20 | 简历叙事：从"识别高风险流程用 FSM"改为"**分层防御**（工具是 security boundary）" | 符合当前 Agent 设计主流思路；强制型 FSM 反而牺牲灵活性 |
| 2026-04-20 | `query_policy` 简化为无参整文件返回，不做 section 切分 | 控制文件体积在 < 2K tokens，直接整文件入上下文信息完整度最高；切分会人为制造 chunking 选择问题，复杂度不值 |
| 2026-04-21 | 工具层统一 **OpenAI function calling** schema | 业界通用规范；便于后续切换任何 OpenAI 兼容 LLM；trace/debug 时格式统一 |
| 2026-04-21 | 移除 M8 可观测性 / M9 成本分析，不落 trace 表 | 简化开发聚焦 Agent 核心；评测阶段由 eval harness 现场采集 latency/token/cost 已足够，运行期全量 trace 系统 ROI 低 |
| 2026-04-21 | 种子数据极简：3 商品 / 2 用户 / 10 订单全部写死，取消 md→DB 同步 | 规模极小且几乎不变，写死最省心；同步逻辑与随机生成都是伪复杂度；md 继续作为商品描述的唯一事实源 |

---

## 10. 开放问题 / 下一步动作

### 立即（W1 起步）
- [x] 确定 SQLite schema DDL（users / products / orders）
- [x] `scripts/seed.py`（md frontmatter → products 表，含 3 条异常订单）
- [ ] ReAct 循环核心数据结构（Message / Action / Observation / Step）
- [ ] 工具通用契约：**OpenAI function calling** schema 定义模板 + 统一错误返回形态
- [ ] 第一个工具 `query_order` 的完整实现模板（function-calling schema + 错误码 + 工具层鉴权 + 测试）
- [ ] qwen-plus SDK 接入与最小 tool-use 调用 demo

### 待定
- [ ] 评测集具体场景细化（W5 之前敲定）
- [ ] 长期记忆摘要的 md 模板定稿（frontmatter 字段、body 结构）
- [ ] 红队攻击用例库来源（自己设计 vs 参考公开数据集）

---

## 维护约定

- **每次开工前**：读 §7 当前进展 + §10 下一步动作
- **每次收工前**：更新 §4/§5 checkbox + §7 当前进展 + §9 决策日志（如有新决策）
- **每周五**：Review §7 里程碑是否需要调整
- **发现新问题**：立刻加到 §10 开放问题，避免遗忘
