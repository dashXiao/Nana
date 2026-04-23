# NanaShop Agent · 开发计划与进度

> **单一事实来源（Single Source of Truth）**
> 开工前读 §7（现状）+ §10（待定）；收工前更新 §7 与 §4 checkbox；有新决策追加 §9 一行。
> 历史变更查 git log，不在本文件流水账。

- **Last updated**: 2026-04-22
- **Owner**: xyh (lostin0320@gmail.com)

---

## 1. 项目定位

自研一个面向电商客服场景的多轮对话 Agent，用于**求职 Agent 开发岗位**，目标是：
- **完整但不过度复杂** —— 覆盖 Agent 开发的关键能力
- **有亮点** —— 意图路由、两层记忆、评估体系、按领域裁剪架构（去 RAG / 去向量）
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

### 3.4 记忆层（per-user 单文件 · 简化版）
路径：`memory/sessions/{user_id}.md`（每用户一个文件，append-only）

```markdown
---
user_id: U001
---

## 2026-04-18 10:23
用户咨询订单 O012 物流，Agent 告知已发货 2 天，用户表示若周五前未到会再次联系。

## 2026-04-20 15:40
用户问了 P001 耳机价格与库存，未下单。
```

- 会话结束（`/reset` / `/user` / `/quit` / 转人工）时 LLM 生成一段摘要 → 追加一节
- 新会话冷启动时读文件，**只取最近 3 段**拼成一个 system 块，避免历史越堆越重
- 由 `LongTermMemory` 类管理，谱系参见 §9 #19

---

## 4. Agent 核心能力（7 个模块）

进度标记：`[ ]` 未开始 / `[~]` 进行中 / `[x]` 完成

- [x] **M1 · ReAct 循环**：自研，内存中维护 Thought/Action/Observation 链（不持久化）
  - [x] 数据结构：`Step` / `ToolInvocation` / `RunResult`
  - [x] 循环调度 + `max_steps` 熔断
  - [x] session 上下文硬覆盖：派发前强制把 `user_id` 改为会话归属，阻断 LLM 伪造
- [x] **M2 · 工具层（5 个）**：**遵循 OpenAI function calling 规范**
  - 每个工具提供标准 `{"type":"function","function":{"name","description","parameters":<JSON Schema>}}` 定义
  - 返回体统一为 JSON-serializable dict；错误用 `{"ok": false, "error_code", "message"}` 形态回传给 LLM
  - 用 Pydantic 校验入参，但对 LLM 暴露的 schema 以 OpenAI function calling 为准
  - [x] `query_orders(user_id)` 订单列表查询：返回当前用户全部订单（入参不含 order_id，架构上消除越权面）
  - [x] `initiate_refund(user_id, order_id)` 退换货发起：NOT_FOUND / UNAUTHORIZED / INVALID_STATE 三道校验，越权由工具层硬挡
  - [x] `query_product(product_ids)` 商品信息查询：DB 交易数据 + md 完整描述；去重保序、部分不存在时返回 products + missing
  - [x] `query_policy()` 政策查询：无参，返回整个 `policies.md` 文件
  - [x] `escalate_to_human()` 转人工：无参的终止型工具，循环在 `agent/core.py` 识别到调用后硬切对话，返回 `"已转接人工"`
- [x] **M3 · 两层记忆**（简化自原三层；谱系参见 §9 #19）
  - [x] 短期 `ShortTermMemory`：Summary-Buffer Memory，用户轮数 >4 时把早期对话压成一条 system 摘要，保留末 1 轮
  - [x] 长期 `LongTermMemory`：per-user 单 md 文件（append-only），冷启动载入最近 3 段，本会话结束时按 `min_turns≥2` 门槛追加
  - [x] `MemoryEvent` 可观测：与 `route_path/confidence` 同构，回填到 `RunResult.memory_events`，REPL 与评测都能消费
- [x] **M4 · 知识检索策略**（非 RAG）
  - [x] Catalog 注入：仅 `product_id + 名称`（价格/库存/描述走工具，防引用过期值）；M5 后 catalog 已从 base 下沉到 `product_inquiry` patch（§9 #17）
  - [x] 商品：LLM 从 catalog 挑 `product_ids` → `query_product(ids)` 全量加载完整 md
  - [x] 政策：`query_policy()` 无参返回整个 `policies.md`（不切分 section），system 硬约束不得臆造
- [x] **M5 · 意图路由**：分层漏斗 + `IntentPolicy` 策略表（7 意图），Framing A「config 派」（详见 §9 #12–#18）
  - [x] 三层漏斗：规则 0 LLM / LLM 一次结构化 JSON / 低置信度 → `other` 兜底；分类器 prompt 带"多意图→other"负样本
  - [x] `IntentPolicy(exposed_tools, prompt_patch, fast_path_tool)` 绑定执行策略；`Agent.run_routed()` 编排分类 → 策略 → fast_path 或带工具子集的 ReAct
  - [x] `RunResult` 带 `intent/route_path/confidence` 供 W5 分桶；escalation 规则命中走 fast_path 零 LLM 成本
  - [x] Slim base + 资料/规则按需挂到 patch（base 仅身份+格式；catalog 挂 product_inquiry；规则挂各意图；`other` 兜底承载全量规则）
- [~] **M6 · 护栏（分层防御）**
  - [x] 工具层鉴权：`initiate_refund` 内部校验 `user_id` 与订单归属；无论对话路径如何都硬拦截（`query_orders` 因入参不含 order_id，已在架构上消除越权面）
  - [ ] Prompt 注入过滤
  - [x] 关键信息（价格/库存/政策）禁止 LLM 臆造，必须走工具（由 system prompt 强约束）
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
│   ├── llm.py               # qwen-plus 客户端（OpenAI 兼容）
│   ├── router.py            # 意图路由
│   ├── tools/               # 5 个工具，一工具一文件；OpenAI function calling schema
│   ├── memory/              # 短期 Summary-Buffer / 长期 per-user 单 md
│   └── guardrails/          # Prompt 注入过滤 / 情绪识别
├── knowledge/
│   ├── policies.md          # 全站统一政策（退换货/配送/FAQ）
│   └── products/            # 每商品一 md，文件名即 product_id
├── memory/
│   └── sessions/            # 每用户一个 {user_id}.md，append-only
├── db/                      # SQLite schema + 连接封装
├── api/                     # FastAPI 入口
├── eval/                    # 评测集 / 指标 / Judge / 红队 / 报告
├── scripts/                 # seed.py / chat.py REPL / smoke_*.py
├── tests/                   # pytest
├── PLAN.md                  # 本文件
└── CLAUDE.md                # 给 Claude Code 的协作约定
```

---

## 7. 里程碑与当前进展

**当前阶段**：**W3 进行中** —— M3 两层记忆骨架已落（§9 #19）；接下来做 qwen-plus 深度优化 & 进入 W4 护栏。

| 周 | 目标 | 状态 |
|---|---|---|
| **W1** | DB Schema + seed + ReAct 循环 + 工具契约 + 首批工具 | **[x] 超额完成**（4 工具 + catalog，见下） |
| **W2** | 补齐第 5 工具 + 意图路由 | **[x]** |
| **W3** | 两层记忆 + qwen-plus 深度优化 | **[~]** 记忆完，优化待续 |
| **W4** | 护栏补全（注入过滤 / 情绪感知）| [ ] |
| **W5** | 评估体系（评测集 + 自动指标 + Judge + 消融 + 红队） | [ ] |
| **W6** | Gradio Demo + Docker + README + 技术博客 | [ ] |

### 已完成（累计）

- **数据层**：SQLite schema + `seed.py`（2 用户 / 3 商品 / 10 订单含 3 异常，全量写死）+ `knowledge/products/*.md` + `knowledge/policies.md`（"发货日 +7 天"统一规则）
- **骨架**：`agent/core.py` ReAct 循环（`Step`/`ToolInvocation`/`RunResult` + `max_steps` + 派发前 `user_id` 硬覆盖）；工具契约 `_contract.py`（OpenAI function calling + 统一错误返回）；qwen-plus client（env 必填，模块级 `MODEL/client`）
- **工具 5/5**：`query_orders` / `query_product`（DB+md 融合）/ `query_policy` / `initiate_refund`（三道校验）/ `escalate_to_human`（终止型，按 name 硬切返回"已转接人工"）
- **M5 意图路由**：`agent/router.py` 漏斗分类器 + 7 意图 `POLICY_MAP` + `Agent.run_routed()` 编排；slim base + 资料/规则按意图下沉到 patch（`other` 兜底承载全量规则）；escalation 走 fast_path
- **M3 两层记忆**：`agent/memory/{events,short_term,long_term}.py`；`run_routed` 入口先压后跑；`RunResult.memory_events` 可观测；REPL 在 `/reset`/`/user`/`/quit`/ 转人工 时 `save_session`
- **测试 & 人工**：89 pytest 全绿；`scripts/smoke_agent.py` S1–S5 共 9 子场景；`scripts/chat.py` REPL 按"路由 / 工具子集 / 记忆事件 / system / 工具调用"顺序打 trace

### 下一步

- qwen-plus 深度优化（W3 下半段）：压缩摘要 prompt 调校、消融对比无记忆基线
- W4 护栏：Prompt 注入过滤 + 情绪感知 → 自动 escalate

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

只记"反直觉 / 会被问'为什么不那样'"的选型。按主题组织，不按时间。

| # | 决策 | 理由 |
|---|---|---|
| 1 | LLM 单模型 qwen-plus + 自研 ReAct 循环 + 工具统一 OpenAI function calling schema | 聚焦原理展现；schema 通用便于跨 LLM 复用；不用 LangChain/LangGraph 以免被抽象遮蔽细节 |
| 2 | 去向量 RAG、去语义记忆、去 Chroma —— 商品走 catalog+全量加载、政策整文档返回、记忆走 md 摘要 | 数据规模 < 15K tokens + 任务导向短 horizon，向量过度工程化；技术栈收敛到 SQLite + 文件系统，决策本身即简历亮点 |
| 3 | 退货核心安全点改**工具层鉴权**（分层防御），不做 FSM 状态机 | 无论对话路径如何都拦得住越权；简历叙事为"tool 即 security boundary"，比 FSM 更主流且更灵活 |
| 4 | ReAct 派发前硬覆盖 tool_call 的 `user_id`（循环入口统一哨卡） | session 上下文不交给 LLM 决定；任何工具自动受益，与工具层鉴权形成分层防御 |
| 5 | 工具预算 5：`query_orders / query_product / query_policy / initiate_refund / escalate_to_human` | 去 `recommend_products`（与客服核心弱相关）、去 `track_logistics`（数据已在 query_orders 内）；`query_order` 改 `query_orders`（用户不记订单号，同时消除跨用户枚举面）|
| 6 | 订单状态枚举 `pending/shipped/delivered/refunded` + `shipped_at` 时间戳 | 存时间戳而非"X 天"避免过期；退款过渡态由对话层管，不污染 schema |
| 7 | 退货规则基准改为**发货日 +7 天**（原方案为"签收日 +7 天无理由 / 15 天质量"）| DB 没有签收字段；原规则会让 Agent 反问用户签收日期，体验破坏 |
| 8 | Catalog 写死在 system prompt（仅 ID + 名称，不含价格/库存/描述） | 3 款商品近乎不变，DB 查询是伪复杂度；价格/库存放 prompt 会诱导 LLM 引用过期值 |
| 9 | 不落运行期 trace 表，原 M8/M9 观测性模块移除 | 评测阶段由 eval harness 现场采集 latency/token/cost 即足够；运行期全量 trace 系统 ROI 低 |
| 10 | 种子数据全量写死（2 用户 / 3 商品 / 10 订单含 3 异常），无随机、无 md→DB 同步 | 规模极小且几乎不变，写死最省心；随机生成与同步逻辑都是伪复杂度 |
| 11 | `escalate_to_human` 的终止语义在 Agent 循环里按 name 硬匹配，不抽 `ToolSpec.terminates` 字段 | YAGNI：当前只有一个终止型工具，通用化抽象待未来第二个出现再做；现在 3 行 if 比一套字段+调度更清晰 |
| 12 | 意图路由走 **Framing A「config 派」**（一个 Agent + 配置表），不拆 sub-agent + orchestrator（Framing B） | 7 个意图的 prompt_patch 本质同构（共享基座 + 小段专属提示），硬包装成 sub-agent 会被面试官识破是 cosmetic；A 的 `(exposed_tools, prompt_patch, fast_path_tool)` 元组未来可直接重命名为 SpecialistAgent 升级到 B，保留路径无损耗 |
| 13 | 路由用**独立一次 LLM 分类调用**（结构化 JSON 输出），不做 `pick_intent` 伪工具混进 ReAct | 独立调用语义清晰、可测、可换模型；混进 ReAct 会让"选意图"和"选动作"耦合，评测分桶也会失真 |
| 14 | `chitchat` 意图不做 fast_path 模板回复，而是 `exposed_tools=()` 仍让 LLM 自然生成 | 模板回"您好！"应对"谢谢你"会很怪；零工具既省 tokens 又杜绝 LLM 对寒暄意外调 `query_orders`，自然度 & 安全性双赢 |
| 15 | 带 `prompt_patch` 的多轮 run：**覆盖** `messages[0]` 的 system，而非沿用 history 原 system | 不同轮可能命中不同意图，patch 必须随轮更新；patch 为空时仍保持"沿用历史 system"的旧行为，不破坏既有测试 |
| 16 | Slim base + 规则下沉：base system prompt 只含身份 + catalog + 格式（~130 字符），原 6 条行为规则全部拆到对应 `POLICY_MAP[intent].prompt_patch`，`other` 兜底 patch 承载完整规则集 | 原长版 base 已包含所有意图规则，patch 只是冗余强调；下沉后 LLM 每轮只看到与当前意图相关的规则，特定意图路径 tokens 降 ~80%；patch 真实承担路由价值（不再是装饰），同时"routing 不只裁工具集，还裁行为约束"成为简历叙事 |
| 17 | 进一步下沉：商品 catalog 从 base 挪到 `product_inquiry` 的 patch，base 砍到身份 + 格式两行（~30 字符） | 只有商品咨询意图需要把"蓝牙耳机"映射到 `product_id`；订单/政策/退款/闲聊看到 catalog 是纯噪声，放 base 相当于每轮多塞 3 行无用资料。"资料按意图需用即挂"与"规则按意图需用即挂"对称，收敛路由心智模型 |
| 18 | 分类器 prompt 加多意图→`other` 的负样本对照（"订单+商品""订单+政策""商品+政策"等 4 条样例），并在单意图标签描述前加"**仅**"字样 | 原 "多意图混合" 规则只一行描述、无样例，qwen-plus 遇"查商品和我的订单"会勉强挑主意图（实测 conf=0.65 命中 order_query，裁掉了 query_product 导致另一维度答不了）。加样例后同一句稳定落 `other` conf=0.98，两把工具都被调到。启示："跨维度 → 兜底" 规则必须用负样本锚定，仅靠抽象描述 LLM 会走捷径 |
| 19 | 两层记忆谱系对照：**短期 = Summary-Buffer（LangChain `ConversationSummaryBufferMemory` / Claude Code auto-compact 谱系）**；**长期 = per-user append-only md（ChatGPT Memory / MemGPT / Letta 谱系）**。入口按"长期 load → 短期 compact → ReAct"顺序；用户轮数 ≥2 才 `save_session`；`load_block` 只取最近 3 段；记忆事件随 `RunResult.memory_events` 回传（非副作用回调）| 两套谱系是主流、各有其位：短期处理会话内膨胀，长期处理老客跨会话召回。Claude Code 是"会话级无状态"——不等于否定长期记忆，而是电商场景有"老客回头"天然诉求，合理场景适配。`min_turns=2` 防 1 轮闲聊污染档案；`最近 3 段` 防历史越堆越重（可供 W5 消融对比）；`MemoryEvent` 与 `route_path/confidence` 同构，保持"可观测=数据回传"的统一心智 |
| 20 | 长期记忆在 Agent 里用 `_lt_cache[user_id]` 做**会话内缓存 + 每轮重注入**，不是"只在第 1 轮 load 一次，之后靠 history 携带" | 续轮 `_build_initial_messages` 在 patch 非空时会替换 `messages[0]` 成 `base + patch`，首轮嵌入的长期块会被吞掉。冷启动 load 后缓存，续轮作为 `memory_block` 参数重注入到新 system 里，抵消替换路径。这与 ChatGPT Memory / Claude Code `CLAUDE.md` / Cursor `.cursorrules` 的"会话启动读一次、期内稳定"契约一致——主流 agent 共识 |

---

## 10. 开放问题

- [ ] 评测集具体场景细化（W5 之前敲定）
- [ ] 红队攻击用例库来源（自己设计 vs 参考公开数据集）
- [ ] 退款流程对"边界日期"订单的处理（例如发货第 7 天）是否需要额外 prompt 话术引导
- [ ] 路由 confidence 阈值 0.6 是凭经验拍的，W5 用真实样本标注后再回调
- [ ] 分类 LLM 的 prompt 已加"多意图→other"负样本（§9 #18）；单意图误分率仍待 W5 真实标注后再迭代 few-shot
- [ ] 意图粒度：`order_query` 是否拆"物流进度" vs "状态确认"、`refund_request` 是否独立出"已发货 vs 未发货"子类——视 W5 分桶数据决定
- [ ] 长期记忆 **合并/淘汰策略**：当前 per-user md 会无限 append，`load_block` 靠只取最近 3 段兜住。超活跃用户若累计几十段，老段落永远进不了冷启动上下文——W5 之前定夺是否做"每 N 段 → 合并成一段元摘要"（对齐 MemGPT/Letta 的 memory consolidation），还是保持 append-only 让 W5 消融数据说话

---

## 维护约定

- **每次开工前**：读 §7（现状）+ §10（待定）
- **每次收工前**：更新 §4/§5 checkbox + §7「已完成」「下一步」；有新决策追加 §9 一行
- **不在 PLAN.md 里留流水账**：每次提交细节（单测数量、commit 信息）一律进 git log
