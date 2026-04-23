# NanaShop Agent · 开发计划与进度

> **单一事实来源（Single Source of Truth）**
> 开工前读 §7（现状）+ §10（待定）；收工前更新 §7 与 §4 checkbox；有新决策追加 §9 一行。
> 历史变更查 git log，不在本文件流水账。

- **Last updated**: 2026-04-23
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
| LLM | **qwen-plus 主 + qwen-turbo 轻**（OpenAI 兼容接口） | 双模型分层：ReAct 走 plus，路由分类 / 记忆摘要走 turbo（§9 #11）|
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
- 由 `LongTermMemory` 类管理，谱系参见 §9 #9

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
- [x] **M3 · 两层记忆**（简化自原三层；谱系参见 §9 #9）
  - [x] 短期 `ShortTermMemory`：Summary-Buffer Memory，用户轮数 >4 时把早期对话压成一条 system 摘要，保留末 1 轮
  - [x] 长期 `LongTermMemory`：per-user 单 md 文件（append-only），冷启动载入最近 3 段，本会话结束时按 `min_turns≥2` 门槛追加
  - [x] `MemoryEvent` 可观测：与 `route_path/confidence` 同构，回填到 `RunResult.memory_events`，REPL 与评测都能消费
- [x] **M4 · 知识检索策略**（非 RAG）
  - [x] Catalog 注入：仅 `product_id + 名称`（价格/库存/描述走工具，防引用过期值）；M5 后 catalog 已从 base 下沉到 `product_inquiry` patch（§9 #8）
  - [x] 商品：LLM 从 catalog 挑 `product_ids` → `query_product(ids)` 全量加载完整 md
  - [x] 政策：`query_policy()` 无参返回整个 `policies.md`（不切分 section），system 硬约束不得臆造
- [x] **M5 · 意图路由**：分层漏斗 + `IntentPolicy` 策略表（8 意图，含 `negative_emotion`），Framing A「config 派」（详见 §9 #7、#8、#10）
  - [x] 三层漏斗：规则 0 LLM / LLM 一次结构化 JSON / 低置信度 → `other` 兜底；分类器 prompt 带"多意图→other"负样本
  - [x] `IntentPolicy(exposed_tools, prompt_patch, fast_path_tool)` 绑定执行策略；`Agent.run_routed()` 编排分类 → 策略 → fast_path 或带工具子集的 ReAct
  - [x] `RunResult` 带 `intent/route_path/confidence` 供 W5 分桶；`escalation` / `negative_emotion` 规则命中都走 fast_path，0 LLM 成本
  - [x] Slim base + 资料/规则按需挂到 patch（base 仅身份+格式；catalog 通过 `_CATALOG_SNIPPET` 常量挂 `product_inquiry` + `other`；规则挂各意图；`other` 兜底承载全量规则）
- [x] **M6 · 护栏（分层防御）**
  - [x] 工具层鉴权：`initiate_refund` 内部校验 `user_id` 与订单归属；无论对话路径如何都硬拦截（`query_orders` 因入参不含 order_id，已在架构上消除越权面）
  - [x] Prompt 注入过滤：`agent/guardrails/injection.py` 规则层三类正则（角色劫持 / 系统提示泄露 / 越狱关键词），`run_routed` 预检命中即短路返回固定拒绝文案（§9 #10）
  - [x] 关键信息（价格/库存/政策）禁止 LLM 臆造，必须走工具（由 system prompt 强约束）
  - [x] 退货流程安全点不靠 FSM，由 system prompt 引导（必须确认才执行）+ 工具层鉴权双层兜底
- [x] **M7 · 情绪感知 + 升级**：作为独立 intent `negative_emotion` 并入路由漏斗（§9 #10）
  - [x] 规则层 12 词正则（垃圾/骗子/气死/投诉 等），内容匹配（search）而非整串
  - [x] LLM 分类器 prompt 扩 8 标签，新增 `negative_emotion` 语义与负样本
  - [x] 与 `escalation` 共用 fast_path（`escalate_to_human` + "已转接人工"），intent 独立供 W5 分桶

---

## 5. 评估体系（最大亮点）

- [ ] **E1 · 评测集**：80–100 条样本，覆盖 6 个 Fixture 剧本
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
  - [ ] 双模型分层对比：全 qwen-plus / 分层（本方案）/ 全 qwen-turbo，看 Task Success × Cost × Latency（§9 #11）
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
│   └── guardrails/          # Prompt 注入过滤（情绪识别并入 router，§9 #10）
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

**当前阶段**：**Agent 功能闭环完成**（M1–M7 全部 [x]，§4），即将进入 **W5 评估体系**。

| 周 | 目标 | 状态 |
|---|---|---|
| **W1** | DB Schema + seed + ReAct 循环 + 工具契约 + 首批工具 | **[x] 超额完成**（4 工具 + catalog，见下） |
| **W2** | 补齐第 5 工具 + 意图路由 | **[x]** |
| **W3** | 两层记忆 | **[x]**（原"qwen-plus 深度优化"口袋项砍掉，相关 prompt 调校已在 §9 #8/#9 零散落地；消融对比并入 W5 E4） |
| **W4** | 护栏补全（注入过滤 / 情绪感知）| **[x]** |
| **W5** | 评估体系（评测集 + 自动指标 + Judge + 消融 + 红队） | [ ] |
| **W6** | Gradio Demo + Docker + README + 技术博客 | [ ] |

### 已完成（累计）

- **数据层**：SQLite schema + `seed.py`（2 用户 / 3 商品 / 10 订单含 3 异常，全量写死）+ `knowledge/products/*.md` + `knowledge/policies.md`（"发货日 +7 天"统一规则）
- **骨架**：`agent/core.py` ReAct 循环（`Step`/`ToolInvocation`/`RunResult` + `max_steps` + 派发前 `user_id` 硬覆盖）；工具契约 `_contract.py`（OpenAI function calling + 统一错误返回）；`agent/llm.py` 双模型 client（`MODEL` 主 / `LIGHT_MODEL` 轻，`chat(model=...)` 参数切换；§9 #11）
- **工具 5/5**：`query_orders` / `query_product`（DB+md 融合）/ `query_policy` / `initiate_refund`（三道校验）/ `escalate_to_human`（终止型，按 name 硬切返回"已转接人工"）
- **M5 意图路由**：`agent/router.py` 漏斗分类器 + 8 意图 `POLICY_MAP`（含 `negative_emotion`）+ `Agent.run_routed()` 编排；slim base + 资料/规则按意图下沉到 patch（`other` 兜底承载全量规则；catalog 通过 `_CATALOG_SNIPPET` 挂 `product_inquiry` + `other`）；`escalation` / `negative_emotion` 都走 fast_path
- **M3 两层记忆**：`agent/memory/{events,short_term,long_term}.py`；`run_routed` 入口先压后跑；`RunResult.memory_events` 可观测；REPL 在 `/reset`/`/user`/`/quit`/ 转人工 时 `save_session`
- **M7 情绪感知**：`negative_emotion` 独立 intent 并入路由漏斗（§9 #10）；规则 12 词 + LLM 8 标签；与 `escalation` 共用 fast_path
- **M6 护栏**：`agent/guardrails/injection.py` 规则层预检（三类：角色劫持 / 提示泄露 / 越狱关键词；正序 + 中文倒装），`run_routed` 最前一道闸；命中 `intent="injection"` / `route_path="guardrails"` / 固定拒绝文案、0 LLM
- **测试 & 人工**：129 pytest 全绿（含双模型分层 3 条契约测试）；`scripts/smoke_agent.py` S1–S5 共 9 子场景；`scripts/chat.py` REPL 按"路由 / 工具子集 / 记忆事件 / system / 工具调用"顺序打 trace

### 下一步

- W5 评测体系：E1 评测集建立（80–100 条样本，覆盖 §5 的 6 个 Fixture 剧本）、E2 自动指标 harness、E3 LLM-as-Judge、E4 消融（记忆/检索/组件三路对比）、E5 红队
- 红队数据出来后，给 `agent/guardrails/injection.py` 加 LLM 分类兜底层（规则漏召回补刀）

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
| 1 | 单模型 qwen-plus + 自研 ReAct + 工具统一 OpenAI function calling schema | 聚焦原理展现；schema 便于跨 LLM 复用；不用 LangChain 以免被抽象遮蔽细节 |
| 2 | 去向量 RAG / 去语义记忆：商品走 catalog+全量加载、政策整文档返回、记忆走 md 摘要 | 数据规模 < 15K tokens + 短 horizon，向量属过度工程化；架构收敛本身即简历亮点 |
| 3 | 退货核心安全改**工具层鉴权**，不做 FSM 状态机 | Tool 即 security boundary，无论对话路径都拦得住越权，比 FSM 更灵活 |
| 4 | ReAct 派发前硬覆盖 tool_call 的 `user_id` | session 上下文不交给 LLM 决定，与工具层鉴权形成分层防御 |
| 5 | 工具预算 5；`query_orders` 不收 `order_id` | 用户记不住订单号，同时从架构上消除跨用户枚举面；去掉与客服弱相关的 recommend / 冗余的 track_logistics |
| 6 | 不落运行期 trace 表 | 评测阶段由 eval harness 现场采集 latency/token/cost 即够用，运行期全量 trace ROI 低 |
| 7 | 意图路由走 **Framing A「config 派」**（单 Agent + 配置表），不拆 sub-agent + orchestrator | 各意图 prompt_patch 本质同构；`(exposed_tools, prompt_patch, fast_path_tool)` 元组将来可平滑升级为 Framing B |
| 8 | Slim base + 规则/资料按意图下沉到 `prompt_patch` | base 仅身份+格式；catalog 与行为规则按意图挂 patch，每轮 tokens 更短；"routing 不只裁工具，还裁行为"成为路由叙事 |
| 9 | 两层记忆：短期 Summary-Buffer（会话内压缩）+ 长期 per-user append-only md（冷启动 load 最近 3 段） | 短期解决会话膨胀，长期解决老客跨会话召回；`_lt_cache` 按轮重注入新 system，抵消 patch 替换 `messages[0]` |
| 10 | 情绪感知并入 router 作独立 intent `negative_emotion`（与 `escalation` 共用 fast_path）；Prompt 注入独立 `guardrails` 层 | 情绪是"意图的延伸"、注入是"威胁行为，不是意图"——guardrails/router/ReAct 三层各司其职；注入纯正则够用，LLM 兜底延后到 W5 红队数据出来再加 |
| 11 | 双模型分层：主 qwen-plus（ReAct 主循环）+ 轻量 qwen-turbo（路由分类 / 短期压缩 / 长期摘要）；`chat(model=...)` 参数注入 | 把"便宜之前跑贵的"思路从规则→LLM 再延伸到模型级：三处轻量任务无 tool use、输入短、输出短，plus 属于杀鸡用牛刀；成本随量下降，函数调用与最终答复质量不妥协。W5 E4 可加一组"全 plus / 分层 / 全 turbo"对比把代价讲清楚 |

---

## 10. 开放问题

- [ ] 评测集具体场景细化（W5 之前敲定）
- [ ] 红队攻击用例库来源（自己设计 vs 参考公开数据集）
- [ ] 退款流程对"边界日期"订单的处理（例如发货第 7 天）是否需要额外 prompt 话术引导
- [ ] 路由 confidence 阈值 0.6 是凭经验拍的，W5 用真实样本标注后再回调
- [ ] 分类 LLM 的 prompt 已加"多意图→other"负样本；单意图误分率仍待 W5 真实标注后再迭代 few-shot
- [ ] 意图粒度：`order_query` 是否拆"物流进度" vs "状态确认"、`refund_request` 是否独立出"已发货 vs 未发货"子类——视 W5 分桶数据决定
- [ ] 长期记忆 **合并/淘汰策略**：当前 per-user md 会无限 append，`load_block` 靠只取最近 3 段兜住。超活跃用户若累计几十段，老段落永远进不了冷启动上下文——W5 之前定夺是否做"每 N 段 → 合并成一段元摘要"（对齐 MemGPT/Letta 的 memory consolidation），还是保持 append-only 让 W5 消融数据说话

---

## 维护约定

- **每次开工前**：读 §7（现状）+ §10（待定）
- **每次收工前**：更新 §4/§5 checkbox + §7「已完成」「下一步」；有新决策追加 §9 一行
- **不在 PLAN.md 里留流水账**：每次提交细节（单测数量、commit 信息）一律进 git log
