# NanaShop Agent

电商客服多轮对话 Agent · 自研 ReAct 循环 + 双模型分层 + 5 工具 + 两层记忆 + 多层防御。

## 项目特色
1. 

## 快速开始

见 [docs/quick-start.md](docs/quick-start.md)

## 项目架构

![架构图](docs/architecture.png)

## 主要目录结构

```
nana/
├─ agent/                  # Agent 核心
│  ├─ core.py              #   ReAct 主循环
│  ├─ llm.py               #   双模型客户端
│  ├─ router.py            #   意图路由
│  ├─ guardrails/          #   入口防御
│  ├─ memory/              #   记忆处理
│  └─ tools/               #   工具
├─ knowledge/              # 商品 / 政策知识
├─ db/                     # SQLite 数据层
├─ memory/sessions/        # 长期记忆
├─ scripts/
│  ├─ webui.py             #   Web UI
│  ├─ chat.py              #   命令行 REPL
│  ├─ seed.py              #   数据初始化
│  ├─ seed_data.py         #   种子数据
│  └─ smoke_*.py           #   冒烟脚本
├─ eval/                   # 评测体系
├─ tests/                  # 测试
└─ docs/                   # 相关文档
```

## 使用展示
#### 咨询商品
![](docs/商品查询.png)

#### 长期记忆
![](docs/长期记忆.png)

#### 防注入护栏
![](docs/防注入.png)

#### 退货
依次查询订单信息与退货政策，拒绝退货
![](docs/退货.png)

## 评测
