"""Guardrails · 输入预检层（注入过滤等）。

与 `agent/router.py` 的职责分离：
- router 把合法用户消息打 intent 标签
- guardrails 判断消息是不是 "应该被拒绝" 的威胁——注入、角色劫持、越狱模板等

命中后 `Agent.run_routed` 短路返回固定拒绝文案，零 LLM 调用，不经过 router。
详细威胁模型与设计权衡见 PLAN §9 #22。
"""

from agent.guardrails.injection import InjectionDetector

__all__ = ["InjectionDetector"]
