"""Guardrails: input preflight checks, including injection filtering.

Responsibilities are separate from agent/router.py:
- The router labels legitimate user messages with an intent.
- Guardrails detect threats to refuse: injection, role hijacking, and jailbreak templates.

On a match, Agent.run_routed returns a fixed refusal without routing or any LLM call.
"""

from agent.guardrails.injection import InjectionDetector

__all__ = ["InjectionDetector"]
