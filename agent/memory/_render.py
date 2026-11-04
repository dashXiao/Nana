"""Shared renderer: flatten messages into text for short- and long-term summarization."""

from __future__ import annotations


def render_for_summary(dialogue: list[dict]) -> str:
    lines: list[str] = []
    for m in dialogue:
        role = m.get("role", "?")
        content = m.get("content") or ""
        if role == "assistant" and m.get("tool_calls"):
            names = [tc.get("function", {}).get("name", "?") for tc in m["tool_calls"]]
            lines.append(f"[assistant→tool_calls] {', '.join(names)}")
            if content:
                lines.append(f"[assistant] {content}")
        elif role == "tool":
            lines.append(f"[tool_result] {content[:200]}")
        else:
            lines.append(f"[{role}] {content}")
    return "\n".join(lines)
