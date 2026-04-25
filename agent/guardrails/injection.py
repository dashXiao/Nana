"""Prompt 注入检测 · 规则黑名单：role_hijack / prompt_leak / jailbreak 三类。"""

from __future__ import annotations

import re

_ROLE_HIJACK = re.compile(
    r"(忽略|无视|ignore|disregard|forget)"
    r".{0,10}"
    r"(前面|之前|上面|所有|previous|prior|all|above|your)"
    r".{0,10}"
    r"(指令|规则|设定|命令|prompt|instruction|rule)",
    re.IGNORECASE | re.DOTALL,
)

# 中文常用倒装"你的 prompt 给我看看"，正序倒序两支都覆盖
_PROMPT_LEAK = re.compile(
    r"(?:"
    r"(?:告诉|输出|显示|打印|给我看|show|reveal|print|repeat|output)"
    r".{0,10}"
    r"(?:你的|your)"
    r".{0,10}"
    r"(?:system|prompt|指令|设定|instruction|提示词)"
    r"|"
    r"(?:你的|your)"
    r".{0,10}"
    r"(?:system|prompt|指令|设定|instruction|提示词)"
    r".{0,15}"
    r"(?:告诉|输出|显示|打印|给我看|show|reveal|print|repeat|output)"
    r")",
    re.IGNORECASE | re.DOTALL,
)

_JAILBREAK_TOKENS = re.compile(
    r"(DAN|jailbreak|越狱|developer mode|dev mode|godmode|do anything now)",
    re.IGNORECASE,
)


class InjectionDetector:
    """detect(msg) 返回命中的模式名 ('role_hijack'/'prompt_leak'/'jailbreak') 或 None。"""

    def detect(self, msg: str) -> str | None:
        text = msg or ""
        if _ROLE_HIJACK.search(text):
            return "role_hijack"
        if _PROMPT_LEAK.search(text):
            return "prompt_leak"
        if _JAILBREAK_TOKENS.search(text):
            return "jailbreak"
        return None
