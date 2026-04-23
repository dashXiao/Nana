"""记忆事件 · 可观测性数据结构。

短/长期记忆的每个动作都生成一条 `MemoryEvent`，随 `RunResult.memory_events`
回传——与 `route_path/confidence` 同构：数据回传而不是副作用回调。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryEvent:
    kind: str       # "compress" | "load_long_term" | "save_long_term"
    detail: str     # compress 事件的 detail 含完整摘要文本，便于 REPL 观察质量
