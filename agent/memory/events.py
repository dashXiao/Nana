"""记忆事件 · 可观测性数据结构。

短/长期记忆的每一次"有意义动作"都生成一条 `MemoryEvent`，随 `RunResult`
回传，供 REPL 肉眼审查 / 评测断言。与 `route_path / confidence` 同构：
**数据返回，而不是副作用回调**，保证链路纯函数化、易于测试。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryEvent:
    """一次记忆层动作的快照。

    - `kind`:
      - "compress"          → 短期记忆触发了摘要压缩
      - "load_long_term"    → 长期记忆加载了（或尝试加载了）历史块
      - "save_long_term"    → 长期记忆把本会话写回文件（或因门槛跳过）
    - `detail`: 人类可读文字；对 compress 事件包含**完整摘要文本**，
      便于 REPL 打印并肉眼判断压缩质量。
    """

    kind: str
    detail: str
