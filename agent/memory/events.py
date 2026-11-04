"""Memory events for observability.

Each short- or long-term memory action emits a MemoryEvent in RunResult.memory_events.
Like route_path and confidence, events are returned as data rather than side-effect callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryEvent:
    kind: str       # "compress" | "load_long_term" | "save_long_term"
    detail: str     # Compression detail includes the full summary for inspection in the REPL.
