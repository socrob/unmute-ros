"""Timestamped event log for the full-duplex evaluation harness.

Every interaction metric in `.claude/EVALUATION_DESIGN.md` (FTED, VSL, ISR, IIR)
is a delta between two events in this log, and the Track-A injector triggers its
barge-ins by tailing it live. So records are flushed on write.

**Off by default.** A quick smoke-test run should not drag eval instrumentation
along; set ``EVAL_EVENT_LOG`` to turn it on:

    EVAL_EVENT_LOG=/path/to/events.jsonl   -> log there
    EVAL_EVENT_LOG=true  (with LOG_DIR)    -> $LOG_DIR/events.jsonl
    unset / false                          -> no-op, zero overhead

Each record is one JSON object per line:

    {"t_mono": 12.3456, "t_wall": "2026-09-05T00:45:01.123456+00:00",
     "seq": 7, "turn": 2, "type": "assistant.audio_first", ...fields}

``t_mono`` (``time.monotonic``) is the metric clock -- immune to NTP steps, so
deltas are trustworthy. ``t_wall`` exists only to line events up against the
other per-run logs in the same directory.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("UnmuteBridge.eval")

# Monitor-only bus events from the action executor (towards_fd
# ROBOT_INTEGRATION_PLAN sec. 3.3) -> their events.jsonl type. Logged here and
# never forwarded to the remote backend: the LLM-facing <action_result> contract
# has no room for them.
ROBOT_MONITOR_EVENTS: dict[str, str] = {
    "robot.system_status": "robot.status",
    "robot.action_started": "action.started",
    "robot.action_progress": "action.progress",
    "robot.action_finished": "action.finished",
}

# Keys EventLog.emit owns: its `event` parameter and the record's own fields.
_RESERVED_KEYS = frozenset({"event", "t_mono", "t_wall", "seq", "turn", "type"})


def robot_monitor_record(data: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """Map a robot monitor bus event to `(events.jsonl type, fields)`, else None.

    The bus `type` is dropped and `name` becomes `action`, as for
    `exec.cancelled`. A payload key that `EventLog.emit` owns is kept under a
    `robot_` prefix so it can neither clash with `event` nor overwrite the record.
    """
    event = ROBOT_MONITOR_EVENTS.get(str(data.get("type")))
    if event is None:
        return None
    fields: dict[str, Any] = {}
    for key, value in data.items():
        if key == "type":
            continue
        if key == "name":
            key = "action"
        elif key in _RESERVED_KEYS:
            key = f"robot_{key}"
        fields[key] = value
    return event, fields


class EventLog:
    """Append-only JSONL sink. Use `EventLog.from_env()`; call `emit()` freely."""

    def __init__(self, path: str | None):
        self.path = path
        self.turn = 0
        self._seq = 0
        self._fh = None
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            # Line-buffered so a live tail (the injector) sees events immediately.
            self._fh = open(path, "a", buffering=1, encoding="utf-8")
            logger.info("Eval event log: %s", path)

    @property
    def enabled(self) -> bool:
        return self._fh is not None

    @classmethod
    def from_env(cls) -> "EventLog":
        raw = (os.environ.get("EVAL_EVENT_LOG") or "").strip()
        if not raw or raw.lower() in ("0", "false", "no", "off"):
            return cls(None)
        if raw.lower() in ("1", "true", "yes", "on"):
            log_dir = os.environ.get("LOG_DIR")
            if not log_dir:
                logger.warning(
                    "EVAL_EVENT_LOG is on but LOG_DIR is unset; event log disabled."
                )
                return cls(None)
            return cls(os.path.join(log_dir, "events.jsonl"))
        return cls(raw)  # an explicit path

    def next_turn(self) -> int:
        """Start a new assistant response cycle; returns the new turn id."""
        self.turn += 1
        return self.turn

    def emit(self, event: str, **fields: Any) -> None:
        """Record one event. A no-op (and cheap) when logging is disabled."""
        if self._fh is None:
            return
        self._seq += 1
        record = {
            "t_mono": round(time.monotonic(), 6),
            "t_wall": datetime.now(timezone.utc).isoformat(),
            "seq": self._seq,
            "turn": self.turn,
            "type": event,
        }
        record.update(fields)
        try:
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:  # instrumentation must never break the bridge
            logger.warning("Could not write eval event %s: %s", type, exc)

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None
