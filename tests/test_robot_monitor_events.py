import json
from pathlib import Path

import pytest

from unmute.eval_events import ROBOT_MONITOR_EVENTS, EventLog, robot_monitor_record


@pytest.mark.parametrize(
    ("bus_type", "event"),
    [
        ("robot.system_status", "robot.status"),
        ("robot.action_started", "action.started"),
        ("robot.action_progress", "action.progress"),
        ("robot.action_finished", "action.finished"),
    ],
)
def test_maps_every_monitor_event(bus_type: str, event: str):
    assert ROBOT_MONITOR_EVENTS[bus_type] == event
    record = robot_monitor_record({"type": bus_type, "name": "move", "phase": "x"})
    assert record == (event, {"action": "move", "phase": "x"})


def test_payloads():
    assert robot_monitor_record(
        {
            "type": "robot.system_status",
            "ready": True,
            "components": {"navigation": "up"},
            "arena": "korea_arena3",
        }
    ) == (
        "robot.status",
        {"ready": True, "components": {"navigation": "up"}, "arena": "korea_arena3"},
    )
    assert robot_monitor_record(
        {
            "type": "robot.action_started",
            "name": "pick",
            "parameters": {"object": "uuid-1"},
        }
    ) == ("action.started", {"action": "pick", "parameters": {"object": "uuid-1"}})


@pytest.mark.parametrize(
    "bus_type",
    ["audio", "robot.exec_cancelled", "bridge.action_result", "robot.text", None],
)
def test_other_types_are_not_monitor_events(bus_type: str | None):
    assert robot_monitor_record({"type": bus_type, "name": "move"}) is None


def test_reserved_keys_cannot_clash_with_emit():
    record = robot_monitor_record(
        {
            "type": "robot.action_progress",
            "name": "move",
            "event": "e",
            "seq": 5,
            "turn": 9,
            "t_mono": 1.0,
            "t_wall": "w",
        }
    )
    assert record is not None
    event, fields = record
    assert event == "action.progress"
    assert fields == {
        "action": "move",
        "robot_event": "e",
        "robot_seq": 5,
        "robot_turn": 9,
        "robot_t_mono": 1.0,
        "robot_t_wall": "w",
    }


def test_event_log_writes_action_finished(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    log = EventLog(str(path))
    record = robot_monitor_record(
        {
            "type": "robot.action_finished",
            "name": "pick",
            "status": "FAILED",
            "message": "unsafe_to_pick",
            "duration_s": 12.5,
        }
    )
    assert record is not None
    log.emit(record[0], **record[1])
    log.close()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    written = json.loads(lines[0])
    assert written["type"] == "action.finished"
    assert written["action"] == "pick"
    assert written["status"] == "FAILED"
    assert written["message"] == "unsafe_to_pick"
    assert written["duration_s"] == 12.5
    assert written["seq"] == 1 and "t_mono" in written and "name" not in written
