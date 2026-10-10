import pytest
from inference.json_constrained import SimpleJSONStateTracker


def test_json_state_tracker_valid_sequence():
    tracker = SimpleJSONStateTracker()
    valid_json = '{"name": "Lily", "age": 5}'

    for char in valid_json:
        valid = tracker.update(char)
        assert valid, f"Tracker falsely rejected valid character: {char}"

    assert tracker.is_completed
    assert tracker.brace_depth == 0


def test_json_state_tracker_rejects_invalid_syntax():
    tracker = SimpleJSONStateTracker()
    # Attempting closing brace before opening brace
    assert not tracker.update("}")

    # Start valid
    tracker = SimpleJSONStateTracker()
    assert tracker.update("{")
    assert tracker.update("}")
    assert tracker.is_completed
    # After completion, trailing non-whitespace characters should be rejected
    assert not tracker.update("x")
