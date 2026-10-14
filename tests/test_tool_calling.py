"""
Tests for Milestone 43: Streaming Tool / Function Calling Parser.
"""

import pytest

from inference.tool_calling import (
    ToolDefinition,
    ToolCall,
    ToolRegistry,
    StreamingToolParser,
)


def test_tool_registry_dispatch():
    registry = ToolRegistry()

    def get_weather(location: str, unit: str = "celsius") -> str:
        return f"Weather in {location}: 22 degrees {unit}"

    def_tool = ToolDefinition(
        name="get_weather",
        description="Fetch current weather",
        parameters={"location": "string", "unit": "string"},
    )
    registry.register(def_tool, get_weather)

    call = ToolCall(name="get_weather", arguments={"location": "Tokyo", "unit": "celsius"}, raw_payload="")
    res = registry.dispatch(call)
    assert res == "Weather in Tokyo: 22 degrees celsius"


def test_streaming_tool_parser_incremental():
    parser = StreamingToolParser()

    # Stream chunks arriving one by one
    chunks = [
        "I will check the weather for you.",
        " <tool_",
        "call>\n",
        '{"name": "get_weather", ',
        '"arguments": {"location": "London"}}',
        "\n</tool_call>",
        " Please wait.",
    ]

    all_user_text = ""
    collected_calls = []

    for c in chunks:
        user_text, calls = parser.feed(c)
        all_user_text += user_text
        collected_calls.extend(calls)

    assert "I will check the weather for you." in all_user_text
    assert "Please wait." in all_user_text
    assert "<tool_call>" not in all_user_text
    assert "</tool_call>" not in all_user_text

    assert len(collected_calls) == 1
    call = collected_calls[0]
    assert call.name == "get_weather"
    assert call.arguments == {"location": "London"}


def test_streaming_tool_parser_no_tools():
    parser = StreamingToolParser()
    text, calls = parser.feed("Hello there! How can I assist you today?")
    assert text == "Hello there! How can I assist you today?"
    assert len(calls) == 0
