"""
Milestone 43: Streaming Tool / Function Calling Parser Engine

Modern agentic LLMs invoke external tools using structured XML / JSON tags:
    <tool_call>
    {"name": "calculator", "arguments": {"expression": "42 * 10"}}
    </tool_call>

This module implements a streaming pushdown parser that monitors token-by-token output:
1. Buffers tokens when entering the tool trigger sequence (<tool_call>).
2. Suppresses raw tool tags from user-facing text stream.
3. Incrementally validates JSON arguments.
4. Emits structured ToolCall objects upon closing tag (</tool_call>) for dispatch.
"""

from dataclasses import dataclass, field
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: Dict[str, Any]


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    raw_payload: str


class ToolRegistry:
    """Registry for registering and dispatching external functions."""

    def __init__(self):
        self.tools: Dict[str, Tuple[ToolDefinition, Callable]] = {}

    def register(self, definition: ToolDefinition, func: Callable) -> None:
        self.tools[definition.name] = (definition, func)

    def dispatch(self, tool_call: ToolCall) -> Any:
        if tool_call.name not in self.tools:
            raise ValueError(f"Tool '{tool_call.name}' not found in registry.")
        _, func = self.tools[tool_call.name]
        return func(**tool_call.arguments)


class StreamingToolParser:
    """
    Incremental state machine for streaming token output with tool call detection.
    """

    START_TAG = "<tool_call>"
    END_TAG = "</tool_call>"

    def __init__(self):
        self.buffer = ""
        self.in_tool_call = False
        self.tool_buffer = ""
        self.completed_tool_calls: List[ToolCall] = []

    def feed(self, delta: str) -> Tuple[str, List[ToolCall]]:
        """
        Feeds a newly arrived text delta from the LLM stream.
        Returns:
            user_text_delta: Text meant for user consumption (excluding tool tags)
            new_tool_calls: List of any ToolCall instances completed in this chunk
        """
        self.buffer += delta
        user_text = ""
        new_calls: List[ToolCall] = []

        while self.buffer:
            if not self.in_tool_call:
                start_idx = self.buffer.find(self.START_TAG)
                if start_idx != -1:
                    # Everything before start tag is regular user text
                    user_text += self.buffer[:start_idx]
                    self.buffer = self.buffer[start_idx + len(self.START_TAG) :]
                    self.in_tool_call = True
                    self.tool_buffer = ""
                else:
                    # Check if buffer ends with a partial prefix of START_TAG
                    partial_match = False
                    for i in range(len(self.START_TAG) - 1, 0, -1):
                        if self.buffer.endswith(self.START_TAG[:i]):
                            user_text += self.buffer[:-i]
                            self.buffer = self.buffer[-i:]
                            partial_match = True
                            break
                    if partial_match:
                        break
                    else:
                        user_text += self.buffer
                        self.buffer = ""
            else:
                end_idx = self.buffer.find(self.END_TAG)
                if end_idx != -1:
                    # Tool call payload completed
                    self.tool_buffer += self.buffer[:end_idx]
                    self.buffer = self.buffer[end_idx + len(self.END_TAG) :]
                    self.in_tool_call = False

                    # Parse JSON payload
                    tool_call = self._parse_payload(self.tool_buffer.strip())
                    if tool_call:
                        new_calls.append(tool_call)
                        self.completed_tool_calls.append(tool_call)
                    self.tool_buffer = ""
                else:
                    # Accumulate in tool buffer
                    self.tool_buffer += self.buffer
                    self.buffer = ""

        return user_text, new_calls

    def _parse_payload(self, raw_str: str) -> Optional[ToolCall]:
        try:
            data = json.loads(raw_str)
            name = data.get("name")
            args = data.get("arguments", {})
            if name:
                return ToolCall(name=name, arguments=args, raw_payload=raw_str)
        except Exception:
            # Fallback regex extraction if JSON is slightly malformed
            match = re.search(r'["\']name["\']\s*:\s*["\']([^"\']+)["\']', raw_str)
            if match:
                name = match.group(1)
                return ToolCall(name=name, arguments={}, raw_payload=raw_str)
        return None
