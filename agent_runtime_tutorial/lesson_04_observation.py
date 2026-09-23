"""Lesson 4: turn a tool result into an observation message."""

from __future__ import annotations

import json

from lesson_02_tool_call_shape import Message, ToolCall, build_weather_tool_call_message
from lesson_03_first_tool import Tool, execute_tool, weather_tool


def execute_and_make_observation(tool_call: ToolCall, tool: Tool) -> Message:
    """Execute one call and package its result for the next LLM request."""

    result = execute_tool(tool=tool, arguments_json=tool_call.arguments)

    return Message(
        role="tool",
        content=json.dumps(result, ensure_ascii=False),
        tool_call_name=tool_call.name,
        tool_call_id=tool_call.id,
    )


def main() -> None:
    assistant_message = build_weather_tool_call_message()
    tool_call = assistant_message.tool_calls[0]

    observation = execute_and_make_observation(
        tool_call=tool_call,
        tool=weather_tool,
    )

    print(f"assistant requested tool_call_id = {tool_call.id}")
    print(f"observation role = {observation.role}")
    print(f"observation tool_call_name = {observation.tool_call_name}")
    print(f"observation tool_call_id = {observation.tool_call_id}")
    print(f"observation content = {observation.content}")


if __name__ == "__main__":
    main()
