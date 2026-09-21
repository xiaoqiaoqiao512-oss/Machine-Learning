"""Lesson 2: represent a model-requested tool call."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ToolCall:
    """A structured request from the model to call one tool."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class Message:
    """A message that may contain text, tool calls, or both."""

    role: Role
    content: Optional[str] = None
    tool_calls: tuple[ToolCall, ...] = ()


def build_weather_tool_call_message() -> Message:
    """Create the kind of message a model might return."""

    return Message(
        role="assistant",
        content=None,
        tool_calls=(
            ToolCall(
                id="call_001",
                name="get_weather",
                arguments='{"city": "北京"}',
            ),
        ),
    )


def print_tool_call_summary(message: Message) -> None:
    for tool_call in message.tool_calls:
        print(f"tool call id = {tool_call.id}")
        print(f"tool name = {tool_call.name}")
        print(f"tool arguments = {tool_call.arguments}")


def main() -> None:
    message = build_weather_tool_call_message()
    print(f"role = {message.role}")
    print_tool_call_summary(message)


if __name__ == "__main__":
    main()
