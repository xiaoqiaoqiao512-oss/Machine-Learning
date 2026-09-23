"""Lesson 5: feed a tool observation back into the conversation."""

from __future__ import annotations

import json
from typing import Sequence

from lesson_02_tool_call_shape import Message, build_weather_tool_call_message
from lesson_03_first_tool import weather_tool
from lesson_04_observation import execute_and_make_observation


class ScriptedDemoLLM:
    """Return a fixed tool request, then answer from a tool observation."""

    def complete(self, messages: Sequence[Message]) -> Message:
        observation = next(
            (message for message in reversed(messages) if message.role == "tool"),
            None,
        )
        if observation is None:
            return build_weather_tool_call_message()

        weather = json.loads(observation.content or "{}")
        if weather.get("rain"):
            advice = "可能下雨，建议带伞。"
        else:
            advice = "目前看起来不需要带伞。"

        return Message(
            role="assistant",
            content=(
                f"{weather.get('city')}当前气温是 "
                f"{weather.get('temperature_c')}°C。{advice}"
            ),
        )


def print_messages(messages: Sequence[Message]) -> None:
    for message in messages:
        if message.role == "assistant" and message.tool_calls:
            for tool_call in message.tool_calls:
                print(
                    f"assistant: tool_call {tool_call.id} "
                    f"{tool_call.name}({tool_call.arguments})"
                )
        elif message.role == "tool":
            print(
                f"tool[{message.tool_call_id}]: {message.content}"
            )
        else:
            print(f"{message.role}: {message.content}")


def main() -> None:
    messages = [
        Message(role="system", content="你是一个简洁、友好的天气助手。"),
        Message(role="user", content="北京今天天气怎么样？需要带伞吗？"),
    ]
    llm = ScriptedDemoLLM()

    assistant_message = llm.complete(messages)
    messages.append(assistant_message)

    for tool_call in assistant_message.tool_calls:
        if tool_call.name != weather_tool.name:
            raise ValueError(f"未知工具：{tool_call.name}")

        observation = execute_and_make_observation(
            tool_call=tool_call,
            tool=weather_tool,
        )

        

    final_message = llm.complete(messages)
    messages.append(final_message)

    print_messages(messages)


if __name__ == "__main__":
    main()
