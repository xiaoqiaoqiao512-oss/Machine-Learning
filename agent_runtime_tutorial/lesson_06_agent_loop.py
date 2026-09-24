"""Lesson 6: turn one tool round-trip into an Agent Loop."""

from __future__ import annotations

from lesson_02_tool_call_shape import Message
from lesson_03_first_tool import weather_tool
from lesson_04_observation import execute_and_make_observation
from lesson_05_round_trip import ScriptedDemoLLM, print_messages


def run_agent(llm: ScriptedDemoLLM) -> list[Message]:
    messages = [
        Message(role="system", content="你是一个简洁、友好的天气助手。"),
        Message(role="user", content="北京今天天气怎么样？需要带伞吗？"),
    ]

    # 目前这里只请求了一次 LLM。
    # 请把下面这两行替换成 Agent Loop：
    max_iterations = 5

    for i in range(max_iterations):
        assistant_message = llm.complete(messages)
        messages.append(assistant_message)

        if not assistant_message.tool_calls:
            break

        for tool_call in assistant_message.tool_calls:
            if tool_call.name == weather_tool.name:
                observation = execute_and_make_observation(
                    tool_call=tool_call,
                    tool=weather_tool,
                )
                messages.append(observation)

    messages.append(Message(role="assistant", content="超过最大次数，停止循环。"))
    return messages


def main() -> None:
    messages = run_agent(ScriptedDemoLLM())
    print_messages(messages)


if __name__ == "__main__":
    main()
