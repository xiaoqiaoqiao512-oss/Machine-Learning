"""Lesson 1: the smallest useful LLM boundary.

Run with the deterministic demo model:

    python3 agent_runtime_tutorial/lesson_01_basic_chat.py

Run with an OpenAI-compatible Chat Completions endpoint:

    OPENAI_API_KEY=... python3 agent_runtime_tutorial/lesson_01_basic_chat.py \
        --provider openai --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from dataclasses import asdict, dataclass
from typing import Literal, Protocol, Sequence


Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    """One item in the conversation sent to, or returned by, an LLM."""

    role: Role
    content: str


class LLM(Protocol):
    """The only capability the rest of our runtime needs from a model."""

    def complete(self, messages: Sequence[Message]) -> Message:
        ...


class DemoLLM:
    """A deterministic stand-in so the lesson runs without network or API key."""

    def complete(self, messages: Sequence[Message]) -> Message:
        latest_user_message = next(
            message.content
            for message in reversed(messages)
            if message.role == "user"
        )
        return Message(
            role="assistant",
            content=f"这是 DemoLLM 的回复。我看到了你的问题：{latest_user_message}",
        )


class OpenAICompatibleLLM:
    """A tiny dependency-free adapter for Chat Completions-compatible APIs."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1/chat/completions",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def complete(self, messages: Sequence[Message]) -> Message:
        request_body = {
            "model": self.model,
            "messages": [asdict(message) for message in messages],
        }
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(request_body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=60) as response:
            response_body = json.load(response)

        assistant_message = response_body["choices"][0]["message"]
        return Message(
            role="assistant",
            content=assistant_message["content"] or "",
        )


def build_llm(provider: str, model: str) -> LLM:
    if provider == "demo":
        return DemoLLM()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "使用 --provider openai 需要先设置 OPENAI_API_KEY；"
            "或者不带参数运行 DemoLLM。"
        )
    return OpenAICompatibleLLM(api_key=api_key, model=model)


def run_two_conversation(llm: LLM) -> list[Message]:
    messages: list[Message] = [
        Message(role="system", content="你是一个简洁、友好的助手。"),
        Message(role="user", content="请用一句话解释 Agent loop 是什么。")
    ]

    demo_assistant_meassage = DemoLLM(messages)
    messages.append(demo_assistant_meassage)

    messages.append(Message(role="user", content="如何构建一个 Agent loop？"))

    demo_assistant_meassage = DemoLLM(messages)
    messages.append(demo_assistant_meassage)

    for message in messages:
        print(f"{message.role}: {message.content}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("demo", "openai"), default="demo")
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    messages: list[Message] = [
        Message(role="system", content="你是一个简洁、友好的助手。"),
        Message(role="user", content="请用一句话解释 Agent Loop 是什么。"),
    ]

    llm = build_llm(args.provider, args.model)
    assistant_message = llm.complete(messages)
    messages.append(assistant_message)

    print("--- messages sent/collected by the runtime ---")
    for message in messages:
        print(f"{message.role}: {message.content}")


if __name__ == "__main__":
    main()