"""Lesson 3: define one tool, without executing it yet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import json


def get_weather(city: str) -> dict[str, Any]:
    """Return deterministic fake weather data for this lesson."""

    fake_weather = {
        "北京": {"temperature_c": 26, "rain": True},
        "上海": {"temperature_c": 29, "rain": False},
    }
    return {
        "city": city,
        **fake_weather.get(city, {"temperature_c": None, "rain": None}),
    }


@dataclass(frozen=True)
class Tool:
    """The metadata and executable handler belonging to one tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]


weather_tool = Tool(
    name="get_weather",
    description="查询指定城市的天气。",
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称"
            }
        },
        "required": ["city"],
        "additionalProperties": False,
    },
    handler=get_weather,
)


def execute_tool(
        tool: Tool,
        arguments_json: str,
)->dict[str, Any]:
    arguments = json.loads(arguments_json)
    return tool.handler(**arguments)


def main() -> None:
    print(f"tool name = {weather_tool.name}")
    print(f"tool description = {weather_tool.description}")
    print(f"tool parameters = {weather_tool.parameters}")
    print(f"python handler = {weather_tool.handler.__name__}")

    print("tool execution result = ", execute_tool(
        tool=weather_tool,
        arguments_json='{"city": "北京"}',
    ))


if __name__ == "__main__":
    main()
