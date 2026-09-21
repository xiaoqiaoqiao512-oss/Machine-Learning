"""Lesson 3: define one tool, without executing it yet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


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
    # TODO: 请你补充这个工具的 JSON Schema。
    parameters={},
    handler=get_weather,
)


def main() -> None:
    print(f"tool name = {weather_tool.name}")
    print(f"tool description = {weather_tool.description}")
    print(f"tool parameters = {weather_tool.parameters}")
    print(f"python handler = {weather_tool.handler.__name__}")


if __name__ == "__main__":
    main()
