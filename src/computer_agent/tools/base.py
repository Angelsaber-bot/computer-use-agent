"""Base definitions shared by every agent tool."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import sys
from typing import Any, ClassVar, Mapping


class ToolValidationError(ValueError):
    """Raised when tool arguments do not match the tool definition."""


@dataclass(frozen=True, slots=True)
class ToolParameter:
    """Definition of one argument accepted by a tool."""

    python_type: type | tuple[type, ...]
    description: str
    required: bool = True
    default: Any = None

    @property
    def json_type(self) -> str:
        """Return the closest JSON type name."""

        python_types = (
            self.python_type
            if isinstance(self.python_type, tuple)
            else (self.python_type,)
        )

        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            dict: "object",
            list: "array",
        }

        json_types = {
            type_map.get(python_type, "object")
            for python_type in python_types
        }

        if json_types == {"integer", "number"}:
            return "number"

        if len(json_types) == 1:
            return json_types.pop()

        return "object"


class BaseTool(ABC):
    """Common interface implemented by every agent tool."""

    name: ClassVar[str]
    description: ClassVar[str]

    parameters: ClassVar[Mapping[str, ToolParameter]] = {}

    supported_platforms: ClassVar[tuple[str, ...]] = (
        "darwin",
        "linux",
        "win32",
    )

    def is_available(self, platform: str | None = None) -> bool:
        """Return whether the tool supports a platform."""

        current_platform = platform or sys.platform
        return current_platform in self.supported_platforms

    def validate_arguments(
        self,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Validate arguments and insert optional defaults."""

        unknown_arguments = set(arguments) - set(self.parameters)

        if unknown_arguments:
            names = ", ".join(sorted(unknown_arguments))
            raise ToolValidationError(
                f"unknown argument(s): {names}"
            )

        validated: dict[str, Any] = {}

        for name, parameter in self.parameters.items():
            if name not in arguments:
                if parameter.required:
                    raise ToolValidationError(
                        f"missing required argument: {name}"
                    )

                validated[name] = parameter.default
                continue

            value = arguments[name]

            if not isinstance(value, parameter.python_type):
                expected = self._type_name(parameter.python_type)
                actual = type(value).__name__

                raise ToolValidationError(
                    f"argument '{name}' must be "
                    f"{expected}, not {actual}"
                )

            validated[name] = value

        return validated

    def schema(self) -> dict[str, Any]:
        """Return a planner-readable description of the tool."""

        properties: dict[str, dict[str, Any]] = {}
        required: list[str] = []

        for name, parameter in self.parameters.items():
            properties[name] = {
                "type": parameter.json_type,
                "description": parameter.description,
            }

            if parameter.required:
                required.append(name)
            else:
                properties[name]["default"] = parameter.default

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
            "supported_platforms": list(
                self.supported_platforms
            ),
        }

    @abstractmethod
    def run(self, **arguments: Any) -> Any:
        """Execute the tool using validated arguments."""

    @staticmethod
    def _type_name(
        expected: type | tuple[type, ...],
    ) -> str:
        expected_types = (
            expected
            if isinstance(expected, tuple)
            else (expected,)
        )

        return " or ".join(
            expected_type.__name__
            for expected_type in expected_types
        )