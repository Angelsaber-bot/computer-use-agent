"""Read-only macOS Accessibility control discovery."""

from __future__ import annotations

import math
from typing import Any

try:
    import AppKit
except ImportError:  # pragma: no cover - depends on host platform
    AppKit = None  # type: ignore[assignment]

try:
    import ApplicationServices
except ImportError:  # pragma: no cover - depends on host platform
    ApplicationServices = None  # type: ignore[assignment]

from computer_agent.perception.models import BoundingBox, UIElement


_ROLE_MAP = {
    "AXTextField": "text_field",
    "AXButton": "button",
    "AXCheckBox": "checkbox",
    "AXPopUpButton": "popup_button",
    "AXRadioButton": "radio_button",
}


class MacOSAccessibility:
    """Read semantic controls from the focused macOS Accessibility window."""

    def __init__(
        self,
        maximum_elements: int = 5000,
        maximum_depth: int = 30,
    ) -> None:
        if (
            isinstance(maximum_elements, bool)
            or not isinstance(maximum_elements, int)
        ):
            raise ValueError("maximum_elements must be an integer")

        if maximum_elements <= 0:
            raise ValueError("maximum_elements must be positive")

        if (
            isinstance(maximum_depth, bool)
            or not isinstance(maximum_depth, int)
        ):
            raise ValueError("maximum_depth must be an integer")

        if maximum_depth < 0:
            raise ValueError("maximum_depth must be non-negative")

        self.maximum_elements = maximum_elements
        self.maximum_depth = maximum_depth

    @staticmethod
    def is_available() -> bool:
        """Return whether the required macOS frameworks are importable."""

        return AppKit is not None and ApplicationServices is not None

    @staticmethod
    def is_trusted() -> bool:
        """Return whether the current process has Accessibility trust."""

        if not MacOSAccessibility.is_available():
            return False

        try:
            return bool(ApplicationServices.AXIsProcessTrusted())
        except Exception:
            return False

    def read_frontmost_controls(self) -> list[UIElement]:
        """Return supported controls from the focused frontmost window."""

        if not self.is_available():
            raise RuntimeError(
                "macOS Accessibility frameworks are unavailable"
            )

        if not self.is_trusted():
            raise RuntimeError(
                "macOS Accessibility permission is not trusted"
            )

        application = self._frontmost_application_element()
        if application is None:
            return []

        focused_window = _copy_attribute(
            application,
            _ax_constant("kAXFocusedWindowAttribute"),
        )
        if focused_window is None:
            return []

        controls: list[UIElement] = []
        self._traverse(
            focused_window,
            depth=0,
            controls=controls,
        )

        return controls

    def _frontmost_application_element(self) -> Any | None:
        workspace = AppKit.NSWorkspace.sharedWorkspace()
        application = workspace.frontmostApplication()

        if application is None:
            return None

        pid = application.processIdentifier()

        return ApplicationServices.AXUIElementCreateApplication(pid)

    def _traverse(
        self,
        element: Any,
        *,
        depth: int,
        controls: list[UIElement],
    ) -> None:
        if len(controls) >= self.maximum_elements:
            return

        role = _copy_attribute(
            element,
            _ax_constant("kAXRoleAttribute"),
        )
        mapped_role = _ROLE_MAP.get(role)

        if mapped_role is not None:
            control = _control_from_element(
                element,
                mapped_role,
            )

            if control is not None:
                controls.append(control)

                if len(controls) >= self.maximum_elements:
                    return

        if depth >= self.maximum_depth:
            return

        children = _copy_attribute(
            element,
            _ax_constant("kAXChildrenAttribute"),
        )

        for child in _iter_children(children):
            self._traverse(
                child,
                depth=depth + 1,
                controls=controls,
            )

            if len(controls) >= self.maximum_elements:
                return


def _ax_constant(name: str) -> str:
    return getattr(ApplicationServices, name)


def _copy_attribute(
    element: Any,
    attribute: str,
) -> Any | None:
    try:
        result = ApplicationServices.AXUIElementCopyAttributeValue(
            element,
            attribute,
            None,
        )
    except Exception:
        return None

    if isinstance(result, tuple) and len(result) == 2:
        error_code, value = result
        if error_code != getattr(ApplicationServices, "kAXErrorSuccess", 0):
            return None

        return value

    return result


def _iter_children(children: Any) -> tuple[Any, ...]:
    if children is None or isinstance(children, (str, bytes)):
        return ()

    try:
        return tuple(children)
    except TypeError:
        return ()


def _control_from_element(
    element: Any,
    mapped_role: str,
) -> UIElement | None:
    bounding_box = _bounding_box_from_element(element)

    if bounding_box is None:
        return None

    return UIElement(
        element_type=mapped_role,
        bounding_box=bounding_box,
        confidence=1.0,
        text=_text_from_element(element),
        identifier=_identifier_from_element(element),
        value=_value_from_element(element),
        enabled=_bool_attribute(element, "kAXEnabledAttribute"),
        focused=_bool_attribute(element, "kAXFocusedAttribute"),
        # Chrome did not reliably expose checkbox/radio state through AXValue
        # or AXSelected during validation.
        selected=None,
        source="accessibility",
    )


def _text_from_element(element: Any) -> str | None:
    title = _non_empty_string_attribute(
        element,
        "kAXTitleAttribute",
    )
    if title is not None:
        return title

    return _non_empty_string_attribute(
        element,
        "kAXDescriptionAttribute",
    )


def _identifier_from_element(element: Any) -> str | None:
    return _non_empty_string_attribute(
        element,
        "kAXDOMIdentifierAttribute",
    )


def _non_empty_string_attribute(
    element: Any,
    attribute_name: str,
) -> str | None:
    value = _copy_attribute(
        element,
        _ax_constant(attribute_name),
    )

    if not isinstance(value, str) or not value.strip():
        return None

    return value.strip()


def _value_from_element(element: Any) -> str | int | float | bool | None:
    value = _copy_attribute(
        element,
        _ax_constant("kAXValueAttribute"),
    )

    if isinstance(value, (str, int, float, bool)):
        return value

    return None


def _bool_attribute(
    element: Any,
    attribute_name: str,
) -> bool | None:
    value = _copy_attribute(
        element,
        _ax_constant(attribute_name),
    )

    if type(value) is bool:
        return value

    return None


def _bounding_box_from_element(element: Any) -> BoundingBox | None:
    position = _copy_attribute(
        element,
        _ax_constant("kAXPositionAttribute"),
    )
    size = _copy_attribute(
        element,
        _ax_constant("kAXSizeAttribute"),
    )

    point = _decode_ax_value(
        position,
        "kAXValueCGPointType",
    )
    dimensions = _decode_ax_value(
        size,
        "kAXValueCGSizeType",
    )

    x = _numeric_component(point, "x", 0)
    y = _numeric_component(point, "y", 1)
    width = _numeric_component(dimensions, "width", 0)
    height = _numeric_component(dimensions, "height", 1)

    values = (x, y, width, height)
    if any(value is None or not math.isfinite(value) for value in values):
        return None

    if x < 0 or y < 0 or width <= 0 or height <= 0:
        return None

    left = math.floor(x)
    top = math.floor(y)
    right = math.ceil(x + width)
    bottom = math.ceil(y + height)

    if right <= left or bottom <= top:
        return None

    try:
        return BoundingBox(
            x=left,
            y=top,
            width=right - left,
            height=bottom - top,
        )
    except ValueError:
        return None


def _decode_ax_value(
    value: Any,
    type_name: str,
) -> Any | None:
    getter = getattr(ApplicationServices, "AXValueGetValue", None)
    value_type = getattr(ApplicationServices, type_name, None)

    if getter is None or value_type is None:
        return value

    try:
        decoded = getter(
            value,
            value_type,
            None,
        )
    except Exception:
        return value

    if (
        isinstance(decoded, tuple)
        and len(decoded) == 2
        and type(decoded[0]) is bool
    ):
        if not decoded[0]:
            return None

        return decoded[1]

    return decoded


def _numeric_component(
    value: Any,
    name: str,
    index: int,
) -> float | None:
    if isinstance(value, dict):
        component = value.get(name)
    elif hasattr(value, name):
        component = getattr(value, name)
    elif isinstance(value, (tuple, list)) and len(value) > index:
        component = value[index]
    else:
        return None

    if isinstance(component, bool) or not isinstance(component, (int, float)):
        return None

    return float(component)
