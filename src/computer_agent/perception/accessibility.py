"""Read-only macOS Accessibility control discovery."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:
    import AppKit
except ImportError:  # pragma: no cover - depends on host platform
    AppKit = None  # type: ignore[assignment]

try:
    import Foundation
except ImportError:  # pragma: no cover - depends on host platform
    Foundation = None  # type: ignore[assignment]

try:
    import ApplicationServices
except ImportError:  # pragma: no cover - depends on host platform
    ApplicationServices = None  # type: ignore[assignment]

try:
    import CoreFoundation
except ImportError:  # pragma: no cover - depends on host platform
    CoreFoundation = None  # type: ignore[assignment]

from computer_agent.perception.models import BoundingBox, UIElement
from computer_agent.perception.viewport import (
    SemanticAXElement,
    Viewport,
)


_ROLE_MAP = {
    "AXTextField": "text_field",
    "AXTextArea": "text_area",
    "AXButton": "button",
    "AXCheckBox": "checkbox",
    "AXPopUpButton": "popup_button",
    "AXRadioButton": "radio_button",
    "AXLink": "link",
    "AXHeading": "heading",
    "AXStaticText": "text",
}

_APPKIT_STATE_REFRESH_SECONDS = 0.01


@dataclass(frozen=True, slots=True)
class AccessibilitySnapshot:
    """One fresh bounded frontmost Accessibility observation."""

    application_name: str | None
    controls: tuple[UIElement, ...]
    semantic_elements: tuple[SemanticAXElement, ...]
    web_areas: tuple[BoundingBox, ...]
    viewport: Viewport | None
    focused_window_bounds: BoundingBox | None
    traversed_nodes: int
    tree_traversals: int
    traversal_truncated: bool = False
    maximum_depth_reached: bool = False
    maximum_nodes_reached: bool = False
    maximum_controls_reached: bool = False

    def __post_init__(self) -> None:
        if self.application_name is not None and not isinstance(
            self.application_name,
            str,
        ):
            raise ValueError("application_name must be a string or None")
        object.__setattr__(self, "controls", tuple(self.controls))
        object.__setattr__(
            self,
            "semantic_elements",
            tuple(self.semantic_elements),
        )
        object.__setattr__(self, "web_areas", tuple(self.web_areas))
        if self.viewport is not None and not isinstance(
            self.viewport,
            Viewport,
        ):
            raise ValueError("viewport must be a Viewport or None")
        if self.focused_window_bounds is not None and not isinstance(
            self.focused_window_bounds,
            BoundingBox,
        ):
            raise ValueError(
                "focused_window_bounds must be a BoundingBox or None"
            )
        for name in ("traversed_nodes", "tree_traversals"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in (
            "traversal_truncated",
            "maximum_depth_reached",
            "maximum_nodes_reached",
            "maximum_controls_reached",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")


@dataclass(slots=True)
class _TraversalState:
    controls: list[UIElement]
    semantic_elements: list[SemanticAXElement]
    web_areas: list[BoundingBox]
    traversed_nodes: int = 0
    maximum_depth_reached: bool = False
    maximum_nodes_reached: bool = False
    maximum_controls_reached: bool = False


class MacOSAccessibility:
    """Read semantic controls from the focused macOS Accessibility window."""

    def __init__(
        self,
        maximum_elements: int = 5000,
        maximum_depth: int = 30,
        maximum_nodes: int = 50000,
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

        if (
            isinstance(maximum_nodes, bool)
            or not isinstance(maximum_nodes, int)
        ):
            raise ValueError("maximum_nodes must be an integer")

        if maximum_nodes <= 0:
            raise ValueError("maximum_nodes must be positive")

        self.maximum_elements = maximum_elements
        self.maximum_depth = maximum_depth
        self.maximum_nodes = maximum_nodes

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
        return list(
            self.read_frontmost_snapshot().controls
        )

    def read_frontmost_snapshot(self) -> AccessibilitySnapshot:
        """Return one bounded snapshot of frontmost Accessibility state."""

        if not self.is_available():
            raise RuntimeError(
                "macOS Accessibility frameworks are unavailable"
            )

        if not self.is_trusted():
            raise RuntimeError(
                "macOS Accessibility permission is not trusted"
            )

        workspace_application = self._frontmost_workspace_application()
        application_name = _localized_application_name(
            workspace_application
        )
        if workspace_application is None:
            return AccessibilitySnapshot(
                application_name=application_name,
                controls=(),
                semantic_elements=(),
                web_areas=(),
                viewport=None,
                focused_window_bounds=None,
                traversed_nodes=0,
                tree_traversals=0,
            )

        pid = workspace_application.processIdentifier()
        application = ApplicationServices.AXUIElementCreateApplication(pid)

        _copy_attribute(
            application,
            _ax_constant("kAXRoleAttribute"),
        )

        focused_window = _copy_attribute(
            application,
            _ax_constant("kAXFocusedWindowAttribute"),
        )
        if focused_window is None:
            return AccessibilitySnapshot(
                application_name=application_name,
                controls=(),
                semantic_elements=(),
                web_areas=(),
                viewport=None,
                focused_window_bounds=None,
                traversed_nodes=0,
                tree_traversals=0,
            )

        focused_element = _copy_attribute(
            application,
            _ax_constant("kAXFocusedUIElementAttribute"),
        )

        traversal = _TraversalState(
            controls=[],
            semantic_elements=[],
            web_areas=[],
        )
        focused_window_bounds = _bounding_box_from_element(
            focused_window
        )
        self._traverse_snapshot(
            focused_window,
            depth=0,
            state=traversal,
            focused_element=focused_element,
        )

        viewport = (
            Viewport(bounds=traversal.web_areas[0])
            if len(traversal.web_areas) == 1
            else None
        )

        return AccessibilitySnapshot(
            application_name=application_name,
            controls=tuple(traversal.controls),
            semantic_elements=tuple(traversal.semantic_elements),
            web_areas=tuple(traversal.web_areas),
            viewport=viewport,
            focused_window_bounds=focused_window_bounds,
            traversed_nodes=traversal.traversed_nodes,
            tree_traversals=1,
            traversal_truncated=(
                traversal.maximum_depth_reached
                or traversal.maximum_nodes_reached
                or traversal.maximum_controls_reached
            ),
            maximum_depth_reached=traversal.maximum_depth_reached,
            maximum_nodes_reached=traversal.maximum_nodes_reached,
            maximum_controls_reached=traversal.maximum_controls_reached,
        )

    def read_frontmost_application_name(self) -> str | None:
        """Return the localized frontmost application name when available."""

        if AppKit is None:
            return None

        try:
            application = self._frontmost_workspace_application()
        except Exception:
            return None

        return _localized_application_name(application)

    def read_frontmost_window_bounds(self) -> BoundingBox | None:
        """Return the focused frontmost window bounds when available."""
        return self.read_frontmost_snapshot().focused_window_bounds

    def read_frontmost_web_areas(self) -> list[BoundingBox]:
        """Return AXWebArea bounds from the focused frontmost window."""
        return list(
            self.read_frontmost_snapshot().web_areas
        )

    def read_frontmost_viewport(self) -> Viewport | None:
        """Return the unique frontmost webpage viewport when available."""
        return self.read_frontmost_snapshot().viewport

    def read_frontmost_semantic_elements(
        self,
    ) -> list[SemanticAXElement]:
        """Return supported Accessibility semantics with optional geometry."""
        return list(
            self.read_frontmost_snapshot().semantic_elements
        )

    def _traverse_snapshot(
        self,
        element: Any,
        *,
        depth: int,
        state: _TraversalState,
        focused_element: Any | None,
    ) -> None:
        if state.traversed_nodes >= self.maximum_nodes:
            state.maximum_nodes_reached = True
            return

        state.traversed_nodes += 1

        role = _copy_attribute(
            element,
            _ax_constant("kAXRoleAttribute"),
        )
        mapped_role = _ROLE_MAP.get(role)
        bounds = None

        if mapped_role is not None:
            bounds = _bounding_box_from_element(element)
            text = _text_from_element(element, mapped_role)
            value = _value_from_element(element, mapped_role)

            state.semantic_elements.append(
                SemanticAXElement(
                    role=role,
                    text=text,
                    bounds=bounds,
                    value=value,
                )
            )

        if role == "AXWebArea":
            bounds = bounds or _bounding_box_from_element(element)
            if bounds is not None:
                state.web_areas.append(bounds)

        if mapped_role is not None and bounds is not None:
            if len(state.controls) < self.maximum_elements:
                state.controls.append(
                    UIElement(
                        element_type=mapped_role,
                        bounding_box=bounds,
                        confidence=1.0,
                        text=text,
                        identifier=_identifier_from_element(element),
                        value=value,
                        enabled=_bool_attribute(
                            element,
                            "kAXEnabledAttribute",
                        ),
                        focused=_focused_from_element(
                            element,
                            focused_element,
                        ),
                        selected=None,
                        source="accessibility",
                    )
                )
                if len(state.controls) >= self.maximum_elements:
                    state.maximum_controls_reached = True
            else:
                state.maximum_controls_reached = True

        if depth >= self.maximum_depth:
            children = _copy_attribute(
                element,
                _ax_constant("kAXChildrenAttribute"),
            )
            if _iter_children(children):
                state.maximum_depth_reached = True
            return

        children = _copy_attribute(
            element,
            _ax_constant("kAXChildrenAttribute"),
        )

        for child in _iter_children(children):
            self._traverse_snapshot(
                child,
                depth=depth + 1,
                state=state,
                focused_element=focused_element,
            )
            if state.maximum_nodes_reached:
                return

    def _frontmost_application_element(self) -> Any | None:
        application = self._frontmost_workspace_application()

        if application is None:
            return None

        pid = application.processIdentifier()

        return ApplicationServices.AXUIElementCreateApplication(pid)

    def _frontmost_workspace_application(self) -> Any | None:
        if not _refresh_appkit_state():
            return None

        workspace = AppKit.NSWorkspace.sharedWorkspace()
        return workspace.frontmostApplication()

    def _traverse(
        self,
        element: Any,
        *,
        depth: int,
        controls: list[UIElement],
        focused_element: Any | None,
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
                focused_element,
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
                focused_element=focused_element,
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


def _refresh_appkit_state() -> bool:
    if Foundation is None:
        return False

    try:
        run_loop = Foundation.NSRunLoop.currentRunLoop()
        deadline = Foundation.NSDate.dateWithTimeIntervalSinceNow_(
            _APPKIT_STATE_REFRESH_SECONDS
        )
        run_loop.runUntilDate_(deadline)
    except Exception:
        return False

    return True


def _localized_application_name(application: Any | None) -> str | None:
    if application is None:
        return None

    try:
        name = application.localizedName()
    except Exception:
        return None

    if not isinstance(name, str) or not name.strip():
        return None

    return name.strip()


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
    focused_element: Any | None,
) -> UIElement | None:
    bounding_box = _bounding_box_from_element(element)

    if bounding_box is None:
        return None

    return UIElement(
        element_type=mapped_role,
        bounding_box=bounding_box,
        confidence=1.0,
        text=_text_from_element(
            element,
            mapped_role,
        ),
        identifier=_identifier_from_element(element),
        value=_value_from_element(
            element,
            mapped_role,
        ),
        enabled=_bool_attribute(element, "kAXEnabledAttribute"),
        focused=_focused_from_element(element, focused_element),
        # Chrome did not reliably expose checkbox/radio state through AXValue
        # or AXSelected during validation.
        selected=None,
        source="accessibility",
    )


def _text_from_element(
    element: Any,
    mapped_role: str,
) -> str | None:
    title = _non_empty_string_attribute(
        element,
        "kAXTitleAttribute",
    )
    if title is not None:
        return title

    description = _non_empty_string_attribute(
        element,
        "kAXDescriptionAttribute",
    )
    if description is not None:
        return description

    if mapped_role == "text":
        return _non_empty_string_attribute(
            element,
            "kAXValueAttribute",
        )

    return None


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


def _value_from_element(
    element: Any,
    mapped_role: str | None = None,
) -> str | int | float | bool | None:
    # Chrome links commonly expose the real href through AXURL rather than
    # AXValue. Preserve it in the existing value field so higher layers can
    # verify the destination without fuzzy title matching.
    if mapped_role == "link":
        url_attribute = getattr(
            ApplicationServices,
            "kAXURLAttribute",
            "AXURL",
        )
        url_value = _copy_attribute(
            element,
            url_attribute,
        )
        if isinstance(url_value, str) and url_value.strip():
            return url_value.strip()

        absolute_string = getattr(
            url_value,
            "absoluteString",
            None,
        )
        if callable(absolute_string):
            try:
                rendered = absolute_string()
            except Exception:
                rendered = None
            if isinstance(rendered, str) and rendered.strip():
                return rendered.strip()

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


def _focused_from_element(
    element: Any,
    focused_element: Any | None,
) -> bool | None:
    if focused_element is not None:
        return _same_accessibility_element(
            element,
            focused_element,
        )

    return _bool_attribute(element, "kAXFocusedAttribute")


def _same_accessibility_element(
    first: Any,
    second: Any,
) -> bool:
    equal = _core_foundation_equal()

    if equal is not None:
        try:
            return bool(
                equal(
                    first,
                    second,
                )
            )
        except Exception:
            pass

    return first is second


def _core_foundation_equal():
    if CoreFoundation is not None:
        equal = getattr(CoreFoundation, "CFEqual", None)
        if equal is not None:
            return equal

    if ApplicationServices is not None:
        return getattr(ApplicationServices, "CFEqual", None)

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
