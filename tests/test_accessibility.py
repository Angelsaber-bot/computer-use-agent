import inspect
import math

import pytest

import computer_agent.perception.accessibility as accessibility_module
from computer_agent.perception import MacOSAccessibility
from computer_agent.perception import MacOSAccessibility as ExportedMacOSAccessibility
from computer_agent.perception import UIElement


class FakeAXValue:
    def __init__(self, value):
        self.value = value


class FakeAXElement:
    def __init__(
        self,
        attributes=None,
        error_attributes=None,
    ):
        self.attributes = attributes or {}
        self.error_attributes = set(error_attributes or ())


class FakeFrontmostApplication:
    def __init__(self, pid, localized_name="Google Chrome"):
        self.pid = pid
        self.localized_name = localized_name
        self.process_identifier_calls = 0
        self.localized_name_calls = 0

    def processIdentifier(self):
        self.process_identifier_calls += 1

        return self.pid

    def localizedName(self):
        self.localized_name_calls += 1

        return self.localized_name


class FakeWorkspace:
    def __init__(self, application, *, event_log=None):
        self.application = application
        self.frontmost_application_calls = 0
        self.event_log = event_log

    def frontmostApplication(self):
        self.frontmost_application_calls += 1
        if self.event_log is not None:
            self.event_log.append("frontmostApplication")

        return self.application


class FakeNSWorkspace:
    def __init__(self, workspace, *, event_log=None):
        self.workspace = workspace
        self.shared_workspace_calls = 0
        self.event_log = event_log

    def sharedWorkspace(self):
        self.shared_workspace_calls += 1
        if self.event_log is not None:
            self.event_log.append("sharedWorkspace")

        return self.workspace


class FakeAppKit:
    def __init__(self, workspace, *, event_log=None):
        self.NSWorkspace = FakeNSWorkspace(
            workspace,
            event_log=event_log,
        )


class FakeRunLoop:
    def __init__(self, foundation):
        self.foundation = foundation

    def runUntilDate_(self, deadline):
        self.foundation.run_until_dates.append(deadline)
        self.foundation.events.append("runUntilDate")

        if self.foundation.refresh_error is not None:
            raise self.foundation.refresh_error

        if self.foundation.on_refresh is not None:
            self.foundation.on_refresh()


class FakeNSRunLoop:
    def __init__(self, foundation):
        self.foundation = foundation
        self.current_run_loop_calls = 0

    def currentRunLoop(self):
        self.current_run_loop_calls += 1
        self.foundation.events.append("currentRunLoop")

        return self.foundation.run_loop


class FakeNSDate:
    def __init__(self, foundation):
        self.foundation = foundation

    def dateWithTimeIntervalSinceNow_(self, interval):
        self.foundation.refresh_intervals.append(interval)
        self.foundation.events.append("dateWithTimeIntervalSinceNow")

        return ("deadline", interval)


class FakeFoundation:
    def __init__(
        self,
        *,
        event_log=None,
        on_refresh=None,
        refresh_error=None,
    ):
        self.events = event_log if event_log is not None else []
        self.on_refresh = on_refresh
        self.refresh_error = refresh_error
        self.refresh_intervals = []
        self.run_until_dates = []
        self.run_loop = FakeRunLoop(self)
        self.NSRunLoop = FakeNSRunLoop(self)
        self.NSDate = FakeNSDate(self)


class FakeApplicationServices:
    kAXFocusedWindowAttribute = "AXFocusedWindow"
    kAXFocusedUIElementAttribute = "AXFocusedUIElement"
    kAXChildrenAttribute = "AXChildren"
    kAXRoleAttribute = "AXRole"
    kAXTitleAttribute = "AXTitle"
    kAXDescriptionAttribute = "AXDescription"
    kAXDOMIdentifierAttribute = "AXDOMIdentifier"
    kAXValueAttribute = "AXValue"
    kAXEnabledAttribute = "AXEnabled"
    kAXFocusedAttribute = "AXFocused"
    kAXPositionAttribute = "AXPosition"
    kAXSizeAttribute = "AXSize"
    kAXErrorSuccess = 0
    kAXValueCGPointType = "point"
    kAXValueCGSizeType = "size"

    def __init__(
        self,
        application_element,
        trusted=True,
    ):
        self.application_element = application_element
        self.trusted = trusted
        self.created_application_pids = []
        self.attribute_reads = []
        self.attribute_writes = []

    def AXIsProcessTrusted(self):
        return self.trusted

    def AXUIElementCreateApplication(self, pid):
        self.created_application_pids.append(pid)

        return self.application_element

    def AXUIElementCopyAttributeValue(
        self,
        element,
        attribute,
        _output,
    ):
        self.attribute_reads.append((element, attribute))

        if attribute in element.error_attributes:
            raise RuntimeError(f"attribute failed: {attribute}")

        if attribute not in element.attributes:
            return 1, None

        return self.kAXErrorSuccess, element.attributes[attribute]

    def AXUIElementSetAttributeValue(
        self,
        element,
        attribute,
        value,
    ):
        self.attribute_writes.append((element, attribute, value))

        return self.kAXErrorSuccess

    def AXValueGetValue(
        self,
        value,
        _value_type,
        _output,
    ):
        if isinstance(value, FakeAXValue):
            return value.value

        return value


class FakeCoreFoundation:
    def __init__(self):
        self.comparisons = []

    def CFEqual(
        self,
        first,
        second,
    ):
        self.comparisons.append((first, second))

        return first is second


def _node(
    *,
    role=None,
    title=None,
    description=None,
    identifier=None,
    value=None,
    enabled=None,
    focused=None,
    position=(10, 20),
    size=(30, 40),
    children=(),
    error_attributes=(),
):
    attributes = {
        "AXChildren": list(children),
    }

    if role is not None:
        attributes["AXRole"] = role

    if title is not None:
        attributes["AXTitle"] = title

    if description is not None:
        attributes["AXDescription"] = description

    if identifier is not None:
        attributes["AXDOMIdentifier"] = identifier

    if value is not None:
        attributes["AXValue"] = value

    if enabled is not None:
        attributes["AXEnabled"] = enabled

    if focused is not None:
        attributes["AXFocused"] = focused

    if position is not None:
        attributes["AXPosition"] = position

    if size is not None:
        attributes["AXSize"] = size

    return FakeAXElement(
        attributes,
        error_attributes,
    )


def _install_fake_accessibility(
    monkeypatch,
    window,
    *,
    trusted=True,
    pid=4242,
    localized_name="Google Chrome",
    foundation=None,
    focused_element=None,
    focused_element_error=False,
    application_role="AXApplication",
    application_role_error=False,
):
    application_attributes = {
        "AXRole": application_role,
        "AXFocusedWindow": window,
    }
    if focused_element is not None:
        application_attributes["AXFocusedUIElement"] = focused_element

    application_error_attributes = set()
    if application_role_error:
        application_error_attributes.add("AXRole")

    if focused_element_error:
        application_error_attributes.add("AXFocusedUIElement")

    application_element = FakeAXElement(
        application_attributes,
        application_error_attributes,
    )
    services = FakeApplicationServices(
        application_element,
        trusted=trusted,
    )
    core_foundation = FakeCoreFoundation()
    application = FakeFrontmostApplication(
        pid,
        localized_name=localized_name,
    )
    if foundation is None:
        foundation = FakeFoundation()

    workspace = FakeWorkspace(
        application,
        event_log=foundation.events,
    )
    appkit = FakeAppKit(
        workspace,
        event_log=foundation.events,
    )

    monkeypatch.setattr(accessibility_module, "AppKit", appkit)
    monkeypatch.setattr(accessibility_module, "Foundation", foundation)
    monkeypatch.setattr(
        accessibility_module,
        "ApplicationServices",
        services,
    )
    monkeypatch.setattr(
        accessibility_module,
        "CoreFoundation",
        core_foundation,
    )

    return (
        appkit,
        services,
        core_foundation,
        application,
        workspace,
        application_element,
    )


def _attribute_read_index(
    services,
    element,
    attribute,
):
    return services.attribute_reads.index((element, attribute))


def _read_controls(
    monkeypatch,
    children,
    *,
    maximum_elements=5000,
    maximum_depth=30,
):
    window = _node(
        role="AXWindow",
        children=children,
    )
    _install_fake_accessibility(monkeypatch, window)

    return MacOSAccessibility(
        maximum_elements=maximum_elements,
        maximum_depth=maximum_depth,
    ).read_frontmost_controls()


def test_macos_accessibility_is_exported_from_perception_package():
    assert ExportedMacOSAccessibility is accessibility_module.MacOSAccessibility


def test_is_available_requires_both_framework_modules(monkeypatch):
    monkeypatch.setattr(accessibility_module, "AppKit", object())
    monkeypatch.setattr(accessibility_module, "ApplicationServices", object())

    assert MacOSAccessibility.is_available() is True

    monkeypatch.setattr(accessibility_module, "AppKit", None)

    assert MacOSAccessibility.is_available() is False

    monkeypatch.setattr(accessibility_module, "AppKit", object())
    monkeypatch.setattr(accessibility_module, "ApplicationServices", None)

    assert MacOSAccessibility.is_available() is False


def test_frontmost_application_lookup_refreshes_run_loop_before_read(
    monkeypatch,
):
    foundation = FakeFoundation()
    window = _node(role="AXWindow")
    _install_fake_accessibility(
        monkeypatch,
        window,
        foundation=foundation,
        localized_name="TextEdit",
    )

    assert MacOSAccessibility().read_frontmost_application_name() == "TextEdit"

    assert foundation.refresh_intervals == [
        accessibility_module._APPKIT_STATE_REFRESH_SECONDS,
    ]
    assert foundation.run_until_dates == [
        (
            "deadline",
            accessibility_module._APPKIT_STATE_REFRESH_SECONDS,
        )
    ]
    assert foundation.events.index("runUntilDate") < foundation.events.index(
        "frontmostApplication"
    )


def test_frontmost_application_name_reflects_change_after_run_loop_refresh(
    monkeypatch,
):
    current_application = FakeFrontmostApplication(
        5252,
        localized_name="TextEdit",
    )
    workspace = None
    foundation = FakeFoundation(
        on_refresh=lambda: setattr(
            workspace,
            "application",
            current_application,
        )
    )
    window = _node(role="AXWindow")
    (
        _appkit,
        _services,
        _core_foundation,
        stale_application,
        workspace,
        _application_element,
    ) = _install_fake_accessibility(
        monkeypatch,
        window,
        foundation=foundation,
        localized_name="Terminal",
    )

    name = MacOSAccessibility().read_frontmost_application_name()

    assert name == "TextEdit"
    assert stale_application.localized_name_calls == 0
    assert current_application.localized_name_calls == 1


def test_read_frontmost_application_name_returns_localized_name(monkeypatch):
    window = _node(role="AXWindow")
    (
        appkit,
        services,
        _core_foundation,
        application,
        workspace,
        _application_element,
    ) = _install_fake_accessibility(
        monkeypatch,
        window,
        localized_name="Safari",
    )

    name = MacOSAccessibility().read_frontmost_application_name()

    assert name == "Safari"
    assert appkit.NSWorkspace.shared_workspace_calls == 1
    assert workspace.frontmost_application_calls == 1
    assert application.localized_name_calls == 1
    assert application.process_identifier_calls == 0
    assert services.created_application_pids == []
    assert services.attribute_reads == []
    assert services.attribute_writes == []


def test_read_frontmost_application_name_returns_none_without_frontmost_app(
    monkeypatch,
):
    foundation = FakeFoundation()
    workspace = FakeWorkspace(
        None,
        event_log=foundation.events,
    )
    appkit = FakeAppKit(
        workspace,
        event_log=foundation.events,
    )
    monkeypatch.setattr(accessibility_module, "AppKit", appkit)
    monkeypatch.setattr(accessibility_module, "Foundation", foundation)

    name = MacOSAccessibility().read_frontmost_application_name()

    assert name is None
    assert appkit.NSWorkspace.shared_workspace_calls == 1
    assert workspace.frontmost_application_calls == 1


def test_read_frontmost_application_name_returns_none_when_refresh_fails(
    monkeypatch,
):
    foundation = FakeFoundation(refresh_error=RuntimeError("run loop failed"))
    window = _node(role="AXWindow")
    (
        appkit,
        _services,
        _core_foundation,
        _application,
        workspace,
        _application_element,
    ) = _install_fake_accessibility(
        monkeypatch,
        window,
        foundation=foundation,
        localized_name="TextEdit",
    )

    name = MacOSAccessibility().read_frontmost_application_name()

    assert name is None
    assert foundation.events == [
        "currentRunLoop",
        "dateWithTimeIntervalSinceNow",
        "runUntilDate",
    ]
    assert appkit.NSWorkspace.shared_workspace_calls == 0
    assert workspace.frontmost_application_calls == 0


def test_read_frontmost_application_name_preserves_trust_behavior(
    monkeypatch,
):
    window = _node(role="AXWindow")
    _install_fake_accessibility(
        monkeypatch,
        window,
        trusted=False,
        localized_name="TextEdit",
    )

    assert MacOSAccessibility().read_frontmost_application_name() == "TextEdit"
    assert MacOSAccessibility.is_trusted() is False
    with pytest.raises(
        RuntimeError,
        match="macOS Accessibility permission is not trusted",
    ):
        MacOSAccessibility().read_frontmost_controls()


def test_frontmost_application_observation_uses_no_shell_or_applescript():
    source = inspect.getsource(accessibility_module)

    assert "NSWorkspace" in source
    assert "subprocess" not in source
    assert "osascript" not in source
    assert "AppleScript" not in source


def test_read_frontmost_controls_fails_when_frameworks_are_unavailable(
    monkeypatch,
):
    monkeypatch.setattr(accessibility_module, "AppKit", None)
    monkeypatch.setattr(accessibility_module, "ApplicationServices", object())

    with pytest.raises(
        RuntimeError,
        match="macOS Accessibility frameworks are unavailable",
    ):
        MacOSAccessibility().read_frontmost_controls()


def test_read_frontmost_controls_fails_when_permission_is_untrusted(
    monkeypatch,
):
    window = _node(role="AXWindow")
    _install_fake_accessibility(
        monkeypatch,
        window,
        trusted=False,
    )

    assert MacOSAccessibility.is_trusted() is False

    with pytest.raises(
        RuntimeError,
        match="macOS Accessibility permission is not trusted",
    ):
        MacOSAccessibility().read_frontmost_controls()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"maximum_elements": True}, "maximum_elements must be an integer"),
        ({"maximum_elements": "5000"}, "maximum_elements must be an integer"),
        ({"maximum_elements": 0}, "maximum_elements must be positive"),
        ({"maximum_elements": -1}, "maximum_elements must be positive"),
        ({"maximum_depth": False}, "maximum_depth must be an integer"),
        ({"maximum_depth": "30"}, "maximum_depth must be an integer"),
        ({"maximum_depth": -1}, "maximum_depth must be non-negative"),
    ],
)
def test_constructor_rejects_invalid_limits(kwargs, message):
    with pytest.raises(ValueError, match=message):
        MacOSAccessibility(**kwargs)


def test_read_frontmost_controls_uses_refreshed_frontmost_application(
    monkeypatch,
):
    control = _node(
        role="AXButton",
        title="ACTIVE_BUTTON_10",
    )
    window = _node(
        role="AXWindow",
        children=[control],
    )
    current_application = FakeFrontmostApplication(
        5252,
        localized_name="TextEdit",
    )
    workspace = None
    foundation = FakeFoundation(
        on_refresh=lambda: setattr(
            workspace,
            "application",
            current_application,
        )
    )
    (
        _appkit,
        services,
        _core_foundation,
        stale_application,
        workspace,
        _application_element,
    ) = _install_fake_accessibility(
        monkeypatch,
        window,
        foundation=foundation,
        pid=1111,
        localized_name="Terminal",
    )

    controls = MacOSAccessibility().read_frontmost_controls()

    assert [control.text for control in controls] == ["ACTIVE_BUTTON_10"]
    assert services.created_application_pids == [5252]
    assert stale_application.process_identifier_calls == 0
    assert current_application.process_identifier_calls == 1
    assert foundation.events.index("runUntilDate") < foundation.events.index(
        "frontmostApplication"
    )


def test_read_frontmost_controls_uses_pid_based_frontmost_application_path(
    monkeypatch,
):
    control = _node(
        role="AXButton",
        title="ACTIVE_BUTTON_10",
    )
    window = _node(
        role="AXWindow",
        children=[control],
    )
    (
        appkit,
        services,
        _core_foundation,
        application,
        workspace,
        application_element,
    ) = _install_fake_accessibility(
        monkeypatch,
        window,
        pid=12345,
    )

    controls = MacOSAccessibility().read_frontmost_controls()

    assert len(controls) == 1
    assert appkit.NSWorkspace.shared_workspace_calls == 1
    assert workspace.frontmost_application_calls == 1
    assert application.process_identifier_calls == 1
    assert services.created_application_pids == [12345]
    assert (
        application_element,
        "AXFocusedWindow",
    ) in services.attribute_reads
    assert services.attribute_reads.count(
        (
            application_element,
            "AXRole",
        )
    ) == 1
    assert services.attribute_reads.count(
        (
            application_element,
            "AXFocusedUIElement",
        )
    ) == 1


def test_application_role_is_requested_once_per_read_frontmost_controls_call(
    monkeypatch,
):
    control = _node(
        role="AXButton",
        title="ACTIVE_BUTTON",
    )
    window = _node(
        role="AXWindow",
        children=[control],
    )
    (
        _appkit,
        services,
        _core_foundation,
        _application,
        _workspace,
        application_element,
    ) = _install_fake_accessibility(monkeypatch, window)
    reader = MacOSAccessibility()

    first_controls = reader.read_frontmost_controls()
    second_controls = reader.read_frontmost_controls()

    assert [control.text for control in first_controls] == ["ACTIVE_BUTTON"]
    assert [control.text for control in second_controls] == ["ACTIVE_BUTTON"]
    assert services.attribute_reads.count(
        (
            application_element,
            "AXRole",
        )
    ) == 2
    assert services.attribute_writes == []


def test_application_role_is_requested_before_focus_and_traversal(
    monkeypatch,
):
    focused_control = _node(
        role="AXTextField",
        title="FOCUSED_FIELD",
        identifier="focused-field",
        value="ready",
        enabled=True,
    )
    window = _node(
        role="AXWindow",
        children=[focused_control],
    )
    (
        _appkit,
        services,
        _core_foundation,
        _application,
        _workspace,
        application_element,
    ) = _install_fake_accessibility(
        monkeypatch,
        window,
        focused_element=focused_control,
    )

    controls = MacOSAccessibility().read_frontmost_controls()

    application_role_index = _attribute_read_index(
        services,
        application_element,
        "AXRole",
    )
    focused_element_index = _attribute_read_index(
        services,
        application_element,
        "AXFocusedUIElement",
    )
    focused_window_index = _attribute_read_index(
        services,
        application_element,
        "AXFocusedWindow",
    )
    window_traversal_index = _attribute_read_index(
        services,
        window,
        "AXRole",
    )

    assert len(controls) == 1
    assert controls[0].element_type == "text_field"
    assert controls[0].text == "FOCUSED_FIELD"
    assert controls[0].identifier == "focused-field"
    assert controls[0].value == "ready"
    assert controls[0].enabled is True
    assert application_role_index < focused_element_index
    assert application_role_index < focused_window_index
    assert application_role_index < window_traversal_index
    assert services.attribute_writes == []


def test_failed_application_role_request_does_not_prevent_window_traversal(
    monkeypatch,
):
    control = _node(
        role="AXButton",
        title="BUTTON_AFTER_FAILED_ROLE_PROBE",
    )
    window = _node(
        role="AXWindow",
        children=[control],
    )
    (
        _appkit,
        services,
        _core_foundation,
        _application,
        _workspace,
        application_element,
    ) = _install_fake_accessibility(
        monkeypatch,
        window,
        application_role_error=True,
    )

    controls = MacOSAccessibility().read_frontmost_controls()

    assert [control.text for control in controls] == [
        "BUTTON_AFTER_FAILED_ROLE_PROBE",
    ]
    assert services.attribute_reads.count(
        (
            application_element,
            "AXRole",
        )
    ) == 1
    assert (
        application_element,
        "AXFocusedWindow",
    ) in services.attribute_reads
    assert services.attribute_writes == []


def test_role_mapping_for_supported_controls_preserves_tree_order(
    monkeypatch,
):
    controls = _read_controls(
        monkeypatch,
        [
            _node(role="AXTextField", title="TEXT"),
            _node(role="AXTextArea", title="TEXT_AREA"),
            _node(role="AXButton", title="BUTTON"),
            _node(role="AXCheckBox", title="CHECKBOX"),
            _node(role="AXPopUpButton", title="POPUP"),
            _node(role="AXRadioButton", title="RADIO"),
        ],
    )

    assert [control.element_type for control in controls] == [
        "text_field",
        "text_area",
        "button",
        "checkbox",
        "popup_button",
        "radio_button",
    ]
    assert all(control.source == "accessibility" for control in controls)
    assert all(control.confidence == 1.0 for control in controls)


def test_ax_text_area_preserves_value_focus_box_and_source(monkeypatch):
    text_area = _node(
        role="AXTextArea",
        title="EDITOR",
        value="CROSS_APP_TRANSFER_10",
        focused=False,
        position=(12, 24),
        size=(320, 180),
    )
    window = _node(
        role="AXWindow",
        children=[text_area],
    )
    _install_fake_accessibility(
        monkeypatch,
        window,
        focused_element=text_area,
    )

    controls = MacOSAccessibility().read_frontmost_controls()

    assert len(controls) == 1
    assert controls[0].element_type == "text_area"
    assert controls[0].value == "CROSS_APP_TRANSFER_10"
    assert controls[0].focused is True
    assert controls[0].bounding_box.x == 12
    assert controls[0].bounding_box.y == 24
    assert controls[0].bounding_box.width == 320
    assert controls[0].bounding_box.height == 180
    assert controls[0].source == "accessibility"


def test_non_focused_ax_text_area_remains_unfocused_when_determinable(
    monkeypatch,
):
    controls = _read_controls(
        monkeypatch,
        [
            _node(
                role="AXTextArea",
                title="EDITOR",
                value="draft",
                focused=False,
            ),
        ],
    )

    assert len(controls) == 1
    assert controls[0].element_type == "text_area"
    assert controls[0].value == "draft"
    assert controls[0].focused is False


def test_existing_ax_text_field_mapping_remains_unchanged(monkeypatch):
    controls = _read_controls(
        monkeypatch,
        [
            _node(
                role="AXTextField",
                title="FIELD",
                value="typed value",
                focused=True,
            ),
        ],
    )

    assert len(controls) == 1
    assert controls[0].element_type == "text_field"
    assert controls[0].value == "typed value"
    assert controls[0].focused is True


def test_title_is_preferred_and_description_is_fallback(monkeypatch):
    controls = _read_controls(
        monkeypatch,
        [
            _node(
                role="AXButton",
                title="TITLE_TEXT",
                description="DESCRIPTION_TEXT",
            ),
            _node(
                role="AXButton",
                title="   ",
                description="DESCRIPTION_FALLBACK",
            ),
            _node(
                role="AXButton",
                title=None,
                description=None,
            ),
        ],
    )

    assert [control.text for control in controls] == [
        "TITLE_TEXT",
        "DESCRIPTION_FALLBACK",
        None,
    ]


def test_identifier_extraction_uses_non_empty_dom_identifier(monkeypatch):
    controls = _read_controls(
        monkeypatch,
        [
            _node(
                role="AXButton",
                title="BUTTON",
                identifier="active-button",
            ),
            _node(
                role="AXButton",
                title="BUTTON",
                identifier="   ",
            ),
        ],
    )

    assert [control.identifier for control in controls] == [
        "active-button",
        None,
    ]


@pytest.mark.parametrize(
    "value",
    [
        "text",
        7,
        7.5,
        True,
        False,
    ],
)
def test_scalar_values_are_preserved(monkeypatch, value):
    controls = _read_controls(
        monkeypatch,
        [
            _node(
                role="AXTextField",
                title="TEXT_FIELD",
                value=value,
            ),
        ],
    )

    assert controls[0].value == value


def test_unsupported_values_become_none(monkeypatch):
    controls = _read_controls(
        monkeypatch,
        [
            _node(
                role="AXTextField",
                title="TEXT_FIELD",
                value=["unsupported"],
            ),
        ],
    )

    assert controls[0].value is None


def test_enabled_and_focused_are_extracted_only_for_bool_values(monkeypatch):
    controls = _read_controls(
        monkeypatch,
        [
            _node(
                role="AXButton",
                title="BUTTON",
                enabled=True,
                focused=False,
            ),
            _node(
                role="AXButton",
                title="BUTTON",
                enabled=1,
                focused="false",
            ),
        ],
    )

    assert controls[0].enabled is True
    assert controls[0].focused is False
    assert controls[1].enabled is None
    assert controls[1].focused is None


def test_application_level_focused_element_is_marked_focused(monkeypatch):
    focused_control = _node(
        role="AXTextField",
        title="FOCUSED_FIELD",
        focused=False,
    )
    other_control = _node(
        role="AXButton",
        title="OTHER_BUTTON",
        focused=True,
    )
    window = _node(
        role="AXWindow",
        children=[
            focused_control,
            other_control,
        ],
    )
    _install_fake_accessibility(
        monkeypatch,
        window,
        focused_element=focused_control,
    )

    controls = MacOSAccessibility().read_frontmost_controls()

    assert [control.text for control in controls] == [
        "FOCUSED_FIELD",
        "OTHER_BUTTON",
    ]
    assert controls[0].focused is True
    assert controls[1].focused is False


def test_application_level_focus_uses_element_identity_not_metadata(
    monkeypatch,
):
    focused_control = _node(
        role="AXTextField",
        title="DUPLICATE_FIELD",
        identifier="duplicate-field",
        value="same value",
        position=(100, 200),
        size=(300, 40),
    )
    duplicate_control = _node(
        role="AXTextField",
        title="DUPLICATE_FIELD",
        identifier="duplicate-field",
        value="same value",
        position=(100, 200),
        size=(300, 40),
    )
    window = _node(
        role="AXWindow",
        children=[
            focused_control,
            duplicate_control,
        ],
    )
    (
        _appkit,
        _services,
        core_foundation,
        _application,
        _workspace,
        _application_element,
    ) = _install_fake_accessibility(
        monkeypatch,
        window,
        focused_element=focused_control,
    )

    controls = MacOSAccessibility().read_frontmost_controls()

    assert [control.text for control in controls] == [
        "DUPLICATE_FIELD",
        "DUPLICATE_FIELD",
    ]
    assert [control.focused for control in controls] == [
        True,
        False,
    ]
    assert (
        duplicate_control,
        focused_control,
    ) in core_foundation.comparisons


def test_application_focus_takes_precedence_over_direct_focused_false(
    monkeypatch,
):
    focused_control = _node(
        role="AXTextField",
        title="FOCUSED_FIELD",
        focused=False,
    )
    window = _node(
        role="AXWindow",
        children=[focused_control],
    )
    _install_fake_accessibility(
        monkeypatch,
        window,
        focused_element=focused_control,
    )

    controls = MacOSAccessibility().read_frontmost_controls()

    assert controls[0].focused is True


def test_failed_application_focused_element_read_falls_back_to_direct_attribute(
    monkeypatch,
):
    focused_control = _node(
        role="AXTextField",
        title="DIRECTLY_FOCUSED_FIELD",
        focused=True,
    )
    unfocused_control = _node(
        role="AXButton",
        title="DIRECTLY_UNFOCUSED_BUTTON",
        focused=False,
    )
    window = _node(
        role="AXWindow",
        children=[
            focused_control,
            unfocused_control,
        ],
    )
    _install_fake_accessibility(
        monkeypatch,
        window,
        focused_element_error=True,
    )

    controls = MacOSAccessibility().read_frontmost_controls()

    assert [control.focused for control in controls] == [
        True,
        False,
    ]


def test_missing_application_focused_element_does_not_crash_traversal(
    monkeypatch,
):
    controls = _read_controls(
        monkeypatch,
        [
            _node(
                role="AXButton",
                title="BUTTON_WITHOUT_FOCUS_METADATA",
            ),
        ],
    )

    assert len(controls) == 1
    assert controls[0].focused is None


def test_selected_remains_none_with_application_level_focus(monkeypatch):
    checkbox = _node(
        role="AXCheckBox",
        title="CHECKBOX",
        focused=True,
    )
    radio = _node(
        role="AXRadioButton",
        title="RADIO",
        focused=True,
    )
    window = _node(
        role="AXWindow",
        children=[
            checkbox,
            radio,
        ],
    )
    _install_fake_accessibility(
        monkeypatch,
        window,
        focused_element=radio,
    )

    controls = MacOSAccessibility().read_frontmost_controls()

    assert [control.focused for control in controls] == [
        False,
        True,
    ]
    assert all(control.selected is None for control in controls)


def test_selected_is_always_none(monkeypatch):
    controls = _read_controls(
        monkeypatch,
        [
            _node(
                role="AXCheckBox",
                title="CHECKBOX",
                value=True,
            ),
            _node(
                role="AXRadioButton",
                title="RADIO",
                value=True,
            ),
        ],
    )

    assert all(control.selected is None for control in controls)


def test_geometry_uses_floor_and_ceil_bounding_box_conversion(monkeypatch):
    controls = _read_controls(
        monkeypatch,
        [
            _node(
                role="AXButton",
                title="BUTTON",
                position=FakeAXValue((10.2, 20.8)),
                size=FakeAXValue((30.1, 40.05)),
            ),
        ],
    )

    assert controls[0].bounding_box.x == 10
    assert controls[0].bounding_box.y == 20
    assert controls[0].bounding_box.width == 31
    assert controls[0].bounding_box.height == 41


def test_invalid_geometry_is_skipped(monkeypatch):
    controls = _read_controls(
        monkeypatch,
        [
            _node(role="AXButton", title="MISSING_POSITION", position=None),
            _node(
                role="AXButton",
                title="NON_FINITE_POSITION",
                position=(math.nan, 20),
            ),
            _node(
                role="AXButton",
                title="NEGATIVE_POSITION",
                position=(-1, 20),
            ),
            _node(
                role="AXButton",
                title="ZERO_WIDTH",
                size=(0, 40),
            ),
            _node(
                role="AXButton",
                title="VALID",
            ),
        ],
    )

    assert [control.text for control in controls] == ["VALID"]


def test_unsupported_roles_are_ignored_but_children_are_traversed(
    monkeypatch,
):
    controls = _read_controls(
        monkeypatch,
        [
            _node(
                role="AXGroup",
                title="IGNORED_GROUP",
                children=[
                    _node(
                        role="AXButton",
                        title="CHILD_BUTTON",
                    ),
                ],
            ),
        ],
    )

    assert [control.text for control in controls] == ["CHILD_BUTTON"]


def test_maximum_elements_limits_returned_controls(monkeypatch):
    controls = _read_controls(
        monkeypatch,
        [
            _node(role="AXButton", title="FIRST"),
            _node(role="AXButton", title="SECOND"),
            _node(role="AXButton", title="THIRD"),
        ],
        maximum_elements=2,
    )

    assert [control.text for control in controls] == [
        "FIRST",
        "SECOND",
    ]


def test_maximum_depth_limits_descending_without_skipping_current_depth(
    monkeypatch,
):
    controls = _read_controls(
        monkeypatch,
        [
            _node(role="AXButton", title="DEPTH_ONE"),
            _node(
                role="AXGroup",
                title="GROUP",
                children=[
                    _node(role="AXButton", title="DEPTH_TWO"),
                ],
            ),
        ],
        maximum_depth=1,
    )

    assert [control.text for control in controls] == ["DEPTH_ONE"]


def test_accessibility_attribute_errors_are_tolerated(monkeypatch):
    controls = _read_controls(
        monkeypatch,
        [
            _node(
                role="AXButton",
                title="TITLE_ERROR",
                description="DESCRIPTION_USED",
                error_attributes={"AXTitle"},
            ),
            _node(
                role="AXButton",
                title="GEOMETRY_ERROR",
                error_attributes={"AXPosition"},
            ),
            _node(
                role="AXGroup",
                children=[
                    _node(
                        role="AXButton",
                        title="CHILD_AFTER_ROLE_ERROR",
                    ),
                ],
                error_attributes={"AXRole"},
            ),
        ],
    )

    assert [control.text for control in controls] == [
        "DESCRIPTION_USED",
        "CHILD_AFTER_ROLE_ERROR",
    ]


def test_accessibility_tree_order_is_preserved(monkeypatch):
    controls = _read_controls(
        monkeypatch,
        [
            _node(role="AXButton", title="FIRST"),
            _node(
                role="AXGroup",
                children=[
                    _node(role="AXButton", title="SECOND"),
                    _node(role="AXButton", title="THIRD"),
                ],
            ),
            _node(role="AXButton", title="FOURTH"),
        ],
    )

    assert [control.text for control in controls] == [
        "FIRST",
        "SECOND",
        "THIRD",
        "FOURTH",
    ]


def test_controls_are_returned_as_ui_elements(monkeypatch):
    controls = _read_controls(
        monkeypatch,
        [
            _node(role="AXButton", title="BUTTON"),
        ],
    )

    assert isinstance(controls[0], UIElement)
