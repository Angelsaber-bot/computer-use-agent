from dataclasses import FrozenInstanceError

import pytest

from computer_agent.core.models import Action
from computer_agent.grounding import (
    ActionGrounder,
    ActionGroundingResult,
    ActionGroundingStatus,
    GroundingResult,
    GroundingStatus,
)
from computer_agent.perception.models import BoundingBox, UIElement


CLICK_REASON = "Click the UI element resolved by deterministic grounding."


def _box(
    x=10,
    y=20,
    width=100,
    height=30,
) -> BoundingBox:
    return BoundingBox(
        x=x,
        y=y,
        width=width,
        height=height,
    )


def _element(
    bounding_box=None,
) -> UIElement:
    return UIElement(
        element_type="button",
        bounding_box=bounding_box or _box(),
        confidence=0.95,
        text="Target",
        enabled=True,
        source="accessibility",
    )


def _resolved(
    bounding_box=None,
) -> GroundingResult:
    return GroundingResult(
        status=GroundingStatus.RESOLVED,
        element=_element(bounding_box),
        candidates=(),
        reason="resolved by text",
    )


def _action() -> Action:
    return Action(
        tool_name="click_mouse",
        arguments={
            "x": 10,
            "y": 20,
        },
        reason=CLICK_REASON,
    )


def test_resolved_element_produces_ready_click_action():
    result = ActionGrounder().ground_click(
        _resolved(),
        (200, 100),
    )

    assert result.status is ActionGroundingStatus.READY
    assert isinstance(result.action, Action)
    assert result.action.tool_name == "click_mouse"
    assert result.action.arguments == {
        "x": 60,
        "y": 35,
    }
    assert isinstance(result.action.arguments["x"], int)
    assert isinstance(result.action.arguments["y"], int)
    assert result.action.reason == CLICK_REASON


@pytest.mark.parametrize(
    ("bounding_box", "expected_arguments"),
    [
        (
            _box(
                x=10,
                y=20,
                width=4,
                height=6,
            ),
            {
                "x": 12,
                "y": 23,
            },
        ),
        (
            _box(
                x=10,
                y=20,
                width=5,
                height=7,
            ),
            {
                "x": 12,
                "y": 23,
            },
        ),
    ],
)
def test_box_center_uses_explicit_floor_policy(
    bounding_box,
    expected_arguments,
):
    result = ActionGrounder().ground_click(
        _resolved(bounding_box),
        (100, 100),
    )

    assert result.status is ActionGroundingStatus.READY
    assert result.action.arguments == expected_arguments


def test_large_integer_coordinates_do_not_require_float_conversion():
    large_coordinate = 10**400

    result = ActionGrounder().ground_click(
        _resolved(
            _box(
                x=large_coordinate,
                y=large_coordinate,
                width=3,
                height=5,
            )
        ),
        (
            large_coordinate + 5,
            large_coordinate + 7,
        ),
    )

    assert result.status is ActionGroundingStatus.READY
    assert result.action.arguments == {
        "x": large_coordinate + 1,
        "y": large_coordinate + 2,
    }


@pytest.mark.parametrize(
    ("status", "source_reason"),
    [
        (
            GroundingStatus.AMBIGUOUS,
            "ambiguous text candidates",
        ),
        (
            GroundingStatus.UNSAFE,
            "text candidates were unsafe",
        ),
        (
            GroundingStatus.NOT_FOUND,
            "no exact identifier or normalized text match",
        ),
    ],
)
def test_non_resolved_grounding_results_block(status, source_reason):
    grounding_result = GroundingResult(
        status=status,
        element=None,
        candidates=(),
        reason=source_reason,
    )

    result = ActionGrounder().ground_click(
        grounding_result,
        (100, 100),
    )

    assert result.status is ActionGroundingStatus.BLOCKED
    assert result.action is None
    assert status.value in result.reason
    assert source_reason in result.reason


@pytest.mark.parametrize(
    ("bounding_box", "expected_point"),
    [
        (
            _box(
                x=0,
                y=10,
                width=1,
                height=1,
            ),
            (0, 10),
        ),
        (
            _box(
                x=99,
                y=10,
                width=1,
                height=1,
            ),
            (99, 10),
        ),
        (
            _box(
                x=10,
                y=0,
                width=1,
                height=1,
            ),
            (10, 0),
        ),
        (
            _box(
                x=10,
                y=79,
                width=1,
                height=1,
            ),
            (10, 79),
        ),
    ],
)
def test_points_on_unsafe_screen_edges_are_blocked(
    bounding_box,
    expected_point,
):
    result = ActionGrounder().ground_click(
        _resolved(bounding_box),
        (100, 80),
    )

    assert result.status is ActionGroundingStatus.BLOCKED
    assert result.action is None
    assert f"({expected_point[0]}, {expected_point[1]})" in result.reason
    assert "violates safe screen bounds" in result.reason


@pytest.mark.parametrize(
    ("bounding_box", "expected_arguments"),
    [
        (
            _box(
                x=1,
                y=1,
                width=1,
                height=1,
            ),
            {
                "x": 1,
                "y": 1,
            },
        ),
        (
            _box(
                x=98,
                y=78,
                width=1,
                height=1,
            ),
            {
                "x": 98,
                "y": 78,
            },
        ),
    ],
)
def test_points_immediately_inside_allowed_margin_are_accepted(
    bounding_box,
    expected_arguments,
):
    result = ActionGrounder().ground_click(
        _resolved(bounding_box),
        (100, 80),
    )

    assert result.status is ActionGroundingStatus.READY
    assert result.action.arguments == expected_arguments


@pytest.mark.parametrize(
    "bounding_box",
    [
        _box(
            x=100,
            y=10,
            width=1,
            height=1,
        ),
        _box(
            x=10,
            y=80,
            width=1,
            height=1,
        ),
    ],
)
def test_points_outside_logical_screen_are_blocked(bounding_box):
    result = ActionGrounder(safe_edge_margin=0).ground_click(
        _resolved(bounding_box),
        (100, 80),
    )

    assert result.status is ActionGroundingStatus.BLOCKED
    assert result.action is None
    assert "violates safe screen bounds" in result.reason


@pytest.mark.parametrize(
    "screen_size",
    [
        (1, 100),
        (100, 1),
        (2, 2),
    ],
)
def test_screen_too_small_for_configured_margin_is_blocked(screen_size):
    result = ActionGrounder().ground_click(
        _resolved(
            _box(
                x=1,
                y=1,
                width=1,
                height=1,
            )
        ),
        screen_size,
    )

    assert result.status is ActionGroundingStatus.BLOCKED
    assert result.action is None
    assert "no usable screen interior" in result.reason


def test_zero_safe_edge_margin_permits_valid_outer_edge_coordinates():
    result = ActionGrounder(safe_edge_margin=0).ground_click(
        _resolved(
            _box(
                x=0,
                y=0,
                width=1,
                height=1,
            )
        ),
        (1, 1),
    )

    assert result.status is ActionGroundingStatus.READY
    assert result.action.arguments == {
        "x": 0,
        "y": 0,
    }


@pytest.mark.parametrize(
    "safe_edge_margin",
    [
        True,
        False,
        -1,
        1.0,
        "1",
        None,
    ],
)
def test_invalid_margins_are_rejected(safe_edge_margin):
    with pytest.raises(ValueError, match="safe_edge_margin"):
        ActionGrounder(safe_edge_margin=safe_edge_margin)


@pytest.mark.parametrize(
    "screen_size",
    [
        [100, 80],
        (100,),
        (100, 80, 1),
        (),
        "100,80",
        None,
        (100.0, 80),
        (100, 80.0),
        (True, 80),
        (100, False),
        (0, 80),
        (100, 0),
        (-1, 80),
        (100, -1),
    ],
)
def test_invalid_screen_sizes_are_rejected(screen_size):
    with pytest.raises(ValueError, match="screen_size"):
        ActionGrounder().ground_click(
            _resolved(),
            screen_size,
        )


@pytest.mark.parametrize(
    "grounding_result",
    [
        None,
        object(),
        "resolved",
        {},
    ],
)
def test_invalid_grounding_result_inputs_are_rejected(grounding_result):
    with pytest.raises(ValueError, match="grounding_result"):
        ActionGrounder().ground_click(
            grounding_result,
            (100, 100),
        )


@pytest.mark.parametrize(
    ("status", "action", "reason", "message"),
    [
        (
            "ready",
            _action(),
            "ready",
            "status",
        ),
        (
            ActionGroundingStatus.READY,
            "click",
            "ready",
            "action",
        ),
        (
            ActionGroundingStatus.READY,
            _action(),
            "",
            "reason",
        ),
        (
            ActionGroundingStatus.READY,
            _action(),
            "   ",
            "reason",
        ),
        (
            ActionGroundingStatus.READY,
            None,
            "ready",
            "READY",
        ),
        (
            ActionGroundingStatus.BLOCKED,
            _action(),
            "blocked",
            "BLOCKED",
        ),
    ],
)
def test_action_grounding_result_invariants(
    status,
    action,
    reason,
    message,
):
    with pytest.raises(ValueError, match=message):
        ActionGroundingResult(
            status=status,
            action=action,
            reason=reason,
        )


def test_action_grounding_result_is_immutable_and_slotted():
    result = ActionGroundingResult(
        status=ActionGroundingStatus.BLOCKED,
        action=None,
        reason="blocked",
    )

    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        result.reason = "changed"
