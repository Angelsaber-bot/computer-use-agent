from dataclasses import FrozenInstanceError

import pytest

from computer_agent.perception import BoundingBox, UIElement


def test_bounding_box_stores_coordinates():
    box = BoundingBox(
        x=10,
        y=20,
        width=30,
        height=40,
    )

    assert box.x == 10
    assert box.y == 20
    assert box.width == 30
    assert box.height == 40


def test_bounding_box_is_immutable():
    box = BoundingBox(
        x=10,
        y=20,
        width=30,
        height=40,
    )

    with pytest.raises(FrozenInstanceError):
        box.x = 99


def test_bounding_box_geometry_properties():
    box = BoundingBox(
        x=10,
        y=20,
        width=30,
        height=40,
    )

    assert box.left == 10
    assert box.top == 20
    assert box.right == 40
    assert box.bottom == 60
    assert box.center == (25.0, 40.0)
    assert box.area == 1200


def test_bounding_box_contains_point():
    box = BoundingBox(
        x=10,
        y=20,
        width=30,
        height=40,
    )

    assert box.contains_point(10, 20) is True
    assert box.contains_point(39, 59) is True
    assert box.contains_point(40, 60) is False
    assert box.contains_point(9, 20) is False
    assert box.contains_point(10, 19) is False


def test_bounding_box_intersection():
    box = BoundingBox(
        x=10,
        y=20,
        width=30,
        height=40,
    )
    other = BoundingBox(
        x=25,
        y=50,
        width=30,
        height=20,
    )

    assert box.intersects(other) is True
    assert box.intersection(other) == BoundingBox(
        x=25,
        y=50,
        width=15,
        height=10,
    )


def test_bounding_box_non_intersection():
    box = BoundingBox(
        x=10,
        y=20,
        width=30,
        height=40,
    )
    other = BoundingBox(
        x=50,
        y=70,
        width=10,
        height=10,
    )

    assert box.intersects(other) is False
    assert box.intersection(other) is None


def test_bounding_box_edge_touching_does_not_intersect():
    box = BoundingBox(
        x=10,
        y=20,
        width=30,
        height=40,
    )
    touching_right_edge = BoundingBox(
        x=40,
        y=20,
        width=10,
        height=40,
    )
    touching_bottom_edge = BoundingBox(
        x=10,
        y=60,
        width=30,
        height=10,
    )

    assert box.intersects(touching_right_edge) is False
    assert box.intersection(touching_right_edge) is None
    assert box.intersects(touching_bottom_edge) is False
    assert box.intersection(touching_bottom_edge) is None


@pytest.mark.parametrize(
    ("arguments", "error_message"),
    [
        (
            {"x": -1, "y": 0, "width": 1, "height": 1},
            "x must be non-negative",
        ),
        (
            {"x": 0, "y": -1, "width": 1, "height": 1},
            "y must be non-negative",
        ),
        (
            {"x": 0, "y": 0, "width": 0, "height": 1},
            "width must be positive",
        ),
        (
            {"x": 0, "y": 0, "width": 1, "height": 0},
            "height must be positive",
        ),
        (
            {"x": 1.5, "y": 0, "width": 1, "height": 1},
            "x must be an integer",
        ),
    ],
)
def test_bounding_box_rejects_invalid_coordinates_and_dimensions(
    arguments,
    error_message,
):
    with pytest.raises(ValueError, match=error_message):
        BoundingBox(**arguments)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("x", True),
        ("y", False),
        ("width", True),
        ("height", False),
    ],
)
def test_bounding_box_rejects_boolean_coordinate_values(
    field_name,
    value,
):
    arguments = {
        "x": 0,
        "y": 0,
        "width": 1,
        "height": 1,
    }
    arguments[field_name] = value

    with pytest.raises(
        ValueError,
        match=f"{field_name} must be an integer",
    ):
        BoundingBox(**arguments)


def test_ui_element_stores_perception_data():
    box = BoundingBox(
        x=10,
        y=20,
        width=30,
        height=40,
    )

    element = UIElement(
        element_type="button",
        bounding_box=box,
        confidence=0.75,
        text="Submit",
    )

    assert element.element_type == "button"
    assert element.bounding_box is box
    assert element.confidence == 0.75
    assert element.text == "Submit"
    assert element.center == box.center


def test_ui_element_is_immutable():
    element = UIElement(
        element_type="button",
        bounding_box=BoundingBox(
            x=10,
            y=20,
            width=30,
            height=40,
        ),
    )

    with pytest.raises(FrozenInstanceError):
        element.text = "Cancel"


@pytest.mark.parametrize("confidence", [0.0, 1.0])
def test_ui_element_accepts_confidence_boundaries(
    confidence,
):
    element = UIElement(
        element_type="button",
        bounding_box=BoundingBox(
            x=10,
            y=20,
            width=30,
            height=40,
        ),
        confidence=confidence,
    )

    assert element.confidence == confidence


@pytest.mark.parametrize(
    ("arguments", "error_message"),
    [
        (
            {
                "element_type": "",
                "bounding_box": BoundingBox(0, 0, 1, 1),
            },
            "element_type must be a non-empty string",
        ),
        (
            {
                "element_type": "   ",
                "bounding_box": BoundingBox(0, 0, 1, 1),
            },
            "element_type must be a non-empty string",
        ),
        (
            {
                "element_type": 123,
                "bounding_box": BoundingBox(0, 0, 1, 1),
            },
            "element_type must be a non-empty string",
        ),
        (
            {
                "element_type": "button",
                "bounding_box": (0, 0, 1, 1),
            },
            "bounding_box must be a BoundingBox",
        ),
        (
            {
                "element_type": "button",
                "bounding_box": BoundingBox(0, 0, 1, 1),
                "confidence": True,
            },
            "confidence must be numeric",
        ),
        (
            {
                "element_type": "button",
                "bounding_box": BoundingBox(0, 0, 1, 1),
                "confidence": "high",
            },
            "confidence must be numeric",
        ),
        (
            {
                "element_type": "button",
                "bounding_box": BoundingBox(0, 0, 1, 1),
                "confidence": -0.01,
            },
            "confidence must be between 0.0 and 1.0",
        ),
        (
            {
                "element_type": "button",
                "bounding_box": BoundingBox(0, 0, 1, 1),
                "confidence": 1.01,
            },
            "confidence must be between 0.0 and 1.0",
        ),
        (
            {
                "element_type": "button",
                "bounding_box": BoundingBox(0, 0, 1, 1),
                "text": 123,
            },
            "text must be a string or None",
        ),
    ],
)
def test_ui_element_rejects_invalid_values(
    arguments,
    error_message,
):
    with pytest.raises(ValueError, match=error_message):
        UIElement(**arguments)
