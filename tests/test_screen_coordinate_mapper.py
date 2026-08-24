from pathlib import Path

import pytest

from computer_agent.perception import (
    BoundingBox,
    ScreenCoordinateMapper,
    ScreenFrame,
    UIElement,
)


def _frame(
    pixel_width=2940,
    pixel_height=1912,
    screen_width=1470,
    screen_height=956,
) -> ScreenFrame:
    return ScreenFrame(
        image_path=Path("screen.png"),
        pixel_width=pixel_width,
        pixel_height=pixel_height,
        screen_width=screen_width,
        screen_height=screen_height,
    )


def test_mapper_converts_exact_retina_scale_box():
    mapper = ScreenCoordinateMapper(
        _frame()
    )

    logical_box = mapper.pixel_box_to_logical(
        BoundingBox(
            x=238,
            y=104,
            width=198,
            height=24,
        )
    )

    assert logical_box == BoundingBox(
        x=119,
        y=52,
        width=99,
        height=12,
    )


def test_mapper_converts_non_integer_scales():
    mapper = ScreenCoordinateMapper(
        _frame(
            pixel_width=1000,
            pixel_height=900,
            screen_width=333,
            screen_height=400,
        )
    )

    assert mapper.pixel_point_to_logical(
        500,
        450,
    ) == pytest.approx(
        (
            166.5,
            200.0,
        )
    )


def test_mapper_uses_floor_and_ceil_for_box_edges():
    mapper = ScreenCoordinateMapper(
        _frame(
            pixel_width=100,
            pixel_height=80,
            screen_width=30,
            screen_height=30,
        )
    )

    logical_box = mapper.pixel_box_to_logical(
        BoundingBox(
            x=1,
            y=1,
            width=9,
            height=9,
        )
    )

    assert logical_box == BoundingBox(
        x=0,
        y=0,
        width=3,
        height=4,
    )


def test_mapper_converts_pixel_point():
    mapper = ScreenCoordinateMapper(
        _frame()
    )

    assert mapper.pixel_point_to_logical(
        294,
        956,
    ) == (
        147.0,
        478.0,
    )


def test_mapper_converts_bounding_box():
    mapper = ScreenCoordinateMapper(
        _frame()
    )

    logical_box = mapper.pixel_box_to_logical(
        BoundingBox(
            x=10,
            y=20,
            width=30,
            height=40,
        )
    )

    assert logical_box == BoundingBox(
        x=5,
        y=10,
        width=15,
        height=20,
    )


def test_mapper_converts_ui_element():
    mapper = ScreenCoordinateMapper(
        _frame()
    )
    element = UIElement(
        element_type="text",
        bounding_box=BoundingBox(
            x=10,
            y=20,
            width=30,
            height=40,
        ),
        confidence=0.75,
        text="Submit",
    )

    mapped = mapper.pixel_element_to_logical(element)

    assert mapped == UIElement(
        element_type="text",
        bounding_box=BoundingBox(
            x=5,
            y=10,
            width=15,
            height=20,
        ),
        confidence=0.75,
        text="Submit",
    )


def test_mapper_preserves_source_objects():
    mapper = ScreenCoordinateMapper(
        _frame()
    )
    box = BoundingBox(
        x=10,
        y=20,
        width=30,
        height=40,
    )
    element = UIElement(
        element_type="text",
        bounding_box=box,
        confidence=0.75,
        text="Submit",
    )

    mapped_box = mapper.pixel_box_to_logical(box)
    mapped_element = mapper.pixel_element_to_logical(element)

    assert box == BoundingBox(
        x=10,
        y=20,
        width=30,
        height=40,
    )
    assert element.bounding_box is box
    assert mapped_box is not box
    assert mapped_element is not element
    assert mapped_element.bounding_box is not box


def test_mapper_accepts_pixel_frame_boundaries():
    mapper = ScreenCoordinateMapper(
        _frame()
    )

    assert mapper.pixel_point_to_logical(
        0,
        0,
    ) == (
        0.0,
        0.0,
    )
    assert mapper.pixel_point_to_logical(
        2939,
        1911,
    ) == (
        1469.5,
        955.5,
    )
    assert mapper.pixel_box_to_logical(
        BoundingBox(
            x=2938,
            y=1910,
            width=2,
            height=2,
        )
    ) == BoundingBox(
        x=1469,
        y=955,
        width=1,
        height=1,
    )


@pytest.mark.parametrize(
    ("method_name", "arguments", "error_message"),
    [
        (
            "pixel_box_to_logical",
            ("not a box",),
            "bounding_box must be a BoundingBox",
        ),
        (
            "pixel_element_to_logical",
            ("not an element",),
            "element must be a UIElement",
        ),
    ],
)
def test_mapper_rejects_invalid_object_types(
    method_name,
    arguments,
    error_message,
):
    mapper = ScreenCoordinateMapper(
        _frame()
    )

    with pytest.raises(
        ValueError,
        match=error_message,
    ):
        getattr(
            mapper,
            method_name,
        )(*arguments)


def test_mapper_rejects_invalid_frame_type():
    with pytest.raises(
        ValueError,
        match="frame must be a ScreenFrame",
    ):
        ScreenCoordinateMapper(
            "not a frame"
        )


@pytest.mark.parametrize(
    ("x", "y", "error_message"),
    [
        (-1, 0, "x must be inside the pixel frame"),
        (2940, 0, "x must be inside the pixel frame"),
        (0, -1, "y must be inside the pixel frame"),
        (0, 1912, "y must be inside the pixel frame"),
    ],
)
def test_mapper_rejects_out_of_bounds_points(
    x,
    y,
    error_message,
):
    mapper = ScreenCoordinateMapper(
        _frame()
    )

    with pytest.raises(
        ValueError,
        match=error_message,
    ):
        mapper.pixel_point_to_logical(
            x,
            y,
        )


@pytest.mark.parametrize(
    "box",
    [
        BoundingBox(
            x=2939,
            y=0,
            width=2,
            height=1,
        ),
        BoundingBox(
            x=0,
            y=1911,
            width=1,
            height=2,
        ),
    ],
)
def test_mapper_rejects_out_of_bounds_boxes(box):
    mapper = ScreenCoordinateMapper(
        _frame()
    )

    with pytest.raises(
        ValueError,
        match="bounding_box must be inside the pixel frame",
    ):
        mapper.pixel_box_to_logical(box)


@pytest.mark.parametrize(
    ("x", "y", "error_message"),
    [
        (True, 0, "x must be numeric"),
        (0, False, "y must be numeric"),
        ("0", 0, "x must be numeric"),
        (0, "0", "y must be numeric"),
        (float("nan"), 0, "x must be finite"),
        (0, float("inf"), "y must be finite"),
    ],
)
def test_mapper_rejects_invalid_point_coordinates(
    x,
    y,
    error_message,
):
    mapper = ScreenCoordinateMapper(
        _frame()
    )

    with pytest.raises(
        ValueError,
        match=error_message,
    ):
        mapper.pixel_point_to_logical(
            x,
            y,
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("pixel_width", 0),
        ("pixel_height", -1),
        ("screen_width", 0),
        ("screen_height", -1),
    ],
)
def test_screen_frame_still_rejects_invalid_dimensions(
    field_name,
    value,
):
    arguments = {
        "image_path": Path("screen.png"),
        "pixel_width": 2940,
        "pixel_height": 1912,
        "screen_width": 1470,
        "screen_height": 956,
    }
    arguments[field_name] = value

    with pytest.raises(
        ValueError,
        match="screen frame dimensions must be positive",
    ):
        ScreenFrame(**arguments)
