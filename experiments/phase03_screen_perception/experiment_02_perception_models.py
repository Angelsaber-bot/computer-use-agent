"""Phase 03 experiment for reusable perception data models."""

from computer_agent.perception import BoundingBox, UIElement


def main() -> int:
    primary_box = BoundingBox(
        x=10,
        y=20,
        width=100,
        height=50,
    )
    overlapping_box = BoundingBox(
        x=60,
        y=40,
        width=80,
        height=40,
    )
    edge_touching_box = BoundingBox(
        x=110,
        y=20,
        width=20,
        height=50,
    )

    intersection = primary_box.intersection(overlapping_box)
    element = UIElement(
        element_type="button",
        bounding_box=primary_box,
        confidence=0.95,
        text="Submit",
    )

    assert primary_box.left == 10
    assert primary_box.top == 20
    assert primary_box.right == 110
    assert primary_box.bottom == 70
    assert primary_box.center == (60.0, 45.0)
    assert primary_box.area == 5000
    assert primary_box.contains_point(10, 20) is True
    assert primary_box.contains_point(109, 69) is True
    assert primary_box.contains_point(110, 70) is False
    assert primary_box.intersects(overlapping_box) is True
    assert intersection == BoundingBox(
        x=60,
        y=40,
        width=50,
        height=30,
    )
    assert primary_box.intersects(edge_touching_box) is False
    assert primary_box.intersection(edge_touching_box) is None
    assert element.center == primary_box.center

    print("Phase 03 Experiment 02: Perception Models")
    print(
        "Primary box: "
        f"left={primary_box.left}, "
        f"top={primary_box.top}, "
        f"right={primary_box.right}, "
        f"bottom={primary_box.bottom}, "
        f"center={primary_box.center}, "
        f"area={primary_box.area}"
    )
    print(
        "Containment: "
        "(10, 20) is inside; "
        "(109, 69) is inside; "
        "(110, 70) is outside"
    )
    print(f"Intersection: {intersection}")
    print(
        "Edge touching: "
        f"intersects={primary_box.intersects(edge_touching_box)}"
    )
    print(
        "UI element: "
        f"type={element.element_type}, "
        f"text={element.text}, "
        f"confidence={element.confidence}, "
        f"center={element.center}"
    )
    print(
        "Phase 03 Experiment 02 completed successfully."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
