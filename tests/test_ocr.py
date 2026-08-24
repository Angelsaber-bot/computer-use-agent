import pytest
from PIL import Image

from computer_agent.perception import (
    BoundingBox,
    TesseractOCR,
    UIElement,
)
from computer_agent.perception import ocr as ocr_module


def _ocr_data(rows):
    data = {
        "text": [],
        "conf": [],
        "left": [],
        "top": [],
        "width": [],
        "height": [],
    }

    for row in rows:
        data["text"].append(row.get("text", "Text"))
        data["conf"].append(row.get("conf", "90"))
        data["left"].append(row.get("left", "1"))
        data["top"].append(row.get("top", "2"))
        data["width"].append(row.get("width", "3"))
        data["height"].append(row.get("height", "4"))

    return data


def _patch_image_to_data(monkeypatch, rows):
    calls = {}

    def image_to_data(
        image,
        lang,
        config,
        output_type,
    ):
        calls["image"] = image
        calls["lang"] = lang
        calls["config"] = config
        calls["output_type"] = output_type

        return _ocr_data(rows)

    monkeypatch.setattr(
        ocr_module.pytesseract,
        "image_to_data",
        image_to_data,
    )

    return calls


def test_tesseract_ocr_converts_valid_ocr_dictionary(monkeypatch):
    image = Image.new(
        "RGB",
        (100, 80),
    )
    calls = _patch_image_to_data(
        monkeypatch,
        [
            {
                "text": "Submit",
                "conf": "87",
                "left": "10",
                "top": "20",
                "width": "30",
                "height": "15",
            }
        ],
    )

    elements = TesseractOCR(
        minimum_confidence=0.5,
    ).recognize(image)

    assert elements == (
        UIElement(
            element_type="text",
            bounding_box=BoundingBox(
                x=10,
                y=20,
                width=30,
                height=15,
            ),
            confidence=0.87,
            text="Submit",
        ),
    )
    assert calls["image"] is not image
    assert calls["image"].size == image.size
    assert calls["lang"] == "eng"
    assert calls["config"] == "--psm 11"
    assert calls["output_type"] == ocr_module.pytesseract.Output.DICT


def test_tesseract_ocr_trims_whitespace_and_filters_empty_text(
    monkeypatch,
):
    image = Image.new(
        "RGB",
        (100, 80),
    )
    _patch_image_to_data(
        monkeypatch,
        [
            {"text": "  Save  "},
            {"text": "   "},
            {"text": ""},
        ],
    )

    elements = TesseractOCR().recognize(image)

    assert [
        element.text for element in elements
    ] == ["Save"]


def test_tesseract_ocr_normalizes_confidence(monkeypatch):
    image = Image.new(
        "RGB",
        (100, 80),
    )
    _patch_image_to_data(
        monkeypatch,
        [
            {
                "text": "Score",
                "conf": "42.5",
            }
        ],
    )

    elements = TesseractOCR().recognize(image)

    assert elements[0].confidence == 0.425


def test_tesseract_ocr_filters_by_minimum_confidence(monkeypatch):
    image = Image.new(
        "RGB",
        (100, 80),
    )
    _patch_image_to_data(
        monkeypatch,
        [
            {
                "text": "Low",
                "conf": "49",
            },
            {
                "text": "Exact",
                "conf": "50",
            },
        ],
    )

    elements = TesseractOCR(
        minimum_confidence=0.5,
    ).recognize(image)

    assert [
        element.text for element in elements
    ] == ["Exact"]


def test_tesseract_ocr_rejects_invalid_image_input():
    with pytest.raises(
        ValueError,
        match="image must be a PIL Image",
    ):
        TesseractOCR().recognize("not an image")


@pytest.mark.parametrize(
    ("minimum_confidence", "error_message"),
    [
        (True, "minimum_confidence must be numeric"),
        (False, "minimum_confidence must be numeric"),
        ("0.5", "minimum_confidence must be numeric"),
        (float("nan"), "minimum_confidence must be finite"),
        (float("inf"), "minimum_confidence must be finite"),
        (float("-inf"), "minimum_confidence must be finite"),
        (-0.01, "minimum_confidence must be between 0.0 and 1.0"),
        (1.01, "minimum_confidence must be between 0.0 and 1.0"),
    ],
)
def test_tesseract_ocr_rejects_invalid_minimum_confidence_values(
    minimum_confidence,
    error_message,
):
    with pytest.raises(
        ValueError,
        match=error_message,
    ):
        TesseractOCR(
            minimum_confidence=minimum_confidence,
        )


@pytest.mark.parametrize(
    "invalid_box_row",
    [
        {"text": "NegativeX", "left": "-1"},
        {"text": "NegativeY", "top": "-1"},
        {"text": "ZeroWidth", "width": "0"},
        {"text": "ZeroHeight", "height": "0"},
        {"text": "InvalidWidth", "width": "bad"},
        {"text": "OutsideRight", "left": "99", "width": "2"},
        {"text": "OutsideBottom", "top": "79", "height": "2"},
    ],
)
def test_tesseract_ocr_ignores_invalid_and_zero_sized_boxes(
    monkeypatch,
    invalid_box_row,
):
    image = Image.new(
        "RGB",
        (100, 80),
    )
    _patch_image_to_data(
        monkeypatch,
        [
            invalid_box_row,
            {
                "text": "Keep",
                "left": "10",
                "top": "20",
                "width": "30",
                "height": "15",
            },
        ],
    )

    elements = TesseractOCR().recognize(image)

    assert [
        element.text for element in elements
    ] == ["Keep"]


def test_tesseract_ocr_does_not_mutate_source_image(monkeypatch):
    image = Image.new(
        "RGB",
        (2, 1),
    )
    image.putdata(
        [
            (10, 20, 30),
            (40, 50, 60),
        ]
    )
    original_bytes = image.tobytes()

    def image_to_data(
        ocr_image,
        lang,
        config,
        output_type,
    ):
        ocr_image.putpixel(
            (0, 0),
            (255, 255, 255),
        )

        return _ocr_data(
            [
                {
                    "text": "Text",
                    "left": "0",
                    "top": "0",
                    "width": "1",
                    "height": "1",
                }
            ]
        )

    monkeypatch.setattr(
        ocr_module.pytesseract,
        "image_to_data",
        image_to_data,
    )

    TesseractOCR().recognize(image)

    assert image.tobytes() == original_bytes


def test_tesseract_ocr_reports_available_executable(monkeypatch):
    monkeypatch.setattr(
        ocr_module.pytesseract,
        "get_tesseract_version",
        lambda: "5.5.1",
    )

    assert TesseractOCR.is_available() is True


def test_tesseract_ocr_reports_missing_executable(monkeypatch):
    def get_tesseract_version():
        raise OSError(
            "missing executable"
        )

    monkeypatch.setattr(
        ocr_module.pytesseract,
        "get_tesseract_version",
        get_tesseract_version,
    )

    assert TesseractOCR.is_available() is False
