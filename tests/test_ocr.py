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
        "page_num": [],
        "block_num": [],
        "par_num": [],
        "line_num": [],
        "text": [],
        "conf": [],
        "left": [],
        "top": [],
        "width": [],
        "height": [],
    }

    for row in rows:
        data["page_num"].append(row.get("page_num", "1"))
        data["block_num"].append(row.get("block_num", "1"))
        data["par_num"].append(row.get("par_num", "1"))
        data["line_num"].append(row.get("line_num", "1"))
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
        call = {
            "image": image,
            "lang": lang,
            "config": config,
            "output_type": output_type,
        }
        calls["calls"] = [
            *calls.get("calls", []),
            call,
        ]
        calls.update(call)

        return _ocr_data(rows)

    monkeypatch.setattr(
        ocr_module.pytesseract,
        "image_to_data",
        image_to_data,
    )

    return calls


def test_tesseract_ocr_uses_default_page_segmentation_mode(
    monkeypatch,
):
    image = Image.new(
        "RGB",
        (100, 80),
    )
    calls = _patch_image_to_data(
        monkeypatch,
        [
            {
                "text": "Default",
            }
        ],
    )

    TesseractOCR().recognize(image)

    assert calls["config"] == "--psm 11"


def test_tesseract_ocr_uses_custom_page_segmentation_mode(
    monkeypatch,
):
    image = Image.new(
        "RGB",
        (100, 80),
    )
    calls = _patch_image_to_data(
        monkeypatch,
        [
            {
                "text": "Custom",
            }
        ],
    )

    TesseractOCR(
        page_segmentation_mode=6,
    ).recognize(image)

    assert calls["config"] == "--psm 6"


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


def test_tesseract_ocr_defaults_to_word_level_elements(monkeypatch):
    image = Image.new(
        "RGB",
        (100, 80),
    )
    _patch_image_to_data(
        monkeypatch,
        [
            {
                "text": "Alpha",
                "conf": "91",
                "left": "10",
                "top": "12",
                "width": "30",
                "height": "14",
                "line_num": "1",
            },
            {
                "text": "Beta",
                "conf": "82",
                "left": "45",
                "top": "12",
                "width": "24",
                "height": "14",
                "line_num": "1",
            },
        ],
    )

    elements = TesseractOCR().recognize(image)

    assert elements == (
        UIElement(
            element_type="text",
            bounding_box=BoundingBox(
                x=10,
                y=12,
                width=30,
                height=14,
            ),
            confidence=0.91,
            text="Alpha",
        ),
        UIElement(
            element_type="text",
            bounding_box=BoundingBox(
                x=45,
                y=12,
                width=24,
                height=14,
            ),
            confidence=0.82,
            text="Beta",
        ),
    )


def test_tesseract_ocr_region_passes_crop_size_to_tesseract(
    monkeypatch,
):
    image = Image.new(
        "RGB",
        (100, 80),
    )
    calls = _patch_image_to_data(
        monkeypatch,
        [
            {
                "text": "Crop",
                "left": "1",
                "top": "2",
                "width": "3",
                "height": "4",
            }
        ],
    )

    TesseractOCR().recognize_region(
        image,
        BoundingBox(
            x=20,
            y=10,
            width=30,
            height=40,
        ),
    )

    assert calls["image"].size == (30, 40)


def test_tesseract_ocr_region_translates_to_global_coordinates(
    monkeypatch,
):
    image = Image.new(
        "RGB",
        (100, 80),
    )
    _patch_image_to_data(
        monkeypatch,
        [
            {
                "text": "Global",
                "left": "5",
                "top": "6",
                "width": "7",
                "height": "8",
            }
        ],
    )

    elements = TesseractOCR().recognize_region(
        image,
        BoundingBox(
            x=20,
            y=10,
            width=30,
            height=40,
        ),
    )

    assert elements[0].bounding_box == BoundingBox(
        x=25,
        y=16,
        width=7,
        height=8,
    )


def test_tesseract_ocr_region_translates_grouped_lines(
    monkeypatch,
):
    image = Image.new(
        "RGB",
        (120, 90),
    )
    _patch_image_to_data(
        monkeypatch,
        [
            {
                "text": "Line",
                "left": "2",
                "top": "3",
                "width": "5",
                "height": "7",
            },
            {
                "text": "Text",
                "left": "10",
                "top": "4",
                "width": "6",
                "height": "5",
            },
        ],
    )

    elements = TesseractOCR(
        group_words_by_line=True,
    ).recognize_region(
        image,
        BoundingBox(
            x=40,
            y=20,
            width=50,
            height=30,
        ),
    )

    assert elements == (
        UIElement(
            element_type="text",
            bounding_box=BoundingBox(
                x=42,
                y=23,
                width=14,
                height=7,
            ),
            confidence=0.9,
            text="Line Text",
        ),
    )


def test_tesseract_ocr_region_preserves_element_metadata(monkeypatch):
    image = Image.new(
        "RGB",
        (100, 80),
    )

    def recognize(
        self,
        crop,
    ):
        assert crop.size == (30, 40)
        return (
            UIElement(
                element_type="button",
                bounding_box=BoundingBox(
                    x=3,
                    y=4,
                    width=5,
                    height=6,
                ),
                confidence=0.7,
                text="Meta",
                identifier="meta-id",
                value="meta-value",
                enabled=True,
                focused=False,
                selected=True,
                source="custom",
            ),
        )

    monkeypatch.setattr(
        TesseractOCR,
        "recognize",
        recognize,
    )

    elements = TesseractOCR().recognize_region(
        image,
        BoundingBox(
            x=20,
            y=10,
            width=30,
            height=40,
        ),
    )

    assert elements == (
        UIElement(
            element_type="button",
            bounding_box=BoundingBox(
                x=23,
                y=14,
                width=5,
                height=6,
            ),
            confidence=0.7,
            text="Meta",
            identifier="meta-id",
            value="meta-value",
            enabled=True,
            focused=False,
            selected=True,
            source="custom",
        ),
    )


@pytest.mark.parametrize(
    "region",
    [
        BoundingBox(
            x=0,
            y=0,
            width=100,
            height=80,
        ),
        BoundingBox(
            x=90,
            y=70,
            width=10,
            height=10,
        ),
    ],
)
def test_tesseract_ocr_region_accepts_regions_touching_image_edges(
    monkeypatch,
    region,
):
    image = Image.new(
        "RGB",
        (100, 80),
    )
    _patch_image_to_data(
        monkeypatch,
        [
            {
                "text": "Edge",
                "left": "0",
                "top": "0",
                "width": "1",
                "height": "1",
            }
        ],
    )

    elements = TesseractOCR().recognize_region(
        image,
        region,
    )

    assert elements[0].bounding_box == BoundingBox(
        x=region.x,
        y=region.y,
        width=1,
        height=1,
    )


@pytest.mark.parametrize(
    ("region", "error_message"),
    [
        ("not a box", "region must be a BoundingBox"),
        (
            BoundingBox(
                x=90,
                y=10,
                width=11,
                height=10,
            ),
            "region must be inside the image",
        ),
        (
            BoundingBox(
                x=10,
                y=70,
                width=10,
                height=11,
            ),
            "region must be inside the image",
        ),
    ],
)
def test_tesseract_ocr_region_rejects_invalid_or_out_of_image_regions(
    region,
    error_message,
):
    image = Image.new(
        "RGB",
        (100, 80),
    )

    with pytest.raises(
        ValueError,
        match=error_message,
    ):
        TesseractOCR().recognize_region(
            image,
            region,
        )


def test_tesseract_ocr_groups_multiple_words_on_one_line(
    monkeypatch,
):
    image = Image.new(
        "RGB",
        (100, 80),
    )
    _patch_image_to_data(
        monkeypatch,
        [
            {
                "text": "Alpha",
                "left": "10",
                "top": "12",
                "width": "30",
                "height": "14",
                "line_num": "1",
            },
            {
                "text": "Beta",
                "left": "45",
                "top": "12",
                "width": "24",
                "height": "14",
                "line_num": "1",
            },
        ],
    )

    elements = TesseractOCR(
        group_words_by_line=True,
    ).recognize(image)

    assert [
        element.text for element in elements
    ] == ["Alpha Beta"]


def test_tesseract_ocr_keeps_separate_lines_separate(
    monkeypatch,
):
    image = Image.new(
        "RGB",
        (100, 80),
    )
    _patch_image_to_data(
        monkeypatch,
        [
            {
                "text": "First",
                "line_num": "1",
            },
            {
                "text": "Second",
                "line_num": "2",
            },
        ],
    )

    elements = TesseractOCR(
        group_words_by_line=True,
    ).recognize(image)

    assert [
        element.text for element in elements
    ] == ["First", "Second"]


def test_tesseract_ocr_line_bounding_box_uses_word_union(
    monkeypatch,
):
    image = Image.new(
        "RGB",
        (100, 80),
    )
    _patch_image_to_data(
        monkeypatch,
        [
            {
                "text": "Wide",
                "left": "10",
                "top": "20",
                "width": "30",
                "height": "10",
            },
            {
                "text": "Tall",
                "left": "50",
                "top": "15",
                "width": "20",
                "height": "25",
            },
        ],
    )

    elements = TesseractOCR(
        group_words_by_line=True,
    ).recognize(image)

    assert elements[0].bounding_box == BoundingBox(
        x=10,
        y=15,
        width=60,
        height=25,
    )


def test_tesseract_ocr_line_confidence_uses_minimum_word_confidence(
    monkeypatch,
):
    image = Image.new(
        "RGB",
        (100, 80),
    )
    _patch_image_to_data(
        monkeypatch,
        [
            {
                "text": "High",
                "conf": "91",
            },
            {
                "text": "Lower",
                "conf": "62",
            },
        ],
    )

    elements = TesseractOCR(
        group_words_by_line=True,
    ).recognize(image)

    assert elements[0].confidence == 0.62


def test_tesseract_ocr_filters_grouped_lines_by_minimum_confidence(
    monkeypatch,
):
    image = Image.new(
        "RGB",
        (100, 80),
    )
    _patch_image_to_data(
        monkeypatch,
        [
            {
                "text": "Reject",
                "conf": "90",
                "line_num": "1",
            },
            {
                "text": "Line",
                "conf": "49",
                "line_num": "1",
            },
            {
                "text": "Keep",
                "conf": "50",
                "line_num": "2",
            },
        ],
    )

    elements = TesseractOCR(
        minimum_confidence=0.5,
        group_words_by_line=True,
    ).recognize(image)

    assert [
        element.text for element in elements
    ] == ["Keep"]


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
    ("page_segmentation_mode", "error_message"),
    [
        (True, "page_segmentation_mode must be an integer"),
        (False, "page_segmentation_mode must be an integer"),
        ("6", "page_segmentation_mode must be an integer"),
        (6.0, "page_segmentation_mode must be an integer"),
        (None, "page_segmentation_mode must be an integer"),
        (-1, "page_segmentation_mode must be between 0 and 13"),
        (14, "page_segmentation_mode must be between 0 and 13"),
    ],
)
def test_tesseract_ocr_rejects_invalid_page_segmentation_modes(
    page_segmentation_mode,
    error_message,
):
    with pytest.raises(
        ValueError,
        match=error_message,
    ):
        TesseractOCR(
            page_segmentation_mode=page_segmentation_mode,
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
