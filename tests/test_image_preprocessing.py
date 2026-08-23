import pytest
from PIL import Image

from computer_agent.perception import ImagePreprocessor


def test_convert_to_grayscale_returns_grayscale_copy():
    image = Image.new(
        "RGB",
        (2, 2),
        color=(10, 20, 30),
    )

    grayscale = ImagePreprocessor.convert_to_grayscale(image)

    assert grayscale.mode == "L"
    assert grayscale.size == image.size
    assert image.mode == "RGB"


def test_resize_calculates_dimensions_deterministically():
    image = Image.new(
        "RGB",
        (10, 7),
    )

    resized = ImagePreprocessor.resize(
        image,
        0.5,
    )

    assert resized.size == (5, 3)


def test_resize_keeps_very_small_outputs_at_least_one_pixel():
    image = Image.new(
        "RGB",
        (2, 2),
    )

    resized = ImagePreprocessor.resize(
        image,
        0.1,
    )

    assert resized.size == (1, 1)


def test_resize_uses_lanczos_resampling(monkeypatch):
    image = Image.new(
        "RGB",
        (10, 10),
    )
    resampling_filters = []
    original_resize = Image.Image.resize

    def resize_spy(
        self,
        size,
        resample=None,
        box=None,
        reducing_gap=None,
    ):
        resampling_filters.append(resample)

        return original_resize(
            self,
            size,
            resample=resample,
            box=box,
            reducing_gap=reducing_gap,
        )

    monkeypatch.setattr(
        Image.Image,
        "resize",
        resize_spy,
    )

    ImagePreprocessor.resize(
        image,
        0.5,
    )

    assert resampling_filters == [
        Image.Resampling.LANCZOS
    ]


@pytest.mark.parametrize(
    ("method_name", "arguments"),
    [
        ("convert_to_grayscale", ()),
        ("resize", (0.5,)),
        ("enhance_contrast", ()),
        ("prepare_for_ocr", ()),
    ],
)
def test_preprocessor_rejects_invalid_image_inputs(
    method_name,
    arguments,
):
    method = getattr(
        ImagePreprocessor,
        method_name,
    )

    with pytest.raises(
        ValueError,
        match="image must be a PIL Image",
    ):
        method(
            "not an image",
            *arguments,
        )


@pytest.mark.parametrize(
    "method_name",
    [
        "resize",
        "prepare_for_ocr",
    ],
)
@pytest.mark.parametrize(
    ("scale_factor", "error_message"),
    [
        (True, "scale_factor must be numeric"),
        (False, "scale_factor must be numeric"),
        ("2", "scale_factor must be numeric"),
        (float("nan"), "scale_factor must be finite"),
        (float("inf"), "scale_factor must be finite"),
        (float("-inf"), "scale_factor must be finite"),
        (0, "scale_factor must be positive"),
        (-0.5, "scale_factor must be positive"),
    ],
)
def test_preprocessor_rejects_invalid_scale_factors(
    method_name,
    scale_factor,
    error_message,
):
    image = Image.new(
        "RGB",
        (2, 2),
    )
    method = getattr(
        ImagePreprocessor,
        method_name,
    )

    with pytest.raises(
        ValueError,
        match=error_message,
    ):
        method(
            image,
            scale_factor,
        )


def test_enhance_contrast_uses_automatic_contrast():
    image = Image.new(
        "L",
        (2, 1),
    )
    image.putdata(
        [100, 110]
    )

    enhanced = ImagePreprocessor.enhance_contrast(image)

    assert enhanced.mode == "L"
    assert enhanced.getextrema() == (0, 255)
    assert image.getextrema() == (100, 110)


def test_prepare_for_ocr_runs_complete_pipeline():
    image = Image.new(
        "RGB",
        (4, 2),
    )
    image.putdata(
        [
            (100, 100, 100),
            (100, 100, 100),
            (110, 110, 110),
            (110, 110, 110),
            (100, 100, 100),
            (100, 100, 100),
            (110, 110, 110),
            (110, 110, 110),
        ]
    )

    prepared = ImagePreprocessor.prepare_for_ocr(
        image,
        scale_factor=0.5,
    )

    assert prepared.mode == "L"
    assert prepared.size == (2, 1)
    assert prepared.getextrema() == (0, 255)


def test_prepare_for_ocr_does_not_mutate_source_image():
    image = Image.new(
        "RGB",
        (2, 1),
    )
    image.putdata(
        [
            (100, 100, 100),
            (110, 110, 110),
        ]
    )
    original_mode = image.mode
    original_size = image.size
    original_bytes = image.tobytes()

    prepared = ImagePreprocessor.prepare_for_ocr(
        image,
    )

    assert prepared.mode == "L"
    assert image.mode == original_mode
    assert image.size == original_size
    assert image.tobytes() == original_bytes
