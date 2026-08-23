"""Deterministic image preprocessing for screen perception."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

from PIL import Image, ImageOps


class ImagePreprocessor:
    """Prepare PIL images for later perception steps."""

    @staticmethod
    def convert_to_grayscale(image: Image.Image) -> Image.Image:
        """Return a grayscale copy of an image."""

        ImagePreprocessor._validate_image(image)

        return image.convert("L")

    @staticmethod
    def resize(
        image: Image.Image,
        scale_factor: float,
    ) -> Image.Image:
        """Return a resized copy of an image."""

        ImagePreprocessor._validate_image(image)
        ImagePreprocessor._validate_scale_factor(scale_factor)

        width = max(
            1,
            int(image.width * scale_factor),
        )
        height = max(
            1,
            int(image.height * scale_factor),
        )

        return image.resize(
            (width, height),
            Image.Resampling.LANCZOS,
        )

    @staticmethod
    def enhance_contrast(image: Image.Image) -> Image.Image:
        """Return an automatic-contrast copy of an image."""

        ImagePreprocessor._validate_image(image)

        return ImageOps.autocontrast(image)

    @staticmethod
    def prepare_for_ocr(
        image: Image.Image,
        scale_factor: float = 1.0,
    ) -> Image.Image:
        """Return a grayscale, optionally resized, contrast-enhanced image."""

        ImagePreprocessor._validate_image(image)
        ImagePreprocessor._validate_scale_factor(scale_factor)

        prepared = ImagePreprocessor.convert_to_grayscale(image)

        if scale_factor != 1.0:
            prepared = ImagePreprocessor.resize(
                prepared,
                scale_factor,
            )

        return ImagePreprocessor.enhance_contrast(prepared)

    @staticmethod
    def _validate_image(image: Any) -> None:
        if not isinstance(image, Image.Image):
            raise ValueError(
                "image must be a PIL Image"
            )

    @staticmethod
    def _validate_scale_factor(scale_factor: Any) -> None:
        if (
            isinstance(scale_factor, bool)
            or not isinstance(scale_factor, Real)
        ):
            raise ValueError(
                "scale_factor must be numeric"
            )

        if not math.isfinite(scale_factor):
            raise ValueError(
                "scale_factor must be finite"
            )

        if scale_factor <= 0:
            raise ValueError(
                "scale_factor must be positive"
            )
