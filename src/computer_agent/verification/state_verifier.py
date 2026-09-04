"""Deterministic verification for observed application state."""

from __future__ import annotations

from typing import Protocol

from computer_agent.perception import MacOSAccessibility
from computer_agent.perception.engine import PerceptionSnapshot
from computer_agent.perception.models import UIElement
from computer_agent.verification.models import (
    StateVerificationResult,
    StateVerificationStatus,
)


class FrontmostApplicationObserver(Protocol):
    """Read the current frontmost application name."""

    def read_frontmost_application_name(self) -> str | None:
        """Return a localized frontmost application name, or None."""


_EDITABLE_ELEMENT_TYPES = frozenset(("text_field", "text_area"))


class StateVerifier:
    """Verify deterministic state conditions from trusted observations."""

    def __init__(
        self,
        *,
        application_observer: FrontmostApplicationObserver | None = None,
    ) -> None:
        if application_observer is None:
            application_observer = MacOSAccessibility()
        elif not callable(
            getattr(
                application_observer,
                "read_frontmost_application_name",
                None,
            )
        ):
            raise ValueError(
                "application_observer must provide "
                "read_frontmost_application_name"
            )

        self._application_observer = application_observer

    @property
    def application_observer(self) -> FrontmostApplicationObserver:
        """Return the configured frontmost-application observer."""

        return self._application_observer

    def verify_frontmost_application(
        self,
        expected_app_name: str,
    ) -> StateVerificationResult:
        """Verify that the exact expected application is frontmost."""

        if (
            not isinstance(expected_app_name, str)
            or not expected_app_name.strip()
        ):
            raise ValueError(
                "expected_app_name must be a non-empty string"
            )

        try:
            observed_app_name = (
                self._application_observer
                .read_frontmost_application_name()
            )
        except Exception as error:
            return StateVerificationResult(
                status=StateVerificationStatus.INCONCLUSIVE,
                reason=(
                    "frontmost application observation failed: "
                    f"{type(error).__name__}: {error}"
                ),
            )

        if observed_app_name is None:
            return StateVerificationResult(
                status=StateVerificationStatus.INCONCLUSIVE,
                reason="frontmost application could not be determined",
            )

        if (
            not isinstance(observed_app_name, str)
            or not observed_app_name.strip()
        ):
            return StateVerificationResult(
                status=StateVerificationStatus.INCONCLUSIVE,
                reason="frontmost application name was not usable",
            )

        if observed_app_name == expected_app_name:
            return StateVerificationResult(
                status=StateVerificationStatus.VERIFIED,
                reason=(
                    "frontmost application matched expected app "
                    f"{expected_app_name}"
                ),
            )

        return StateVerificationResult(
            status=StateVerificationStatus.FAILED,
            reason=(
                "frontmost application was "
                f"{observed_app_name}, expected {expected_app_name}"
            ),
        )

    def verify_focused_editable_value(
        self,
        snapshot: PerceptionSnapshot,
        expected_value: str,
    ) -> StateVerificationResult:
        """Verify exact value in one focused Accessibility editable."""

        if not isinstance(snapshot, PerceptionSnapshot):
            raise ValueError("snapshot must be a PerceptionSnapshot")

        if not isinstance(expected_value, str) or not expected_value.strip():
            raise ValueError("expected_value must be a non-empty string")

        focused_editables = tuple(
            element
            for element in snapshot.accessibility_elements
            if _is_focused_editable(element)
        )

        if not focused_editables:
            return StateVerificationResult(
                status=StateVerificationStatus.INCONCLUSIVE,
                reason="no focused editable accessibility element found",
            )

        if len(focused_editables) > 1:
            return StateVerificationResult(
                status=StateVerificationStatus.INCONCLUSIVE,
                reason="multiple focused editable accessibility elements found",
            )

        value = focused_editables[0].value
        if not isinstance(value, str):
            return StateVerificationResult(
                status=StateVerificationStatus.INCONCLUSIVE,
                reason="focused editable value was not a string",
            )

        if value == expected_value:
            return StateVerificationResult(
                status=StateVerificationStatus.VERIFIED,
                reason="focused editable value matched expected value",
            )

        return StateVerificationResult(
            status=StateVerificationStatus.FAILED,
            reason="focused editable value did not match expected value",
        )


def _is_focused_editable(element: UIElement) -> bool:
    return (
        element.element_type in _EDITABLE_ELEMENT_TYPES
        and element.focused is True
    )
