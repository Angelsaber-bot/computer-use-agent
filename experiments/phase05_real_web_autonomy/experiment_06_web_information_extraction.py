"""Phase 05 Experiment 06: web information extraction."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
import sys
import time

from computer_agent.perception import (
    MacOSAccessibility,
    NewsExtractionResult,
    NewsExtractionStatus,
    SectionExtractionResult,
    SectionExtractionStatus,
    extract_news_records,
    extract_semantic_section,
    normalize_ui_text,
)


TARGET_URL = "https://www.python.org/"
EXPECTED_APPLICATION_NAME = "Google Chrome"
TARGET_HEADING_TEXT = "Latest News"
EXPECTED_FOLLOWING_HEADING_TEXT = "Upcoming Events"
EXPECTED_RECORD_COUNT = 5
DEFAULT_WAIT_SECONDS = 8


class WebInformationExtractionStatus(str, Enum):
    """Experiment-level acceptance outcomes."""

    PASSED = "passed"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class WebInformationExtractionResult:
    """Read-only live extraction result."""

    status: WebInformationExtractionStatus
    reason: str
    application_name: str | None
    semantic_element_count: int
    section_result: SectionExtractionResult | None
    news_result: NewsExtractionResult | None
    action_execution_count: int


def _wait_seconds(value: str) -> int:
    try:
        seconds = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--wait-seconds must be an integer"
        ) from error

    if not 0 <= seconds <= 30:
        raise argparse.ArgumentTypeError(
            "--wait-seconds must be from 0 through 30"
        )

    return seconds


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read python.org Accessibility semantics and deterministically "
            "extract the Latest News records."
        )
    )
    parser.add_argument(
        "--wait-seconds",
        type=_wait_seconds,
        default=DEFAULT_WAIT_SECONDS,
        help="Seconds to count down before the read-only observation.",
    )
    return parser.parse_args()


def _countdown(
    seconds: int,
    *,
    sleeper=time.sleep,
) -> None:
    for remaining in range(
        seconds,
        0,
        -1,
    ):
        print(f"{remaining}...")
        sleeper(1)


def _run_acceptance(
    *,
    accessibility: MacOSAccessibility,
) -> WebInformationExtractionResult:
    application_name = accessibility.read_frontmost_application_name()

    if application_name != EXPECTED_APPLICATION_NAME:
        result = WebInformationExtractionResult(
            status=WebInformationExtractionStatus.BLOCKED,
            reason=(
                "frontmost application is not "
                f"{EXPECTED_APPLICATION_NAME}"
            ),
            application_name=application_name,
            semantic_element_count=0,
            section_result=None,
            news_result=None,
            action_execution_count=0,
        )
        _print_result(result)
        return result

    semantic_elements = tuple(
        accessibility.read_frontmost_semantic_elements()
    )

    section_result = extract_semantic_section(
        semantic_elements,
        heading_text=TARGET_HEADING_TEXT,
    )
    if section_result.status is not SectionExtractionStatus.EXTRACTED:
        result = WebInformationExtractionResult(
            status=WebInformationExtractionStatus.FAILED,
            reason=section_result.reason,
            application_name=application_name,
            semantic_element_count=len(semantic_elements),
            section_result=section_result,
            news_result=None,
            action_execution_count=0,
        )
        _print_result(result)
        return result

    if not _has_expected_following_heading(section_result):
        result = WebInformationExtractionResult(
            status=WebInformationExtractionStatus.FAILED,
            reason=(
                "section did not end before the expected following heading"
            ),
            application_name=application_name,
            semantic_element_count=len(semantic_elements),
            section_result=section_result,
            news_result=None,
            action_execution_count=0,
        )
        _print_result(result)
        return result

    news_result = extract_news_records(section_result.section)
    if news_result.status is not NewsExtractionStatus.EXTRACTED:
        result = WebInformationExtractionResult(
            status=WebInformationExtractionStatus.FAILED,
            reason=news_result.reason,
            application_name=application_name,
            semantic_element_count=len(semantic_elements),
            section_result=section_result,
            news_result=news_result,
            action_execution_count=0,
        )
        _print_result(result)
        return result

    if len(news_result.records) != EXPECTED_RECORD_COUNT:
        result = WebInformationExtractionResult(
            status=WebInformationExtractionStatus.FAILED,
            reason=(
                "extracted record count did not match expected live count"
            ),
            application_name=application_name,
            semantic_element_count=len(semantic_elements),
            section_result=section_result,
            news_result=news_result,
            action_execution_count=0,
        )
        _print_result(result)
        return result

    result = WebInformationExtractionResult(
        status=WebInformationExtractionStatus.PASSED,
        reason="all live read-only extraction checks passed",
        application_name=application_name,
        semantic_element_count=len(semantic_elements),
        section_result=section_result,
        news_result=news_result,
        action_execution_count=0,
    )
    _print_result(result)
    return result


def _has_expected_following_heading(
    section_result: SectionExtractionResult,
) -> bool:
    following = section_result.following_heading
    return (
        following is not None
        and following.role == "AXHeading"
        and normalize_ui_text(following.text)
        == normalize_ui_text(EXPECTED_FOLLOWING_HEADING_TEXT)
    )


def _print_result(
    result: WebInformationExtractionResult,
) -> None:
    print("Extraction status:", result.status.value)
    print("Extraction reason:", result.reason)
    print("Frontmost application:", result.application_name)
    print("Raw semantic element count:", result.semantic_element_count)
    print("Action execution count:", result.action_execution_count)

    if result.section_result is None:
        print("Section status: None")
    else:
        _print_section_result(result.section_result)

    if result.news_result is None:
        print("News status: None")
    else:
        _print_news_result(result.news_result)


def _print_section_result(
    result: SectionExtractionResult,
) -> None:
    print("Section status:", result.status.value)
    print("Section reason:", result.reason)
    print("Matching headings:", len(result.matching_headings))

    if result.section is not None:
        print("Section heading:", repr(result.section.heading.text))
        print("Section element count:", len(result.section.elements))

    if result.following_heading is None:
        print("Following heading: None")
    else:
        print("Following heading:", repr(result.following_heading.text))


def _print_news_result(
    result: NewsExtractionResult,
) -> None:
    print("News status:", result.status.value)
    print("News reason:", result.reason)
    print("News record count:", len(result.records))

    for index, record in enumerate(
        result.records,
        start=1,
    ):
        print(f"{index}. date={record.date!r} title={record.title!r}")

    print("News issue count:", len(result.issues))
    for issue in result.issues:
        print(
            "Issue: "
            f"kind={issue.kind.value} "
            f"index={issue.element_index} "
            f"text={issue.text!r} "
            f"reason={issue.reason}"
        )


def _exit_code(status: WebInformationExtractionStatus) -> int:
    if status is WebInformationExtractionStatus.PASSED:
        return 0

    return 1


def main() -> int:
    args = _parse_args()

    print("Phase 05 Experiment 06: Web Information Extraction")
    print(f"Target URL: {TARGET_URL}")
    print(f"Browser: {EXPECTED_APPLICATION_NAME}")
    print(f"Target heading: {TARGET_HEADING_TEXT!r}")
    print("Read-only mode: no computer actions will be executed.")

    if sys.platform != "darwin":
        print(
            "Extraction status:",
            WebInformationExtractionStatus.BLOCKED.value,
        )
        print("Extraction reason: this experiment requires macOS.")
        return 1

    accessibility = MacOSAccessibility()

    if not accessibility.is_available():
        print(
            "Extraction status:",
            WebInformationExtractionStatus.BLOCKED.value,
        )
        print(
            "Extraction reason: "
            "macOS Accessibility is unavailable."
        )
        return 1

    if not accessibility.is_trusted():
        print(
            "Extraction status:",
            WebInformationExtractionStatus.BLOCKED.value,
        )
        print(
            "Extraction reason: "
            "Accessibility permission is not trusted."
        )
        return 1

    if args.wait_seconds > 0:
        print(
            "Switch to Chrome with python.org open. "
            "Read-only observation starts after countdown:"
        )
        _countdown(args.wait_seconds)

    result = _run_acceptance(
        accessibility=accessibility,
    )
    return _exit_code(result.status)


if __name__ == "__main__":
    raise SystemExit(main())
