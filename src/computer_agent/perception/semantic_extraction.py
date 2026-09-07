"""Pure semantic extraction helpers for Accessibility element streams."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from enum import Enum
import re

from computer_agent.perception.fusion import normalize_ui_text
from computer_agent.perception.viewport import SemanticAXElement


HEADING_ROLE = "AXHeading"
LINK_ROLE = "AXLink"
STATIC_TEXT_ROLE = "AXStaticText"

_FULL_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_YEAR_FRAGMENT_RE = re.compile(r"^\d{4}-$")
_MONTH_DAY_FRAGMENT_RE = re.compile(r"^\d{2}-\d{2}$")
_DATE_LIKE_RE = re.compile(r"^(?:\d{4}[-/]\d{0,2}|\d{2}[-/]\d{2})")


class SectionExtractionStatus(str, Enum):
    """Deterministic outcomes for semantic section extraction."""

    EXTRACTED = "extracted"
    MISSING_HEADING = "missing_heading"
    DUPLICATE_HEADING = "duplicate_heading"
    EMPTY_SECTION = "empty_section"


class NewsExtractionStatus(str, Enum):
    """Deterministic outcomes for news-record extraction."""

    EXTRACTED = "extracted"
    EMPTY_SECTION = "empty_section"
    NO_RECORDS = "no_records"
    MALFORMED_RECORDS = "malformed_records"


class NewsExtractionIssueKind(str, Enum):
    """Explicit malformed-input categories for news extraction."""

    MALFORMED_INCOMPLETE_DATE = "malformed_incomplete_date"
    TITLE_WITHOUT_COMPLETE_DATE = "title_without_complete_date"
    DATE_WITHOUT_FOLLOWING_TITLE = "date_without_following_title"


@dataclass(frozen=True, slots=True)
class SemanticSection:
    """A heading-delimited slice of semantic Accessibility elements."""

    heading: SemanticAXElement
    elements: tuple[SemanticAXElement, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.heading, SemanticAXElement):
            raise ValueError("heading must be a SemanticAXElement")

        elements = _validate_semantic_elements(
            self.elements,
            name="elements",
        )
        object.__setattr__(
            self,
            "elements",
            elements,
        )


@dataclass(frozen=True, slots=True)
class SectionExtractionResult:
    """Result of locating one semantic section."""

    status: SectionExtractionStatus
    section: SemanticSection | None
    matching_headings: tuple[SemanticAXElement, ...]
    following_heading: SemanticAXElement | None
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, SectionExtractionStatus):
            raise ValueError(
                "status must be a SectionExtractionStatus"
            )

        if self.section is not None and not isinstance(
            self.section,
            SemanticSection,
        ):
            raise ValueError(
                "section must be a SemanticSection or None"
            )

        matching_headings = _validate_semantic_elements(
            self.matching_headings,
            name="matching_headings",
        )
        object.__setattr__(
            self,
            "matching_headings",
            matching_headings,
        )

        if self.following_heading is not None and not isinstance(
            self.following_heading,
            SemanticAXElement,
        ):
            raise ValueError(
                "following_heading must be a SemanticAXElement or None"
            )

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")


@dataclass(frozen=True, slots=True)
class NewsRecord:
    """A deterministic news record extracted from semantic structure."""

    date: str
    title: str

    def __post_init__(self) -> None:
        _validate_non_empty_string(
            "date",
            self.date,
        )
        _validate_non_empty_string(
            "title",
            self.title,
        )


@dataclass(frozen=True, slots=True)
class NewsExtractionIssue:
    """One explicit issue encountered while parsing news records."""

    kind: NewsExtractionIssueKind
    element_index: int
    text: str | None
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, NewsExtractionIssueKind):
            raise ValueError(
                "kind must be a NewsExtractionIssueKind"
            )

        if (
            isinstance(self.element_index, bool)
            or not isinstance(self.element_index, int)
            or self.element_index < 0
        ):
            raise ValueError(
                "element_index must be a non-negative integer"
            )

        if self.text is not None and not isinstance(self.text, str):
            raise ValueError("text must be a string or None")

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")


@dataclass(frozen=True, slots=True)
class NewsExtractionResult:
    """Result of parsing news records from a semantic section."""

    status: NewsExtractionStatus
    records: tuple[NewsRecord, ...]
    issues: tuple[NewsExtractionIssue, ...]
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, NewsExtractionStatus):
            raise ValueError(
                "status must be a NewsExtractionStatus"
            )

        if not isinstance(self.records, tuple) or any(
            not isinstance(record, NewsRecord) for record in self.records
        ):
            raise ValueError("records must be a tuple of NewsRecord objects")

        if not isinstance(self.issues, tuple) or any(
            not isinstance(issue, NewsExtractionIssue)
            for issue in self.issues
        ):
            raise ValueError(
                "issues must be a tuple of NewsExtractionIssue objects"
            )

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")


def extract_semantic_section(
    semantic_elements: Sequence[SemanticAXElement],
    *,
    heading_text: str,
    heading_role: str = HEADING_ROLE,
) -> SectionExtractionResult:
    """Locate exactly one heading-delimited semantic section."""

    elements = _validate_semantic_elements(
        semantic_elements,
        name="semantic_elements",
    )
    _validate_non_empty_string(
        "heading_text",
        heading_text,
    )
    _validate_non_empty_string(
        "heading_role",
        heading_role,
    )

    normalized_heading_text = normalize_ui_text(heading_text)
    matches = tuple(
        element
        for element in elements
        if element.role == heading_role
        and normalize_ui_text(element.text) == normalized_heading_text
    )

    if not matches:
        return SectionExtractionResult(
            status=SectionExtractionStatus.MISSING_HEADING,
            section=None,
            matching_headings=(),
            following_heading=None,
            reason="target heading was not found",
        )

    if len(matches) > 1:
        return SectionExtractionResult(
            status=SectionExtractionStatus.DUPLICATE_HEADING,
            section=None,
            matching_headings=matches,
            following_heading=None,
            reason="target heading matched more than once",
        )

    heading = matches[0]
    start_index = elements.index(heading) + 1
    end_index = len(elements)
    following_heading = None

    for index in range(
        start_index,
        len(elements),
    ):
        if elements[index].role == heading_role:
            end_index = index
            following_heading = elements[index]
            break

    section = SemanticSection(
        heading=heading,
        elements=elements[start_index:end_index],
    )
    if not section.elements:
        return SectionExtractionResult(
            status=SectionExtractionStatus.EMPTY_SECTION,
            section=section,
            matching_headings=matches,
            following_heading=following_heading,
            reason="target heading contains no semantic elements",
        )

    return SectionExtractionResult(
        status=SectionExtractionStatus.EXTRACTED,
        section=section,
        matching_headings=matches,
        following_heading=following_heading,
        reason="target section extracted",
    )


def extract_news_records(
    section: SemanticSection,
) -> NewsExtractionResult:
    """Parse date/title news records from one semantic section."""

    if not isinstance(section, SemanticSection):
        raise ValueError("section must be a SemanticSection")

    if not section.elements:
        return NewsExtractionResult(
            status=NewsExtractionStatus.EMPTY_SECTION,
            records=(),
            issues=(),
            reason="section contains no semantic elements",
        )

    records: list[NewsRecord] = []
    issues: list[NewsExtractionIssue] = []
    pending_date: str | None = None
    pending_date_index: int | None = None
    pending_year_fragment: str | None = None
    pending_year_index: int | None = None
    last_record_title: str | None = None

    for index, element in enumerate(section.elements):
        text = _clean_text(element.text)
        if text is None:
            continue

        if _is_more_link(text):
            continue

        if (
            element.role == STATIC_TEXT_ROLE
            and pending_date is None
            and pending_year_fragment is None
            and last_record_title is not None
            and normalize_ui_text(text) == normalize_ui_text(last_record_title)
        ):
            continue

        if element.role == STATIC_TEXT_ROLE:
            date_kind = _date_text_kind(text)

            if date_kind == "full":
                if _is_valid_date(text):
                    if pending_date is not None:
                        _add_date_without_title_issue(
                            issues,
                            pending_date_index,
                            pending_date,
                        )
                    if pending_year_fragment is not None:
                        _add_malformed_date_issue(
                            issues,
                            pending_year_index,
                            pending_year_fragment,
                            "date fragment was not completed",
                        )
                    pending_date = text
                    pending_date_index = index
                    pending_year_fragment = None
                    pending_year_index = None
                else:
                    _clear_pending_before_malformed_date(
                        issues,
                        pending_date,
                        pending_date_index,
                        pending_year_fragment,
                        pending_year_index,
                    )
                    pending_date = None
                    pending_date_index = None
                    pending_year_fragment = None
                    pending_year_index = None
                    _add_malformed_date_issue(
                        issues,
                        index,
                        text,
                        "date is not a valid calendar date",
                    )
                continue

            if date_kind == "year_fragment":
                if pending_date is not None:
                    _add_date_without_title_issue(
                        issues,
                        pending_date_index,
                        pending_date,
                    )
                if pending_year_fragment is not None:
                    _add_malformed_date_issue(
                        issues,
                        pending_year_index,
                        pending_year_fragment,
                        "date fragment was not completed",
                    )
                pending_date = None
                pending_date_index = None
                pending_year_fragment = text
                pending_year_index = index
                continue

            if date_kind == "month_day_fragment":
                if pending_year_fragment is None:
                    _clear_pending_before_malformed_date(
                        issues,
                        pending_date,
                        pending_date_index,
                        None,
                        None,
                    )
                    pending_date = None
                    pending_date_index = None
                    _add_malformed_date_issue(
                        issues,
                        index,
                        text,
                        "date fragment has no preceding year",
                    )
                    continue

                combined_date = f"{pending_year_fragment}{text}"
                if _is_valid_date(combined_date):
                    pending_date = combined_date
                    pending_date_index = pending_year_index
                    pending_year_fragment = None
                    pending_year_index = None
                else:
                    _add_malformed_date_issue(
                        issues,
                        pending_year_index,
                        combined_date,
                        "combined date is not a valid calendar date",
                    )
                    pending_date = None
                    pending_date_index = None
                    pending_year_fragment = None
                    pending_year_index = None
                continue

            if date_kind == "malformed":
                _clear_pending_before_malformed_date(
                    issues,
                    pending_date,
                    pending_date_index,
                    pending_year_fragment,
                    pending_year_index,
                )
                pending_date = None
                pending_date_index = None
                pending_year_fragment = None
                pending_year_index = None
                _add_malformed_date_issue(
                    issues,
                    index,
                    text,
                    "date text is malformed or incomplete",
                )
                continue

        if element.role == LINK_ROLE:
            if pending_year_fragment is not None:
                _add_malformed_date_issue(
                    issues,
                    pending_year_index,
                    pending_year_fragment,
                    "date fragment was not completed before title",
                )
                pending_year_fragment = None
                pending_year_index = None

            if pending_date is None:
                issues.append(
                    NewsExtractionIssue(
                        kind=(
                            NewsExtractionIssueKind
                            .TITLE_WITHOUT_COMPLETE_DATE
                        ),
                        element_index=index,
                        text=text,
                        reason="title link appeared before a complete date",
                    )
                )
                continue

            records.append(
                NewsRecord(
                    date=pending_date,
                    title=text,
                )
            )
            last_record_title = text
            pending_date = None
            pending_date_index = None

    if pending_date is not None:
        _add_date_without_title_issue(
            issues,
            pending_date_index,
            pending_date,
        )

    if pending_year_fragment is not None:
        _add_malformed_date_issue(
            issues,
            pending_year_index,
            pending_year_fragment,
            "date fragment was not completed",
        )

    if issues:
        return NewsExtractionResult(
            status=NewsExtractionStatus.MALFORMED_RECORDS,
            records=tuple(records),
            issues=tuple(issues),
            reason="one or more news records were malformed",
        )

    if not records:
        return NewsExtractionResult(
            status=NewsExtractionStatus.NO_RECORDS,
            records=(),
            issues=(),
            reason="section contained no complete news records",
        )

    return NewsExtractionResult(
        status=NewsExtractionStatus.EXTRACTED,
        records=tuple(records),
        issues=(),
        reason="news records extracted",
    )


def _validate_semantic_elements(
    value: object,
    *,
    name: str,
) -> tuple[SemanticAXElement, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(
            f"{name} must be a sequence of SemanticAXElement objects"
        )

    elements = tuple(value)
    if any(not isinstance(element, SemanticAXElement) for element in elements):
        raise ValueError(
            f"{name} must be a sequence of SemanticAXElement objects"
        )

    return elements


def _validate_non_empty_string(
    name: str,
    value: object,
) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _clean_text(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None

    cleaned = value.strip()
    if not cleaned:
        return None

    return cleaned


def _is_more_link(text: str) -> bool:
    normalized = normalize_ui_text(text).replace(
        ">",
        "",
    ).strip()
    return normalized == "more"


def _date_text_kind(text: str) -> str | None:
    if _FULL_DATE_RE.match(text):
        return "full"

    if _YEAR_FRAGMENT_RE.match(text):
        return "year_fragment"

    if _MONTH_DAY_FRAGMENT_RE.match(text):
        return "month_day_fragment"

    if _DATE_LIKE_RE.match(text):
        return "malformed"

    return None


def _is_valid_date(value: str) -> bool:
    try:
        year_text, month_text, day_text = value.split("-")
        date(
            int(year_text),
            int(month_text),
            int(day_text),
        )
    except ValueError:
        return False

    return True


def _clear_pending_before_malformed_date(
    issues: list[NewsExtractionIssue],
    pending_date: str | None,
    pending_date_index: int | None,
    pending_year_fragment: str | None,
    pending_year_index: int | None,
) -> None:
    if pending_date is not None:
        _add_date_without_title_issue(
            issues,
            pending_date_index,
            pending_date,
        )

    if pending_year_fragment is not None:
        _add_malformed_date_issue(
            issues,
            pending_year_index,
            pending_year_fragment,
            "date fragment was not completed",
        )


def _add_date_without_title_issue(
    issues: list[NewsExtractionIssue],
    element_index: int | None,
    text: str | None,
) -> None:
    issues.append(
        NewsExtractionIssue(
            kind=NewsExtractionIssueKind.DATE_WITHOUT_FOLLOWING_TITLE,
            element_index=0
            if element_index is None
            else element_index,
            text=text,
            reason="complete date was not followed by a title link",
        )
    )


def _add_malformed_date_issue(
    issues: list[NewsExtractionIssue],
    element_index: int | None,
    text: str | None,
    reason: str,
) -> None:
    issues.append(
        NewsExtractionIssue(
            kind=NewsExtractionIssueKind.MALFORMED_INCOMPLETE_DATE,
            element_index=0
            if element_index is None
            else element_index,
            text=text,
            reason=reason,
        )
    )
