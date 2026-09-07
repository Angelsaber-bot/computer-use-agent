from computer_agent.perception import SemanticAXElement
from experiments.phase05_real_web_autonomy import (
    experiment_06_web_information_extraction as experiment,
)


TEST_TITLES = (
    "News item one",
    "News item two",
    "News item three",
    "News item four",
    "News item five",
)


class FakeAccessibility:
    def __init__(
        self,
        *,
        application_name="Google Chrome",
        elements=(),
    ):
        self.application_name = application_name
        self.elements = tuple(elements)
        self.calls = []

    def read_frontmost_application_name(self):
        self.calls.append("read_frontmost_application_name")
        return self.application_name

    def read_frontmost_semantic_elements(self):
        self.calls.append("read_frontmost_semantic_elements")
        return list(self.elements)


def _element(
    role,
    text,
):
    return SemanticAXElement(
        role=role,
        text=text,
        bounds=None,
    )


def _heading(text):
    return _element(
        "AXHeading",
        text,
    )


def _text(text):
    return _element(
        "AXStaticText",
        text,
    )


def _link(text):
    return _element(
        "AXLink",
        text,
    )


def _valid_live_elements():
    date_parts = (
        ("2026-", "09-01"),
        ("2026-", "09-01"),
        ("2026-", "08-29"),
        ("2026-", "08-28"),
        ("2026-", "08-28"),
    )

    elements = [
        _heading("Welcome"),
        _link("Docs"),
        _heading("Latest News"),
    ]
    for (year_fragment, month_day), title in zip(
        date_parts,
        TEST_TITLES,
        strict=True,
    ):
        elements.extend(
            (
                _text(year_fragment),
                _text(month_day),
                _link(title),
                _text(title),
            )
        )
    elements.extend(
        (
            _link(">>>More"),
            _heading("Upcoming Events"),
            _link("Outside link"),
        )
    )
    return tuple(elements)


def test_run_acceptance_passes_with_five_live_records(capsys):
    accessibility = FakeAccessibility(
        elements=_valid_live_elements(),
    )

    result = experiment._run_acceptance(
        accessibility=accessibility,
    )
    output = capsys.readouterr().out

    assert result.status is experiment.WebInformationExtractionStatus.PASSED
    assert result.action_execution_count == 0
    assert result.news_result is not None
    assert len(result.news_result.records) == 5
    assert [
        record.title
        for record in result.news_result.records
    ] == list(TEST_TITLES)
    assert accessibility.calls == [
        "read_frontmost_application_name",
        "read_frontmost_semantic_elements",
    ]
    assert "Action execution count: 0" in output
    assert "News record count: 5" in output


def test_run_acceptance_blocks_when_frontmost_app_is_not_chrome(capsys):
    accessibility = FakeAccessibility(
        application_name="Safari",
        elements=_valid_live_elements(),
    )

    result = experiment._run_acceptance(
        accessibility=accessibility,
    )
    output = capsys.readouterr().out

    assert result.status is experiment.WebInformationExtractionStatus.BLOCKED
    assert result.semantic_element_count == 0
    assert result.action_execution_count == 0
    assert accessibility.calls == ["read_frontmost_application_name"]
    assert "Extraction status: blocked" in output


def test_run_acceptance_fails_when_heading_is_missing():
    accessibility = FakeAccessibility(
        elements=(
            _heading("Upcoming Events"),
            _link("Outside"),
        ),
    )

    result = experiment._run_acceptance(
        accessibility=accessibility,
    )

    assert result.status is experiment.WebInformationExtractionStatus.FAILED
    assert result.section_result is not None
    assert (
        result.section_result.status.value
        == "missing_heading"
    )
    assert result.action_execution_count == 0


def test_run_acceptance_fails_when_following_heading_differs():
    elements = list(_valid_live_elements())
    following_index = next(
        index
        for index, element in enumerate(elements)
        if element.role == "AXHeading" and element.text == "Upcoming Events"
    )
    elements[following_index] = _heading("Community")
    accessibility = FakeAccessibility(
        elements=tuple(elements),
    )

    result = experiment._run_acceptance(
        accessibility=accessibility,
    )

    assert result.status is experiment.WebInformationExtractionStatus.FAILED
    assert (
        result.reason
        == "section did not end before the expected following heading"
    )
    assert result.news_result is None
    assert result.action_execution_count == 0


def test_run_acceptance_fails_when_records_are_malformed():
    accessibility = FakeAccessibility(
        elements=(
            _heading("Latest News"),
            _text("2026-13-40"),
            _link("Malformed item title"),
            _heading("Upcoming Events"),
        ),
    )

    result = experiment._run_acceptance(
        accessibility=accessibility,
    )

    assert result.status is experiment.WebInformationExtractionStatus.FAILED
    assert result.news_result is not None
    assert result.news_result.status.value == "malformed_records"
    assert result.action_execution_count == 0


def test_countdown_prints_each_visible_second(capsys):
    sleeps = []

    experiment._countdown(
        3,
        sleeper=sleeps.append,
    )

    assert capsys.readouterr().out.splitlines() == [
        "3...",
        "2...",
        "1...",
    ]
    assert sleeps == [
        1,
        1,
        1,
    ]
