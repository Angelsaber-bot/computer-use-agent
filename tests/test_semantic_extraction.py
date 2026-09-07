from computer_agent.perception import (
    NewsExtractionIssueKind,
    NewsExtractionStatus,
    SectionExtractionStatus,
    SemanticAXElement,
    SemanticSection,
    extract_news_records,
    extract_semantic_section,
)


def _element(
    role,
    text,
    *,
    bounds=None,
):
    return SemanticAXElement(
        role=role,
        text=text,
        bounds=bounds,
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


def _section(
    *elements,
):
    return SemanticSection(
        heading=_heading("Latest News"),
        elements=tuple(elements),
    )


def _python_news_elements():
    records = (
        (
            "2026-",
            "09-01",
            "The 2026 PSF Board Election is Open!",
        ),
        (
            "2026-",
            "09-01",
            (
                "Inaugural Python Packaging Council Election: "
                "Voting is now open!"
            ),
        ),
        (
            "2026-",
            "08-29",
            "Python 3.15.0 candidate 2 is here!",
        ),
        (
            "2026-",
            "08-28",
            "Kojo Idrissa: 2026 PSF Board Election Candidate Interview",
        ),
        (
            "2026-",
            "08-28",
            "Ramya Ravi: 2026 PSF Board Election Candidate Interview",
        ),
    )

    elements = []
    for year_fragment, month_day, title in records:
        elements.extend(
            (
                _text(year_fragment),
                _text(month_day),
                _link(title),
                _text(title),
            )
        )
    elements.append(
        _link(">>>More")
    )
    return tuple(elements)


def test_unique_heading_returns_correct_section_slice():
    elements = (
        _heading("Main"),
        _link("Docs"),
        _heading("Latest News"),
        _text("2026-"),
        _text("09-01"),
        _link("News title"),
        _heading("Upcoming Events"),
        _link("Outside"),
    )

    result = extract_semantic_section(
        elements,
        heading_text="latest news",
    )

    assert result.status is SectionExtractionStatus.EXTRACTED
    assert result.section is not None
    assert result.section.elements == elements[3:6]
    assert result.following_heading == elements[6]


def test_section_stops_before_next_heading():
    elements = (
        _heading("Latest News"),
        _link("Inside"),
        _heading("Upcoming Events"),
        _link("Outside"),
    )

    result = extract_semantic_section(
        elements,
        heading_text="Latest News",
    )

    assert result.status is SectionExtractionStatus.EXTRACTED
    assert result.section is not None
    assert result.section.elements == (_link("Inside"),)


def test_bounds_none_is_accepted_for_section_and_records():
    elements = (
        _heading("Latest News"),
        _text("2026-"),
        _text("09-01"),
        _link("News title"),
        _heading("Upcoming Events"),
    )

    section_result = extract_semantic_section(
        elements,
        heading_text="Latest News",
    )
    news_result = extract_news_records(section_result.section)

    assert section_result.status is SectionExtractionStatus.EXTRACTED
    assert news_result.status is NewsExtractionStatus.EXTRACTED
    assert [
        (record.date, record.title)
        for record in news_result.records
    ] == [("2026-09-01", "News title")]


def test_extraction_does_not_assume_fixed_element_indices():
    elements = (
        _link("Skip first link"),
        _text("Intro"),
        _heading("Not News"),
        _text("Still outside"),
        _heading("Latest News"),
        _text("2026-09-01"),
        _link("News title"),
        _heading("Upcoming Events"),
    )

    section_result = extract_semantic_section(
        elements,
        heading_text="Latest News",
    )
    news_result = extract_news_records(section_result.section)

    assert section_result.status is SectionExtractionStatus.EXTRACTED
    assert news_result.records[0].title == "News title"


def test_missing_heading_is_explicit():
    result = extract_semantic_section(
        (
            _heading("Upcoming Events"),
        ),
        heading_text="Latest News",
    )

    assert result.status is SectionExtractionStatus.MISSING_HEADING
    assert result.section is None
    assert result.matching_headings == ()


def test_duplicate_heading_is_explicit():
    first = _heading("Latest News")
    second = _heading("latest news")

    result = extract_semantic_section(
        (
            first,
            _link("One"),
            second,
        ),
        heading_text="Latest News",
    )

    assert result.status is SectionExtractionStatus.DUPLICATE_HEADING
    assert result.section is None
    assert result.matching_headings == (
        first,
        second,
    )


def test_one_valid_date_title_record():
    result = extract_news_records(
        _section(
            _text("2026-09-01"),
            _link("One title"),
        )
    )

    assert result.status is NewsExtractionStatus.EXTRACTED
    assert [
        (record.date, record.title)
        for record in result.records
    ] == [("2026-09-01", "One title")]
    assert result.issues == ()


def test_empty_section_returns_empty_section_status():
    result = extract_news_records(
        _section()
    )

    assert result.status is NewsExtractionStatus.EMPTY_SECTION
    assert result.records == ()
    assert result.issues == ()


def test_section_with_content_but_no_records_returns_no_records():
    result = extract_news_records(
        _section(
            _text("Some prose"),
            _text("More prose"),
        )
    )

    assert result.status is NewsExtractionStatus.NO_RECORDS
    assert result.records == ()
    assert result.issues == ()


def test_split_date_fragments_combine_correctly():
    result = extract_news_records(
        _section(
            _text("2026-"),
            _text("09-01"),
            _link("Split date title"),
        )
    )

    assert result.status is NewsExtractionStatus.EXTRACTED
    assert result.records[0].date == "2026-09-01"


def test_incomplete_year_fragment_records_malformed_issue():
    result = extract_news_records(
        _section(
            _text("2026-"),
        )
    )

    assert result.status is NewsExtractionStatus.MALFORMED_RECORDS
    assert result.records == ()
    assert len(result.issues) == 1
    assert (
        result.issues[0].kind
        is NewsExtractionIssueKind.MALFORMED_INCOMPLETE_DATE
    )


def test_month_day_fragment_without_year_records_malformed_issue():
    result = extract_news_records(
        _section(
            _text("09-01"),
        )
    )

    assert result.status is NewsExtractionStatus.MALFORMED_RECORDS
    assert result.records == ()
    assert len(result.issues) == 1
    assert (
        result.issues[0].kind
        is NewsExtractionIssueKind.MALFORMED_INCOMPLETE_DATE
    )


def test_five_valid_records_preserve_source_order():
    result = extract_news_records(
        _section(
            *_python_news_elements(),
        )
    )

    assert result.status is NewsExtractionStatus.EXTRACTED
    assert [
        record.title
        for record in result.records
    ] == [
        "The 2026 PSF Board Election is Open!",
        (
            "Inaugural Python Packaging Council Election: "
            "Voting is now open!"
        ),
        "Python 3.15.0 candidate 2 is here!",
        "Kojo Idrissa: 2026 PSF Board Election Candidate Interview",
        "Ramya Ravi: 2026 PSF Board Election Candidate Interview",
    ]


def test_static_text_duplicate_title_does_not_duplicate_record():
    result = extract_news_records(
        _section(
            _text("2026-"),
            _text("09-01"),
            _link("Repeated title"),
            _text("Repeated title"),
        )
    )

    assert result.status is NewsExtractionStatus.EXTRACTED
    assert len(result.records) == 1
    assert result.records[0].title == "Repeated title"


def test_more_link_is_ignored():
    result = extract_news_records(
        _section(
            _text("2026-09-01"),
            _link("One title"),
            _link(">>>More"),
        )
    )

    assert result.status is NewsExtractionStatus.EXTRACTED
    assert len(result.records) == 1
    assert result.issues == ()


def test_malformed_date_records_issue_without_inventing_record():
    result = extract_news_records(
        _section(
            _text("2026-13-40"),
            _link("Title after bad date"),
        )
    )

    assert result.status is NewsExtractionStatus.MALFORMED_RECORDS
    assert result.records == ()
    assert [
        issue.kind
        for issue in result.issues
    ] == [
        NewsExtractionIssueKind.MALFORMED_INCOMPLETE_DATE,
        NewsExtractionIssueKind.TITLE_WITHOUT_COMPLETE_DATE,
    ]


def test_date_without_following_title_records_issue():
    result = extract_news_records(
        _section(
            _text("2026-09-01"),
            _text("duplicate-free prose"),
        )
    )

    assert result.status is NewsExtractionStatus.MALFORMED_RECORDS
    assert result.records == ()
    assert len(result.issues) == 1
    assert (
        result.issues[0].kind
        is NewsExtractionIssueKind.DATE_WITHOUT_FOLLOWING_TITLE
    )


def test_title_without_complete_date_records_issue():
    result = extract_news_records(
        _section(
            _link("Orphan title"),
        )
    )

    assert result.status is NewsExtractionStatus.MALFORMED_RECORDS
    assert result.records == ()
    assert len(result.issues) == 1
    assert (
        result.issues[0].kind
        is NewsExtractionIssueKind.TITLE_WITHOUT_COMPLETE_DATE
    )


def test_unrelated_elements_outside_the_section_are_ignored():
    elements = (
        _text("2026-13-40"),
        _link("Outside malformed title"),
        _heading("Latest News"),
        _text("2026-09-01"),
        _link("Inside title"),
        _heading("Upcoming Events"),
        _link("Outside after"),
    )

    section_result = extract_semantic_section(
        elements,
        heading_text="Latest News",
    )
    result = extract_news_records(section_result.section)

    assert result.status is NewsExtractionStatus.EXTRACTED
    assert [
        (record.date, record.title)
        for record in result.records
    ] == [("2026-09-01", "Inside title")]
