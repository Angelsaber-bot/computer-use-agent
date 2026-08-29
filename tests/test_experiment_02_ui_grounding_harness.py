from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from computer_agent.perception import (
    BoundingBox,
    PerceptionSnapshot,
    ScreenFrame,
    UIElement,
)
from experiments.phase04_ui_grounding_task_reasoning import (
    experiment_02_ui_grounding as experiment,
)


def _captured_at():
    return datetime(
        2026,
        8,
        28,
        12,
        0,
        0,
        tzinfo=timezone.utc,
    )


def _element(
    text,
    *,
    source=None,
) -> UIElement:
    return UIElement(
        element_type="text",
        bounding_box=BoundingBox(
            x=1,
            y=2,
            width=3,
            height=4,
        ),
        confidence=0.9,
        text=text,
        source=source,
    )


def _snapshot(
    image_path,
    *,
    accessibility_texts=(),
    ocr_texts=(),
    fused_texts=(),
) -> PerceptionSnapshot:
    size = (20, 10)
    return PerceptionSnapshot(
        frame=ScreenFrame(
            image_path=Path(image_path),
            pixel_width=size[0],
            pixel_height=size[1],
            screen_width=size[0],
            screen_height=size[1],
            captured_at=_captured_at(),
        ),
        image=Image.new(
            "RGB",
            size,
        ),
        accessibility_elements=tuple(
            _element(
                text,
                source="accessibility",
            )
            for text in accessibility_texts
        ),
        ocr_elements=tuple(
            _element(
                text,
                source="ocr",
            )
            for text in ocr_texts
        ),
        fused_elements=tuple(
            _element(
                text,
                source="hybrid",
            )
            for text in fused_texts
        ),
        warnings=(),
    )


def test_fixture_identity_accepts_marker_from_accessibility_evidence(tmp_path):
    snapshot = _snapshot(
        tmp_path / "candidate.png",
        accessibility_texts=(
            "phase04 ui grounding fixture 02",
        ),
    )

    assert experiment._fixture_identity_observed(snapshot)


def test_fixture_identity_accepts_marker_from_ocr_evidence(tmp_path):
    snapshot = _snapshot(
        tmp_path / "candidate.png",
        ocr_texts=(
            "prefix PHASE04_UI_GROUNDING_FIXTURE_02 suffix",
        ),
    )

    assert experiment._fixture_identity_observed(snapshot)


def test_fixture_identity_rejects_unrelated_observation_evidence(tmp_path):
    snapshot = _snapshot(
        tmp_path / "candidate.png",
        accessibility_texts=(
            "IDENTIFIER_TARGET_02",
        ),
        ocr_texts=(
            "Some unrelated focused window",
        ),
        fused_texts=(
            experiment.FIXTURE_MARKER,
        ),
    )

    assert not experiment._fixture_identity_observed(snapshot)


def test_fixture_failure_stops_before_grounding_cases(tmp_path, capsys):
    candidate_path = tmp_path / "candidate.png"
    formal_path = tmp_path / "formal.png"
    snapshot = _snapshot(
        candidate_path,
        ocr_texts=(
            "Wrong window",
        ),
    )
    grounding_called = False

    def grounding_runner(observed_snapshot):
        nonlocal grounding_called
        grounding_called = True
        return ()

    exit_code = experiment._evaluate_observed_snapshot(
        snapshot,
        candidate_evidence_path=candidate_path,
        formal_evidence_path=formal_path,
        grounding_runner=grounding_runner,
        acceptance_checker=lambda *_args, **_kwargs: [],
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert not grounding_called
    assert (
        "Environment error: expected Phase 04 Experiment 02 fixture "
        "was not observed."
    ) in output
    assert "wrong window may be focused" in output
    assert "Grounding cases were not run." in output
    assert "Case:" not in output
    assert "not_found" not in output


def test_acceptance_failure_does_not_overwrite_formal_evidence(
    tmp_path,
    capsys,
):
    candidate_path = tmp_path / "candidate.png"
    formal_path = tmp_path / "formal.png"
    candidate_path.write_bytes(b"candidate evidence")
    formal_path.write_bytes(b"existing formal evidence")
    snapshot = _snapshot(
        candidate_path,
        ocr_texts=(
            experiment.FIXTURE_MARKER,
        ),
    )

    def acceptance_checker(
        observed_snapshot,
        results,
        *,
        candidate_evidence_path,
    ):
        assert observed_snapshot is snapshot
        assert results == ("synthetic result",)
        assert Path(candidate_evidence_path) == candidate_path
        return ["forced synthetic acceptance failure"]

    exit_code = experiment._evaluate_observed_snapshot(
        snapshot,
        candidate_evidence_path=candidate_path,
        formal_evidence_path=formal_path,
        grounding_runner=lambda observed_snapshot: ("synthetic result",),
        acceptance_checker=acceptance_checker,
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert formal_path.read_bytes() == b"existing formal evidence"
    assert candidate_path.read_bytes() == b"candidate evidence"
    assert "forced synthetic acceptance failure" in output
    assert "Candidate screenshot retained for debugging" in output


def test_successful_acceptance_promotes_candidate_evidence(
    tmp_path,
    capsys,
):
    candidate_path = tmp_path / "candidate.png"
    formal_path = tmp_path / "formal.png"
    candidate_path.write_bytes(b"candidate evidence")
    formal_path.write_bytes(b"existing formal evidence")
    snapshot = _snapshot(
        candidate_path,
        ocr_texts=(
            experiment.FIXTURE_MARKER,
        ),
    )

    exit_code = experiment._evaluate_observed_snapshot(
        snapshot,
        candidate_evidence_path=candidate_path,
        formal_evidence_path=formal_path,
        grounding_runner=lambda observed_snapshot: ("synthetic result",),
        acceptance_checker=lambda *_args, **_kwargs: [],
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert formal_path.read_bytes() == b"candidate evidence"
    assert not candidate_path.exists()
    assert "Formal evidence updated from candidate" in output
    assert "Live acceptance result: passed" in output
