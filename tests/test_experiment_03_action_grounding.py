from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
import pytest

from computer_agent.grounding import ActionGroundingStatus, GroundingStatus
from computer_agent.perception import (
    BoundingBox,
    PerceptionSnapshot,
    ScreenFrame,
    UIElement,
)
from experiments.phase04_ui_grounding_task_reasoning import (
    experiment_03_action_grounding as experiment,
)


def _captured_at():
    return datetime(2026, 8, 29, 12, 0, 0, tzinfo=timezone.utc)


def _element(
    text,
    *,
    identifier=None,
    element_type="button",
    enabled=True,
    source="accessibility",
    x=10,
    y=10,
) -> UIElement:
    return UIElement(
        element_type=element_type,
        bounding_box=BoundingBox(x=x, y=y, width=20, height=10),
        confidence=0.95,
        text=text,
        identifier=identifier,
        enabled=enabled,
        source=source,
    )


def _snapshot(
    image_path,
    *,
    accessibility_elements=(),
    ocr_elements=(),
    fused_elements=(),
) -> PerceptionSnapshot:
    return PerceptionSnapshot(
        frame=ScreenFrame(
            image_path=Path(image_path),
            pixel_width=100,
            pixel_height=80,
            screen_width=100,
            screen_height=80,
            captured_at=_captured_at(),
        ),
        image=Image.new("RGB", (100, 80)),
        accessibility_elements=tuple(accessibility_elements),
        ocr_elements=tuple(ocr_elements),
        fused_elements=tuple(fused_elements),
        warnings=(),
    )


def _valid_snapshot(candidate_path) -> PerceptionSnapshot:
    marker = _element(
        experiment.FIXTURE_MARKER,
        element_type="text",
    )
    resolved = _element(
        experiment.IDENTIFIER_TEXT,
        identifier=experiment.IDENTIFIER_TARGET_ID,
    )
    disabled = _element(
        experiment.DISABLED_ONLY_TEXT,
        enabled=False,
        x=40,
    )

    return _snapshot(
        candidate_path,
        accessibility_elements=(marker,),
        fused_elements=(resolved, disabled),
    )


def _write_evidence_files(tmp_path):
    candidate_path = tmp_path / "candidate.png"
    formal_path = tmp_path / "formal.png"
    candidate_path.write_bytes(b"candidate evidence")
    formal_path.write_bytes(b"existing formal evidence")
    return candidate_path, formal_path


def test_fixture_identity_accepts_marker_from_accessibility_evidence(tmp_path):
    snapshot = _snapshot(
        tmp_path / "candidate.png",
        accessibility_elements=(
            _element(
                "prefix PHASE04_UI_GROUNDING_FIXTURE_02 suffix",
                element_type="text",
            ),
        ),
    )

    assert experiment._fixture_identity_observed(snapshot)


def test_fixture_identity_accepts_marker_from_ocr_evidence(tmp_path):
    snapshot = _snapshot(
        tmp_path / "candidate.png",
        ocr_elements=(
            _element(
                "phase04 ui grounding fixture 02",
                element_type="text",
                source="ocr",
            ),
        ),
    )

    assert experiment._fixture_identity_observed(snapshot)


def test_fused_only_marker_evidence_is_rejected(tmp_path):
    snapshot = _snapshot(
        tmp_path / "candidate.png",
        fused_elements=(
            _element(
                experiment.FIXTURE_MARKER,
                element_type="text",
                source="hybrid",
            ),
        ),
    )

    assert not experiment._fixture_identity_observed(snapshot)


def test_valid_snapshot_runs_cases_and_promotes_candidate_evidence(tmp_path, capsys):
    candidate_path, formal_path = _write_evidence_files(tmp_path)

    exit_code = experiment._evaluate_observed_snapshot(
        _valid_snapshot(candidate_path),
        candidate_evidence_path=candidate_path,
        formal_evidence_path=formal_path,
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert formal_path.read_bytes() == b"candidate evidence"
    assert not candidate_path.exists()
    assert "Live acceptance result: passed" in output
    assert "Execution skipped: no generated Action was executed." in output


def test_missing_fixture_identity_preserves_evidence(
    tmp_path,
    capsys,
    monkeypatch,
):
    candidate_path, formal_path = _write_evidence_files(tmp_path)

    def fail_if_called(_snapshot):
        pytest.fail("grounding cases should have been skipped")

    monkeypatch.setattr(experiment, "_run_action_cases", fail_if_called)
    exit_code = experiment._evaluate_observed_snapshot(
        _snapshot(
            candidate_path,
            fused_elements=(
                _element(
                    experiment.FIXTURE_MARKER,
                    element_type="text",
                    source="hybrid",
                ),
            ),
        ),
        candidate_evidence_path=candidate_path,
        formal_evidence_path=formal_path,
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert candidate_path.read_bytes() == b"candidate evidence"
    assert formal_path.read_bytes() == b"existing formal evidence"
    assert "Grounding and action-grounding cases were not run." in output


def test_acceptance_failure_preserves_evidence(tmp_path, capsys):
    candidate_path, formal_path = _write_evidence_files(tmp_path)

    exit_code = experiment._evaluate_observed_snapshot(
        _valid_snapshot(candidate_path),
        candidate_evidence_path=candidate_path,
        formal_evidence_path=formal_path,
        acceptance_checker=lambda *_args, **_kwargs: ["forced failure"],
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert candidate_path.read_bytes() == b"candidate evidence"
    assert formal_path.read_bytes() == b"existing formal evidence"
    assert "forced failure" in output


def test_evidence_promotion_failure_preserves_evidence(
    tmp_path,
    capsys,
    monkeypatch,
):
    candidate_path, formal_path = _write_evidence_files(tmp_path)

    def raise_os_error(_source, _target):
        raise OSError("forced replace failure")

    monkeypatch.setattr(experiment.os, "replace", raise_os_error)
    exit_code = experiment._evaluate_observed_snapshot(
        _valid_snapshot(candidate_path),
        candidate_evidence_path=candidate_path,
        formal_evidence_path=formal_path,
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert candidate_path.read_bytes() == b"candidate evidence"
    assert formal_path.read_bytes() == b"existing formal evidence"
    assert "Evidence promotion failed: OSError: forced replace failure" in output
    assert "Candidate screenshot retained for debugging" in output


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("forced observation failure"),
        OSError("forced observation failure"),
    ],
)
def test_main_observation_failures_preserve_evidence(
    tmp_path,
    capsys,
    monkeypatch,
    error,
):
    candidate_path = tmp_path / "candidate.png"
    formal_path = tmp_path / "formal.png"
    fixture_path = tmp_path / "fixture.html"
    formal_path.write_bytes(b"existing formal evidence")
    fixture_path.write_text("fixture", encoding="utf-8")

    class FailingEngine:
        def observe(self):
            raise error

    def fail_if_called(*_args, **_kwargs):
        pytest.fail("grounding cases and evidence promotion should be skipped")

    monkeypatch.setattr(experiment, "_parse_args", lambda: argparse_args(0))
    monkeypatch.setattr(experiment.sys, "platform", "darwin")
    monkeypatch.setattr(experiment, "FIXTURE_PATH", fixture_path)
    monkeypatch.setattr(experiment, "CANDIDATE_EVIDENCE_PATH", candidate_path)
    monkeypatch.setattr(experiment, "EVIDENCE_PATH", formal_path)
    monkeypatch.setattr(experiment, "_build_engine", lambda: FailingEngine())
    monkeypatch.setattr(experiment, "_run_action_cases", fail_if_called)
    monkeypatch.setattr(experiment, "_promote_candidate_evidence", fail_if_called)

    exit_code = experiment.main()

    output = capsys.readouterr().out
    assert exit_code == 1
    assert f"Observation failed: {type(error).__name__}: {error}" in output
    assert not candidate_path.exists()
    assert formal_path.read_bytes() == b"existing formal evidence"
    assert "Candidate screenshot retained" not in output


def argparse_args(wait_seconds):
    return type("Args", (), {"wait_seconds": wait_seconds})()


def test_successful_results_include_ready_action_and_blocked_action(capsys, tmp_path):
    candidate_path = tmp_path / "candidate.png"
    results = experiment._run_action_cases(_valid_snapshot(candidate_path))

    capsys.readouterr()
    resolved_case, resolved_grounding, resolved_action = results[0]
    blocked_case, blocked_grounding, blocked_action = results[1]

    assert resolved_case.name == experiment.IDENTIFIER_TEXT
    assert resolved_grounding.status is GroundingStatus.RESOLVED
    assert resolved_action.status is ActionGroundingStatus.READY
    assert resolved_action.action is not None
    assert resolved_action.action.tool_name == "click_mouse"

    assert blocked_case.name == experiment.DISABLED_ONLY_TEXT
    assert blocked_grounding.status is GroundingStatus.UNSAFE
    assert blocked_action.status is ActionGroundingStatus.BLOCKED
    assert blocked_action.action is None
