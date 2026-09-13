"""Visible evidence-grounded task-state panel."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QLabel,
    QPlainTextEdit,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from computer_agent.task import TaskStateSnapshot


class TaskStatePanel(QWidget):
    """Render semantic task state without exposing hidden reasoning."""

    def __init__(
        self,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        summary_group = QGroupBox(
            "Task Summary"
        )
        summary_layout = QFormLayout(
            summary_group
        )
        summary_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        self._goal_value = QLabel("None")
        self._goal_value.setObjectName(
            "taskStateGoal"
        )
        self._goal_value.setWordWrap(True)
        self._goal_value.setMinimumHeight(0)
        self._goal_value.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        summary_layout.addRow(
            "Goal:",
            self._goal_value,
        )

        self._status_value = QLabel("None")
        self._status_value.setObjectName(
            "taskStateStatus"
        )
        summary_layout.addRow(
            "Status:",
            self._status_value,
        )

        self._task_id_value = QLabel("None")
        self._task_id_value.setObjectName(
            "taskStateId"
        )
        self._task_id_value.setTextInteractionFlags(
            self._task_id_value.textInteractionFlags()
            | Qt.TextInteractionFlag.TextSelectableByMouse
        )
        summary_layout.addRow(
            "Task ID:",
            self._task_id_value,
        )

        layout.addWidget(
            summary_group
        )

        progress_group = QGroupBox("Task Progress")
        progress_layout = QVBoxLayout(progress_group)

        self._progress_view = QPlainTextEdit()
        self._progress_view.setObjectName("progressView")
        self._progress_view.setReadOnly(True)
        self._progress_view.setFixedHeight(95)
        progress_layout.addWidget(self._progress_view)

        layout.addWidget(progress_group)

        evidence_group = QGroupBox("Evidence")
        evidence_layout = QVBoxLayout(evidence_group)

        self._evidence_view = QPlainTextEdit()
        self._evidence_view.setObjectName("evidenceView")
        self._evidence_view.setReadOnly(True)
        self._evidence_view.setFixedHeight(175)
        evidence_layout.addWidget(self._evidence_view)

        layout.addWidget(evidence_group)

        artifacts_group = QGroupBox(
            "Resources / Artifacts"
        )
        artifacts_layout = QVBoxLayout(artifacts_group)

        self._artifacts_view = QPlainTextEdit()
        self._artifacts_view.setObjectName(
            "artifactsView"
        )
        self._artifacts_view.setReadOnly(True)
        self._artifacts_view.setFixedHeight(90)
        artifacts_layout.addWidget(
            self._artifacts_view
        )

        layout.addWidget(artifacts_group)

        effect_group = QGroupBox("Side Effects")
        effect_layout = QVBoxLayout(effect_group)

        self._side_effects_view = QPlainTextEdit()
        self._side_effects_view.setObjectName(
            "sideEffectsView"
        )
        self._side_effects_view.setReadOnly(True)
        self._side_effects_view.setFixedHeight(90)
        effect_layout.addWidget(self._side_effects_view)

        layout.addWidget(effect_group)

        completion_group = QGroupBox("Completion")
        completion_layout = QVBoxLayout(completion_group)

        self._completion_status = QLabel("BLOCKED")
        self._completion_status.setObjectName(
            "completionStatus"
        )
        self._completion_status.setStyleSheet(
            "font-size: 16px; font-weight: 600;"
        )
        completion_layout.addWidget(
            self._completion_status
        )

        self._completion_view = QPlainTextEdit()
        self._completion_view.setObjectName(
            "completionView"
        )
        self._completion_view.setReadOnly(True)
        self._completion_view.setFixedHeight(90)
        completion_layout.addWidget(
            self._completion_view
        )

        layout.addWidget(completion_group)

        self.clear()

    def clear(self) -> None:
        self._goal_value.setText("None")
        self._status_value.setText("None")
        self._task_id_value.setText("None")

        self._progress_view.setPlainText(
            "No semantic task progress yet."
        )
        self._evidence_view.setPlainText(
            "No evidence recorded yet."
        )
        self._artifacts_view.setPlainText(
            "No resources recorded yet."
        )
        self._side_effects_view.setPlainText(
            "No side effects recorded yet."
        )
        self._completion_status.setText("BLOCKED")
        self._completion_view.setPlainText(
            "Task has not started."
        )

    def render(
        self,
        snapshot: TaskStateSnapshot,
    ) -> None:
        if not isinstance(snapshot, TaskStateSnapshot):
            raise ValueError(
                "snapshot must be a TaskStateSnapshot"
            )

        self._goal_value.setText(
            snapshot.goal
        )
        self._status_value.setText(
            snapshot.status.upper()
        )
        self._task_id_value.setText(
            snapshot.task_id
        )

        self._render_progress(snapshot)
        self._render_evidence(snapshot)
        self._render_artifacts(snapshot)
        self._render_side_effects(snapshot)
        self._render_completion(snapshot)

    def _render_progress(
        self,
        snapshot: TaskStateSnapshot,
    ) -> None:
        if not snapshot.subgoals:
            text = "No subgoals recorded yet."
        else:
            lines = []

            for item in snapshot.subgoals:
                marker = (
                    "✓"
                    if item.status == "verified"
                    else "?"
                )
                lines.append(
                    f"{marker} [{item.status.upper()}] "
                    f"{item.description}"
                )

            text = "\n".join(lines)

        self._progress_view.setPlainText(text)

    def _render_evidence(
        self,
        snapshot: TaskStateSnapshot,
    ) -> None:
        if not snapshot.evidence:
            self._evidence_view.setPlainText(
                "No evidence recorded yet."
            )
            return

        blocks = []

        for item in snapshot.evidence:
            observed = (
                item.observed_at
                .astimezone()
                .strftime("%H:%M:%S")
            )

            blocks.append(
                f"[{item.freshness.upper()}] "
                f"{item.summary}\n"
                f"Source: {item.source}\n"
                f"Observed: {observed}"
            )

        self._evidence_view.setPlainText(
            "\n\n".join(blocks)
        )

    def _render_artifacts(
        self,
        snapshot: TaskStateSnapshot,
    ) -> None:
        if not snapshot.artifacts:
            self._artifacts_view.setPlainText(
                "No resources recorded yet."
            )
            return

        self._artifacts_view.setPlainText(
            "\n\n".join(
                f"{item.description}\n"
                f"{item.location}\n"
                f"ID: {item.artifact_id}"
                for item in snapshot.artifacts
            )
        )

    def _render_side_effects(
        self,
        snapshot: TaskStateSnapshot,
    ) -> None:
        if not snapshot.side_effects:
            self._side_effects_view.setPlainText(
                "No side effects recorded yet."
            )
            return

        self._side_effects_view.setPlainText(
            "\n".join(
                _side_effect_line(item)
                for item in snapshot.side_effects
            )
        )

    def _render_completion(
        self,
        snapshot: TaskStateSnapshot,
    ) -> None:
        if snapshot.completion_allowed:
            self._completion_status.setText("ALLOWED")
            self._completion_view.setPlainText(
                "No semantic completion blockers remain."
            )
            return

        self._completion_status.setText("BLOCKED")

        if not snapshot.completion_blockers:
            self._completion_view.setPlainText(
                "Completion is not currently allowed."
            )
            return

        self._completion_view.setPlainText(
            "\n".join(
                f"- {blocker}"
                for blocker
                in snapshot.completion_blockers
            )
        )


def _side_effect_line(
    item,
) -> str:
    parts = [
        f"[{item.state.upper()}] {item.description}",
    ]

    if item.idempotent is not None:
        parts.append(
            f"idempotent={item.idempotent}"
        )

    if item.action_key is not None:
        parts.append(
            f"action_key={item.action_key}"
        )

    return " | ".join(parts)
