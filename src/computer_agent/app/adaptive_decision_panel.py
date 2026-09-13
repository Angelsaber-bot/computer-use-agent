"""Visible adaptive next-step decision panel."""

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

from computer_agent.reasoning import (
    AdaptiveDecisionSnapshot,
)


class AdaptiveDecisionPanel(QWidget):
    """Render adaptive decision state without hidden reasoning."""

    def __init__(
        self,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("adaptiveDecisionPanel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        observation_group = QGroupBox(
            "Current Observation"
        )
        observation_layout = QFormLayout(
            observation_group
        )

        self._source_value = QLabel("None")
        self._source_value.setObjectName(
            "decisionSource"
        )
        observation_layout.addRow(
            "Source:",
            self._source_value,
        )

        self._application_value = QLabel("None")
        self._application_value.setObjectName(
            "observationApplication"
        )
        observation_layout.addRow(
            "Application:",
            self._application_value,
        )

        self._window_value = QLabel("None")
        self._window_value.setObjectName(
            "observationWindow"
        )
        observation_layout.addRow(
            "Window:",
            self._window_value,
        )

        self._visible_text_view = QPlainTextEdit()
        self._visible_text_view.setObjectName(
            "observationText"
        )
        self._visible_text_view.setReadOnly(True)
        self._visible_text_view.setFixedHeight(80)
        observation_layout.addRow(
            "Visible:",
            self._visible_text_view,
        )

        layout.addWidget(observation_group)

        attempts_group = QGroupBox(
            "Decision Trace"
        )
        attempts_layout = QVBoxLayout(
            attempts_group
        )

        self._attempts_view = QPlainTextEdit()
        self._attempts_view.setObjectName(
            "decisionAttempts"
        )
        self._attempts_view.setReadOnly(True)
        self._attempts_view.setFixedHeight(110)
        attempts_layout.addWidget(
            self._attempts_view
        )

        layout.addWidget(attempts_group)

        safety_group = QGroupBox(
            "Safety / Validation"
        )
        safety_layout = QFormLayout(
            safety_group
        )

        self._attempt_count_value = QLabel("0")
        self._attempt_count_value.setObjectName(
            "attemptCount"
        )
        safety_layout.addRow(
            "Model attempts:",
            self._attempt_count_value,
        )

        self._replan_value = QLabel("NO")
        self._replan_value.setObjectName(
            "safetyReplan"
        )
        safety_layout.addRow(
            "Safety replan:",
            self._replan_value,
        )

        self._validation_value = QLabel("None")
        self._validation_value.setObjectName(
            "validationStatus"
        )
        safety_layout.addRow(
            "Validation:",
            self._validation_value,
        )

        self._blocked_keys_view = QPlainTextEdit()
        self._blocked_keys_view.setObjectName(
            "blockedActionKeys"
        )
        self._blocked_keys_view.setReadOnly(True)
        self._blocked_keys_view.setFixedHeight(55)
        safety_layout.addRow(
            "Blocked:",
            self._blocked_keys_view,
        )

        self._blocked_reason = QLabel("None")
        self._blocked_reason.setObjectName(
            "blockedReason"
        )
        self._blocked_reason.setWordWrap(True)
        safety_layout.addRow(
            "Reason:",
            self._blocked_reason,
        )

        layout.addWidget(safety_group)

        final_group = QGroupBox(
            "Final Decision"
        )
        final_layout = QFormLayout(final_group)
        final_layout.setRowWrapPolicy(
            QFormLayout.RowWrapPolicy.WrapLongRows
        )
        final_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        self._final_type_value = QLabel("None")
        self._final_type_value.setObjectName(
            "finalDecisionType"
        )
        final_layout.addRow(
            "Decision:",
            self._final_type_value,
        )

        self._operation_value = QLabel("None")
        self._operation_value.setObjectName(
            "finalOperation"
        )
        final_layout.addRow(
            "Operation:",
            self._operation_value,
        )

        self._target_value = QLabel("None")
        self._target_value.setObjectName(
            "finalTarget"
        )
        _configure_wrapped_value_label(
            self._target_value
        )
        final_layout.addRow(
            "Target:",
            self._target_value,
        )

        self._effect_value = QLabel("None")
        self._effect_value.setObjectName(
            "finalEffect"
        )
        _configure_wrapped_value_label(
            self._effect_value
        )
        final_layout.addRow(
            "Expected effect:",
            self._effect_value,
        )

        self._final_summary_value = QLabel("None")
        self._final_summary_value.setObjectName(
            "finalSummary"
        )
        _configure_wrapped_value_label(
            self._final_summary_value
        )
        final_layout.addRow(
            "Summary/question:",
            self._final_summary_value,
        )

        layout.addWidget(final_group)
        layout.addStretch()

        self.clear()

    def clear(self) -> None:
        self._source_value.setText("None")
        self._application_value.setText("None")
        self._window_value.setText("None")
        self._visible_text_view.setPlainText(
            "No current observation yet."
        )
        self._attempts_view.setPlainText(
            "No adaptive decision yet."
        )
        self._attempt_count_value.setText("0")
        self._replan_value.setText("NO")
        self._validation_value.setText("None")
        self._blocked_keys_view.setPlainText(
            "No blocked action keys."
        )
        self._blocked_reason.setText("None")
        self._final_type_value.setText("None")
        self._operation_value.setText("None")
        self._target_value.setText("None")
        self._effect_value.setText("None")
        self._final_summary_value.setText("None")

    def render(
        self,
        snapshot: AdaptiveDecisionSnapshot,
    ) -> None:
        if not isinstance(
            snapshot,
            AdaptiveDecisionSnapshot,
        ):
            raise ValueError(
                "snapshot must be an AdaptiveDecisionSnapshot"
            )

        self._source_value.setText(
            snapshot.decision_source
            or "None"
        )

        self._application_value.setText(
            snapshot.observation_application
            or "None"
        )
        self._window_value.setText(
            snapshot.observation_window
            or "None"
        )
        self._visible_text_view.setPlainText(
            " | ".join(snapshot.observation_text)
            if snapshot.observation_text
            else "No visible text recorded."
        )

        self._attempt_count_value.setText(
            str(snapshot.model_attempts)
        )
        self._replan_value.setText(
            "YES"
            if snapshot.safety_replan_used
            else "NO"
        )
        self._validation_value.setText(
            snapshot.final_status
        )
        self._blocked_keys_view.setPlainText(
            "\n".join(snapshot.blocked_action_keys)
            if snapshot.blocked_action_keys
            else "No blocked action keys."
        )
        self._blocked_reason.setText(
            snapshot.blocked_reason
            or "None"
        )

        self._attempts_view.setPlainText(
            _format_attempts(snapshot)
        )

        self._final_type_value.setText(
            snapshot.final_decision_type
            or "None"
        )
        self._operation_value.setText(
            snapshot.operation
            or "None"
        )
        self._target_value.setText(
            snapshot.target_text
            or "None"
        )
        self._effect_value.setText(
            snapshot.expected_effect
            or "None"
        )
        self._final_summary_value.setText(
            snapshot.completion_summary
            or snapshot.question
            or "None"
        )


def _configure_wrapped_value_label(
    label: QLabel,
) -> None:
    label.setWordWrap(True)
    label.setMinimumWidth(0)
    label.setAlignment(
        Qt.AlignmentFlag.AlignLeft
        | Qt.AlignmentFlag.AlignTop
    )
    label.setSizePolicy(
        QSizePolicy.Policy.Expanding,
        QSizePolicy.Policy.MinimumExpanding,
    )


def _format_attempts(
    snapshot: AdaptiveDecisionSnapshot,
) -> str:
    if not snapshot.attempts:
        return "No adaptive decision yet."

    lines: list[str] = []

    for attempt in snapshot.attempts:
        decision = attempt.decision_type or "BLOCKED"
        target = (
            f" -> {attempt.target_text}"
            if attempt.target_text
            else ""
        )

        lines.append(
            f"{attempt.attempt_number}. "
            f"{decision}{target}  {attempt.result}"
        )

        if attempt.result != "ACCEPTED":
            lines.append(
                f"   {attempt.reason}"
            )

    return "\n".join(lines)
