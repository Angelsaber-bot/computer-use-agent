"""Launch the Computer Agent desktop workspace."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from computer_agent.app.live_web_worker import (
    create_live_web_worker,
)
from computer_agent.app.main_window import (
    MainWindow,
)
from computer_agent.app.storage import (
    create_default_task_store,
)


def main() -> int:
    """Launch the production desktop Agent Workspace."""
    app = QApplication(
        sys.argv
    )

    window = MainWindow(
        semantic_worker_factory=(
            create_live_web_worker
        ),
        task_store=(
            create_default_task_store()
        ),
    )

    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
