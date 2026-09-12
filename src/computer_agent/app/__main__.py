"""Launch the Computer Agent desktop workspace."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from computer_agent.app.main_window import MainWindow


def main() -> int:
    """Launch the desktop Agent Workspace."""

    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
