"""Запуск приложения: python run.py"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Каталог данных = папка database/ рядом с проектом
os.environ.setdefault("NIRSEG_DATA_DIR", str(PROJECT_ROOT / "database"))
# Гарантируем, что проект виден как пакет gui.*
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PyQt6.QtWidgets import QApplication
from gui.mainwindow import SegmentationApp, APP_STYLESHEET


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("CYBER MED AI")
    app.setStyleSheet(APP_STYLESHEET)
    window = SegmentationApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()