import sys
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QMainWindow, QStackedWidget

if sys.platform == "win32":
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
        "Multilaser.Iris.LiveQr.1"
    )

from app.src.UIX.components.shared import GLOBAL_STYLE
from app.src.UIX.main_menu.view import MainMenuView
from app.src.UIX.modules.decoding.live_qr.view import LiveQrView

APP_ICON_PATH = Path(__file__).resolve().parents[1] / "assets" / "img" / "logo.png"


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Iris - Live QR")
        self.setMinimumSize(960, 680)
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        self.setStyleSheet(GLOBAL_STYLE)
        if APP_ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(APP_ICON_PATH)))

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._live_qr_view = None

        self._main_menu = MainMenuView(
            on_live_qr=self._go_live_qr,
            on_exit=self.close,
        )
        self._stack.addWidget(self._main_menu)
        self._stack.setCurrentWidget(self._main_menu)

    def _go_home(self) -> None:
        if self._live_qr_view:
            self._stack.removeWidget(self._live_qr_view)
            self._live_qr_view.deleteLater()
            self._live_qr_view = None
        self._stack.setCurrentWidget(self._main_menu)

    def _go_live_qr(self):
        if not self._live_qr_view:
            QTimer.singleShot(0, self._create_and_open_live_qr)
            return
        self._stack.setCurrentWidget(self._live_qr_view)

    def _create_and_open_live_qr(self):
        if not self._live_qr_view:
            self._live_qr_view = LiveQrView(on_back=self._go_home)
            self._stack.addWidget(self._live_qr_view)
        self._stack.setCurrentWidget(self._live_qr_view)


def main():
    app = QApplication(sys.argv)
    app.setOrganizationName("Iris")
    app.setApplicationName("Iris")
    app.setStyle("Fusion")
    if APP_ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(APP_ICON_PATH)))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
