from pathlib import Path

from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QGraphicsDropShadowEffect, QSizePolicy
)
from PyQt6.QtCore import QPoint, Qt, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QColor, QCursor, QFontMetrics, QPixmap

C = {
    "bg": "#060B14",
    "surface": "#0B1120",
    "card": "#111C2E",
    "card_hover": "#162236",
    "primary": "#00FF88",
    "primary_dim": "#00A855",
    "secondary": "#22D3EE",
    "danger": "#F87171",
    "success": "#4ADE80",
    "warning": "#FBBF24",
    "text": "#E8F4F0",
    "text_muted": "#52687A",
    "border": "#1A3048",
    "border_hi": "#204060",
    "surface_alt": "#0E1726",
    "appbar_edge": "#163047",
    "window_ctrl_bg": "#0F1928",
    "window_ctrl_hover": "#1A2A3D",
}

APP_LOGO_PATH = Path(__file__).resolve().parents[2] / "assets" / "img" / "logo.png"

GLOBAL_STYLE = f"""
    * {{
        font-family: 'Consolas', 'Courier New', monospace;
    }}
    QWidget {{
        background-color: {C['bg']};
        color: {C['text']};
        font-size: 13px;
    }}
    QMainWindow, QDialog {{
        background-color: {C['bg']};
    }}
    QScrollArea, QAbstractScrollArea {{
        border: none;
        background: transparent;
    }}
    QScrollBar:vertical {{
        background: {C['surface']};
        width: 4px;
        margin: 0;
        border-radius: 2px;
    }}
    QScrollBar::handle:vertical {{
        background: {C['primary']};
        border-radius: 2px;
        min-height: 50px;
    }}
    QScrollBar::add-line:vertical,
    QScrollBar::sub-line:vertical {{ height: 0; border: none; }}
    QScrollBar::add-page:vertical,
    QScrollBar::sub-page:vertical {{ background: none; }}
    QLineEdit {{
        background: {C['card']};
        border: 1px solid {C['border']};
        border-radius: 6px;
        padding: 10px 14px;
        color: {C['text']};
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 13px;
        selection-background-color: {C['primary']};
        selection-color: {C['bg']};
    }}
    QLineEdit:focus {{
        border-color: {C['primary']};
        background: {C['card_hover']};
    }}
    QLabel {{ background: transparent; }}
    QProgressBar {{
        background: {C['border']};
        border: none;
        border-radius: 2px;
        text-align: center;
    }}
    QProgressBar::chunk {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {C['primary']}, stop:1 {C['secondary']});
        border-radius: 2px;
    }}
"""


class IrisAppBar(QFrame):
    """Barra superior universal com gradiente e layout responsivo."""

    def __init__(self, title: str = "Iris", show_back_button: bool = False, on_back=None, parent=None):
        super().__init__(parent)
        self._title_text = title.upper()
        self._drag_offset = None
        self._drag_ratio_x = 0.5
        self._drag_press_y = 0
        self.setFixedHeight(58)
        self.setObjectName("IrisAppBar")
        self.setStyleSheet(f"""
            #IrisAppBar {{
                background: {C['window_ctrl_bg']};
                border-bottom: 1px solid {C['appbar_edge']};
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(10)

        if show_back_button and on_back:
            back_btn = QPushButton("←")
            back_btn.setFixedSize(30, 30)
            back_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            back_btn.setStyleSheet(f"""
                QPushButton {{
                    background: transparent;
                    border: 1px solid {C['primary']};
                    border-radius: 15px;
                    color: {C['primary']};
                    font-size: 14px;
                    font-weight: bold;
                    padding-bottom: 2px;
                }}
                QPushButton:hover {{
                    background: {C['primary']};
                    color: {C['bg']};
                }}
                QPushButton:pressed {{
                    background: {C['primary_dim']};
                }}
            """)
            back_btn.clicked.connect(on_back)
            layout.addWidget(back_btn)

        mark = QLabel()
        mark.setFixedSize(18, 18)
        mark.setStyleSheet("background: transparent;")
        logo = QPixmap(str(APP_LOGO_PATH))
        if not logo.isNull():
            mark.setPixmap(
                logo.scaled(
                    14,
                    14,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        layout.addWidget(mark)

        self._title_lbl = QLabel(self._title_text)
        self._title_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._title_lbl.setStyleSheet(f"""
            color: {C['text']};
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 3px;
        """)
        layout.addWidget(self._title_lbl, stretch=1)

        self._status_lbl = QLabel("● ONLINE")
        self._status_lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self._status_lbl.setStyleSheet(f"""
            color: {C['secondary']};
            font-size: 10px;
            letter-spacing: 2px;
        """)
        layout.addWidget(self._status_lbl)

        self._controls = QFrame()
        self._controls.setStyleSheet("background: transparent; border: none;")
        controls_layout = QHBoxLayout(self._controls)
        controls_layout.setContentsMargins(6, 0, 0, 0)
        controls_layout.setSpacing(4)
        controls_layout.addWidget(self._make_window_button("—", self._minimize_window))
        controls_layout.addWidget(self._make_window_button("□", self._toggle_maximize))
        controls_layout.addWidget(self._make_window_button("✕", self._close_window, danger=True))
        layout.addWidget(self._controls)
        self._sync_compact_state()

    def _make_window_button(self, text: str, on_click, danger: bool = False) -> QPushButton:
        btn = QPushButton(text)
        btn.setFixedSize(28, 28)
        btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        hover = C["danger"] if danger else C["primary"]
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {C['window_ctrl_bg']};
                border: 1px solid {C['appbar_edge']};
                border-radius: 4px;
                color: {C['text']};
                font-size: 11px;
                font-weight: 700;
                padding-bottom: 1px;
            }}
            QPushButton:hover {{
                background: {C['danger'] if danger else C['window_ctrl_hover']};
                border-color: {hover};
                color: {C['text'] if danger else C['primary']};
            }}
            QPushButton:pressed {{
                background: {C['card_hover']};
                border-color: {hover};
                color: {hover};
            }}
        """)
        btn.clicked.connect(on_click)
        return btn

    def _window_target(self):
        return self.window()

    def _minimize_window(self) -> None:
        target = self._window_target()
        if target is not None:
            target.showMinimized()

    def _toggle_maximize(self) -> None:
        target = self._window_target()
        if target is None:
            return
        if target.isMaximized():
            target.showNormal()
        else:
            target.showMaximized()

    def _close_window(self) -> None:
        target = self._window_target()
        if target is not None:
            target.close()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            target = self._window_target()
            if target is not None:
                local_pos = event.position().toPoint()
                self._drag_ratio_x = local_pos.x() / max(self.width(), 1)
                self._drag_press_y = local_pos.y()
                if not target.isMaximized():
                    self._drag_offset = event.globalPosition().toPoint() - target.frameGeometry().topLeft()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            target = self._window_target()
            if target is not None:
                global_pos = event.globalPosition().toPoint()
                if target.isMaximized():
                    normal_geom = target.normalGeometry()
                    if not normal_geom.isValid():
                        normal_geom = target.geometry()
                    restored_width = max(normal_geom.width(), target.minimumWidth())
                    restored_height = max(normal_geom.height(), target.minimumHeight())
                    titlebar_y = min(self._drag_press_y, self.height() - 1)
                    self._drag_offset = QPoint(
                        int(restored_width * self._drag_ratio_x),
                        titlebar_y,
                    )
                    target.showNormal()
                    target.resize(restored_width, restored_height)
                    target.move(global_pos - self._drag_offset)
                elif self._drag_offset is not None:
                    target.move(global_pos - self._drag_offset)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_offset = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._toggle_maximize()
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event) -> None:
        self._sync_compact_state()
        super().resizeEvent(event)

    def _sync_compact_state(self) -> None:
        compact = self.width() < 980
        self._status_lbl.setVisible(not compact)
        title_margin = 220 if compact else 320
        available = max(120, self.width() - title_margin)
        metrics = QFontMetrics(self._title_lbl.font())
        self._title_lbl.setText(
            metrics.elidedText(
                self._title_text,
                Qt.TextElideMode.ElideRight,
                available,
            )
        )


class IrisButton(QPushButton):
    """Botao com animacao de glow ao passar o mouse."""

    def __init__(self, text: str, on_click=None, width: int = 260, parent=None):
        super().__init__(text.upper(), parent)
        self.setFixedSize(width, 48)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: 1px solid {C['primary']};
                border-radius: 3px;
                color: {C['primary']};
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 3px;
            }}
            QPushButton:hover {{
                background: {C['primary']};
                color: {C['bg']};
                border-color: {C['primary']};
            }}
            QPushButton:pressed {{
                background: {C['primary_dim']};
                border-color: {C['primary_dim']};
            }}
            QPushButton:disabled {{
                border-color: {C['text_muted']};
                color: {C['text_muted']};
            }}
        """)

        if on_click:
            self.clicked.connect(on_click)

        self._glow = QGraphicsDropShadowEffect(self)
        self._glow.setColor(QColor(C["primary"]))
        self._glow.setBlurRadius(0.0)
        self._glow.setOffset(0.0, 0.0)
        self.setGraphicsEffect(self._glow)

        self._anim = QPropertyAnimation(self._glow, b"blurRadius", self)
        self._anim.setDuration(180)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def enterEvent(self, event):
        self._anim.stop()
        self._anim.setStartValue(self._glow.blurRadius())
        self._anim.setEndValue(22.0)
        self._anim.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._anim.stop()
        self._anim.setStartValue(self._glow.blurRadius())
        self._anim.setEndValue(0.0)
        self._anim.start()
        super().leaveEvent(event)
