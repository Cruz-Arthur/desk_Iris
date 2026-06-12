"""
UIX/components/shared.py
========================
Sistema de design "instrumento óptico".

Hierarquia cromática:
- grafite      → corpo do instrumento (bg / surface / card)
- âmbar óptico → único acento de interação (controles, foco, estado)
- verde fósforo→ exclusivamente decodificação bem-sucedida (cor = informação)

Tipografia:
- Bahnschrift (DIN industrial, nativa Windows) → display e labels técnicos
- Segoe UI                                     → corpo
- Consolas                                     → dados (payloads, números)
"""

import math

from pathlib import Path

from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QGraphicsDropShadowEffect,
    QSizePolicy, QWidget,
)
from PyQt6.QtCore import (
    QPoint, QPointF, QRectF, Qt, QPropertyAnimation, QEasingCurve,
    QTimer, QVariantAnimation,
)
from PyQt6.QtGui import (
    QColor, QCursor, QFontMetrics, QPainter, QPainterPath, QPen, QPixmap,
    QRadialGradient,
)

C = {
    # corpo do instrumento
    "bg":            "#0B0E13",
    "surface":       "#11161E",
    "card":          "#171E28",
    "card_hover":    "#1D2632",
    "surface_alt":   "#131922",
    # acento único: âmbar óptico (coating de lente)
    "primary":       "#FFB454",
    "primary_dim":   "#C98A3B",
    # informação secundária: aço
    "secondary":     "#8FA8BF",
    # semânticas — verde aparece SOMENTE quando um código é decodificado
    "danger":        "#FF7A7A",
    "success":       "#4ADE80",
    "warning":       "#FFB454",
    # texto
    "text":          "#EAF0F6",
    "text_muted":    "#8B9AAB",
    # bordas
    "border":        "#222C39",
    "border_hi":     "#32404F",
    "appbar_edge":   "#222C39",
    "window_ctrl_bg":    "#11161E",
    "window_ctrl_hover": "#1D2632",
}

# Pilha tipográfica
F_DISPLAY = "'Bahnschrift', 'Segoe UI', sans-serif"
F_BODY    = "'Segoe UI', sans-serif"
F_DATA    = "'Consolas', 'Courier New', monospace"

APP_LOGO_PATH = Path(__file__).resolve().parents[2] / "assets" / "img" / "logo.png"

GLOBAL_STYLE = f"""
    QWidget {{
        background-color: {C['bg']};
        color: {C['text']};
        font-family: {F_BODY};
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
        background: transparent;
        width: 8px;
        margin: 2px;
    }}
    QScrollBar::handle:vertical {{
        background: {C['border_hi']};
        border-radius: 4px;
        min-height: 40px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {C['primary_dim']};
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
        font-family: {F_DATA};
        font-size: 13px;
        selection-background-color: {C['primary']};
        selection-color: {C['bg']};
    }}
    QLineEdit:focus {{
        border-color: {C['primary']};
        background: {C['card_hover']};
    }}
    QLabel {{ background: transparent; }}
"""


# ─────────────────────────────────────────────────────────────────────────────
# IrisAperture — a assinatura do sistema
# ─────────────────────────────────────────────────────────────────────────────

class IrisAperture(QWidget):
    """
    Diafragma de íris desenhado com geometria real de lâminas tangentes.

    O furo central é o N-gono formado pelas retas tangentes a um círculo de
    raio ``r = lerp(min, max, openness)``; ao abrir, o conjunto gira — o
    movimento icônico de um diafragma fotográfico.

    O estado físico da íris comunica o estado do sistema:
        fechada  → parado
        abrindo  → inicializando
        aberta   → ao vivo
    """

    _N_BLADES   = 7
    _MIN_OPEN   = 0.10
    _MAX_OPEN   = 0.78
    _TWIST_RAD  = 0.85   # rotação total do furo entre fechado e aberto

    def __init__(self, diameter: int = 120, openness: float = 0.10,
                 parent=None) -> None:
        super().__init__(parent)
        self._openness = max(0.0, min(1.0, openness))
        self.setFixedSize(diameter, diameter)

        self._anim = QVariantAnimation(self)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._anim.valueChanged.connect(self._on_anim)

        # Respiração (sine) — usada no estado "inicializando"
        self._breath_timer = QTimer(self)
        self._breath_timer.setInterval(33)
        self._breath_timer.timeout.connect(self._breath_tick)
        self._breath_phase = 0.0
        self._breath_lo    = 0.15
        self._breath_hi    = 0.70

    # ── API ──────────────────────────────────────────────────────────────────

    def openness(self) -> float:
        return self._openness

    def set_openness(self, f: float) -> None:
        self._openness = max(0.0, min(1.0, f))
        self.update()

    def animate_to(self, f: float, ms: int = 650) -> None:
        self.stop_breathing()
        self._anim.stop()
        self._anim.setStartValue(self._openness)
        self._anim.setEndValue(max(0.0, min(1.0, f)))
        self._anim.setDuration(ms)
        self._anim.start()

    def start_breathing(self, lo: float = 0.15, hi: float = 0.70) -> None:
        self._anim.stop()
        self._breath_lo, self._breath_hi = lo, hi
        if not self._breath_timer.isActive():
            self._breath_timer.start()

    def stop_breathing(self) -> None:
        self._breath_timer.stop()

    # ── Interno ──────────────────────────────────────────────────────────────

    def _on_anim(self, v) -> None:
        self._openness = float(v)
        self.update()

    def _breath_tick(self) -> None:
        self._breath_phase += 0.045
        mid = (self._breath_lo + self._breath_hi) / 2.0
        amp = (self._breath_hi - self._breath_lo) / 2.0
        self._openness = mid + amp * math.sin(self._breath_phase)
        self.update()

    def hideEvent(self, event) -> None:
        self._breath_timer.stop()
        super().hideEvent(event)

    # ── Pintura ──────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        side = min(self.width(), self.height())
        if side < 8:
            return

        cx = self.width() / 2.0
        cy = self.height() / 2.0
        R  = side / 2.0 - 2.0                       # raio externo (barril)
        n  = self._N_BLADES
        t  = self._openness
        r  = (self._MIN_OPEN + (self._MAX_OPEN - self._MIN_OPEN) * t) * R
        base = t * self._TWIST_RAD - math.pi / 2.0  # giro ao abrir

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 1. disco das lâminas (material grafite, leve gradiente radial)
        grad = QRadialGradient(QPointF(cx, cy), R)
        grad.setColorAt(0.0, QColor("#222C39"))
        grad.setColorAt(1.0, QColor("#151B24"))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(grad)
        p.drawEllipse(QPointF(cx, cy), R, R)

        # 2. costuras das lâminas — do vértice do furo até o barril,
        #    numa direção só (o "swirl" característico do diafragma)
        seam = QPen(QColor("#0B0E13"), max(1.0, side * 0.014),
                    Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        p.setPen(seam)
        half_chord = math.sqrt(max(R * R - r * r, 0.0))
        t0 = r * math.tan(math.pi / n)
        for i in range(n):
            phi = base + 2.0 * math.pi * i / n
            px_, py_ = r * math.cos(phi), r * math.sin(phi)
            dx_, dy_ = -math.sin(phi), math.cos(phi)
            a = QPointF(cx + px_ + dx_ * t0,         cy + py_ + dy_ * t0)
            b = QPointF(cx + px_ + dx_ * half_chord, cy + py_ + dy_ * half_chord)
            p.drawLine(a, b)

        # 3. furo — N-gono das tangentes, recortado até o fundo
        rv = r / math.cos(math.pi / n)              # circunraio dos vértices
        hole = QPainterPath()
        for i in range(n):
            ang = base + 2.0 * math.pi * i / n + math.pi / n
            pt = QPointF(cx + rv * math.cos(ang), cy + rv * math.sin(ang))
            if i == 0:
                hole.moveTo(pt)
            else:
                hole.lineTo(pt)
        hole.closeSubpath()

        p.setPen(QPen(QColor(255, 180, 84, 200), max(1.0, side * 0.012)))
        p.setBrush(QColor(C["bg"]))
        p.drawPath(hole)

        # 4. barril externo
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor("#32404F"), max(1.5, side * 0.02)))
        p.drawEllipse(QPointF(cx, cy), R, R)
        p.setPen(QPen(QColor(255, 180, 84, 55), 1.0))
        p.drawEllipse(QPointF(cx, cy), R - max(2.0, side * 0.035),
                      R - max(2.0, side * 0.035))
        p.end()


# ─────────────────────────────────────────────────────────────────────────────
# Barra superior
# ─────────────────────────────────────────────────────────────────────────────

class IrisAppBar(QFrame):
    """Barra superior universal — arrasto de janela e controles nativos."""

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
                background: {C['surface']};
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
                QPushButton:focus {{
                    border: 2px solid {C['text']};
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
            font-family: {F_DISPLAY};
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 4px;
        """)
        layout.addWidget(self._title_lbl, stretch=1)

        self._status_lbl = QLabel("IRIS SYSTEMS")
        self._status_lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self._status_lbl.setStyleSheet(f"""
            color: {C['text_muted']};
            font-family: {F_DISPLAY};
            font-size: 9px;
            letter-spacing: 3px;
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
            QPushButton:focus {{
                border-color: {hover};
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
    """Botão de ação primária — contorno âmbar com glow no hover."""

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
                font-family: {F_DISPLAY};
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 3px;
            }}
            QPushButton:hover {{
                background: {C['primary']};
                color: {C['bg']};
                border-color: {C['primary']};
            }}
            QPushButton:focus {{
                border: 2px solid {C['text']};
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
