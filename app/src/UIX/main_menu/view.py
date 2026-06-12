"""
UIX/main_menu/view.py
=====================
Menu principal — "bancada do instrumento".

Painel de marca à esquerda (íris hero respirando + wordmark), módulos à
direita. Sem barra de filtro: com um módulo, filtro é cerimônia. O grid
permanece — novos módulos entram sem refatorar.

Acessibilidade: cards focáveis por Tab, ativados por Enter/Espaço; o foco
desenha os mesmos colchetes de viewfinder do hover.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.src.UIX.components.shared import (
    APP_LOGO_PATH, C, F_BODY, F_DATA, F_DISPLAY, IrisAperture, IrisAppBar,
)
from app.src.UIX.components.star_field import StarFieldPanel


_CARD_MIN_W   = 240
_CARD_MAX_W   = 560
_CARD_MIN_H   = 150
_CARD_MAX_H   = 320
_MAX_COLS     = 3
_PREF_CARD_W  = 380


# ─────────────────────────────────────────────────────────────────────────────
# Painel de marca
# ─────────────────────────────────────────────────────────────────────────────

class _BrandPanel(StarFieldPanel):

    def __init__(self, on_exit, parent=None):
        super().__init__(accent_edge="right", n_stars=40, parent=parent)
        self.setMinimumWidth(280)
        self.setMaximumWidth(360)

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 36, 28, 24)
        root.setSpacing(0)

        root.addStretch(2)

        # Íris hero — respiração lenta: o instrumento está vivo, em repouso
        self._aperture = IrisAperture(diameter=148, openness=0.35)
        self._aperture.start_breathing(lo=0.28, hi=0.46)
        root.addWidget(self._aperture, alignment=Qt.AlignmentFlag.AlignCenter)
        root.addSpacing(26)

        brand = QLabel("IRIS")
        brand.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand.setStyleSheet(
            f"color: {C['text']}; font-family: {F_DISPLAY};"
            "font-size: 32px; font-weight: 700; letter-spacing: 14px;"
            "padding-left: 14px;"   # compensa o letter-spacing do último glifo
        )
        root.addWidget(brand)
        root.addSpacing(10)

        sep = QFrame()
        sep.setFixedSize(56, 1)
        sep.setStyleSheet("background: rgba(255,180,84,0.45);")
        root.addWidget(sep, alignment=Qt.AlignmentFlag.AlignCenter)
        root.addSpacing(10)

        tagline = QLabel("ESTAÇÃO DE LEITURA ÓPTICA")
        tagline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tagline.setStyleSheet(
            f"color: {C['text_muted']}; font-family: {F_DISPLAY};"
            "font-size: 10px; letter-spacing: 4px;"
        )
        root.addWidget(tagline)

        root.addStretch(3)

        exit_btn = QPushButton("✕   ENCERRAR SESSÃO")
        exit_btn.setFixedHeight(40)
        exit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        exit_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: 1px solid rgba(255,122,122,0.30);
                border-radius: 8px;
                color: {C['danger']};
                font-family: {F_DISPLAY};
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 2px;
                padding: 0 14px;
                text-align: left;
            }}
            QPushButton:hover {{
                background: rgba(255,122,122,0.12);
                border-color: {C['danger']};
            }}
            QPushButton:focus {{
                border: 2px solid {C['danger']};
            }}
            QPushButton:pressed {{ background: rgba(255,122,122,0.20); }}
        """)
        if on_exit:
            exit_btn.clicked.connect(on_exit)
        root.addWidget(exit_btn)
        root.addSpacing(14)

        foot = QLabel("IRIS  ·  SYSTEMS")
        foot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        foot.setStyleSheet(
            f"color: rgba(234,240,246,0.18); font-family: {F_DISPLAY};"
            "font-size: 9px; letter-spacing: 2px;"
        )
        root.addWidget(foot)


# ─────────────────────────────────────────────────────────────────────────────
# Card de módulo
# ─────────────────────────────────────────────────────────────────────────────

class _ModuleCard(QFrame):
    """
    Card focável. Hover ou foco: o diafragma abre e colchetes de viewfinder
    enquadram o card — o sistema "mira" no módulo escolhido.
    """

    def __init__(self, title: str, subtitle: str, tag: str,
                 callback=None, parent=None):
        super().__init__(parent)
        self._callback = callback
        self._engaged  = False

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(_CARD_MIN_W, _CARD_MIN_H)
        self.setMaximumSize(_CARD_MAX_W, _CARD_MAX_H)
        self.setObjectName("ModCard")
        self._apply_style(False)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 16)
        root.setSpacing(0)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(10)

        self._aperture = IrisAperture(diameter=40, openness=0.12)
        top.addWidget(self._aperture, alignment=Qt.AlignmentFlag.AlignTop)

        tag_lbl = QLabel(tag.upper())
        tag_lbl.setStyleSheet(
            f"color: {C['text_muted']}; font-family: {F_DISPLAY};"
            "font-size: 9px; font-weight: 700; letter-spacing: 2px;"
        )
        top.addStretch()
        top.addWidget(tag_lbl, alignment=Qt.AlignmentFlag.AlignTop)
        root.addLayout(top)

        root.addStretch()

        title_lbl = QLabel(title)
        title_lbl.setWordWrap(True)
        title_lbl.setStyleSheet(
            f"color: {C['text']}; font-family: {F_DISPLAY};"
            "font-size: 18px; font-weight: 700; letter-spacing: 1px;"
        )
        root.addWidget(title_lbl)
        root.addSpacing(6)

        sub_lbl = QLabel(subtitle)
        sub_lbl.setWordWrap(True)
        sub_lbl.setStyleSheet(
            f"color: {C['text_muted']}; font-family: {F_BODY}; font-size: 12px;"
        )
        root.addWidget(sub_lbl)
        root.addSpacing(12)

        self._hint = QLabel("ABRIR  →")
        self._hint.setStyleSheet(
            f"color: {C['primary']}; font-family: {F_DISPLAY};"
            "font-size: 10px; font-weight: 700; letter-spacing: 3px;"
        )
        self._hint.setVisible(False)
        root.addWidget(self._hint)

    # ── Estado visual ────────────────────────────────────────────────────────

    def _apply_style(self, engaged: bool) -> None:
        border = C["primary"] if engaged else C["border"]
        bg     = C["card_hover"] if engaged else C["card"]
        self.setStyleSheet(f"""
            #ModCard {{
                background: {bg};
                border: 1px solid {border};
                border-radius: 14px;
            }}
        """)

    def _engage(self, on: bool) -> None:
        if on == self._engaged:
            return
        self._engaged = on
        self._apply_style(on)
        self._hint.setVisible(on)
        self._aperture.animate_to(0.80 if on else 0.12, ms=420)
        self.update()

    # ── Interação ────────────────────────────────────────────────────────────

    def enterEvent(self, event):
        self._engage(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        if not self.hasFocus():
            self._engage(False)
        super().leaveEvent(event)

    def focusInEvent(self, event):
        self._engage(True)
        super().focusInEvent(event)

    def focusOutEvent(self, event):
        if not self.underMouse():
            self._engage(False)
        super().focusOutEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._callback:
            self._callback()
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            if self._callback:
                self._callback()
            return
        super().keyPressEvent(event)

    # ── Colchetes de viewfinder ──────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if not self._engaged:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(C["primary"]), 2.0,
                   Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        w, h = self.width(), self.height()
        m, L = 7.0, 16.0
        for x, y, dx, dy in (
            (m, m, 1, 1), (w - m, m, -1, 1),
            (w - m, h - m, -1, -1), (m, h - m, 1, -1),
        ):
            p.drawLine(QPointF(x, y), QPointF(x + dx * L, y))
            p.drawLine(QPointF(x, y), QPointF(x, y + dy * L))
        p.end()


# ─────────────────────────────────────────────────────────────────────────────
# Área de módulos
# ─────────────────────────────────────────────────────────────────────────────

class _ModuleArea(QFrame):
    _MARGIN_H = 32

    def __init__(self, all_items: list[dict], parent=None):
        super().__init__(parent)
        self._all_items    = all_items
        self._current_cols = 0
        self._current_rows = 0
        self.setStyleSheet(f"background: {C['bg']}; border: none;")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Cabeçalho da seção
        head = QFrame()
        head.setFixedHeight(58)
        head.setStyleSheet("background: transparent; border: none;")
        hl = QHBoxLayout(head)
        hl.setContentsMargins(self._MARGIN_H, 0, self._MARGIN_H, 0)
        hl.setSpacing(12)

        title = QLabel("MÓDULOS")
        title.setStyleSheet(
            f"color: {C['text']}; font-family: {F_DISPLAY};"
            "font-size: 13px; font-weight: 700; letter-spacing: 5px;"
        )
        hl.addWidget(title)

        count = QLabel(f"{len(all_items):02d}")
        count.setStyleSheet(
            f"color: {C['primary']}; font-family: {F_DATA};"
            "font-size: 11px; font-weight: 700;"
        )
        hl.addWidget(count)
        hl.addStretch()
        root.addWidget(head)

        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {C['border']}; border: none;")
        root.addWidget(sep)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        scroll_widget = QWidget()
        scroll_widget.setStyleSheet(f"background: {C['bg']};")
        outer = QVBoxLayout(scroll_widget)
        outer.setContentsMargins(self._MARGIN_H, 22, self._MARGIN_H, 32)
        outer.setSpacing(0)

        self._grid_container = QWidget()
        self._grid_container.setStyleSheet("background: transparent;")
        self._grid_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._grid_layout = QGridLayout(self._grid_container)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        self._grid_layout.setHorizontalSpacing(16)
        self._grid_layout.setVerticalSpacing(16)

        outer.addWidget(self._grid_container)
        scroll.setWidget(scroll_widget)
        root.addWidget(scroll, stretch=1)

        self._render_grid()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        new_cols = self._calc_cols()
        if new_cols != self._current_cols:
            self._current_cols = new_cols
            self._render_grid()

    def _calc_cols(self) -> int:
        w = self.width()
        if w == 0:
            return 1
        available = w - 2 * self._MARGIN_H
        spacing   = self._grid_layout.horizontalSpacing()
        cols = max(1, (available + spacing) // (_PREF_CARD_W + spacing))
        return min(cols, _MAX_COLS)

    def _render_grid(self) -> None:
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)

        cols = self._current_cols or 1
        for col in range(_MAX_COLS + 1):
            self._grid_layout.setColumnStretch(col, 0)
        for row in range(self._current_rows + 1):
            self._grid_layout.setRowStretch(row, 0)
        for col in range(cols):
            self._grid_layout.setColumnStretch(col, 1)

        n_rows = max(1, -(-len(self._all_items) // cols))
        for idx, item in enumerate(self._all_items):
            card = _ModuleCard(
                title=item["title"],
                subtitle=item["subtitle"],
                tag=item.get("tag", ""),
                callback=item["callback"],
            )
            self._grid_layout.addWidget(card, idx // cols, idx % cols)

        for row in range(n_rows):
            self._grid_layout.setRowStretch(row, 1)
        self._current_rows = n_rows


# ─────────────────────────────────────────────────────────────────────────────
# View principal
# ─────────────────────────────────────────────────────────────────────────────

class MainMenuView(QWidget):

    def __init__(self, on_live_qr, on_exit, parent=None):
        super().__init__(parent)

        all_items = [
            {
                "key":      "live_qr",
                "title":    "Live QR",
                "subtitle": "Leitura contínua de códigos QR em câmera única, "
                            "com registro de cada decodificação.",
                "tag":      "DETECÇÃO",
                "callback": on_live_qr,
            }
        ]

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(IrisAppBar("Iris"))

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        body.addWidget(_BrandPanel(on_exit=on_exit), stretch=1)
        body.addWidget(_ModuleArea(all_items), stretch=4)

        root.addLayout(body, stretch=1)
