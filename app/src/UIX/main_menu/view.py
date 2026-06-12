from __future__ import annotations

import math
import random

from PyQt6.QtCore import Qt, QPointF, QRectF, QTimer
from PyQt6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QPixmap
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

from app.src.UIX.components.shared import APP_LOGO_PATH, C, IrisAppBar
from app.src.UIX.components.star_field import StarFieldPanel


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Category palette ──────────────────────────────────────────────────────────

_CAT: dict[str, dict] = {
    "todos":    {"label": "TODOS",    "color": "#00FF88"},
    "deteccao": {"label": "DETECÇÃO", "color": "#22D3EE"},
}

_CAT_ORDER = ["todos", "deteccao"]

_MODULE_CATEGORY: dict[str, str] = {
    "live_qr": "deteccao",
}

_CARD_MIN_W   = 160
_CARD_MAX_W   = 600
_CARD_MIN_H   = 110
_CARD_MAX_H   = 400
_COLS_DEFAULT = 2
_MAX_COLS     = 3


# ── Navigation button (sidebar) ───────────────────────────────────────────────

def _nav_btn(label: str, callback=None, *, danger: bool = False) -> QPushButton:
    btn = QPushButton(label)
    btn.setFixedHeight(40)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)

    if danger:
        fg, bg_h, bdr = "#F87171", "rgba(248,113,113,0.12)", "rgba(248,113,113,0.30)"
    else:
        fg, bg_h, bdr = "rgba(241,245,249,0.72)", "rgba(255,255,255,0.07)", "rgba(255,255,255,0.12)"

    btn.setStyleSheet(f"""
        QPushButton {{
            background: transparent;
            border: 1px solid {bdr};
            border-radius: 8px;
            color: {fg};
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.5px;
            padding: 0 14px;
            text-align: left;
        }}
        QPushButton:hover {{
            background: {bg_h};
            border-color: {fg};
        }}
        QPushButton:pressed {{ background: {bg_h}; }}
    """)
    if callback:
        btn.clicked.connect(callback)
    return btn


# ── Sidebar ───────────────────────────────────────────────────────────────────

class _SystemNav(StarFieldPanel):

    def __init__(self, on_exit, parent=None):
        super().__init__(accent_edge="right", n_stars=45, parent=parent)
        self.setMinimumWidth(260)
        self.setMaximumWidth(340)

        root = QVBoxLayout(self)
        root.setContentsMargins(22, 28, 22, 22)
        root.setSpacing(0)

        logo_lbl = QLabel()
        logo_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pix = QPixmap(str(APP_LOGO_PATH))
        if not pix.isNull():
            pix = pix.scaledToWidth(72, Qt.TransformationMode.SmoothTransformation)
            logo_lbl.setPixmap(pix)
        root.addWidget(logo_lbl)
        root.addSpacing(14)

        brand = QLabel("IRIS")
        brand.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand.setStyleSheet(
            "color: #00FF88; font-size: 26px; font-weight: 900; letter-spacing: 8px;"
        )
        root.addWidget(brand)
        root.addSpacing(10)

        sep = QFrame()
        sep.setFixedSize(48, 1)
        sep.setStyleSheet("background: rgba(0,255,136,0.35);")
        root.addWidget(sep, alignment=Qt.AlignmentFlag.AlignCenter)
        root.addSpacing(8)

        tagline = QLabel("Live QR")
        tagline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tagline.setStyleSheet(
            "color: rgba(241,245,249,0.38); font-size: 10px; letter-spacing: 4px;"
        )
        root.addWidget(tagline)
        root.addSpacing(32)

        section = QLabel("SISTEMA")
        section.setStyleSheet(
            f"color: {C['secondary']}; font-size: 9px; font-weight: 700;"
            "letter-spacing: 3px; padding-left: 4px;"
        )
        root.addWidget(section)
        root.addSpacing(10)

        root.addWidget(_nav_btn("✕   Encerrar", on_exit, danger=True))

        root.addStretch()

        foot = QLabel("Iris  ·  Grupo Multilaser")
        foot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        foot.setStyleSheet(
            "color: rgba(241,245,249,0.12); font-size: 9px; letter-spacing: 1px;"
        )
        root.addWidget(foot)


# ── Module card ───────────────────────────────────────────────────────────────

_CAT_GLYPH: dict[str, str] = {
    "deteccao": "◎",
}


class _ModuleCard(QFrame):

    def __init__(self, title, subtitle, callback=None, color=None, category="", parent=None):
        super().__init__(parent)
        self._callback  = callback
        self._color_str = color or C["primary"]
        self._hovered   = False
        self._category  = category
        self._glyph     = _CAT_GLYPH.get(category, "◈")
        self._title_color    = C["text"]
        self._subtitle_color = C["text_muted"]

        self._particles: list[dict] = []
        self._spawning  = False
        self._anim      = QTimer(self)
        self._anim.setInterval(30)
        self._anim.timeout.connect(self._tick)

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumWidth(_CARD_MIN_W)
        self.setMaximumWidth(_CARD_MAX_W)
        self.setMinimumHeight(_CARD_MIN_H)
        self.setMaximumHeight(_CARD_MAX_H)

        self.setStyleSheet(f"""
            QFrame {{
                background: {C['card']};
                border: 1px solid {_rgba(self._color_str, 0.30)};
                border-radius: 16px;
            }}
            QFrame:hover {{
                background: {C['card_hover']};
                border: 1px solid {self._color_str};
            }}
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(0)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(6)

        cat_text = _CAT.get(category, {}).get("label", "")
        self._cat_lbl = QLabel(cat_text)
        self._cat_lbl.setStyleSheet(
            f"color: {_rgba(self._color_str, 0.60)};"
            "font-size: 9px; font-weight: 700; letter-spacing: 1.8px;"
            "border: none; background: transparent;"
            "font-family: 'Consolas', 'Courier New', monospace;"
        )
        top_row.addWidget(self._cat_lbl)
        top_row.addStretch()

        dot = QFrame()
        dot.setFixedSize(7, 7)
        dot.setStyleSheet(
            f"background: {self._color_str}; border-radius: 3px; border: none;"
        )
        top_row.addWidget(dot)
        root.addLayout(top_row)
        root.addStretch()

        self._title_lbl = QLabel(title)
        self._title_lbl.setWordWrap(True)
        self._title_lbl.setStyleSheet(
            f"color: {self._title_color};"
            "font-size: 14px; font-weight: 700; border: none; background: transparent;"
        )
        root.addWidget(self._title_lbl)
        root.addSpacing(4)

        self._subtitle_lbl = QLabel(subtitle)
        self._subtitle_lbl.setWordWrap(True)
        self._subtitle_lbl.setStyleSheet(
            f"color: {self._subtitle_color};"
            "font-size: 11px; border: none; background: transparent;"
        )
        root.addWidget(self._subtitle_lbl)

    def mousePressEvent(self, event):
        if self._callback:
            self._callback()
        super().mousePressEvent(event)

    def enterEvent(self, event):
        self._hovered  = True
        self._spawning = True
        if not self._anim.isActive():
            self._anim.start()
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered  = False
        self._spawning = False
        self.update()
        super().leaveEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        h = self.height()
        title_sz = max(13, min(int(h * 0.115), 24))
        sub_sz   = max(10, min(int(h * 0.090), 18))
        cat_sz   = max( 9, min(int(h * 0.068), 13))
        self._title_lbl.setStyleSheet(
            f"color: {self._title_color}; font-size: {title_sz}px; font-weight: 700;"
            "border: none; background: transparent;"
        )
        self._subtitle_lbl.setStyleSheet(
            f"color: {self._subtitle_color}; font-size: {sub_sz}px;"
            "border: none; background: transparent;"
        )
        self._cat_lbl.setStyleSheet(
            f"color: {_rgba(self._color_str, 0.60)}; font-size: {cat_sz}px;"
            "font-weight: 700; letter-spacing: 1.8px; border: none; background: transparent;"
            "font-family: 'Consolas', 'Courier New', monospace;"
        )

    _glyph_pt_cache: dict[str, float] = {}

    @classmethod
    def _pt_per_px(cls, glyph: str) -> float:
        if glyph not in cls._glyph_pt_cache:
            from PyQt6.QtGui import QFont, QFontMetricsF
            ref_pt = 80
            f = QFont()
            f.setPointSize(ref_pt)
            br = QFontMetricsF(f).boundingRect(glyph)
            h_px = br.height()
            cls._glyph_pt_cache[glyph] = (ref_pt / h_px) if h_px > 0 else 1.0
        return cls._glyph_pt_cache[glyph]

    def _corner_geometry(self, w, h):
        side      = min(w, h)
        pad       = max(6, int(side * 0.06))
        n         = 5
        max_cover = int(side * 0.28)
        step      = max(6, max_cover // (n + 1))
        return pad, step, step, n

    def _line_endpoints(self, w, h):
        pad, first, step, n = self._corner_geometry(w, h)
        lines = []
        for i in range(n):
            d = first + i * step
            x1, y1 = float(w - pad),     float(h - pad - d)
            x2, y2 = float(w - pad - d), float(h - pad)
            if y1 >= 0 and x2 >= 0:
                lines.append((x1, y1, x2, y2))
        return lines

    def _spawn_particle(self) -> None:
        w, h  = self.width(), self.height()
        lines = self._line_endpoints(w, h)
        if not lines:
            return
        x1, y1, x2, y2 = random.choice(lines)
        t  = random.random()
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        angle_rad = math.radians(random.uniform(95, 175))
        speed     = random.uniform(0.5, 2.0)
        self._particles.append({
            "x": px, "y": py,
            "vx": math.cos(angle_rad) * speed,
            "vy": -math.sin(angle_rad) * speed,
            "life": 1.0, "decay": random.uniform(0.025, 0.050),
            "size": random.uniform(1.2, 2.8),
        })

    def _tick(self) -> None:
        if self._spawning:
            for _ in range(random.randint(1, 2)):
                self._spawn_particle()
        for pt in self._particles:
            pt["x"] += pt["vx"]
            pt["y"] += pt["vy"]
            pt["life"] -= pt["decay"]
        self._particles = [pt for pt in self._particles if pt["life"] > 0]
        if not self._particles and not self._spawning:
            self._anim.stop()
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        clip = QPainterPath()
        clip.addRoundedRect(QRectF(self.rect()), 16.0, 16.0)
        painter.setClipPath(clip)

        # Glyph watermark
        w, h = self.width(), self.height()
        target_px = int(h * 0.50)
        pt_size   = max(20, int(target_px * self._pt_per_px(self._glyph)))
        color = QColor(self._color_str)
        color.setAlphaF(0.07 if not self._hovered else 0.11)
        painter.setPen(color)
        font = painter.font()
        font.setPointSize(pt_size)
        font.setWeight(font.Weight.Bold)
        painter.setFont(font)
        painter.drawText(
            QRectF(w * 0.30, 0, w * 0.70, float(h)),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            self._glyph,
        )

        # Accent bar
        bar_h = max(2, int(h * 0.022))
        alpha = 0.50 if self._hovered else 0.28
        color = QColor(self._color_str)
        color.setAlphaF(alpha)
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QRectF(1, h - bar_h - 1, w - 2, bar_h), 3.0, 3.0)

        # Corner lines
        alpha = 0.65 if self._hovered else 0.38
        color = QColor(self._color_str)
        color.setAlphaF(alpha)
        pen = QPen(color, 1.2, Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for x1, y1, x2, y2 in self._line_endpoints(w, h):
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        # Particles
        if self._particles:
            color = QColor(self._color_str)
            painter.setPen(Qt.PenStyle.NoPen)
            for pt in self._particles:
                color.setAlphaF(max(0.0, pt["life"]) * 0.90)
                painter.setBrush(QBrush(color))
                r = pt["size"] / 2.0
                painter.drawEllipse(QPointF(pt["x"], pt["y"]), r, r)

        painter.end()


# ── Filter bar ────────────────────────────────────────────────────────────────

class _FilterBar(QFrame):

    def __init__(self, on_select, parent=None):
        super().__init__(parent)
        self.setFixedHeight(56)
        self.setStyleSheet("background: transparent; border: none;")
        self._on_select = on_select
        self._buttons: dict[str, QPushButton] = {}

        layout = QHBoxLayout(self)
        layout.setContentsMargins(28, 11, 28, 11)
        layout.setSpacing(10)

        for key in _CAT_ORDER:
            cat = _CAT[key]
            btn = QPushButton(cat["label"])
            btn.setFixedHeight(34)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda checked, k=key: self._activate(k))
            self._buttons[key] = btn
            layout.addWidget(btn)

        layout.addStretch()
        self._activate("todos", notify=False)

    def _pill_style(self, color: str, active: bool) -> str:
        if active:
            return f"""
                QPushButton {{
                    background: {color};
                    border: 1px solid {color};
                    border-radius: 17px;
                    color: {C['bg']};
                    font-size: 11px; font-weight: 800;
                    letter-spacing: 1px; padding: 0 18px;
                }}
            """
        return f"""
            QPushButton {{
                background: {_rgba(color, 0.08)};
                border: 1px solid {_rgba(color, 0.28)};
                border-radius: 17px;
                color: {_rgba(color, 0.65)};
                font-size: 11px; font-weight: 600;
                letter-spacing: 1px; padding: 0 18px;
            }}
            QPushButton:hover {{
                background: {_rgba(color, 0.16)};
                border-color: {_rgba(color, 0.60)};
                color: {color};
            }}
        """

    def _activate(self, key: str, notify: bool = True) -> None:
        for k, btn in self._buttons.items():
            btn.setStyleSheet(self._pill_style(_CAT[k]["color"], k == key))
        if notify:
            self._on_select(key)


# ── Content area ──────────────────────────────────────────────────────────────

class _ContentArea(QFrame):
    _MARGIN_H    = 28
    _SCROLLBAR_W = 8
    _PREF_CARD_W = 380

    def __init__(self, all_items: list[dict], parent=None):
        super().__init__(parent)
        self._all_items    = all_items
        self._active_cat   = "todos"
        self._current_cols = 0
        self._current_rows = 0
        self.setStyleSheet(f"background: {C['bg']}; border: none;")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(_FilterBar(self._on_filter))

        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {C['border']}; border: none;")
        root.addWidget(sep)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{ background: {C['bg']}; border: none; }}
            QScrollBar:vertical {{
                background: transparent; width: 8px; margin: 8px 4px 8px 0;
            }}
            QScrollBar::handle:vertical {{
                background: {C['border_hi']}; border-radius: 4px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
        """)

        scroll_widget = QWidget()
        scroll_widget.setStyleSheet(f"background: {C['bg']};")
        outer = QVBoxLayout(scroll_widget)
        outer.setContentsMargins(28, 18, 28, 32)
        outer.setSpacing(0)

        self._grid_container = QWidget()
        self._grid_container.setStyleSheet("background: transparent;")
        self._grid_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._grid_layout = QGridLayout(self._grid_container)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        self._grid_layout.setHorizontalSpacing(14)
        self._grid_layout.setVerticalSpacing(14)

        outer.addWidget(self._grid_container)
        scroll.setWidget(scroll_widget)
        root.addWidget(scroll, stretch=1)

        self._render_grid("todos")

    def _on_filter(self, cat_key: str) -> None:
        self._active_cat = cat_key
        self._render_grid(cat_key)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        new_cols = self._calc_cols()
        if new_cols != self._current_cols:
            self._current_cols = new_cols
            self._render_grid(self._active_cat)

    def _calc_cols(self) -> int:
        w = self.width()
        if w == 0:
            return _COLS_DEFAULT
        available = w - 2 * self._MARGIN_H - self._SCROLLBAR_W
        spacing   = self._grid_layout.horizontalSpacing()
        cols = max(1, (available + spacing) // (self._PREF_CARD_W + spacing))
        return min(max(cols, 1), _MAX_COLS)

    def _render_grid(self, cat_key: str) -> None:
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)

        cols = self._current_cols or _COLS_DEFAULT
        for col in range(_MAX_COLS + 1):
            self._grid_layout.setColumnStretch(col, 0)
        for row in range(self._current_rows + 1):
            self._grid_layout.setRowStretch(row, 0)
        for col in range(cols):
            self._grid_layout.setColumnStretch(col, 1)

        items = (
            self._all_items if cat_key == "todos"
            else [i for i in self._all_items if i["category"] == cat_key]
        )
        n_rows = max(1, -(-len(items) // cols))

        for idx, item in enumerate(items):
            cat   = item["category"]
            color = _CAT[cat]["color"]
            card  = _ModuleCard(
                title=item["title"],
                subtitle=item["subtitle"],
                callback=item["callback"],
                color=color,
                category=cat,
            )
            self._grid_layout.addWidget(card, idx // cols, idx % cols)

        for row in range(n_rows):
            self._grid_layout.setRowStretch(row, 1)
        self._current_rows = n_rows


# ── Main menu view ────────────────────────────────────────────────────────────

class MainMenuView(QWidget):

    def __init__(self, on_live_qr, on_exit, parent=None):
        super().__init__(parent)

        all_items = [
            {
                "key":      "live_qr",
                "title":    "Live QR",
                "subtitle": "Leitura contínua de QR em câmera única.",
                "callback": on_live_qr,
                "locked":   False,
                "category": "deteccao",
            }
        ]

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(IrisAppBar("Iris"))

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        body.addWidget(_SystemNav(on_exit=on_exit), stretch=1)
        body.addWidget(_ContentArea(all_items), stretch=4)

        root.addLayout(body, stretch=1)
