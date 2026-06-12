"""
StarFieldPanel — reusable animated star-field background widget.

Subclass it and add child widgets normally; they render on top of the
animation.  The 60 fps timer is owned by the panel and stops when the
widget is destroyed.

Usage
-----
    class MySidebar(StarFieldPanel):
        def __init__(self, parent=None):
            super().__init__(accent_edge="right", n_stars=45, parent=parent)
            lay = QVBoxLayout(self)
            lay.addWidget(QLabel("Hello"))

    class MyBrandPanel(StarFieldPanel):
        def __init__(self, parent=None):
            super().__init__(accent_edge="right", parent=parent)
            ...
"""
from __future__ import annotations

import math
import random

from PyQt6.QtCore import Qt, QPointF, QTimer
from PyQt6.QtGui import QColor, QLinearGradient, QPainter, QPen
from PyQt6.QtWidgets import QWidget


# ── Physics constants ─────────────────────────────────────────────────────────

_DEFAULT_N     = 75
_DEFAULT_SPD   = 0.55
_DEFAULT_ANGLE = 35.0   # degrees — left-downward drift
_REPEL_R       = 130    # mouse repulsion radius (px)
_LINE_R        = 110    # constellation line radius (px)
_DECAY         = 0.91   # velocity decay per frame back toward base

_BG_TOP    = "#060A11"
_BG_BOTTOM = "#0C1422"


# ── Internal particle ─────────────────────────────────────────────────────────

class _Star:
    __slots__ = ("x", "y", "r", "alpha", "vx", "vy")

    def __init__(self, w: int, h: int, bvx: float, bvy: float) -> None:
        self.x     = random.uniform(0, w)
        self.y     = random.uniform(0, h)
        self.r     = random.uniform(0.9, 2.4)
        self.alpha = random.randint(55, 200)
        self.vx    = bvx
        self.vy    = bvy


# ── Public component ──────────────────────────────────────────────────────────

class StarFieldPanel(QWidget):
    """
    Animated star-field background widget.

    Parameters
    ----------
    accent_edge : 'right' | 'left' | 'bottom' | 'top' | None
        Which edge gets a faint green gradient accent line.
    bg_top, bg_bottom : str
        Hex colours for the vertical background gradient.
    n_stars : int
        Number of particles (default 75).
    speed : float
        Base drift speed in px/frame (default 0.55).
    angle_deg : float
        Drift direction in degrees, measured clockwise from left-horizontal.
        35° → left-and-slightly-down (default).
    """

    def __init__(
        self,
        *,
        accent_edge: str | None = "right",
        bg_top:      str        = _BG_TOP,
        bg_bottom:   str        = _BG_BOTTOM,
        n_stars:     int        = _DEFAULT_N,
        speed:       float      = _DEFAULT_SPD,
        angle_deg:   float      = _DEFAULT_ANGLE,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)

        rad        = math.radians(angle_deg)
        self._bvx  = -speed * math.cos(rad)
        self._bvy  =  speed * math.sin(rad)

        self._n_stars = n_stars
        self._accent  = accent_edge
        self._bg_top  = QColor(bg_top)
        self._bg_btm  = QColor(bg_bottom)
        self._stars: list[_Star] = []
        self._mouse   = QPointF(-999.0, -999.0)

        self._timer = QTimer(self)
        self._timer.setInterval(16)          # ~60 fps
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._stars:
            self._spawn_stars()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._spawn_stars()

    def _spawn_stars(self) -> None:
        w = max(self.width(),  1)
        h = max(self.height(), 1)
        self._stars = [_Star(w, h, self._bvx, self._bvy) for _ in range(self._n_stars)]

    # ── Mouse ─────────────────────────────────────────────────────────────────

    def mouseMoveEvent(self, event) -> None:
        self._mouse = event.position()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        self._mouse = QPointF(-999.0, -999.0)
        super().leaveEvent(event)

    # ── Physics tick ──────────────────────────────────────────────────────────

    def _tick(self) -> None:
        w  = max(self.width(),  1)
        h  = max(self.height(), 1)
        mx = self._mouse.x()
        my = self._mouse.y()

        for s in self._stars:
            dx   = s.x - mx
            dy   = s.y - my
            dist = math.hypot(dx, dy)

            if 0.1 < dist < _REPEL_R:
                f     = ((_REPEL_R - dist) / _REPEL_R) * 2.8
                s.vx += dx / dist * f
                s.vy += dy / dist * f

            s.vx = s.vx * _DECAY + self._bvx * (1 - _DECAY)
            s.vy = s.vy * _DECAY + self._bvy * (1 - _DECAY)

            spd = math.hypot(s.vx, s.vy)
            if spd > 5.0:
                s.vx = s.vx / spd * 5.0
                s.vy = s.vy / spd * 5.0

            s.x += s.vx
            s.y += s.vy

            if   s.x < -4:     s.x = w + 4
            elif s.x > w + 4:  s.x = -4
            if   s.y < -4:     s.y = h + 4
            elif s.y > h + 4:  s.y = -4

        self.update()

    # ── Paint ─────────────────────────────────────────────────────────────────

    def paintEvent(self, _) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background gradient
        grad = QLinearGradient(0, 0, 0, self.height())
        grad.setColorAt(0.0, self._bg_top)
        grad.setColorAt(1.0, self._bg_btm)
        p.fillRect(self.rect(), grad)

        mx = self._mouse.x()
        my = self._mouse.y()

        # Constellation lines: cursor → nearby stars
        for s in self._stars:
            dx, dy = s.x - mx, s.y - my
            dist   = math.hypot(dx, dy)
            if 0.1 < dist < _LINE_R:
                a   = int(55 * (1 - dist / _LINE_R))
                pen = QPen(QColor(0, 255, 136, a))
                pen.setWidthF(0.6)
                p.setPen(pen)
                p.drawLine(int(mx), int(my), int(s.x), int(s.y))

        # Stars
        p.setPen(Qt.PenStyle.NoPen)
        for s in self._stars:
            dx, dy = s.x - mx, s.y - my
            dist   = math.hypot(dx, dy)
            boost_a = 0
            boost_r = 0.0
            if dist < _REPEL_R:
                t       = 1 - dist / _REPEL_R
                boost_a = int(t * 120)
                boost_r = t * 1.6
            p.setBrush(QColor(0, 255, 136, min(255, s.alpha + boost_a)))
            p.drawEllipse(QPointF(s.x, s.y), s.r + boost_r, s.r + boost_r)

        # Accent edge line
        if self._accent:
            ag = QLinearGradient(0, 0, 0, self.height())
            ag.setColorAt(0.0, QColor(0, 255, 136,  0))
            ag.setColorAt(0.5, QColor(0, 255, 136, 50))
            ag.setColorAt(1.0, QColor(0, 255, 136,  0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(ag)
            e = self._accent
            if   e == "right":  p.drawRect(self.width() - 1, 0, 1, self.height())
            elif e == "left":   p.drawRect(0, 0, 1, self.height())
            elif e == "bottom": p.drawRect(0, self.height() - 1, self.width(), 1)
            elif e == "top":    p.drawRect(0, 0, self.width(), 1)

        p.end()
