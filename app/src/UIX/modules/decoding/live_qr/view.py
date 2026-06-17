"""
UIX/modules/decoding/live_qr/view.py
======================================
Tela de detecção em tempo real via webcam — v3 "viewfinder".

Estados do instrumento (a íris conta a história, o texto confirma):
    idle    → íris fechada, overlay em espera
    opening → íris respirando, câmera inicializando (sem tela preta)
    live    → íris aberta, feed com retículo de viewfinder

Cor é informação: âmbar = instrumento/atenção; verde = decodificado.

Teclado: S inicia/para · E edges · Ctrl+L limpa leituras.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Any

import cv2
cv2.setUseOptimized(True)
cv2.setNumThreads(max(1, (__import__('os').cpu_count() or 2) // 2))
import numpy as np
from PyQt6.QtCore import (
    Qt,
    QThread,
    pyqtSignal,
    QTimer,
    QMutex,
    QMutexLocker,
    QRect,
)
from PyQt6.QtGui import (
    QColor,
    QCursor,
    QImage,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStackedLayout,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.src.UIX.components.shared import (
    C, F_BODY, F_DATA, F_DISPLAY, IrisAperture, IrisAppBar, IrisButton,
)
from app.src.engine.modules.decoding.live_qr.decoder import QrDecoder
from app.src.engine.modules.decoding.live_qr.detector import Detection, IrisDetector
from app.src.engine.modules.decoding.live_qr.tracker import GhostDetection, QrTracker, TrackedDetection
from app.src.infrastructure.video.camera import SingleCameraManager
from app.src.infrastructure.video.enhance import EdgeEnhancer
from app.src.infrastructure.websocket import QrWebSocketServer


# ─────────────────────────────────────────────────────────────────────────────
# Constantes visuais
# ─────────────────────────────────────────────────────────────────────────────

# BGR — âmbar (localizado, ainda sem leitura) / verde fósforo (decodificado)
_BOX_COLOR     = (84, 180, 255)
_BOX_HI_COLOR  = (128, 222, 74)
_GHOST_COLOR   = (255, 60, 200)      # BGR magenta elétrico — predição de velocidade
_LABEL_INK     = (19, 14, 11)        # BGR de #0B0E13
_BOX_THICKNESS = 2
_LABEL_FONT    = cv2.FONT_HERSHEY_SIMPLEX
_HIST_MAX      = 50
_RENDER_INTERVAL_MS = int((1 / 33) * 1000)  # ~30 ms
# Distância mínima (px) entre predição e detecção para suprimir ghost estático.
_GHOST_MIN_DIST = 5.0

_SCANLINE_STEP  = 0.004   # fração da altura por tick de 16 ms
_FLASH_MS       = 380     # ms — flash do card ao repetir leitura
_REARM_S        = 1.0     # s — ausência mínima para contar novo "bip"

# Definição dos controles de câmera expostos na sidebar.
# (label, CAP_PROP_*, min, max, default)
_CAM_CTRL_DEFS = [
    ("BRILHO",    cv2.CAP_PROP_BRIGHTNESS,  0,   255, 128),
    ("CONTRASTE", cv2.CAP_PROP_CONTRAST,    0,   255, 128),
    ("SATURAÇÃO", cv2.CAP_PROP_SATURATION,  0,   255, 128),
    ("NITIDEZ",   cv2.CAP_PROP_SHARPNESS,   0,   255, 128),
    ("GANHO",     cv2.CAP_PROP_GAIN,        0,   255,  64),
    ("EXPOSIÇÃO", cv2.CAP_PROP_EXPOSURE,  -13,    0,  -5),
    ("FOCO",      cv2.CAP_PROP_FOCUS,       0,   255, 100),
    ("ZOOM",      cv2.CAP_PROP_ZOOM,       100,  800, 100),
]

# Propriedades de modo automático: prop_id → (auto_prop_id, val_auto, val_manual)
# DSHOW: CAP_PROP_AUTO_EXPOSURE = 3 (auto) / 1 (manual)
_CAM_AUTO: Dict[int, tuple] = {
    cv2.CAP_PROP_EXPOSURE: (cv2.CAP_PROP_AUTO_EXPOSURE, 3, 1),
    cv2.CAP_PROP_FOCUS:    (cv2.CAP_PROP_AUTOFOCUS,     1, 0),
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de anotação (usados por ambos os workers via pull-render)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_ghost_bbox(
    img: np.ndarray,
    cx: float, cy: float,
    w: float, h: float,
    track_id: int,
    show_label: bool = False,
) -> None:
    """Draw a single ghost (predicted) bbox onto `img` (drawn on overlay, blended later)."""
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)
    c  = _GHOST_COLOR

    cv2.rectangle(img, (x1, y1), (x2, y2), c, 1)

    corner = max(int(min(w, h) // 6), 6)
    for bx, by, dx, dy in [
        (x1, y1, +1, +1), (x2, y1, -1, +1),
        (x2, y2, -1, -1), (x1, y2, +1, -1),
    ]:
        cv2.line(img, (bx, by), (bx + dx * corner, by), c, 1)
        cv2.line(img, (bx, by), (bx, by + dy * corner), c, 1)

    if show_label:
        lbl = f"#{track_id}"
        sc, th = 0.38, 1
        (lw, lh), _ = cv2.getTextSize(lbl, _LABEL_FONT, sc, th)
        cv2.rectangle(img, (x1, y1), (x1 + lw + 6, y1 + lh + 6), c, cv2.FILLED)
        cv2.putText(img, lbl, (x1 + 3, y1 + lh + 3), _LABEL_FONT, sc, _LABEL_INK, th, cv2.LINE_AA)


def _annotate(
    frame: np.ndarray,
    results: List[Dict[str, Any]],
    ghosts: Optional[List[GhostDetection]] = None,
) -> np.ndarray:
    """
    Draw detections and velocity-prediction ghost boxes.

    Order:
      1. Ghost boxes drawn directly at full opacity (branco-acinzentado, 1 px).
         Skipped per-detection if predicted centroid < _GHOST_MIN_DIST from actual.
      2. Real detection boxes on top.
    """
    # ── 1. Ghost boxes ─────────────────────────────────────────────────────────
    for res in results:
        d    = res["detection"]
        pcx  = d.pred_cx
        pcy  = d.pred_cy
        acx  = (d.x1 + d.x2) / 2.0
        acy  = (d.y1 + d.y2) / 2.0
        dist = ((pcx - acx) ** 2 + (pcy - acy) ** 2) ** 0.5
        if dist >= _GHOST_MIN_DIST:
            _draw_ghost_bbox(frame, pcx, pcy, float(d.width), float(d.height),
                             d.track_id, show_label=False)

    for g in (ghosts or []):
        _draw_ghost_bbox(frame, g.pred_cx, g.pred_cy,
                         g.est_width, g.est_height,
                         g.track_id, show_label=True)

    # ── 2. Real detection boxes ────────────────────────────────────────────────
    for res in results:
        d        = res["detection"]
        raw_text = res.get("text")
        track_id = res.get("track_id", 0)
        color    = _BOX_HI_COLOR if raw_text else _BOX_COLOR

        cv2.rectangle(frame, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), color, _BOX_THICKNESS)

        corner = min(d.width, d.height) // 5
        for bx, by, dx, dy in [
            (int(d.x1), int(d.y1), +1, +1),
            (int(d.x2), int(d.y1), -1, +1),
            (int(d.x2), int(d.y2), -1, -1),
            (int(d.x1), int(d.y2), +1, -1),
        ]:
            cv2.line(frame, (bx, by), (bx + dx * corner, by), color, _BOX_THICKNESS + 1)
            cv2.line(frame, (bx, by), (bx, by + dy * corner), color, _BOX_THICKNESS + 1)

        id_lbl = f"#{track_id}"
        id_scale, id_thick = 0.40, 1
        (iw, ih), _ = cv2.getTextSize(id_lbl, _LABEL_FONT, id_scale, id_thick)
        ix1, iy1 = int(d.x1), int(d.y1)
        cv2.rectangle(frame, (ix1, iy1), (ix1 + iw + 6, iy1 + ih + 6), color, cv2.FILLED)
        cv2.putText(frame, id_lbl, (ix1 + 3, iy1 + ih + 3),
                    _LABEL_FONT, id_scale, _LABEL_INK, id_thick, cv2.LINE_AA)

        if raw_text:
            content = str(raw_text)
            if len(content) > 20:
                content = content[:17] + "..."
        else:
            content = "LENDO..."
        scale, thick = 0.42, 1
        (tw, th), _ = cv2.getTextSize(content, _LABEL_FONT, scale, thick)
        ty = int(d.y1) - 8 if int(d.y1) - 8 > th else int(d.y2) + th + 8
        cv2.rectangle(frame,
                      (int(d.x1), ty - th - 4), (int(d.x1) + tw + 8, ty + 4),
                      color, cv2.FILLED)
        cv2.putText(frame, content, (int(d.x1) + 4, ty),
                    _LABEL_FONT, scale, _LABEL_INK, thick, cv2.LINE_AA)

    return frame


def _merge_with_text(
    detections: List[TrackedDetection],
    decode_results: List[Dict[str, Any]],
    max_dist: int = 100,
) -> List[Dict[str, Any]]:
    """Une TrackedDetections frescas com textos do decoder por proximidade de centróide.

    Prioriza match por track_id; cai para distância se o track_id não estiver
    nos resultados do decoder (decoder é mais lento, pode estar stale).
    """
    # Índice rápido: track_id → texto decodificado
    id_to_text: Dict[int, Optional[str]] = {}
    for r in decode_results:
        tid = r.get("track_id")
        if tid is not None:
            id_to_text[tid] = r.get("text")

    merged = []
    for d in detections:
        text: Optional[str] = None

        # 1. Match direto por track_id
        if d.track_id in id_to_text:
            text = id_to_text[d.track_id]
        else:
            # 2. Fallback: centróide mais próximo
            cx   = (d.x1 + d.x2) // 2
            cy   = (d.y1 + d.y2) // 2
            best = max_dist * max_dist
            for r in decode_results:
                rd    = r["detection"]
                rdcx  = (rd.x1 + rd.x2) // 2
                rdcy  = (rd.y1 + rd.y2) // 2
                dist2 = (rdcx - cx) ** 2 + (rdcy - cy) ** 2
                if dist2 < best:
                    best = dist2
                    text = r.get("text")

        merged.append({"detection": d, "text": text, "track_id": d.track_id})
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Worker 1 — Tracking (só YOLO, velocidade máxima)
# ─────────────────────────────────────────────────────────────────────────────

class _TrackingWorker(QThread):
    """Roda YOLO + QrTracker e emite TrackedDetections imediatamente.

    O QrTracker atribui IDs estáveis entre frames sem depender de decodificação.
    A UI vê posição e ID atualizados a cada inferência (~20–30 fps com CPU).
    """

    boxes_ready = pyqtSignal(object, object, object)  # (List[TrackedDetection], List[GhostDetection], frame)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._engine: Optional[IrisDetector] = None
        self._running = True
        self._mutex   = QMutex()
        self._pending: Optional[np.ndarray] = None

    def submit_frame(self, frame: np.ndarray) -> None:
        with QMutexLocker(self._mutex):
            self._pending = frame   # caller already owns this buffer

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        self._engine  = IrisDetector()
        qr_tracker    = QrTracker(max_missed_s=0.6)
        while self._running:
            with QMutexLocker(self._mutex):
                frame         = self._pending
                self._pending = None
            if frame is None:
                self.msleep(10)
                continue
            try:
                detections      = self._engine.detect(frame)
                tracked, ghosts = qr_tracker.update(detections)
                self.boxes_ready.emit(tracked, ghosts, frame)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Worker 2 — Decoding (só QR decode, LIFO, assíncrono)
# ─────────────────────────────────────────────────────────────────────────────

class _DecodingWorker(QThread):
    """Recebe (frame, detecções) do tracker e tenta decodificar os QR Codes.

    Roda em paralelo com o tracker: o tracker nunca espera o decode.
    Buffer LIFO: se o decoder ainda está ocupado quando chega um novo par,
    o par antigo é descartado — só importa o mais recente.
    """

    decode_ready = pyqtSignal(object)   # List[Dict[str, Any]]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._decoder  = QrDecoder()
        self._running  = True
        self._mutex    = QMutex()
        self._pending: Optional[tuple] = None   # (frame, List[TrackedDetection])

    def submit(self, frame: np.ndarray, detections: List[TrackedDetection]) -> None:
        with QMutexLocker(self._mutex):
            if not detections:
                self._pending = None
                return
            self._pending = (frame, list(detections))

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        while self._running:
            with QMutexLocker(self._mutex):
                item          = self._pending
                self._pending = None
            if item is None:
                self.msleep(15)
                continue
            frame, detections = item
            try:
                boxes = [
                    np.array(
                        [[d.x1, d.y1], [d.x2, d.y1], [d.x2, d.y2], [d.x1, d.y2]],
                        dtype=np.int32,
                    )
                    for d in detections
                ]
                hints = [
                    (d.pred1_cx, d.pred1_cy, float(d.width), float(d.height), d.vel_mag)
                    for d in detections
                ]
                decoded = self._decoder.decode(frame, boxes, hints=hints)
                results = [
                    {"detection": d, "text": text, "track_id": d.track_id}
                    for d, (text, _, _, _) in zip(detections, decoded)
                ]
                self.decode_ready.emit(results)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Feed com retículo de viewfinder + linha de varredura
# ─────────────────────────────────────────────────────────────────────────────

class _ViewfinderFeed(QLabel):
    """
    QLabel da câmera com sobreposições de instrumento óptico:
    colchetes de retículo nos cantos + linha de varredura âmbar.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scan_pos = 0.0
        self._active   = False
        self._timer    = QTimer(self)
        self._timer.setInterval(16)   # ~60 fps
        self._timer.timeout.connect(self._tick)

    def start_scan(self) -> None:
        self._active = True
        self._timer.start()

    def stop_scan(self) -> None:
        self._active = False
        self._timer.stop()
        self.update()

    def _tick(self) -> None:
        self._scan_pos = (self._scan_pos + _SCANLINE_STEP) % 1.0
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if not self._active or self.width() <= 0 or self.height() <= 0:
            return

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Linha de varredura âmbar com halo
        y = int(self._scan_pos * self.height())
        top_y = max(0, y - 20)
        bot_y = min(self.height(), y + 20)
        grad  = QLinearGradient(0, top_y, 0, bot_y)
        grad.setColorAt(0.0, QColor(255, 180, 84, 0))
        grad.setColorAt(0.5, QColor(255, 180, 84, 34))
        grad.setColorAt(1.0, QColor(255, 180, 84, 0))
        p.fillRect(QRect(0, top_y, self.width(), bot_y - top_y), grad)
        pen = QPen(QColor(255, 180, 84, 70))
        pen.setWidth(1)
        p.setPen(pen)
        p.drawLine(0, y, self.width(), y)

        # Colchetes de retículo nos quatro cantos
        pen = QPen(QColor(255, 180, 84, 120), 2.0,
                   Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        w, h = self.width(), self.height()
        m, L = 14, 26
        for x, yy, dx, dy in (
            (m, m, 1, 1), (w - m, m, -1, 1),
            (w - m, h - m, -1, -1), (m, h - m, 1, -1),
        ):
            p.drawLine(x, yy, x + dx * L, yy)
            p.drawLine(x, yy, x, yy + dy * L)
        p.end()


# ─────────────────────────────────────────────────────────────────────────────
# Badge de contagem de bips
# ─────────────────────────────────────────────────────────────────────────────

class _CountBadge(QLabel):
    """Pílula ×N à direita do card — aparece a partir da segunda leitura."""

    _S_NORMAL = (
        f"color: {C['primary']};"
        "background: rgba(255,180,84,0.12);"
        "border: 1px solid rgba(255,180,84,0.45);"
        "border-radius: 9px;"
        f"font-family: {F_DATA};"
        "font-size: 10px; font-weight: 700;"
        "padding: 0px 7px;"
    )
    _S_FLASH = (
        f"color: {C['bg']};"
        f"background: {C['primary']};"
        f"border: 1px solid {C['primary']};"
        "border-radius: 9px;"
        f"font-family: {F_DATA};"
        "font-size: 10px; font-weight: 700;"
        "padding: 0px 7px;"
    )

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedHeight(19)
        self.setMinimumWidth(38)
        self.setStyleSheet(self._S_NORMAL)
        self.hide()

    def pulse(self) -> None:
        self.setStyleSheet(self._S_FLASH)
        QTimer.singleShot(_FLASH_MS, self._restore)

    def _restore(self) -> None:
        try:
            self.setStyleSheet(self._S_NORMAL)
        except RuntimeError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Card do registro de leituras
# ─────────────────────────────────────────────────────────────────────────────

class _HistoryCard(QFrame):
    _S_NORMAL = (
        f"#HistCard {{ background:{C['card']}; border:1px solid {C['border']};"
        f" border-left:3px solid {C['success']}; border-radius:4px; }}"
    )
    _S_FLASH = (
        f"#HistCard {{ background:{C['card_hover']}; border:1px solid {C['primary']};"
        f" border-left:3px solid {C['primary']}; border-radius:4px; }}"
    )

    def __init__(self, index: int, text_value: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("HistCard")
        self.text_value = str(text_value)
        self._count = 1
        self.setStyleSheet(self._S_NORMAL)
        self.setFixedHeight(38)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 0, 8, 0)
        lay.setSpacing(8)

        idx = QLabel(f"{index:03d}")
        idx.setFixedWidth(32)
        idx.setStyleSheet(
            f"color:{C['text_muted']}; font-family:{F_DATA};"
            "font-size:9px; font-weight:700; background:transparent;"
        )
        lay.addWidget(idx)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFixedWidth(1)
        sep.setFixedHeight(18)
        sep.setStyleSheet(f"background:{C['border_hi']}; border:none;")
        lay.addWidget(sep)

        display = self.text_value[:30] + "…" if len(self.text_value) > 30 else self.text_value
        val = QLabel(display)
        val.setToolTip(self.text_value)
        val.setStyleSheet(
            f"color:{C['text']}; font-family:{F_DATA}; font-size:10px;"
            " background:transparent;"
        )
        lay.addWidget(val, stretch=1)

        self._badge = _CountBadge()
        lay.addWidget(self._badge)

    def increment(self) -> None:
        self._count += 1
        self._badge.setText(f"×{self._count}")
        self._badge.show()
        self._badge.pulse()
        self.setStyleSheet(self._S_FLASH)
        QTimer.singleShot(_FLASH_MS, self._restore)

    def _restore(self) -> None:
        try:
            self.setStyleSheet(self._S_NORMAL)
        except RuntimeError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Slider individual de câmera
# ─────────────────────────────────────────────────────────────────────────────

class _SliderRow(QWidget):
    """Label + valor + QSlider horizontal + toggle AUTO opcional."""

    prop_changed = pyqtSignal(int, float)   # (prop_id, value)
    auto_changed = pyqtSignal(int, float)   # (auto_prop_id, value)

    def __init__(self, label: str, prop_id: int, lo: int, hi: int,
                 default: int, parent=None) -> None:
        super().__init__(parent)
        self._prop_id  = prop_id
        self._lo       = lo
        self._hi       = hi
        self._default  = default
        self._auto_info = _CAM_AUTO.get(prop_id)
        self._is_auto   = False

        self.setStyleSheet("background:transparent;")
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 8, 12, 6)
        root.setSpacing(4)

        # ── header: label · AUTO · valor ─────────────────────────────────────
        hdr = QHBoxLayout()
        hdr.setContentsMargins(0, 0, 0, 0)
        hdr.setSpacing(6)

        lbl_w = QLabel(label)
        lbl_w.setStyleSheet(
            f"color:{C['text_muted']}; font-family:{F_DISPLAY};"
            "font-size:9px; font-weight:700; letter-spacing:1.5px;"
        )
        hdr.addWidget(lbl_w, stretch=1)

        if self._auto_info:
            self._auto_btn = QPushButton("AUTO")
            self._auto_btn.setFixedSize(40, 17)
            self._auto_btn.setCheckable(True)
            self._auto_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            self._auto_btn.clicked.connect(self._on_auto_toggled)
            self._apply_auto_style(False)
            hdr.addWidget(self._auto_btn)

        self._val_lbl = QLabel(str(default))
        self._val_lbl.setFixedWidth(32)
        self._val_lbl.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._val_lbl.setStyleSheet(
            f"color:{C['primary']}; font-family:{F_DATA};"
            "font-size:11px; font-weight:700;"
        )
        hdr.addWidget(self._val_lbl)
        root.addLayout(hdr)

        # ── slider ────────────────────────────────────────────────────────────
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(lo, hi)
        self._slider.setValue(default)
        self._slider.setFixedHeight(20)
        self._slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 3px;
                background: rgba(255,180,84,0.15);
                border-radius: 2px;
            }}
            QSlider::sub-page:horizontal {{
                background: rgba(255,180,84,0.65);
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                width: 12px; height: 12px;
                background: {C['primary']};
                border-radius: 6px;
                margin: -5px 0;
            }}
            QSlider::groove:horizontal:disabled {{
                background: rgba(255,255,255,0.06);
            }}
            QSlider::sub-page:horizontal:disabled {{
                background: rgba(255,255,255,0.1);
            }}
            QSlider::handle:horizontal:disabled {{
                background: {C['text_muted']};
            }}
        """)
        self._slider.valueChanged.connect(lambda v: self._val_lbl.setText(str(v)))
        self._slider.sliderReleased.connect(self._on_released)
        root.addWidget(self._slider)

    # ── slots ─────────────────────────────────────────────────────────────────

    def _on_released(self) -> None:
        self.prop_changed.emit(self._prop_id, float(self._slider.value()))

    def _on_auto_toggled(self, checked: bool) -> None:
        self._is_auto = checked
        self._apply_auto_style(checked)
        self._slider.setEnabled(not checked)
        if self._auto_info:
            auto_prop_id, auto_on, auto_off = self._auto_info
            self.auto_changed.emit(auto_prop_id, float(auto_on if checked else auto_off))

    def _apply_auto_style(self, active: bool) -> None:
        if active:
            self._auto_btn.setStyleSheet(
                f"background:{C['primary']}; color:{C['bg']}; border:none;"
                f" border-radius:3px; font-family:{F_DISPLAY}; font-size:8px;"
                " font-weight:700; letter-spacing:1px;"
            )
        else:
            self._auto_btn.setStyleSheet(
                f"background:transparent; color:{C['text_muted']};"
                f" border:1px solid {C['border']}; border-radius:3px;"
                f" font-family:{F_DISPLAY}; font-size:8px;"
                " font-weight:700; letter-spacing:1px;"
            )

    # ── API ───────────────────────────────────────────────────────────────────

    def set_value(self, value: float) -> None:
        v = int(max(self._lo, min(self._hi, round(value))))
        self._slider.blockSignals(True)
        self._slider.setValue(v)
        self._slider.blockSignals(False)
        self._val_lbl.setText(str(v))

    def reset_to_default(self) -> None:
        self.set_value(self._default)
        self.prop_changed.emit(self._prop_id, float(self._default))
        if self._auto_info and self._is_auto:
            self._is_auto = False
            self._auto_btn.setChecked(False)
            self._apply_auto_style(False)
            self._slider.setEnabled(True)
            auto_prop_id, _, auto_off = self._auto_info
            self.auto_changed.emit(auto_prop_id, float(auto_off))

    @property
    def prop_id(self) -> int:
        return self._prop_id


# ─────────────────────────────────────────────────────────────────────────────
# Painel de controles de câmera
# ─────────────────────────────────────────────────────────────────────────────

class _CameraControls(QFrame):
    """Scroll de _SliderRow com botões de sincronização e reset."""

    def __init__(self, cam_getter, parent=None) -> None:
        super().__init__(parent)
        self._cam_getter = cam_getter
        self._rows: List[_SliderRow] = []

        self.setStyleSheet("background:transparent;")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── botão de leitura ──────────────────────────────────────────────────
        sync_btn = QPushButton("↺  LER DA CÂMERA")
        sync_btn.setFixedHeight(36)
        sync_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        sync_btn.setToolTip("Sincronizar sliders com os valores atuais do driver")
        sync_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                border-bottom: 1px solid {C['border']};
                color: {C['text_muted']};
                font-family: {F_DISPLAY}; font-size: 10px;
                font-weight: 600; letter-spacing: 2px;
            }}
            QPushButton:hover {{ color:{C['primary']}; }}
            QPushButton:pressed {{ background:rgba(255,180,84,0.05); color:{C['primary']}; }}
        """)
        sync_btn.clicked.connect(self.sync_from_camera)
        root.addWidget(sync_btn)

        # ── sliders ───────────────────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background:transparent; border:none;")

        container = QWidget()
        container.setStyleSheet("background:transparent;")
        vl = QVBoxLayout(container)
        vl.setContentsMargins(0, 4, 0, 4)
        vl.setSpacing(0)

        for label, prop_id, lo, hi, default in _CAM_CTRL_DEFS:
            row = _SliderRow(label, prop_id, lo, hi, default)
            row.prop_changed.connect(self._on_prop)
            row.auto_changed.connect(self._on_auto)
            self._rows.append(row)
            vl.addWidget(row)
            div = QFrame()
            div.setFixedHeight(1)
            div.setStyleSheet(f"background:{C['border']}; border:none;")
            vl.addWidget(div)

        vl.addStretch()
        scroll.setWidget(container)
        root.addWidget(scroll, stretch=1)

        # ── botão de reset ────────────────────────────────────────────────────
        reset_btn = QPushButton("⌫  RESTAURAR PADRÕES")
        reset_btn.setFixedHeight(34)
        reset_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        reset_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; border: none;
                border-top: 1px solid {C['border']};
                color: {C['text_muted']};
                font-family: {F_DISPLAY}; font-size: 10px;
                font-weight: 600; letter-spacing: 2px;
            }}
            QPushButton:hover {{ color:{C['danger']}; background:rgba(255,122,122,0.06); }}
            QPushButton:pressed {{ background:rgba(255,122,122,0.12); }}
        """)
        reset_btn.clicked.connect(self._reset_all)
        root.addWidget(reset_btn)

    # ── persistência ──────────────────────────────────────────────────────────

    @staticmethod
    def _settings_path() -> "Path":
        import os
        from pathlib import Path
        base = Path(os.environ.get("APPDATA") or Path.home())
        p = base / "Iris" / "camera_settings.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _save(self) -> None:
        import json
        data: dict = {"props": {}, "auto": {}}
        for row in self._rows:
            data["props"][str(row.prop_id)] = float(row._slider.value())
            if row._auto_info:
                auto_prop_id, auto_on, auto_off = row._auto_info
                data["auto"][str(auto_prop_id)] = float(auto_on if row._is_auto else auto_off)
        try:
            self._settings_path().write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def load_and_apply(self) -> None:
        """Lê configurações salvas e aplica na câmera + sliders."""
        import json
        path = self._settings_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return

        cam = self._cam_getter()
        props = data.get("props", {})
        autos = data.get("auto", {})

        for row in self._rows:
            # Auto mode primeiro
            if row._auto_info:
                auto_prop_id, auto_on, _ = row._auto_info
                saved_auto = autos.get(str(auto_prop_id))
                if saved_auto is not None:
                    is_auto = (int(round(saved_auto)) == auto_on)
                    if is_auto != row._is_auto:
                        row._auto_btn.setChecked(is_auto)
                        row._on_auto_toggled(is_auto)
                    if cam is not None:
                        cam.set_property(auto_prop_id, float(saved_auto))
            # Valor numérico
            saved_val = props.get(str(row.prop_id))
            if saved_val is not None:
                v = float(saved_val)
                if row._lo <= v <= row._hi:
                    row.set_value(v)
                    if cam is not None:
                        cam.set_property(row.prop_id, v)

    # ── slots ─────────────────────────────────────────────────────────────────

    def _on_prop(self, prop_id: int, value: float) -> None:
        cam = self._cam_getter()
        if cam is not None:
            cam.set_property(prop_id, value)
        self._save()

    def _on_auto(self, auto_prop_id: int, value: float) -> None:
        cam = self._cam_getter()
        if cam is not None:
            cam.set_property(auto_prop_id, value)
        self._save()

    def sync_from_camera(self) -> None:
        """Lê valores actuais do driver e atualiza todos os sliders."""
        cam = self._cam_getter()
        if cam is None:
            return
        for row in self._rows:
            # Sincroniza estado AUTO primeiro (exposure, focus)
            if row._auto_info:
                auto_prop_id, auto_on, _ = row._auto_info
                aval = cam.get_property(auto_prop_id)
                if aval is not None:
                    is_auto = (int(round(aval)) == auto_on)
                    if is_auto != row._is_auto:
                        row._auto_btn.setChecked(is_auto)
                        row._on_auto_toggled(is_auto)
            # Sincroniza valor numérico
            val = cam.get_property(row.prop_id)
            if val is not None and row._lo <= val <= row._hi:
                row.set_value(val)

    def _reset_all(self) -> None:
        for row in self._rows:
            row.reset_to_default()
        self._save()


# ─────────────────────────────────────────────────────────────────────────────
# Painel lateral — registro de leituras
# ─────────────────────────────────────────────────────────────────────────────

class _SidePanel(QFrame):
    def __init__(self, cam_getter=None, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(290)
        self.setObjectName("SidePanel")
        self.setStyleSheet(f"""
            #SidePanel {{
                background: {C['surface']};
                border-left: 1px solid {C['border']};
            }}
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Tab bar ──────────────────────────────────────────────────────────
        tab_bar = QFrame()
        tab_bar.setObjectName("SPTabBar")
        tab_bar.setFixedHeight(44)
        tab_bar.setStyleSheet(
            f"#SPTabBar {{ background:{C['card']}; border-bottom:1px solid {C['border']}; }}"
        )
        tl = QHBoxLayout(tab_bar)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(0)

        self._tab_reads = QPushButton("LEITURAS")
        self._tab_cam   = QPushButton("CÂMERA")
        for btn in (self._tab_reads, self._tab_cam):
            btn.setFixedHeight(44)
            btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        tl.addWidget(self._tab_reads, stretch=1)
        sep_v = QFrame()
        sep_v.setFixedWidth(1)
        sep_v.setStyleSheet(f"background:{C['border']}; border:none;")
        tl.addWidget(sep_v)
        tl.addWidget(self._tab_cam, stretch=1)
        root.addWidget(tab_bar)

        # ── Stacked pages ────────────────────────────────────────────────────
        self._pages = QStackedWidget()
        root.addWidget(self._pages, stretch=1)

        # ── Page 0: Leituras ─────────────────────────────────────────────────
        p_reads = QWidget()
        p_reads.setStyleSheet("background:transparent;")
        pl = QVBoxLayout(p_reads)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(0)

        stats = QFrame()
        stats.setObjectName("SPS")
        stats.setFixedHeight(58)
        stats.setStyleSheet(
            f"#SPS {{ background:{C['surface']}; border-bottom:1px solid {C['border']}; }}"
        )
        sl = QHBoxLayout(stats)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(0)
        self._stat_reads = self._make_stat("LEITURAS", "0", C["success"])
        self._stat_scene = self._make_stat("NA CENA",  "0", C["secondary"])
        self._stat_fps   = self._make_stat("FPS",      "—", C["text_muted"])
        for i, w in enumerate([self._stat_reads, self._stat_scene, self._stat_fps]):
            sl.addWidget(w, stretch=1)
            if i < 2:
                d = QFrame(); d.setFixedWidth(1)
                d.setStyleSheet(f"background:{C['border']}; border:none;")
                sl.addWidget(d)
        pl.addWidget(stats)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background:transparent; border:none;")
        self._hist_container = QWidget()
        self._hist_container.setStyleSheet("background:transparent;")
        self._hist_layout = QVBoxLayout(self._hist_container)
        self._hist_layout.setContentsMargins(8, 8, 8, 8)
        self._hist_layout.setSpacing(4)
        self._empty_lbl = QLabel(
            "Nenhuma leitura ainda.\nAproxime um código QR da câmera."
        )
        self._empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_lbl.setWordWrap(True)
        self._empty_lbl.setStyleSheet(
            f"color:{C['text_muted']}; font-family:{F_BODY};"
            "font-size:11px; padding:22px 10px; background:transparent;"
        )
        self._hist_layout.addWidget(self._empty_lbl)
        self._hist_layout.addStretch()
        scroll.setWidget(self._hist_container)
        pl.addWidget(scroll, stretch=1)

        clear_btn = QPushButton("⌫  LIMPAR LEITURAS")
        clear_btn.setFixedHeight(34)
        clear_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        clear_btn.setToolTip("Limpar registro de leituras (Ctrl+L)")
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; border: none;
                border-top: 1px solid {C['border']};
                color: {C['text_muted']};
                font-family: {F_DISPLAY}; font-size: 10px;
                font-weight: 600; letter-spacing: 2px;
            }}
            QPushButton:hover {{ color:{C['danger']}; background:rgba(255,122,122,0.06); }}
            QPushButton:focus {{ color:{C['danger']}; border-top:1px solid {C['danger']}; }}
            QPushButton:pressed {{ color:{C['danger']}; background:rgba(255,122,122,0.12); }}
        """)
        clear_btn.clicked.connect(self.clear_history)
        pl.addWidget(clear_btn)

        self._pages.addWidget(p_reads)   # index 0

        # ── Page 1: Câmera ───────────────────────────────────────────────────
        self._cam_controls = _CameraControls(cam_getter or (lambda: None))
        self._pages.addWidget(self._cam_controls)   # index 1

        # ── Tab wiring ────────────────────────────────────────────────────────
        self._tab_reads.clicked.connect(lambda: self._switch_tab(0))
        self._tab_cam.clicked.connect(lambda: self._switch_tab(1))
        self._apply_tab_styles(0)

        # ── State ─────────────────────────────────────────────────────────────
        self._history_count = 0
        self._seen: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _make_stat(label: str, value: str, color: str) -> QWidget:
        w = QWidget()
        w.setStyleSheet("background:transparent;")
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, 6, 0, 6)
        vl.setSpacing(1)
        vl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        val = QLabel(value)
        val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        val.setObjectName("val")
        val.setStyleSheet(
            f"color:{color}; font-family:{F_DATA};"
            "font-size:16px; font-weight:700; background:transparent;"
        )
        lbl = QLabel(label)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(
            f"color:{C['text_muted']}; font-family:{F_DISPLAY};"
            "font-size:9px; letter-spacing:1.5px; background:transparent;"
        )
        vl.addWidget(val)
        vl.addWidget(lbl)
        return w

    # ── Tab helpers ───────────────────────────────────────────────────────────

    def _switch_tab(self, idx: int) -> None:
        self._pages.setCurrentIndex(idx)
        self._apply_tab_styles(idx)
        if idx == 1:
            self._cam_controls.sync_from_camera()

    def _apply_tab_styles(self, active: int) -> None:
        _a = f"""
            QPushButton {{
                background: transparent; border: none;
                border-bottom: 2px solid {C['primary']};
                color: {C['primary']};
                font-family: {F_DISPLAY}; font-size: 10px;
                font-weight: 700; letter-spacing: 3px;
            }}
        """
        _i = f"""
            QPushButton {{
                background: transparent; border: none;
                border-bottom: 2px solid transparent;
                color: {C['text_muted']};
                font-family: {F_DISPLAY}; font-size: 10px;
                font-weight: 600; letter-spacing: 3px;
            }}
            QPushButton:hover {{ color:{C['text']}; }}
        """
        self._tab_reads.setStyleSheet(_a if active == 0 else _i)
        self._tab_cam.setStyleSheet(_a if active == 1 else _i)

    def _get_stat_val(self, w: QWidget) -> QLabel:
        return w.findChild(QLabel, "val")

    def update_stats(self, in_scene: int, fps: float) -> None:
        self._get_stat_val(self._stat_scene).setText(str(in_scene))
        self._get_stat_val(self._stat_fps).setText(f"{fps:.1f}")

    def add_detection(self, text_value: str) -> None:
        now      = time.monotonic()
        text_str = str(text_value)

        entry = self._seen.get(text_str)
        if entry is not None:
            gap = now - entry["last_ts"]
            entry["last_ts"] = now
            if gap >= _REARM_S:
                # Código saiu de cena e voltou: novo bip → contador ×N
                self._history_count += 1
                self._sync_counters()
                entry["card"].increment()
                self._heartbeat()
            return

        # Primeira leitura deste código
        self._history_count += 1
        card = _HistoryCard(self._history_count, text_str)
        self._seen[text_str] = {"card": card, "last_ts": now}
        self._empty_lbl.hide()
        self._hist_layout.insertWidget(1, card)   # 0 = empty_lbl
        self._sync_counters()

        # Evita crescimento sem limite: derruba o card mais antigo
        if self._hist_layout.count() - 2 > _HIST_MAX:   # − empty − stretch
            item = self._hist_layout.takeAt(self._hist_layout.count() - 2)
            w = item.widget() if item else None
            if w is not None:
                self._seen.pop(getattr(w, "text_value", ""), None)
                w.deleteLater()
        self._heartbeat()

    def _sync_counters(self) -> None:
        self._get_stat_val(self._stat_reads).setText(str(self._history_count))

    def _heartbeat(self) -> None:
        """Pulsa o tab LEITURAS em verde por 200ms a cada nova leitura."""
        try:
            self._tab_reads.setStyleSheet(f"""
                QPushButton {{
                    background: transparent; border: none;
                    border-bottom: 2px solid {C['success']};
                    color: {C['success']};
                    font-family: {F_DISPLAY}; font-size: 10px;
                    font-weight: 700; letter-spacing: 3px;
                }}
            """)
            QTimer.singleShot(200, self._hb_off)
        except RuntimeError:
            pass

    def _hb_off(self) -> None:
        try:
            self._apply_tab_styles(self._pages.currentIndex())
        except RuntimeError:
            pass

    def clear_history(self) -> None:
        # Mantém empty_lbl (índice 0) e o stretch final
        while self._hist_layout.count() > 2:
            item = self._hist_layout.takeAt(1)
            if item and item.widget():
                item.widget().deleteLater()
        self._history_count = 0
        self._seen.clear()
        self._sync_counters()
        self._empty_lbl.show()


# ─────────────────────────────────────────────────────────────────────────────
# View principal
# ─────────────────────────────────────────────────────────────────────────────

class LiveQrView(QWidget):
    """Tela de detecção ao vivo usando IrisDetector."""

    # Tracker emite bboxes (rápido); decoder emite textos (assíncrono)
    _boxes_sig  = pyqtSignal(object, object)   # (List[TrackedDetection], frame)
    _decode_sig = pyqtSignal(object)           # List[Dict]

    _STATE_TEXT = {
        "idle":    "EM ESPERA",
        "opening": "ABRINDO CÂMERA",
        "live":    "AO VIVO",
    }

    def __init__(self, on_back=None, camera=None, ws_server=None, headless: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._on_back  = on_back
        self._state    = "idle"
        # Em modo headless os workers continuam rodando mesmo quando o widget
        # fica oculto (ex: usuário navega para o menu principal ou janela esconde).
        self._headless = headless

        # Estado de rastreamento: atualizado pelo _TrackingWorker a cada frame
        self._last_detections:    List[TrackedDetection] = []
        self._last_ghosts:        List[GhostDetection]   = []
        # Estado de decodificação: atualizado pelo _DecodingWorker, pode ser stale
        self._last_decode_results: List[Dict[str, Any]]  = []

        self._last_ts    = time.monotonic()
        self._fps_alpha  = 0.2
        self._fps_smooth = 0.0

        self._tracker:        Optional[_TrackingWorker] = None
        self._decoder_worker: Optional[_DecodingWorker] = None

        self._ui_frame_mutex  = QMutex()
        self._latest_ui_frame: Optional[np.ndarray] = None

        self._edge_enhancer  = EdgeEnhancer()
        self._edges_enabled  = False

        # Câmera pré-aquecida pelo MainWindow (passada via `camera`).
        # Se não recebida, cria uma própria (fallback para uso standalone).
        self._cam_owned = camera is None
        if camera is not None:
            self._cam = camera
        else:
            self._cam = SingleCameraManager(camera_index=0, force_mjpg=True)
            self._cam.start()

        self._cam_subscribed = False  # rastreia subscrição, não se a câmera roda

        # Servidor WebSocket — usa o servidor externo se fornecido (MainWindow),
        # caso contrário cria e gerencia o próprio.
        self._ws_owned = ws_server is None
        self._ws_server = ws_server if ws_server is not None else QrWebSocketServer()
        if self._ws_owned:
            self._ws_server.start()

        # Timer de renderização da UI
        self._render_timer = QTimer(self)
        self._render_timer.timeout.connect(self._pull_and_render)

        self._build()
        self._build_shortcuts()
        self._set_state("idle")

    # ── Construção da UI ──────────────────────────────────────────────────────

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(
            IrisAppBar("Live QR", show_back_button=True, on_back=self._handle_back)
        )

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # ── Área do feed ─────────────────────────────────────────────────────
        feed_wrapper = QWidget()
        feed_wrapper.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        feed_wrapper.setStyleSheet(f"background:{C['bg']};")
        fw = QVBoxLayout(feed_wrapper)
        fw.setContentsMargins(0, 0, 0, 0)
        fw.setSpacing(0)

        # HUD — o estado do instrumento mora aqui
        hud = QFrame()
        hud.setObjectName("Hud")
        hud.setFixedHeight(46)
        hud.setStyleSheet(f"""
            #Hud {{
                background: {C['surface']};
                border-bottom: 1px solid {C['border']};
            }}
        """)
        cl = QHBoxLayout(hud)
        cl.setContentsMargins(14, 0, 12, 0)
        cl.setSpacing(12)

        # Íris em miniatura: fechada/respirando/aberta = idle/opening/live
        self._hud_iris = IrisAperture(diameter=24, openness=0.10)
        cl.addWidget(self._hud_iris)

        self._status_lbl = QLabel(self._STATE_TEXT["idle"])
        self._status_lbl.setStyleSheet(
            f"color:{C['text_muted']}; font-family:{F_DISPLAY}; font-size:10px;"
            " font-weight:700; letter-spacing:3px; background:transparent;"
        )
        cl.addWidget(self._status_lbl)

        self._res_lbl = QLabel("")
        self._res_lbl.setStyleSheet(
            f"color:{C['text_muted']}; font-family:{F_DATA}; font-size:10px;"
            " background:transparent;"
        )
        cl.addWidget(self._res_lbl)
        cl.addStretch()

        self._edges_active = False
        self._edges_btn = QPushButton("◇  EDGES")
        self._edges_btn.setFixedHeight(26)
        self._edges_btn.setMinimumWidth(84)
        self._edges_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._edges_btn.setToolTip("Realce de bordas — CLAHE (E)")
        self._edges_btn.clicked.connect(self._toggle_edges)
        self._apply_edges_style()
        cl.addWidget(self._edges_btn)

        self._toggle_btn = IrisButton("▶  INICIAR", on_click=self._toggle_capture, width=130)
        self._toggle_btn.setToolTip("Iniciar/parar leitura (S)")
        cl.addWidget(self._toggle_btn)
        fw.addWidget(hud)

        # Feed da câmera × overlay de espera
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._view_stack   = QStackedLayout(container)
        self._idle_overlay = self._build_idle_overlay()
        self._feed_label   = _ViewfinderFeed()
        self._feed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._feed_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._feed_label.setStyleSheet("background:#000000;")
        self._view_stack.addWidget(self._idle_overlay)
        self._view_stack.addWidget(self._feed_label)
        fw.addWidget(container, stretch=1)

        body.addWidget(feed_wrapper, stretch=1)

        self._side_panel = _SidePanel(cam_getter=lambda: self._cam)
        body.addWidget(self._side_panel)

        root.addLayout(body, stretch=1)

    def _build_idle_overlay(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet(f"background:{C['bg']};")
        vl = QVBoxLayout(w)
        vl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vl.setSpacing(18)

        self._overlay_iris = IrisAperture(diameter=150, openness=0.10)
        vl.addWidget(self._overlay_iris, alignment=Qt.AlignmentFlag.AlignCenter)

        self._overlay_title = QLabel("INSTRUMENTO EM ESPERA")
        self._overlay_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._overlay_title.setStyleSheet(
            f"color:{C['text_muted']}; font-family:{F_DISPLAY}; font-size:12px;"
            " font-weight:700; letter-spacing:5px; background:transparent;"
        )
        vl.addWidget(self._overlay_title)

        self._overlay_sub = QLabel("Pressione INICIAR ou tecle S")
        self._overlay_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._overlay_sub.setStyleSheet(
            f"color:{C['border_hi']}; font-family:{F_BODY}; font-size:11px;"
            " background:transparent;"
        )
        vl.addWidget(self._overlay_sub)
        return w

    def _build_shortcuts(self) -> None:
        sc_toggle = QShortcut(QKeySequence("S"), self)
        sc_toggle.activated.connect(self._toggle_capture)
        sc_edges = QShortcut(QKeySequence("E"), self)
        sc_edges.activated.connect(self._toggle_edges)
        sc_clear = QShortcut(QKeySequence("Ctrl+L"), self)
        sc_clear.activated.connect(self._side_panel.clear_history)

    # ── Máquina de estados visual ─────────────────────────────────────────────

    def _set_state(self, state: str) -> None:
        """idle | opening | live — íris, overlay, HUD e botão em uníssono."""
        self._state = state
        try:
            self._status_lbl.setText(self._STATE_TEXT[state])

            if state == "idle":
                color = C["text_muted"]
                self._toggle_btn.setText("▶  INICIAR")
                self._res_lbl.setText("")
                self._hud_iris.stop_breathing()
                self._hud_iris.animate_to(0.10, ms=400)
                self._overlay_iris.stop_breathing()
                self._overlay_iris.animate_to(0.10, ms=500)
                self._overlay_title.setText("INSTRUMENTO EM ESPERA")
                self._overlay_sub.setText("Pressione INICIAR ou tecle S")
                self._view_stack.setCurrentIndex(0)

            elif state == "opening":
                color = C["primary"]
                self._toggle_btn.setText("■  PARAR")
                self._hud_iris.start_breathing(lo=0.15, hi=0.70)
                self._overlay_iris.start_breathing(lo=0.15, hi=0.70)
                self._overlay_title.setText("ABRINDO CÂMERA")
                self._overlay_sub.setText("Aguardando o primeiro frame…")
                self._view_stack.setCurrentIndex(0)

            else:  # live
                color = C["primary"]
                self._toggle_btn.setText("■  PARAR")
                self._hud_iris.stop_breathing()
                self._hud_iris.animate_to(0.82, ms=450)
                self._overlay_iris.stop_breathing()
                self._view_stack.setCurrentIndex(1)
                res = self._cam.resolution if self._cam else None
                if res:
                    self._res_lbl.setText(f"{res[0]}×{res[1]}")

            self._status_lbl.setStyleSheet(
                f"color:{color}; font-family:{F_DISPLAY}; font-size:10px;"
                " font-weight:700; letter-spacing:3px; background:transparent;"
            )
        except RuntimeError:
            pass

    # ── Controle de captura ───────────────────────────────────────────────────

    def _toggle_edges(self) -> None:
        self._edges_active = not self._edges_active
        self._edges_enabled = self._edges_active
        self._apply_edges_style()

    def _apply_edges_style(self) -> None:
        if self._edges_active:
            self._edges_btn.setText("◈  EDGES")
            self._edges_btn.setStyleSheet(f"""
                QPushButton {{
                    background: {C['primary']};
                    border: none; border-radius: 13px; color: {C['bg']};
                    font-family: {F_DISPLAY};
                    font-size: 10px; font-weight: 800; letter-spacing: 2px; padding: 0 12px;
                }}
                QPushButton:hover   {{ background: {C['primary_dim']}; }}
                QPushButton:focus   {{ border: 2px solid {C['text']}; }}
                QPushButton:pressed {{ background: {C['primary_dim']}; }}
            """)
        else:
            self._edges_btn.setText("◇  EDGES")
            self._edges_btn.setStyleSheet(f"""
                QPushButton {{
                    background: transparent; border: 1px solid {C['text_muted']};
                    border-radius: 13px; color: {C['text_muted']};
                    font-family: {F_DISPLAY};
                    font-size: 10px; font-weight: 600; letter-spacing: 2px; padding: 0 12px;
                }}
                QPushButton:hover   {{ border-color:{C['text']}; color:{C['text']}; }}
                QPushButton:focus   {{ border: 2px solid {C['primary']}; color:{C['primary']}; }}
                QPushButton:pressed {{ border-color:{C['primary']}; color:{C['primary']}; }}
            """)

    def _toggle_capture(self) -> None:
        if self._cam_subscribed:
            self._stop_capture()
        else:
            self._start_capture()

    def _start_capture(self) -> None:
        if self._cam_subscribed:
            return

        if self._tracker is None or not self._tracker.isRunning():
            self._tracker = _TrackingWorker()
            self._tracker.boxes_ready.connect(self._on_tracker_boxes)
            self._tracker.start()

        if self._decoder_worker is None or not self._decoder_worker.isRunning():
            self._decoder_worker = _DecodingWorker()
            self._decoder_worker.decode_ready.connect(self._on_decoder_result)
            self._decoder_worker.start()

        self._cam.subscribe(self._on_raw_frame)
        self._cam_subscribed = True

        # Aplica configurações salvas assim que a câmera está subscrita
        if hasattr(self, "_side_panel") and hasattr(self._side_panel, "_cam_controls"):
            self._side_panel._cam_controls.load_and_apply()

        # O feed só aparece quando o primeiro frame chegar (_pull_and_render);
        # até lá o overlay "ABRINDO CÂMERA" conta a verdade — sem tela preta.
        # Em modo headless não há nada a renderizar — o timer só desperdiça CPU.
        self._set_state("opening")
        self._last_ts = time.monotonic()
        if not self._headless:
            self._render_timer.start(_RENDER_INTERVAL_MS)
            self._feed_label.start_scan()

    def _stop_capture(self) -> None:
        self._render_timer.stop()
        self._feed_label.stop_scan()

        if self._cam:
            self._cam.unsubscribe(self._on_raw_frame)
            if self._cam_owned:
                self._cam.stop()
                self._cam = None
        self._cam_subscribed = False

        if self._tracker:
            self._tracker.stop()
            self._tracker.wait(3000)
            self._tracker = None

        if self._decoder_worker:
            self._decoder_worker.stop()
            self._decoder_worker.wait(3000)
            self._decoder_worker = None

        self._set_state("idle")

    # ── Callbacks de câmera ───────────────────────────────────────────────────

    def _on_raw_frame(self, frame: np.ndarray) -> None:
        # Headless sem render ativo: tracker não precisa de buffer próprio pois a
        # câmera não reutiliza o array entre callbacks — passa direto, sem cópia.
        render_active = self._render_timer.isActive()
        if self._edges_enabled:
            frame = self._edge_enhancer.apply(frame)  # nova array — cópia implícita
        elif render_active:
            frame = frame.copy()   # isola buffer compartilhado com o render timer
        tracker = self._tracker
        if tracker is not None and tracker.isRunning():
            tracker.submit_frame(frame)
        if render_active:
            with QMutexLocker(self._ui_frame_mutex):
                self._latest_ui_frame = frame   # no extra copy — same owned buffer

    def _pull_and_render(self) -> None:
        with QMutexLocker(self._ui_frame_mutex):
            frame = self._latest_ui_frame
            self._latest_ui_frame = None
        if frame is None:
            return
        if self._state == "opening":
            self._set_state("live")
        display = frame
        if self._last_detections or self._last_ghosts:
            merged  = _merge_with_text(self._last_detections, self._last_decode_results)
            display = _annotate(display.copy(), merged, self._last_ghosts)
        self._render_frame(display)

    # ── Slots de análise ──────────────────────────────────────────────────────

    def _on_tracker_boxes(
        self,
        detections: List[TrackedDetection],
        ghosts: List[GhostDetection],
        frame: np.ndarray,
    ) -> None:
        """Chamado pelo _TrackingWorker a cada frame inferido (thread-safe via Qt queue)."""
        now = time.monotonic()
        instant_fps = 1.0 / max(now - self._last_ts, 1e-6)
        self._fps_smooth = (
            self._fps_alpha * instant_fps + (1 - self._fps_alpha) * self._fps_smooth
        )
        self._last_ts         = now
        self._last_detections = detections or []
        self._last_ghosts     = ghosts     or []

        decoder = self._decoder_worker
        if decoder is not None and decoder.isRunning():
            decoder.submit(frame, detections)

        self._side_panel.update_stats(len(self._last_detections), self._fps_smooth)

    def _on_decoder_result(self, results: List[Dict[str, Any]]) -> None:
        """Chamado pelo _DecodingWorker quando um decode completa."""
        self._last_decode_results = results or []
        codes: List[str] = []
        for r in results:
            text = r.get("text")
            if text:
                self._side_panel.add_detection(text)
                codes.append(text)
        if codes:
            self._ws_server.send(codes)

    # ── Renderização ──────────────────────────────────────────────────────────

    def _render_frame(self, frame: np.ndarray) -> None:
        if frame is None:
            return
        if self._feed_label.width() <= 0 or self._feed_label.height() <= 0:
            return
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg).scaled(
            self._feed_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self._feed_label.setPixmap(pixmap)

    # ── Erros ─────────────────────────────────────────────────────────────────

    def _on_error(self, msg: str) -> None:
        self._stop_capture()
        self._status_lbl.setText(f"ERRO: {msg}")
        self._status_lbl.setStyleSheet(
            f"color:{C['danger']}; font-family:{F_DISPLAY}; font-size:10px;"
            " font-weight:700; letter-spacing:3px; background:transparent;"
        )

    # ── Ciclo de vida do widget ───────────────────────────────────────────────

    def _handle_back(self) -> None:
        if not self._headless:
            self._release_resources()
        if self._on_back:
            self._on_back()

    def hideEvent(self, event) -> None:
        # Em modo headless os workers continuam vivos — apenas a janela some.
        if not self._headless:
            self._release_resources()
        super().hideEvent(event)

    def closeEvent(self, event) -> None:
        self._release_resources()
        super().closeEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # Auto-inicia: o operador não deveria precisar clicar para trabalhar.
        QTimer.singleShot(300, self._start_capture)

    def enable_render(self) -> None:
        """Liga o render timer (chamado ao mostrar a janela em modo headless)."""
        if not self._render_timer.isActive() and self._cam_subscribed:
            self._render_timer.start(_RENDER_INTERVAL_MS)
            self._feed_label.start_scan()

    def disable_render(self) -> None:
        """Desliga o render timer sem parar workers (chamado ao esconder em modo headless)."""
        self._render_timer.stop()
        self._feed_label.stop_scan()

    def _release_resources(self) -> None:
        self._render_timer.stop()
        self._feed_label.stop_scan()
        if self._ws_owned:
            self._ws_server.stop()

        if self._cam:
            self._cam.unsubscribe(self._on_raw_frame)
            if self._cam_owned:
                self._cam.stop()
                self._cam = None
        self._cam_subscribed = False

        if self._tracker:
            self._tracker.stop()
            self._tracker.wait(3000)
            self._tracker = None

        if self._decoder_worker:
            self._decoder_worker.stop()
            self._decoder_worker.wait(3000)
            self._decoder_worker = None

        with QMutexLocker(self._ui_frame_mutex):
            self._latest_ui_frame = None

        self._last_detections     = []
        self._last_ghosts         = []
        self._last_decode_results = []
        self._set_state("idle")
