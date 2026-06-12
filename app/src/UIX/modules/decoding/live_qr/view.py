"""
UIX/modules/decoding/live_qr/view.py
======================================
Tela de detecção em tempo real via webcam — v2 redesenhada.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Any, Tuple

import cv2
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
    QLinearGradient,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from app.src.UIX.components.shared import C, IrisAppBar, IrisButton
from app.src.engine.modules.decoding.live_qr.decoder import QrDecoder
from app.src.engine.modules.decoding.live_qr.detector import Detection, IrisDetector
from app.src.infrastructure.video.camera import SingleCameraManager


# ─────────────────────────────────────────────────────────────────────────────
# Constantes visuais
# ─────────────────────────────────────────────────────────────────────────────

_BOX_COLOR     = (0x4C, 0x28, 0x14)
_BOX_HI_COLOR  = (0x4A, 0xDE, 0x74)
_BOX_THICKNESS = 2
_LABEL_FONT    = cv2.FONT_HERSHEY_SIMPLEX
_HIST_MAX      = 50
_RENDER_INTERVAL_MS = int((1 / 33) * 1000)  # ~30 ms

# Animações
_SCANLINE_STEP    = 0.004   # fração da altura por tick de 16 ms
_PULSE_INTERVAL   = 700     # ms — pisca do ponto de status
_FLASH_MS         = 380     # ms — duração do flash do card


# ─────────────────────────────────────────────────────────────────────────────
# Worker de análise pesada (lógica inalterada)
# ─────────────────────────────────────────────────────────────────────────────

class _AnalysisWorker(QThread):
    """Roda em background e processa o frame mais recente disponível."""

    frame_ready = pyqtSignal(object, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._engine   = IrisDetector()
        self._decoder  = QrDecoder()
        self._running  = True
        self._mutex    = QMutex()
        self._pending: Optional[np.ndarray] = None
        self._busy     = False

    def submit_frame(self, frame: np.ndarray) -> None:
        with QMutexLocker(self._mutex):
            self._pending = frame.copy()

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        while self._running:
            with QMutexLocker(self._mutex):
                frame = self._pending
                self._pending = None
                if frame is not None:
                    self._busy = True

            if frame is None:
                self.msleep(15)
                continue

            try:
                annotated, results = self._process(frame)
                self.frame_ready.emit(annotated, results)
            except Exception as exc:
                print(f"[_AnalysisWorker] Erro crítico na inferência: {exc}")
            finally:
                with QMutexLocker(self._mutex):
                    self._busy = False

    def _process(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, List[Dict[str, Any]]]:
        detections: List[Detection] = self._engine.detect(frame)
        results = []
        boxes = []
        for d in detections:
            pts = np.array([
                [d.x1, d.y1], [d.x2, d.y1], [d.x2, d.y2], [d.x1, d.y2]
            ], dtype=np.int32)
            boxes.append(pts)
        decoded_tuples = self._decoder.decode(frame, boxes)
        for d, (text, _patch, _ox, _oy) in zip(detections, decoded_tuples):
            results.append({"detection": d, "text": text})
        return np.array([]), results

    @staticmethod
    def _annotate(
        frame: np.ndarray,
        results: List[Dict[str, Any]],
    ) -> np.ndarray:
        for res in results:
            d = res["detection"]
            raw_text = res["text"]
            color = _BOX_HI_COLOR if raw_text else _BOX_COLOR
            cv2.rectangle(frame, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), color, _BOX_THICKNESS)
            corner = min(d.width, d.height) // 5
            for cx, cy, dx, dy in [
                (int(d.x1), int(d.y1), +1, +1),
                (int(d.x2), int(d.y1), -1, +1),
                (int(d.x2), int(d.y2), -1, -1),
                (int(d.x1), int(d.y2), +1, -1),
            ]:
                cv2.line(frame, (cx, cy), (int(cx + dx * corner), cy), color, _BOX_THICKNESS + 1)
                cv2.line(frame, (cx, cy), (cx, int(cy + dy * corner)), color, _BOX_THICKNESS + 1)
            label = str(raw_text) if raw_text else "Nao Lido"
            if len(label) > 20:
                label = label[:17] + "..."
            scale, thick = 0.45, 1
            (tw, th), _ = cv2.getTextSize(label, _LABEL_FONT, scale, thick)
            ty = int(d.y1) - 8 if int(d.y1) - 8 > th else int(d.y2) + th + 8
            cv2.rectangle(frame,
                          (int(d.x1), ty - th - 4), (int(d.x1) + tw + 8, ty + 4),
                          color, cv2.FILLED)
            cv2.putText(frame, label, (int(d.x1) + 4, ty),
                        _LABEL_FONT, scale, (0x06, 0x0B, 0x14), thick, cv2.LINE_AA)
        return frame


# ─────────────────────────────────────────────────────────────────────────────
# Feed com overlay de linha de varredura
# ─────────────────────────────────────────────────────────────────────────────

class _ScanlineFeed(QLabel):
    """
    QLabel da câmera com linha de varredura animada pintada via paintEvent.
    A linha representa visualmente que o sistema está ativamente varrendo.
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

        y = int(self._scan_pos * self.height())

        p = QPainter(self)
        # Halo gradiente: 20 px acima e abaixo da linha
        top_y  = max(0, y - 20)
        bot_y  = min(self.height(), y + 20)
        grad   = QLinearGradient(0, top_y, 0, bot_y)
        grad.setColorAt(0.0, QColor(0, 255, 136, 0))
        grad.setColorAt(0.5, QColor(0, 255, 136, 40))
        grad.setColorAt(1.0, QColor(0, 255, 136, 0))
        p.fillRect(QRect(0, top_y, self.width(), bot_y - top_y), grad)

        # Linha principal
        pen = QPen(QColor(0, 255, 136, 80))
        pen.setWidth(1)
        p.setPen(pen)
        p.drawLine(0, y, self.width(), y)
        p.end()


# ─────────────────────────────────────────────────────────────────────────────
# Badge de contagem de duplicados
# ─────────────────────────────────────────────────────────────────────────────

class _CountBadge(QLabel):
    """Pílula à direita do card — aparece somente quando count > 1."""

    _S_NORMAL = (
        f"color: {C['primary']};"
        f"background: rgba(0,255,136,0.12);"
        f"border: 1px solid rgba(0,255,136,0.45);"
        "border-radius: 8px;"
        "font-size: 9px; font-weight: 700; letter-spacing: 1px;"
        "padding: 0px 6px;"
    )
    _S_FLASH = (
        f"color: {C['bg']};"
        f"background: {C['primary']};"
        f"border: 1px solid {C['primary']};"
        "border-radius: 8px;"
        "font-size: 9px; font-weight: 700; letter-spacing: 1px;"
        "padding: 0px 6px;"
    )

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedHeight(18)
        self.setMinimumWidth(36)
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
# Card de histórico redesenhado
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
        self._count = 1
        self.setStyleSheet(self._S_NORMAL)
        self.setFixedHeight(38)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 0, 8, 0)
        lay.setSpacing(8)

        idx = QLabel(f"#{index:03d}")
        idx.setFixedWidth(36)
        idx.setStyleSheet(
            f"color:{C['primary']}; font-size:9px; font-weight:700;"
            " letter-spacing:1px; background:transparent;"
        )
        lay.addWidget(idx)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFixedWidth(1)
        sep.setFixedHeight(18)
        sep.setStyleSheet(f"background:{C['border_hi']}; border:none;")
        lay.addWidget(sep)

        text_str = str(text_value)
        display  = text_str[:32] + "…" if len(text_str) > 32 else text_str
        val      = QLabel(display)
        val.setStyleSheet(
            f"color:{C['text']}; font-size:10px;"
            " letter-spacing:0.3px; background:transparent;"
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
# Painel lateral redesenhado
# ─────────────────────────────────────────────────────────────────────────────

class _SidePanel(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(280)
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

        # ── Header ──────────────────────────────────────────────────────────
        header = QFrame()
        header.setObjectName("SPH")
        header.setFixedHeight(44)
        header.setStyleSheet(f"""
            #SPH {{
                background: {C['card']};
                border-bottom: 2px solid {C['primary']};
            }}
        """)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(12, 0, 12, 0)
        hl.setSpacing(8)

        icon = QLabel("◈")
        icon.setStyleSheet(f"color:{C['primary']}; font-size:14px; background:transparent;")
        hl.addWidget(icon)

        title = QLabel("SCAN LOG")
        title.setStyleSheet(
            f"color:{C['primary']}; font-size:10px; font-weight:700;"
            " letter-spacing:3px; background:transparent;"
        )
        hl.addWidget(title, stretch=1)

        self._hb = QLabel("●")
        self._hb.setStyleSheet(f"color:{C['text_muted']}; font-size:8px; background:transparent;")
        hl.addWidget(self._hb)

        self._count_lbl = QLabel("0")
        self._count_lbl.setFixedWidth(28)
        self._count_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._count_lbl.setStyleSheet(
            f"color:{C['text_muted']}; font-size:10px; letter-spacing:1px; background:transparent;"
        )
        hl.addWidget(self._count_lbl)
        root.addWidget(header)

        # ── Stats ────────────────────────────────────────────────────────────
        stats = QFrame()
        stats.setObjectName("SPS")
        stats.setFixedHeight(56)
        stats.setStyleSheet(
            f"#SPS {{ background:{C['surface']}; border-bottom:1px solid {C['border']}; }}"
        )
        sl = QHBoxLayout(stats)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(0)
        self._stat_detected = self._make_stat("LOCALIZADOS", "0", C["secondary"])
        self._stat_reads    = self._make_stat("LIDOS",       "0", C["success"])
        self._stat_fps      = self._make_stat("FPS",         "—", C["text_muted"])
        for i, w in enumerate([self._stat_detected, self._stat_reads, self._stat_fps]):
            sl.addWidget(w, stretch=1)
            if i < 2:
                div = QFrame()
                div.setFixedWidth(1)
                div.setStyleSheet(f"background:{C['border']}; border:none;")
                sl.addWidget(div)
        root.addWidget(stats)

        # ── Scroll ────────────────────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background:transparent; border:none;")
        self._hist_container = QWidget()
        self._hist_container.setStyleSheet("background:transparent;")
        self._hist_layout = QVBoxLayout(self._hist_container)
        self._hist_layout.setContentsMargins(8, 8, 8, 8)
        self._hist_layout.setSpacing(4)
        self._hist_layout.addStretch()
        scroll.setWidget(self._hist_container)
        root.addWidget(scroll, stretch=1)

        # ── Footer ────────────────────────────────────────────────────────────
        clear_btn = QPushButton("⌫  LIMPAR HISTÓRICO")
        clear_btn.setFixedHeight(32)
        clear_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; border: none;
                border-top: 1px solid {C['border']};
                color: {C['text_muted']};
                font-family: 'Consolas', monospace; font-size: 9px;
                font-weight: 600; letter-spacing: 2px;
            }}
            QPushButton:hover {{
                color: {C['danger']};
                background: rgba(248, 113, 113, 0.06);
            }}
            QPushButton:pressed {{
                color: {C['danger']};
                background: rgba(248, 113, 113, 0.12);
            }}
        """)
        clear_btn.clicked.connect(self._clear_history)
        root.addWidget(clear_btn)

        self._history_count = 0
        # text → (timestamp, card)  —  rastreia duplicados dentro de 2 s
        self._recent_texts: Dict[str, Tuple[float, _HistoryCard]] = {}

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
            f"color:{color}; font-size:16px; font-weight:700; background:transparent;"
        )
        lbl = QLabel(label)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(
            f"color:{C['text_muted']}; font-size:7px; letter-spacing:1.5px; background:transparent;"
        )
        vl.addWidget(val)
        vl.addWidget(lbl)
        return w

    def _get_stat_val(self, w: QWidget) -> QLabel:
        return w.findChild(QLabel, "val")

    def update_stats(self, detected: int, read_count: int, fps: float) -> None:
        self._get_stat_val(self._stat_detected).setText(str(detected))
        self._get_stat_val(self._stat_reads).setText(str(read_count))
        self._get_stat_val(self._stat_fps).setText(f"{fps:.1f}")

    def add_detection(self, text_value: str) -> None:
        now      = time.monotonic()
        text_str = str(text_value)

        # Expirar entradas com mais de 2 s
        self._recent_texts = {
            k: v for k, v in self._recent_texts.items() if now - v[0] < 2.0
        }

        if text_str in self._recent_texts:
            # Duplicado: atualiza timestamp e incrementa badge
            ts, card = self._recent_texts[text_str]
            self._recent_texts[text_str] = (now, card)
            card.increment()
            self._heartbeat()
            return

        # Nova leitura
        self._history_count += 1
        self._count_lbl.setText(str(self._history_count))
        card = _HistoryCard(self._history_count, text_str)
        self._recent_texts[text_str] = (now, card)
        self._hist_layout.insertWidget(self._hist_layout.count() - 1, card)
        if self._hist_layout.count() - 1 > _HIST_MAX:
            item = self._hist_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        self._heartbeat()

    def _heartbeat(self) -> None:
        """Pulsa o indicador ● no header por 200 ms a cada leitura."""
        self._hb.setStyleSheet(
            f"color:{C['success']}; font-size:8px; background:transparent;"
        )
        QTimer.singleShot(200, self._hb_off)

    def _hb_off(self) -> None:
        try:
            self._hb.setStyleSheet(
                f"color:{C['text_muted']}; font-size:8px; background:transparent;"
            )
        except RuntimeError:
            pass

    def _clear_history(self) -> None:
        while self._hist_layout.count() > 1:
            item = self._hist_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        self._history_count = 0
        self._count_lbl.setText("0")
        self._recent_texts.clear()


# ─────────────────────────────────────────────────────────────────────────────
# View principal
# ─────────────────────────────────────────────────────────────────────────────

class LiveQrView(QWidget):
    """Tela de detecção ao vivo usando IrisDetector."""

    _frame_signal = pyqtSignal(object, object)

    def __init__(self, on_back=None, parent=None) -> None:
        super().__init__(parent)
        self._on_back = on_back

        self.last_annotated: Optional[np.ndarray] = None
        self.last_results: List[Dict[str, Any]]   = []

        self._last_ts    = time.monotonic()
        self._fps_alpha  = 0.2
        self._fps_smooth = 0.0

        self._worker: Optional[_AnalysisWorker]      = None
        self._cam:    Optional[SingleCameraManager]  = None

        self._ui_frame_mutex  = QMutex()
        self._latest_ui_frame: Optional[np.ndarray] = None

        # Timer de renderização da UI
        self._render_timer = QTimer(self)
        self._render_timer.timeout.connect(self._pull_and_render)

        # Timer de pulso do ponto de status
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(_PULSE_INTERVAL)
        self._pulse_timer.timeout.connect(self._toggle_pulse)
        self._pulse_state = False

        self._frame_signal.connect(self._on_analysis_done)
        self._build()

    # ── Construção da UI ──────────────────────────────────────────────────────

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(
            IrisAppBar("Live QR Detector", show_back_button=True, on_back=self._handle_back)
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

        # Barra de controle
        ctrl_bar = QFrame()
        ctrl_bar.setObjectName("CtrlBar")
        ctrl_bar.setFixedHeight(42)
        ctrl_bar.setStyleSheet(f"""
            #CtrlBar {{
                background: {C['surface']};
                border-bottom: 1px solid {C['border']};
            }}
        """)
        cl = QHBoxLayout(ctrl_bar)
        cl.setContentsMargins(14, 0, 12, 0)
        cl.setSpacing(10)

        self._status_dot = QLabel("●")
        self._status_dot.setFixedWidth(14)
        self._status_dot.setStyleSheet(
            f"color:{C['text_muted']}; background:transparent; font-size:10px;"
        )
        cl.addWidget(self._status_dot)

        self._status_lbl = QLabel("CÂMERA INATIVA")
        self._status_lbl.setStyleSheet(
            f"color:{C['text_muted']}; font-size:9px;"
            " font-weight:600; letter-spacing:2px; background:transparent;"
        )
        cl.addWidget(self._status_lbl)
        cl.addStretch()

        self._toggle_btn = IrisButton("▶  INICIAR", on_click=self._toggle_capture, width=130)
        cl.addWidget(self._toggle_btn)
        fw.addWidget(ctrl_bar)

        # Feed da câmera
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._view_stack   = QStackedLayout(container)
        self._idle_overlay = self._build_idle_overlay()
        self._feed_label   = _ScanlineFeed()
        self._feed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._feed_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._feed_label.setStyleSheet("background:#000000;")
        self._view_stack.addWidget(self._idle_overlay)
        self._view_stack.addWidget(self._feed_label)
        fw.addWidget(container, stretch=1)

        body.addWidget(feed_wrapper, stretch=1)

        self._side_panel = _SidePanel()
        body.addWidget(self._side_panel)

        root.addLayout(body, stretch=1)
        root.addWidget(self._build_statusbar())

    def _build_idle_overlay(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet(f"background:{C['bg']};")
        vl = QVBoxLayout(w)
        vl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vl.setSpacing(16)

        self._idle_icon = QLabel("⊕")
        self._idle_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._idle_icon.setStyleSheet(
            f"color:{C['border_hi']}; font-size:60px; background:transparent;"
        )
        vl.addWidget(self._idle_icon)

        line1 = QLabel("CÂMERA AGUARDANDO")
        line1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        line1.setStyleSheet(
            f"color:{C['text_muted']}; font-size:11px;"
            " font-weight:700; letter-spacing:4px; background:transparent;"
        )
        vl.addWidget(line1)

        line2 = QLabel("Pressione INICIAR para ativar")
        line2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        line2.setStyleSheet(
            f"color:{C['border_hi']}; font-size:10px;"
            " letter-spacing:2px; background:transparent;"
        )
        vl.addWidget(line2)

        # Timer de pulso do ícone idle
        self._idle_timer = QTimer(self)
        self._idle_timer.setInterval(1200)
        self._idle_timer.timeout.connect(self._toggle_idle)
        self._idle_state = False
        return w

    def _toggle_idle(self) -> None:
        self._idle_state = not self._idle_state
        color = C["primary_dim"] if self._idle_state else C["border_hi"]
        try:
            self._idle_icon.setStyleSheet(
                f"color:{color}; font-size:60px; background:transparent;"
            )
        except RuntimeError:
            pass

    @staticmethod
    def _build_statusbar() -> QFrame:
        bar = QFrame()
        bar.setObjectName("StatusBar")
        bar.setFixedHeight(30)
        bar.setStyleSheet(f"""
            #StatusBar {{
                background: {C['surface']};
                border-top: 1px solid {C['border']};
            }}
        """)
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(20, 0, 20, 0)
        bl.setSpacing(20)
        bl.addStretch()
        for hex_color, label in [("#14284C", "LOCALIZADO"), ("#4ADE80", "DECODIFICADO")]:
            dot = QLabel("■")
            dot.setStyleSheet(f"color:{hex_color}; font-size:10px; background:transparent;")
            txt = QLabel(label)
            txt.setStyleSheet(
                f"color:{C['text_muted']}; font-size:9px;"
                " letter-spacing:1px; background:transparent;"
            )
            bl.addWidget(dot)
            bl.addWidget(txt)
        bl.addStretch()
        return bar

    # ── Animações de status ───────────────────────────────────────────────────

    def _toggle_pulse(self) -> None:
        self._pulse_state = not self._pulse_state
        color = C["primary"] if self._pulse_state else C["primary_dim"]
        try:
            self._status_dot.setStyleSheet(
                f"color:{color}; background:transparent; font-size:10px;"
            )
        except RuntimeError:
            pass

    # ── Controle de captura ───────────────────────────────────────────────────

    def _toggle_capture(self) -> None:
        if self._cam and self._cam.is_running:
            self._stop_capture()
        else:
            self._start_capture()

    def _start_capture(self) -> None:
        if self._cam and self._cam.is_running:
            return

        if self._worker is None or not self._worker.isRunning():
            self._worker = _AnalysisWorker()
            self._worker.frame_ready.connect(self._on_worker_frame)
            self._worker.start()

        try:
            self._cam = SingleCameraManager(camera_index=0)
            self._cam.start()
            self._cam.subscribe(self._on_raw_frame)
        except RuntimeError as exc:
            self._on_error(str(exc))
            return

        self._toggle_btn.setText("■  PARAR")
        self._status_dot.setStyleSheet(
            f"color:{C['primary']}; background:transparent; font-size:10px;"
        )
        self._status_lbl.setText("CÂMERA ATIVA")
        self._status_lbl.setStyleSheet(
            f"color:{C['primary']}; font-size:9px;"
            " font-weight:600; letter-spacing:2px; background:transparent;"
        )
        self._view_stack.setCurrentIndex(1)
        self._last_ts = time.monotonic()
        self._render_timer.start(_RENDER_INTERVAL_MS)
        self._pulse_timer.start()
        self._idle_timer.stop()
        self._feed_label.start_scan()

    def _stop_capture(self) -> None:
        self._render_timer.stop()
        self._pulse_timer.stop()
        self._feed_label.stop_scan()

        if self._cam:
            self._cam.stop()
            self._cam = None

        self._toggle_btn.setText("▶  INICIAR")
        self._status_dot.setStyleSheet(
            f"color:{C['text_muted']}; background:transparent; font-size:10px;"
        )
        self._status_lbl.setText("CÂMERA INATIVA")
        self._status_lbl.setStyleSheet(
            f"color:{C['text_muted']}; font-size:9px;"
            " font-weight:600; letter-spacing:2px; background:transparent;"
        )
        self._view_stack.setCurrentIndex(0)
        self._idle_timer.start()

    # ── Callbacks de câmera ───────────────────────────────────────────────────

    def _on_raw_frame(self, frame: np.ndarray) -> None:
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.submit_frame(frame)
        with QMutexLocker(self._ui_frame_mutex):
            self._latest_ui_frame = frame.copy()

    def _pull_and_render(self) -> None:
        with QMutexLocker(self._ui_frame_mutex):
            frame = self._latest_ui_frame
            self._latest_ui_frame = None
        if frame is None:
            return
        display = frame
        if self.last_results:
            display = _AnalysisWorker._annotate(display, self.last_results)
        self._render_frame(display)

    # ── Slots de análise ──────────────────────────────────────────────────────

    def _on_worker_frame(
        self, annotated: np.ndarray, results: List[Dict[str, Any]]
    ) -> None:
        self._frame_signal.emit(annotated, results)

    def _on_analysis_done(
        self, annotated: np.ndarray, results: List[Dict[str, Any]]
    ) -> None:
        now = time.monotonic()
        instant_fps = 1.0 / max(now - self._last_ts, 1e-6)
        self._fps_smooth = (
            self._fps_alpha * instant_fps + (1 - self._fps_alpha) * self._fps_smooth
        )
        self._last_ts   = now
        self.last_results = results or []

        detected   = len(results)
        read_count = sum(1 for r in results if r["text"])
        self._side_panel.update_stats(detected, read_count, self._fps_smooth)
        for r in results:
            if r["text"]:
                self._side_panel.add_detection(r["text"])

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
            f"color:{C['danger']}; font-size:9px;"
            " font-weight:600; letter-spacing:2px; background:transparent;"
        )

    # ── Ciclo de vida do widget ───────────────────────────────────────────────

    def _handle_back(self) -> None:
        self._release_resources()
        if self._on_back:
            self._on_back()

    def hideEvent(self, event) -> None:
        self._release_resources()
        super().hideEvent(event)

    def closeEvent(self, event) -> None:
        self._release_resources()
        super().closeEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._idle_timer.start()
        QTimer.singleShot(300, self._start_capture)

    def _release_resources(self) -> None:
        self._render_timer.stop()
        self._pulse_timer.stop()
        self._idle_timer.stop()
        self._feed_label.stop_scan()

        if self._cam:
            self._cam.stop()
            self._cam = None

        if self._worker:
            self._worker.stop()
            self._worker.wait(3000)
            self._worker = None

        with QMutexLocker(self._ui_frame_mutex):
            self._latest_ui_frame = None

        self.last_annotated = None
        self.last_results   = []
        self._view_stack.setCurrentIndex(0)
        self._toggle_btn.setText("▶  INICIAR")
        self._status_dot.setStyleSheet(
            f"color:{C['text_muted']}; background:transparent; font-size:10px;"
        )
        self._status_lbl.setText("CÂMERA INATIVA")
        self._status_lbl.setStyleSheet(
            f"color:{C['text_muted']}; font-size:9px;"
            " font-weight:600; letter-spacing:2px; background:transparent;"
        )
