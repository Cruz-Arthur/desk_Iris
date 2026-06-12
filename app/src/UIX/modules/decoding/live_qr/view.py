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
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from app.src.UIX.components.shared import (
    C, F_BODY, F_DATA, F_DISPLAY, IrisAperture, IrisAppBar, IrisButton,
)
from app.src.engine.modules.decoding.live_qr.decoder import QrDecoder
from app.src.engine.modules.decoding.live_qr.detector import Detection, IrisDetector
from app.src.infrastructure.video.camera import SingleCameraManager
from app.src.infrastructure.video.enhance import EdgeEnhancer


# ─────────────────────────────────────────────────────────────────────────────
# Constantes visuais
# ─────────────────────────────────────────────────────────────────────────────

# BGR — âmbar (localizado, ainda sem leitura) / verde fósforo (decodificado)
_BOX_COLOR     = (84, 180, 255)
_BOX_HI_COLOR  = (128, 222, 74)
_LABEL_INK     = (19, 14, 11)        # BGR de #0B0E13
_BOX_THICKNESS = 2
_LABEL_FONT    = cv2.FONT_HERSHEY_SIMPLEX
_HIST_MAX      = 50
_RENDER_INTERVAL_MS = int((1 / 33) * 1000)  # ~30 ms

_SCANLINE_STEP  = 0.004   # fração da altura por tick de 16 ms
_FLASH_MS       = 380     # ms — flash do card ao repetir leitura
_REARM_S        = 1.0     # s — ausência mínima para contar novo "bip"


# ─────────────────────────────────────────────────────────────────────────────
# Worker de análise pesada (lógica inalterada)
# ─────────────────────────────────────────────────────────────────────────────

class _AnalysisWorker(QThread):
    """Roda em background e processa o frame mais recente disponível."""

    frame_ready = pyqtSignal(object, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        # IrisDetector é inicializado dentro de run() para não bloquear a UI
        # (DirectML compila shaders na primeira carga — pode levar vários segundos)
        self._engine: Optional[IrisDetector] = None
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
        self._engine = IrisDetector()  # inicializa na thread do worker
        while self._running:
            with QMutexLocker(self._mutex):
                frame = self._pending
                self._pending = None
                if frame is not None:
                    self._busy = True

            if frame is None:
                self.msleep(15)
                continue

            if self._engine is None:
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
            label = str(raw_text) if raw_text else "LENDO..."
            if len(label) > 20:
                label = label[:17] + "..."
            scale, thick = 0.45, 1
            (tw, th), _ = cv2.getTextSize(label, _LABEL_FONT, scale, thick)
            ty = int(d.y1) - 8 if int(d.y1) - 8 > th else int(d.y2) + th + 8
            cv2.rectangle(frame,
                          (int(d.x1), ty - th - 4), (int(d.x1) + tw + 8, ty + 4),
                          color, cv2.FILLED)
            cv2.putText(frame, label, (int(d.x1) + 4, ty),
                        _LABEL_FONT, scale, _LABEL_INK, thick, cv2.LINE_AA)
        return frame


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
# Painel lateral — registro de leituras
# ─────────────────────────────────────────────────────────────────────────────

class _SidePanel(QFrame):
    def __init__(self, parent=None) -> None:
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
        hl.setContentsMargins(14, 0, 14, 0)
        hl.setSpacing(8)

        title = QLabel("LEITURAS")
        title.setStyleSheet(
            f"color:{C['primary']}; font-family:{F_DISPLAY};"
            "font-size:11px; font-weight:700; letter-spacing:4px;"
            " background:transparent;"
        )
        hl.addWidget(title, stretch=1)

        self._hb = QLabel("●")
        self._hb.setStyleSheet(f"color:{C['text_muted']}; font-size:8px; background:transparent;")
        hl.addWidget(self._hb)

        self._count_lbl = QLabel("0")
        self._count_lbl.setFixedWidth(36)
        self._count_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._count_lbl.setStyleSheet(
            f"color:{C['text']}; font-family:{F_DATA};"
            "font-size:11px; font-weight:700; background:transparent;"
        )
        hl.addWidget(self._count_lbl)
        root.addWidget(header)

        # ── Stats ────────────────────────────────────────────────────────────
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
                div = QFrame()
                div.setFixedWidth(1)
                div.setStyleSheet(f"background:{C['border']}; border:none;")
                sl.addWidget(div)
        root.addWidget(stats)

        # ── Scroll do registro ────────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background:transparent; border:none;")
        self._hist_container = QWidget()
        self._hist_container.setStyleSheet("background:transparent;")
        self._hist_layout = QVBoxLayout(self._hist_container)
        self._hist_layout.setContentsMargins(8, 8, 8, 8)
        self._hist_layout.setSpacing(4)

        # Estado vazio: convite à ação, não decoração
        self._empty_lbl = QLabel(
            "Nenhuma leitura ainda.\nAproxime um código QR da câmera."
        )
        self._empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_lbl.setWordWrap(True)
        self._empty_lbl.setStyleSheet(
            f"color:{C['text_muted']}; font-family:{F_BODY};"
            "font-size:11px; padding: 22px 10px; background:transparent;"
        )
        self._hist_layout.addWidget(self._empty_lbl)
        self._hist_layout.addStretch()
        scroll.setWidget(self._hist_container)
        root.addWidget(scroll, stretch=1)

        # ── Footer ────────────────────────────────────────────────────────────
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
            QPushButton:hover {{
                color: {C['danger']};
                background: rgba(255, 122, 122, 0.06);
            }}
            QPushButton:focus {{
                color: {C['danger']};
                border-top: 1px solid {C['danger']};
            }}
            QPushButton:pressed {{
                color: {C['danger']};
                background: rgba(255, 122, 122, 0.12);
            }}
        """)
        clear_btn.clicked.connect(self.clear_history)
        root.addWidget(clear_btn)

        self._history_count = 0       # total de bips (inclui repetições)
        # text → {"card": _HistoryCard, "last_ts": float}
        # Um novo "bip" do mesmo código só conta após _REARM_S de ausência —
        # presença contínua no frame NÃO infla o contador.
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
        self._count_lbl.setText(str(self._history_count))
        self._get_stat_val(self._stat_reads).setText(str(self._history_count))

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

    _frame_signal = pyqtSignal(object, object)

    _STATE_TEXT = {
        "idle":    "EM ESPERA",
        "opening": "ABRINDO CÂMERA",
        "live":    "AO VIVO",
    }

    def __init__(self, on_back=None, camera=None, parent=None) -> None:
        super().__init__(parent)
        self._on_back = on_back
        self._state   = "idle"

        self.last_annotated: Optional[np.ndarray] = None
        self.last_results: List[Dict[str, Any]]   = []

        self._last_ts    = time.monotonic()
        self._fps_alpha  = 0.2
        self._fps_smooth = 0.0

        self._worker: Optional[_AnalysisWorker]      = None

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

        # Timer de renderização da UI
        self._render_timer = QTimer(self)
        self._render_timer.timeout.connect(self._pull_and_render)

        self._frame_signal.connect(self._on_analysis_done)
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

        self._side_panel = _SidePanel()
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

        if self._worker is None or not self._worker.isRunning():
            self._worker = _AnalysisWorker()
            self._worker.frame_ready.connect(self._on_worker_frame)
            self._worker.start()

        self._cam.subscribe(self._on_raw_frame)
        self._cam_subscribed = True

        # O feed só aparece quando o primeiro frame chegar (_pull_and_render);
        # até lá o overlay "ABRINDO CÂMERA" conta a verdade — sem tela preta.
        self._set_state("opening")
        self._last_ts = time.monotonic()
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
        self._set_state("idle")

    # ── Callbacks de câmera ───────────────────────────────────────────────────

    def _on_raw_frame(self, frame: np.ndarray) -> None:
        if self._edges_enabled:
            frame = self._edge_enhancer.apply(frame)
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
        if self._state == "opening":
            # Primeiro frame chegou — o instrumento abre
            self._set_state("live")
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

        self._side_panel.update_stats(len(results), self._fps_smooth)
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
            f"color:{C['danger']}; font-family:{F_DISPLAY}; font-size:10px;"
            " font-weight:700; letter-spacing:3px; background:transparent;"
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
        # Auto-inicia: o operador não deveria precisar clicar para trabalhar.
        QTimer.singleShot(300, self._start_capture)

    def _release_resources(self) -> None:
        self._render_timer.stop()
        self._feed_label.stop_scan()

        if self._cam:
            self._cam.unsubscribe(self._on_raw_frame)
            if self._cam_owned:
                self._cam.stop()
                self._cam = None
        self._cam_subscribed = False

        if self._worker:
            self._worker.stop()
            self._worker.wait(3000)
            self._worker = None

        with QMutexLocker(self._ui_frame_mutex):
            self._latest_ui_frame = None

        self.last_annotated = None
        self.last_results   = []
        self._set_state("idle")
