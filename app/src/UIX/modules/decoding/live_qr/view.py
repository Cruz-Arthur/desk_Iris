"""
UIX/live/live_layout.py
========================
Tela de detecção em tempo real via webcam.
"""

from __future__ import annotations

import time
from typing import List, Optional, Dict, Any

import cv2
import numpy as np
from PyQt6.QtCore import (
    Qt,
    QThread,
    pyqtSignal,
    QTimer,
    QMutex,
    QMutexLocker,
)
from PyQt6.QtGui import QImage, QPixmap, QCursor
from PyQt6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QFrame,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedLayout,
)

# ── Importações do Core ───────────────────────────────────────────────────────
from app.src.UIX.components.shared import IrisAppBar, IrisButton, C
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
_RENDER_INTERVAL_MS = int((1 / 33) * 1000) # ~30ms


# ─────────────────────────────────────────────────────────────────────────────
# Worker de análise pesada
# ─────────────────────────────────────────────────────────────────────────────

class _AnalysisWorker(QThread):
    """
    Roda em background e processa o frame mais recente disponível.
    Agora atua como um conduíte enxuto: o tracking espacial é nativo do QrDecoder.
    """

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
        """Sobrescreve _pending com o frame mais recente (LIFO estrito).
        .copy() é sempre obrigatório — o buffer do CameraManager é reutilizado
        imediatamente pelo próximo frame e corromperia _pending sem a cópia."""
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
                [d.x1, d.y1],
                [d.x2, d.y1],
                [d.x2, d.y2],
                [d.x1, d.y2]
            ], dtype=np.int32)
            boxes.append(pts)

        # O QrDecoder recusa o processamento de imagens cacheadas e retorna None instantaneamente
        decoded_tuples = self._decoder.decode(frame, boxes)

        for d, (text, _patch, _ox, _oy) in zip(detections, decoded_tuples):
            results.append({
                "detection": d,
                "text": text
            })

        # O processamento visual inútil no background causava pressão de alocação.
        # Retorna array vazio para respeitar a tipagem, pois o receiver descarta o objeto imagem.
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

            # Cast estrito para string para evitar crashes no OpenCV bindings (C++)
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
# Card de histórico
# ─────────────────────────────────────────────────────────────────────────────

class _HistoryCard(QFrame):
    def __init__(self, index: int, text_value: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("HistCard")
        self.setStyleSheet(f"""
            #HistCard {{
                background: {C['card']};
                border: 1px solid {C['border']};
                border-left: 3px solid {C['success']};
                border-radius: 4px;
            }}
        """)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(10)

        idx_lbl = QLabel(f"#{index:03d}")
        idx_lbl.setFixedWidth(38)
        idx_lbl.setStyleSheet(f"""
            color: {C['primary']}; font-size: 10px;
            font-weight: 700; letter-spacing: 1px; background: transparent;
        """)
        lay.addWidget(idx_lbl)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFixedWidth(1)
        sep.setStyleSheet(f"background: {C['border_hi']}; border: none;")
        lay.addWidget(sep)

        # Sanitização também garantida na UI
        text_str = str(text_value)
        display_text = text_str[:35] + "..." if len(text_str) > 35 else text_str

        val_lbl = QLabel(f"Valor: {display_text}")
        val_lbl.setWordWrap(True)
        val_lbl.setStyleSheet(f"""
            color: {C['text']}; font-size: 11px;
            letter-spacing: 0.5px; background: transparent;
        """)
        lay.addWidget(val_lbl, stretch=1)


# ─────────────────────────────────────────────────────────────────────────────
# Painel lateral
# ─────────────────────────────────────────────────────────────────────────────

class _SidePanel(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(300)
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

        header = QFrame()
        header.setFixedHeight(48)
        header.setStyleSheet(f"background: {C['card']}; border-bottom: 1px solid {C['border']};")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(14, 0, 14, 0)
        title = QLabel("◆  LOG DE LEITURAS")
        title.setStyleSheet(f"""
            color: {C['primary']}; font-size: 11px;
            font-weight: 700; letter-spacing: 3px; background: transparent;
        """)
        hl.addWidget(title)
        hl.addStretch()
        self._count_lbl = QLabel("0")
        self._count_lbl.setStyleSheet(f"""
            color: {C['text_muted']}; font-size: 10px;
            letter-spacing: 1px; background: transparent;
        """)
        hl.addWidget(self._count_lbl)
        root.addWidget(header)

        stats = QFrame()
        stats.setFixedHeight(52)
        stats.setStyleSheet(f"background: {C['surface']}; border-bottom: 1px solid {C['border']};")
        sl = QHBoxLayout(stats)
        sl.setContentsMargins(14, 8, 14, 8)
        sl.setSpacing(0)
        self._stat_detected = self._make_stat("LOCALIZADOS", "0",  C['secondary'])
        self._stat_reads    = self._make_stat("QRs LIDOS",    "0",  C['success'])
        self._stat_fps      = self._make_stat("FPS",         "—",  C['text_muted'])
        for w in [self._stat_detected, self._stat_reads, self._stat_fps]:
            sl.addWidget(w, stretch=1)
        root.addWidget(stats)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent; border: none;")
        self._hist_container = QWidget()
        self._hist_container.setStyleSheet("background: transparent;")
        self._hist_layout = QVBoxLayout(self._hist_container)
        self._hist_layout.setContentsMargins(10, 10, 10, 10)
        self._hist_layout.setSpacing(6)
        self._hist_layout.addStretch()
        scroll.setWidget(self._hist_container)
        root.addWidget(scroll, stretch=1)

        clear_btn = QPushButton("LIMPAR HISTÓRICO")
        clear_btn.setFixedHeight(36)
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
        """)
        clear_btn.clicked.connect(self._clear_history)
        root.addWidget(clear_btn)

        self._history_count = 0
        self._recent_texts: dict[str, float] = {}

    @staticmethod
    def _make_stat(label: str, value: str, color: str) -> QWidget:
        w = QWidget()
        w.setStyleSheet("background: transparent;")
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(2)
        vl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        val = QLabel(value)
        val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        val.setObjectName("val")
        val.setStyleSheet(f"""
            color: {color}; font-size: 14px;
            font-weight: 700; background: transparent;
        """)
        lbl = QLabel(label)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(f"""
            color: {C['text_muted']}; font-size: 8px;
            letter-spacing: 1px; background: transparent;
        """)
        vl.addWidget(val)
        vl.addWidget(lbl)
        return w

    def _get_stat_val(self, stat_widget: QWidget) -> QLabel:
        return stat_widget.findChild(QLabel, "val")

    def update_stats(
        self,
        detected: int,
        read_count: int,
        fps: float,
    ) -> None:
        self._get_stat_val(self._stat_detected).setText(str(detected))
        self._get_stat_val(self._stat_reads).setText(str(read_count))
        self._get_stat_val(self._stat_fps).setText(f"{fps:.1f}")

    def add_detection(self, text_value: str) -> None:
        now = time.monotonic()
        text_str = str(text_value)

        self._recent_texts = {
            k: v for k, v in self._recent_texts.items() if now - v < 2.0
        }

        if text_str in self._recent_texts:
            return

        self._recent_texts[text_str] = now
        self._history_count += 1
        self._count_lbl.setText(str(self._history_count))
        card = _HistoryCard(self._history_count, text_str)
        self._hist_layout.insertWidget(self._hist_layout.count() - 1, card)
        if self._hist_layout.count() - 1 > _HIST_MAX:
            item = self._hist_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

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
    """
    Tela de detecção ao vivo usando IrisDetector.
    """

    _frame_signal = pyqtSignal(object, object)
    # _raw_frame_signal REMOVIDO para erradicar o vazamento da fila de eventos do Qt.

    def __init__(self, on_back=None, parent=None) -> None:
        super().__init__(parent)
        self._on_back = on_back

        self.last_annotated: Optional[np.ndarray] = None
        self.last_results: List[Dict[str, Any]] = []

        self._last_ts        = time.monotonic()
        self._fps_alpha      = 0.2
        self._fps_smooth     = 0.0

        self._worker: Optional[_AnalysisWorker] = None
        self._cam:    Optional[SingleCameraManager]   = None

        # Mutex de controle do render da UI
        self._ui_frame_mutex = QMutex()
        self._latest_ui_frame: Optional[np.ndarray] = None

        # QTimer dita o rate do UI Thread limitando-o artificialmente de maneira sadia
        self._render_timer = QTimer(self)
        self._render_timer.timeout.connect(self._pull_and_render)

        self._frame_signal.connect(self._on_analysis_done)
        self._build()

    # ── Construção da UI ──────────────────────────────────────────────────────

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(
            IrisAppBar("Live Detector", show_back_button=True, on_back=self._handle_back)
        )

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        feed_wrapper = QWidget()
        feed_wrapper.setStyleSheet(f"background: {C['bg']};")
        fw = QVBoxLayout(feed_wrapper)
        fw.setContentsMargins(0, 0, 0, 0)
        fw.setSpacing(0)

        ctrl_bar = QFrame()
        ctrl_bar.setFixedHeight(48)
        ctrl_bar.setStyleSheet(f"""
            background: {C['surface']};
            border-bottom: 1px solid {C['border']};
        """)
        cl = QHBoxLayout(ctrl_bar)
        cl.setContentsMargins(16, 0, 16, 0)

        self._status_dot = QLabel("●")
        self._status_dot.setStyleSheet(f"color: {C['text_muted']}; background: transparent;")
        cl.addWidget(self._status_dot)

        self._status_lbl = QLabel("CÂMERA INATIVA")
        self._status_lbl.setStyleSheet(f"""
            color: {C['text_muted']}; font-size: 10px;
            font-weight: 600; letter-spacing: 2px; background: transparent;
        """)
        cl.addWidget(self._status_lbl)
        cl.addStretch()

        self._toggle_btn = IrisButton("▶  INICIAR", on_click=self._toggle_capture, width=140)
        cl.addWidget(self._toggle_btn)
        fw.addWidget(ctrl_bar)

        container = QWidget()
        container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._view_stack = QStackedLayout(container)
        self._idle_overlay = self._build_idle_overlay()
        self._feed_label   = QLabel()
        self._feed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._feed_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._feed_label.setStyleSheet("background: black;")
        self._view_stack.addWidget(self._idle_overlay)
        self._view_stack.addWidget(self._feed_label)
        fw.addWidget(container, stretch=1)

        body.addWidget(feed_wrapper, stretch=1)

        self._side_panel = _SidePanel()
        body.addWidget(self._side_panel)

        root.addLayout(body, stretch=1)
        root.addWidget(self._build_legend())

    def _build_idle_overlay(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet(f"background: {C['bg']};")
        vl = QVBoxLayout(w)
        vl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vl.setSpacing(12)
        icon = QLabel("◈")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet(f"""
            color: {C['border_hi']}; font-size: 64px; background: transparent;
        """)
        vl.addWidget(icon)
        msg = QLabel("Pressione INICIAR para ativar a câmera")
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        msg.setStyleSheet(f"""
            color: {C['text_muted']}; font-size: 12px;
            letter-spacing: 2px; background: transparent;
        """)
        vl.addWidget(msg)
        return w

    @staticmethod
    def _build_legend() -> QFrame:
        bar = QFrame()
        bar.setFixedHeight(36)
        bar.setStyleSheet(f"background: {C['surface']}; border-top: 1px solid {C['border']};")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(20, 0, 20, 0)
        bl.setSpacing(24)
        bl.addStretch()
        for color_hex, label in [
            ("#14284C", "Localizado (Não Lido)"),
            ("#4ADE80", "Lido com Sucesso"),
        ]:
            dot = QLabel("■")
            dot.setStyleSheet(f"color: {color_hex}; font-size: 12px; background: transparent;")
            txt = QLabel(label)
            txt.setStyleSheet(f"""
                color: {C['text_muted']}; font-size: 10px;
                letter-spacing: 1px; background: transparent;
            """)
            bl.addWidget(dot)
            bl.addWidget(txt)
        bl.addStretch()
        return bar

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
        self._status_dot.setStyleSheet(f"color: {C['primary']}; background: transparent;")
        self._status_lbl.setText("CÂMERA ATIVA")
        self._status_lbl.setStyleSheet(f"""
            color: {C['primary']}; font-size: 10px;
            font-weight: 600; letter-spacing: 2px; background: transparent;
        """)
        self._view_stack.setCurrentIndex(1)
        self._last_ts = time.monotonic()

        # Inicializa o loop de captura de estado assíncrono para a tela.
        self._render_timer.start(_RENDER_INTERVAL_MS)

    def _stop_capture(self) -> None:
        self._render_timer.stop()

        if self._cam:
            self._cam.stop()
            self._cam = None

        self._toggle_btn.setText("▶  INICIAR")
        self._status_dot.setStyleSheet(f"color: {C['text_muted']}; background: transparent;")
        self._status_lbl.setText("CÂMERA INATIVA")
        self._status_lbl.setStyleSheet(f"""
            color: {C['text_muted']}; font-size: 10px;
            font-weight: 600; letter-spacing: 2px; background: transparent;
        """)
        self._view_stack.setCurrentIndex(0)

    # ── Callbacks de câmera (thread do CameraManager) ─────────────────────────

    def _on_raw_frame(self, frame: np.ndarray) -> None:
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.submit_frame(frame)

        # Salva silenciosamente o ponteiro em buffer LIFO. Zero filas para a UI tratar.
        with QMutexLocker(self._ui_frame_mutex):
            self._latest_ui_frame = frame.copy()

    def _pull_and_render(self) -> None:
        # PULL-MODEL: Disparado exclusivamente pelo QTimer a cada ~30ms
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
        self,
        annotated: np.ndarray,
        results: List[Dict[str, Any]],
    ) -> None:
        self._frame_signal.emit(annotated, results)

    def _on_analysis_done(
        self,
        annotated: np.ndarray,
        results: List[Dict[str, Any]],
    ) -> None:
        now = time.monotonic()
        instant_fps  = 1.0 / max(now - self._last_ts, 1e-6)
        self._fps_smooth = (
            self._fps_alpha * instant_fps
            + (1 - self._fps_alpha) * self._fps_smooth
        )
        self._last_ts = now

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
        self._status_lbl.setStyleSheet(f"""
            color: {C['danger']}; font-size: 10px;
            font-weight: 600; letter-spacing: 2px; background: transparent;
        """)

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
        QTimer.singleShot(300, self._start_capture)

    def _release_resources(self) -> None:
        self._render_timer.stop()

        if self._cam:
            self._cam.stop()
            self._cam = None

        if self._worker:
            self._worker.stop()
            self._worker.wait(3000)
            self._worker = None

        with QMutexLocker(self._ui_frame_mutex):
            self._latest_ui_frame = None

        self.last_annotated  = None
        self.last_results    = []
        self._view_stack.setCurrentIndex(0)
        self._toggle_btn.setText("▶  INICIAR")
        self._status_dot.setStyleSheet(f"color: {C['text_muted']}; background: transparent;")
        self._status_lbl.setText("CÂMERA INATIVA")
        self._status_lbl.setStyleSheet(f"""
            color: {C['text_muted']}; font-size: 10px;
            font-weight: 600; letter-spacing: 2px; background: transparent;
        """)
