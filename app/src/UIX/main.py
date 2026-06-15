import sys
import threading
import time
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtWidgets import (
    QApplication, QLabel, QMainWindow, QSizePolicy,
    QStackedWidget, QVBoxLayout, QWidget,
)

if sys.platform == "win32":
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
        "Iris.LiveQr.1"
    )

from app.src.UIX.components.shared import C, F_BODY, F_DISPLAY, GLOBAL_STYLE, IrisAperture
from app.src.UIX.main_menu.view import MainMenuView
from app.src.UIX.modules.decoding.live_qr.view import LiveQrView
from app.src.infrastructure.video.camera import SingleCameraManager

APP_ICON_PATH = Path(__file__).resolve().parents[1] / "assets" / "img" / "logo.png"
SVG_HEADER    = Path(__file__).resolve().parents[4] / "docs" / "header.svg"

_LOADING_MIN_MS  = 1_000   # tempo mínimo de exibição da loading screen
_CAM_TIMEOUT_MS  = 6_000   # fallback: prossegue mesmo sem câmera


# ─────────────────────────────────────────────────────────────────────────────
# Tela de loading
# ─────────────────────────────────────────────────────────────────────────────

def _build_loading_screen() -> QWidget:
    """
    Widget de loading: SVG do README centralizado + íris respirando embaixo.
    Segue o mesmo padrão de _build_idle_overlay (view.py) para garantir
    que funcione dentro do QStackedWidget.
    """
    w = QWidget()
    w.setStyleSheet(f"background: {C['bg']};")

    vl = QVBoxLayout(w)
    vl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    vl.setSpacing(0)
    vl.setContentsMargins(0, 32, 0, 32)

    # ── SVG do README ─────────────────────────────────────────────────────────
    # Qt renderiza SVG estático (sem suporte a animações SMIL).
    # A animação visual fica por conta da IrisAperture abaixo.
    if SVG_HEADER.exists():
        svg = QSvgWidget(str(SVG_HEADER))
        # Preserva proporção 900:400 e expande com a janela
        svg.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        svg.setMinimumWidth(400)
        svg.setFixedHeight(320)  # ≈ 900×400 escalado para caber na tela
        svg.setStyleSheet("background: transparent;")
        vl.addWidget(svg)
        vl.addSpacing(28)
    else:
        # Fallback se SVG não existir: só a íris grande
        vl.addSpacing(40)

    # ── Íris respirando — único elemento animado (sinal de atividade) ─────────
    iris = IrisAperture(diameter=80, openness=0.12)
    vl.addWidget(iris, alignment=Qt.AlignmentFlag.AlignCenter)
    iris.start_breathing(lo=0.12, hi=0.68)
    vl.addSpacing(16)

    # ── Status ────────────────────────────────────────────────────────────────
    sub = QLabel("INICIALIZANDO CÂMERA")
    sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
    sub.setStyleSheet(
        f"color: {C['text_muted']}; font-family: {F_DISPLAY};"
        " font-size: 10px; font-weight: 600; letter-spacing: 4px;"
    )
    vl.addWidget(sub)

    w._iris = iris  # type: ignore[attr-defined]
    return w


# ─────────────────────────────────────────────────────────────────────────────
# Janela principal
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    _cam_ready_sig = pyqtSignal()   # ponte câmera-thread → UI-thread

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
        self._cam: SingleCameraManager | None = None
        self._initialized    = False
        self._cam_notified   = False   # evita dismiss duplo
        self._loading_start  = 0.0     # monotonic quando _start_init rodou

        # ── Loading screen: índice 0 (exibido por padrão pelo QStackedWidget) ─
        self._loading_widget = _build_loading_screen()
        self._stack.addWidget(self._loading_widget)

        # ── Menu principal: índice 1 ───────────────────────────────────────────
        self._main_menu = MainMenuView(
            on_live_qr=self._go_live_qr,
            on_exit=self.close,
        )
        self._stack.addWidget(self._main_menu)
        self._stack.setCurrentIndex(0)

        self._cam_ready_sig.connect(self._on_cam_frame_received)

        self._cam_timeout = QTimer(self)
        self._cam_timeout.setSingleShot(True)
        self._cam_timeout.setInterval(_CAM_TIMEOUT_MS)
        self._cam_timeout.timeout.connect(self._dismiss_loading)

    # ── Ciclo de vida ─────────────────────────────────────────────────────────

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.raise_()
        self.activateWindow()
        if not self._initialized:
            self._initialized = True
            # 1.5 s de delay: loading screen visível antes da câmera iniciar
            QTimer.singleShot(1_500, self._start_init)

    def _start_init(self) -> None:
        """Câmera e warmup iniciam aqui — loading screen já está na tela."""
        self._loading_start = time.monotonic()

        # Câmera em thread separada (SingleCameraManager é assíncrono) —
        # a UI nunca bloqueia aguardando abertura do dispositivo.
        self._cam = SingleCameraManager(camera_index=0, force_mjpg=True)
        self._cam.subscribe(self._cam_frame_probe)
        self._cam.start()
        self._cam_timeout.start()

        # ONNX warmup em daemon thread — não bloqueia event loop
        threading.Thread(
            target=self._prewarm_detector,
            name="IrisDetector-Warmup",
            daemon=True,
        ).start()

    # ── Câmera ────────────────────────────────────────────────────────────────

    def _cam_frame_probe(self, _frame) -> None:
        """Thread da câmera → emite sinal para UI thread (não bloqueia nada)."""
        self._cam_ready_sig.emit()

    def _on_cam_frame_received(self) -> None:
        """UI thread: câmera deu primeiro frame — respeita mínimo de 1 s."""
        if self._cam is not None:
            try:
                self._cam.unsubscribe(self._cam_frame_probe)
            except Exception:
                pass

        elapsed_ms = int((time.monotonic() - self._loading_start) * 1000)
        delay_ms   = max(0, _LOADING_MIN_MS - elapsed_ms)
        QTimer.singleShot(delay_ms, self._dismiss_loading)

    def _dismiss_loading(self) -> None:
        if self._cam_notified:
            return
        self._cam_notified = True
        self._cam_timeout.stop()

        iris = getattr(self._loading_widget, "_iris", None)
        if iris is not None:
            iris.stop_breathing()

        self._stack.setCurrentWidget(self._main_menu)
        QTimer.singleShot(200, self._cleanup_loading)

    def _cleanup_loading(self) -> None:
        if self._loading_widget is not None:
            self._stack.removeWidget(self._loading_widget)
            self._loading_widget.deleteLater()
            self._loading_widget = None  # type: ignore[assignment]

    # ── Warmup do detector ────────────────────────────────────────────────────

    @staticmethod
    def _prewarm_detector() -> None:
        """
        Isola D3D12/DML em subprocesso separado — o processo principal nunca
        toca D3D12 durante a compilação de shaders, eliminando o freeze de UI.

        Fluxo:
        1. Subprocesso: compila shaders D3D12 → grava cache em disco (~5–20 s)
        2. Este thread espera o subprocesso terminar (daemon, não bloqueia UI)
        3. Processo principal cria InferenceSession com cache de disco (~<1 s)
           e armazena em _default_session para reuso instantâneo no módulo
        """
        import subprocess  # noqa: PLC0415

        from app.src.engine.modules.decoding.live_qr.detector import (  # noqa: PLC0415
            _DEFAULT_MODEL_PATH,
        )

        warmup_script = (
            Path(__file__).resolve().parents[1]
            / "engine" / "modules" / "decoding" / "live_qr" / "_dml_warmup.py"
        )

        # ── Passo 1: subprocesso aquece cache D3D12 em disco ─────────────────
        if warmup_script.exists():
            try:
                result = subprocess.run(
                    [sys.executable, str(warmup_script), str(_DEFAULT_MODEL_PATH)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=90,
                )
                if result.returncode == 0:
                    print("[WARMUP] Subprocesso DML concluído — cache em disco pronto")
                else:
                    print("[WARMUP] Subprocesso DML encerrou com erro (continuando)")
            except subprocess.TimeoutExpired:
                print("[WARMUP] Subprocesso DML excedeu timeout")
            except Exception as exc:
                print(f"[WARMUP] Subprocesso DML falhou: {exc}")
        else:
            print(f"[WARMUP] Script não encontrado: {warmup_script}")

        # ── Passo 2: sessão no processo principal via cache de disco ──────────
        # D3D12 já está aquecido → criação rápida, sem compilação, sem freeze
        try:
            from app.src.engine.modules.decoding.live_qr.detector import IrisDetector  # noqa: PLC0415
            IrisDetector()
            print("[WARMUP] IrisDetector pronto — sessão em cache de memória")
        except Exception as exc:
            print(f"[WARMUP] IrisDetector falhou: {exc}")

    # ── Navegação ─────────────────────────────────────────────────────────────

    def _go_home(self) -> None:
        if self._live_qr_view:
            self._stack.removeWidget(self._live_qr_view)
            self._live_qr_view.deleteLater()
            self._live_qr_view = None
        self._stack.setCurrentWidget(self._main_menu)

    def _go_live_qr(self) -> None:
        if not self._live_qr_view:
            QTimer.singleShot(0, self._create_and_open_live_qr)
            return
        self._stack.setCurrentWidget(self._live_qr_view)

    def _create_and_open_live_qr(self) -> None:
        if not self._live_qr_view:
            self._live_qr_view = LiveQrView(on_back=self._go_home, camera=self._cam)
            self._stack.addWidget(self._live_qr_view)
        self._stack.setCurrentWidget(self._live_qr_view)

    def closeEvent(self, event) -> None:
        if self._cam:
            self._cam.stop()
            self._cam = None
        super().closeEvent(event)


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
