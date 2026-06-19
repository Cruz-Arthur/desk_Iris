import atexit
import logging
import sys
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon
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
from app.src.infrastructure.websocket import QrWebSocketServer

from app.src.utils.paths import ASSETS_DIR as _ASSETS_DIR, DOCS_DIR as _DOCS_DIR
APP_ICON_PATH = _ASSETS_DIR / "img" / "logo.png"
SVG_HEADER    = _DOCS_DIR / "header.svg"

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

    # ── Logo PNG ──────────────────────────────────────────────────────────────
    if APP_ICON_PATH.exists():
        from PyQt6.QtGui import QPixmap
        logo = QLabel()
        pix = QPixmap(str(APP_ICON_PATH)).scaled(
            120, 120,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        logo.setPixmap(pix)
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setStyleSheet("background: transparent;")
        vl.addWidget(logo)
        vl.addSpacing(24)
    else:
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

    _cam_ready_sig  = pyqtSignal()        # ponte câmera-thread → UI-thread
    _ui_command_sig = pyqtSignal(str, bool)  # ponte asyncio-thread → UI-thread

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

        self._live_qr_view: LiveQrView | None = None
        self._cam: SingleCameraManager | None = None
        self._initialized    = False
        self._cam_notified   = False   # evita dismiss duplo
        self._loading_start  = 0.0     # monotonic quando _start_init rodou

        # ── WebSocket server — criado aqui para sobreviver ao ciclo de vida da UI
        # exit_on_disconnect=False: o SyncAssistente gerencia o ciclo de vida do
        # Iris (inicia + mata via taskkill). Se o Iris se matasse a cada queda de
        # conexão, brigaria com a reconexão do Sync → loop de morte (suicídio →
        # relança → suicídio). Quem mata o Iris é só o Sync, explicitamente.
        self._ws_server = QrWebSocketServer(
            on_command=self._on_ws_command,
            exit_on_disconnect=False,
        )
        self._ws_server.start()
        atexit.register(self._ws_server.broadcast_closing)  # cobre crashes e saídas não-Qt

        # ── Push periódico de status (FPS real, estado, clientes) a cada 1s ───
        # Garante que o cliente receba o FPS continuamente, mesmo sem detecção
        # de QR e sem depender de polling do lado do cliente.
        self._status_push_timer = QTimer(self)
        self._status_push_timer.setInterval(1000)
        self._status_push_timer.timeout.connect(self._push_status)
        self._status_push_timer.start()

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
        self._ui_command_sig.connect(self._handle_ui_command)

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

        # Cria o LiveQrView em modo headless — workers de câmera/decode iniciam
        # imediatamente e o WebSocket fica ativo. A janela some sem deixar rastros.
        self._create_live_qr_headless()
        QTimer.singleShot(200, self._cleanup_loading)
        QTimer.singleShot(400, self.hide)  # some após a loading screen sumir

    def _create_live_qr_headless(self) -> None:
        if self._live_qr_view is None:
            self._live_qr_view = LiveQrView(
                on_back=self._go_home,
                camera=self._cam,
                ws_server=self._ws_server,
                headless=True,
            )
            self._stack.addWidget(self._live_qr_view)
        # Inicia captura manualmente — showEvent não dispara em widget oculto
        QTimer.singleShot(300, self._live_qr_view._start_capture)

    # ── Comando WebSocket (display_ui) ────────────────────────────────────────

    def _push_status(self) -> None:
        """Empurra status atual para os clientes WS (chamado na UI thread a cada 1s)."""
        view = self._live_qr_view
        self._ws_server.broadcast_status(
            capture=view._cam_subscribed if view else False,
            state=view._state          if view else "idle",
            fps=view._fps_smooth       if view else 0.0,
        )

    def _on_ws_command(self, name: str, value) -> None:
        """Chamado da thread asyncio — rebridgea para UI thread via sinal Qt."""
        import json as _json
        if name == "display_ui":
            self._ui_command_sig.emit(name, bool(value))
        elif name in ("start_capture", "stop_capture"):
            self._ui_command_sig.emit(name, True)
        elif name == "get_status":
            ws = value
            view = self._live_qr_view
            payload = _json.dumps({
                "type": "status",
                "capture": view._cam_subscribed if view else False,
                "state": view._state if view else "idle",
                "fps": round(view._fps_smooth, 1) if view else 0.0,
                "clients": len(self._ws_server._clients),
            })
            self._ws_server.reply(ws, payload)
    def _handle_ui_command(self, name: str, value: bool) -> None:
        """Executa na UI thread — seguro para manipular widgets."""
        if name == "display_ui":
            if value:
                self._stack.setCurrentWidget(self._main_menu)
                self.show()
                self.raise_()
                self.activateWindow()
                if self._live_qr_view is not None:
                    self._live_qr_view.enable_render()
            else:
                if self._live_qr_view is not None:
                    self._live_qr_view.disable_render()
                self.hide()
        elif name == "start_capture":
            if self._live_qr_view is not None:
                self._live_qr_view._start_capture()
        elif name == "stop_capture":
            if self._live_qr_view is not None:
                self._live_qr_view._stop_capture()

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

        from app.src.utils.paths import APP_ROOT as _APP_ROOT
        warmup_script = (
            _APP_ROOT / "app" / "src" / "engine" / "modules" / "decoding" / "live_qr" / "_dml_warmup.py"
        )

        # Em app frozen (PyInstaller onefile) sys.executable é o próprio Iris.exe,
        # não um interpretador Python — relançá-lo dispararia outra instância
        # (barrada pelo mutex) em vez de aquecer o cache. Pula direto para o
        # passo 2, criando a sessão no próprio processo.
        _frozen = getattr(sys, "frozen", False)

        # ── Passo 1: subprocesso aquece cache D3D12 em disco ─────────────────
        if not _frozen and warmup_script.exists():
            try:
                result = subprocess.run(
                    [sys.executable, str(warmup_script), str(_DEFAULT_MODEL_PATH)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=90,
                )
                if result.returncode == 0:
                    pass
                else:
                    pass
            except subprocess.TimeoutExpired:
                pass
            except Exception:
                pass

        # ── Passo 2: sessão no processo principal via cache de disco ──────────
        # D3D12 já está aquecido → criação rápida, sem compilação, sem freeze
        try:
            from app.src.engine.modules.decoding.live_qr.detector import IrisDetector  # noqa: PLC0415
            IrisDetector()
        except Exception:
            pass

    # ── Navegação ─────────────────────────────────────────────────────────────

    def _go_home(self) -> None:
        # Em modo headless mantemos o LiveQrView vivo — não destruímos.
        self._stack.setCurrentWidget(self._main_menu)

    def _go_live_qr(self) -> None:
        if self._live_qr_view is None:
            self._live_qr_view = LiveQrView(
                on_back=self._go_home,
                camera=self._cam,
                ws_server=self._ws_server,
                headless=True,
            )
            self._stack.addWidget(self._live_qr_view)
        self._stack.setCurrentWidget(self._live_qr_view)

    def closeEvent(self, event) -> None:
        self._ws_server.broadcast_closing()  # avisa clientes antes de fechar
        self._ws_server.stop()
        if self._cam:
            self._cam.stop()
            self._cam = None
        super().closeEvent(event)


def _install_crash_logging() -> None:
    """
    Captura tracebacks de crash em %APPDATA%/Iris/crash.log — essencial no exe
    frozen (windowed), que não tem console. Cobre:
      • segfaults / aborts nativos (faulthandler) — ex.: D3D12/DML, OpenCV
      • exceções Python não tratadas (sys.excepthook)
      • exceções não tratadas em threads (threading.excepthook)
      • exceções em slots Qt (PyQt6 roteia para sys.excepthook antes de abortar)
    """
    import faulthandler
    import traceback
    from app.src.utils.paths import CONFIG_DIR
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        f = open(CONFIG_DIR / "crash.log", "a", encoding="utf-8", buffering=1)
        faulthandler.enable(f)

        def _hook(exc_type, exc, tb):
            f.write("\n==== UNCAUGHT EXCEPTION ====\n")
            traceback.print_exception(exc_type, exc, tb, file=f)
            f.flush()
            logger.error("Crash capturado em crash.log", exc_info=(exc_type, exc, tb))

        sys.excepthook = _hook

        def _thook(args):
            f.write("\n==== UNCAUGHT THREAD EXCEPTION ====\n")
            traceback.print_exception(
                args.exc_type, args.exc_value, args.exc_traceback, file=f
            )
            f.flush()

        threading.excepthook = _thook
    except Exception as exc:
        logger.warning("Falha ao instalar crash logging: %s", exc)


def main():
    _install_crash_logging()
    app = QApplication(sys.argv)
    app.setOrganizationName("Iris")
    app.setApplicationName("Iris")
    app.setStyle("Fusion")
    # Impede que o app encerre ao esconder a janela principal (modo headless).
    app.setQuitOnLastWindowClosed(False)
    if APP_ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(APP_ICON_PATH)))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
