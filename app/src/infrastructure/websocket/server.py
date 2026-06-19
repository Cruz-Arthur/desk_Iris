"""
infrastructure/websocket/server.py
-----------------------------------
WebSocket server — transmite QR codes lidos para clientes externos (e.g. app C#).

Protocolo de envio:
    Formato: "@fps=<valor> <código1><sep><código2>..."
    O prefixo "@fps=" informa o FPS real do pipeline no momento da detecção.
    Exemplo com dois QRs a 28.5 fps: "@fps=28.5 ABC123#XYZ456"
    Exemplo com QR único:            "@fps=30.0 ABC123"

Protocolo de recepção (mensagens do cliente → servidor):
    change_separator_character: X   — troca o separador para o caractere X
    display_ui: True                — solicita exibição da interface gráfica
    display_ui: False               — solicita modo headless (interface oculta)
    start_capture                   — inicia pipeline câmera+detector
    stop_capture                    — para pipeline câmera+detector
    get_status                      — responde JSON com estado atual
    ping                            — responde "pong"

O servidor roda em thread dedicada com seu próprio event loop asyncio,
permitindo integração sem bloqueio com o loop principal do PyQt6.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from typing import Any, Callable, List, Optional, Set

logger = logging.getLogger(__name__)

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False
    logger.warning(
        "websockets não instalado — servidor WebSocket desativado. "
        "Instale com: pip install websockets"
    )

_DEFAULT_HOST = "0.0.0.0"
_DEFAULT_PORT = 8765


class QrWebSocketServer:
    """
    Servidor WebSocket que transmite códigos QR decodificados em tempo real.

    Uso:
        server = QrWebSocketServer()
        server.start()          # inicia em background (não bloqueia)
        server.send(["ABC"])    # chamável de qualquer thread
        server.stop()           # encerra o servidor
    """

    def __init__(
        self,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
        on_command: Optional[Callable[[str, Any], None]] = None,
        exit_on_disconnect: bool = False,
    ) -> None:
        self._host = host
        self._port = port
        self._separator = self._load_separator()  # persistido entre execuções
        self._clients: Set["WebSocketServerProtocol"] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event: Optional[asyncio.Event] = None
        # Callback chamado (de thread asyncio) quando comando de controle chega.
        # Exemplo: on_command("display_ui", True)
        # Use sinais Qt para rebridgear para a UI thread.
        self._on_command = on_command
        # Quando True, encerra o processo imediatamente ao ficar sem clientes.
        self._exit_on_disconnect = exit_on_disconnect
        # Só conta após o primeiro cliente ter conectado — não termina na subida.
        self._had_client = False
        self._closing_sent = False  # garante envio único mesmo com atexit + closeEvent

    # ── Persistência do separador ───────────────────────────────────────────────

    @staticmethod
    def _settings_path():
        from app.src.utils.paths import CONFIG_DIR
        return CONFIG_DIR / "ws_settings.json"

    def _load_separator(self) -> str:
        """Lê o separador salvo; default '#' se ausente ou inválido."""
        try:
            path = self._settings_path()
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                sep = data.get("separator")
                if isinstance(sep, str) and len(sep) == 1:
                    return sep
        except Exception as exc:
            logger.warning("Falha ao carregar separador persistido: %s", exc)
        return "#"

    def _save_separator(self) -> None:
        """Grava o separador atual para as próximas inicializações."""
        try:
            path = self._settings_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"separator": self._separator}), encoding="utf-8")
            logger.info("Separador persistido: %r", self._separator)
        except Exception as exc:
            logger.warning("Falha ao salvar separador: %s", exc)

    # ── Ciclo de vida ──────────────────────────────────────────────────────────

    def start(self) -> None:
        if not _WS_AVAILABLE:
            return
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="qr-ws-server"
        )
        self._thread.start()
        logger.info("QrWebSocketServer iniciado em ws://%s:%d", self._host, self._port)

    def stop(self) -> None:
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    # ── API pública (thread-safe) ──────────────────────────────────────────────

    def reply(self, ws: "WebSocketServerProtocol", payload: str) -> None:
        """Envia uma resposta diretamente a um cliente específico (thread-safe)."""
        if self._loop is None or not self._loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(self._send_one(ws, payload), self._loop)

    def broadcast_status(
        self, capture: bool, state: str, fps: float, clients: int = -1
    ) -> None:
        """
        Empurra o estado atual (incl. FPS real) para todos os clientes.

        Push periódico — desacopla a entrega de FPS da detecção de QR e de
        qualquer polling do cliente. Usa o mesmo caminho de _broadcast já
        comprovado por send(). Chamável de qualquer thread (e.g. Qt).
        """
        if self._loop is None or not self._loop.is_running():
            return
        if not self._clients:
            return
        if clients < 0:
            clients = len(self._clients)
        payload = json.dumps({
            "type": "status",
            "capture": bool(capture),
            "state": str(state),
            "fps": round(float(fps), 1),
            "clients": clients,
        })
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), self._loop)

    def broadcast_closing(self) -> None:
        """
        Envia {"type":"closing"} para todos os clientes e aguarda entrega (bloqueante).

        Deve ser chamado antes de stop() — no closeEvent e via atexit — para que
        o cliente C# saiba que o processo está encerrando intencionalmente e possa
        iniciar a reconexão sem esperar o timeout de socket.
        Seguro para chamar múltiplas vezes: só envia uma vez.
        """
        if self._closing_sent:
            return
        self._closing_sent = True
        if self._loop is None or not self._loop.is_running() or not self._clients:
            return
        future = asyncio.run_coroutine_threadsafe(
            self._broadcast('{"type":"closing"}'), self._loop
        )
        try:
            future.result(timeout=1.0)
        except Exception:
            pass

    def send(self, codes: List[str], fps: float = 0.0) -> None:
        """
        Envia os códigos QR para todos os clientes conectados.

        Formato: "@fps=30.2 ABC123#XYZ456"
        O prefixo "@fps=<valor> " sempre precede os códigos para que o cliente
        possa extraí-lo sem quebrar parsers que apenas fazem split no separador.

        Deve ser chamado da thread do Qt — internamente agenda o coroutine
        no event loop do servidor sem bloquear.
        """
        if not codes or self._loop is None or not self._loop.is_running():
            return
        if not self._clients:
            return
        payload = f"@fps={fps:.1f} {self._separator.join(codes)}"
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), self._loop)

    @property
    def separator(self) -> str:
        return self._separator

    @property
    def port(self) -> int:
        return self._port

    # ── Internals ──────────────────────────────────────────────────────────────

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as exc:
            logger.error("WebSocket server encerrado com erro: %s", exc)
        finally:
            self._loop.close()

    async def _serve(self) -> None:
        self._stop_event = asyncio.Event()
        async with websockets.serve(self._handler, self._host, self._port):
            await self._stop_event.wait()

    async def _handler(self, ws: "WebSocketServerProtocol") -> None:
        self._clients.add(ws)
        self._had_client = True
        self._start_time = getattr(self, "_start_time", time.monotonic())
        logger.info("Cliente WS conectado: %s", ws.remote_address)
        try:
            async for message in ws:
                self._process_incoming(str(message).strip(), ws)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._clients.discard(ws)
            logger.info("Cliente WS desconectado: %s", ws.remote_address)
            if self._exit_on_disconnect and self._had_client and not self._clients:
                logger.info("Último cliente desconectou — encerrando processo.")
                os._exit(0)

    def _process_incoming(self, message: str, ws: "WebSocketServerProtocol") -> None:
        if message.startswith("change_separator_character:"):
            raw = message.split(":", 1)[1].strip().strip('"').strip("'")
            if raw:
                self._separator = raw[0]
                self._save_separator()  # persiste para as próximas inicializações
                logger.info("Separador alterado para %r", self._separator)
            else:
                logger.warning("change_separator_character recebido sem caractere válido")

        elif message.startswith("display_ui:"):
            val = message.split(":", 1)[1].strip().lower()
            show = val in ("true", "1", "yes")
            logger.info("display_ui: %s", show)
            if self._on_command:
                try:
                    self._on_command("display_ui", show)
                except Exception as exc:
                    logger.error("on_command error: %s", exc)

        elif message == "ping":
            self.reply(ws, "pong")

        elif message in ("start_capture", "stop_capture"):
            if self._on_command:
                try:
                    self._on_command(message, None)
                except Exception as exc:
                    logger.error("on_command error: %s", exc)

        elif message == "get_status":
            if self._on_command:
                try:
                    self._on_command("get_status", ws)
                except Exception as exc:
                    logger.error("on_command error: %s", exc)

        else:
            logger.debug("Mensagem WS ignorada: %r", message)

    async def _broadcast(self, payload: str) -> None:
        dead: Set["WebSocketServerProtocol"] = set()
        for ws in set(self._clients):
            try:
                await ws.send(payload)
            except Exception:
                dead.add(ws)
        self._clients -= dead

    async def _send_one(self, ws: "WebSocketServerProtocol", payload: str) -> None:
        try:
            await ws.send(payload)
        except Exception:
            self._clients.discard(ws)
