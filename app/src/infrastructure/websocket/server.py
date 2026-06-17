"""
infrastructure/websocket/server.py
-----------------------------------
WebSocket server — transmite QR codes lidos para clientes externos (e.g. app C#).

Protocolo de envio:
    Múltiplos códigos lidos no mesmo frame são separados pelo caractere separador
    (padrão "#"). Exemplo com dois QRs simultâneos: "ABC123#XYZ456"

Protocolo de recepção (mensagens do cliente → servidor):
    change_separator_character: X   — troca o separador para o caractere X
    display_ui: True                — solicita exibição da interface gráfica
    display_ui: False               — solicita modo headless (interface oculta)

O servidor roda em thread dedicada com seu próprio event loop asyncio,
permitindo integração sem bloqueio com o loop principal do PyQt6.
"""

from __future__ import annotations

import asyncio
import logging
import threading
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
    ) -> None:
        self._host = host
        self._port = port
        self._separator = "#"
        self._clients: Set["WebSocketServerProtocol"] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event: Optional[asyncio.Event] = None
        # Callback chamado (de thread asyncio) quando comando de controle chega.
        # Exemplo: on_command("display_ui", True)
        # Use sinais Qt para rebridgear para a UI thread.
        self._on_command = on_command

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

    def send(self, codes: List[str]) -> None:
        """
        Envia os códigos QR para todos os clientes conectados.

        Deve ser chamado da thread do Qt — internamente agenda o coroutine
        no event loop do servidor sem bloquear.
        """
        if not codes or self._loop is None or not self._loop.is_running():
            return
        if not self._clients:
            return
        payload = self._separator.join(codes)
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
        logger.info("Cliente WS conectado: %s", ws.remote_address)
        try:
            async for message in ws:
                self._process_incoming(str(message).strip())
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._clients.discard(ws)
            logger.info("Cliente WS desconectado: %s", ws.remote_address)

    def _process_incoming(self, message: str) -> None:
        if message.startswith("change_separator_character:"):
            raw = message.split(":", 1)[1].strip().strip('"').strip("'")
            if raw:
                self._separator = raw
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
