from __future__ import annotations

"""
video/camera.py
---------------

Infraestrutura central de captura de câmera.

Responsabilidades:
- gerenciar captura de câmera única
- gerenciar múltiplas câmeras
- manter apenas o último frame disponível (LIFO)
- tolerar índices inválidos / câmeras ausentes
- monitorar perda de sinal
- tentar reconexão automática
- expor estado de câmera para a UI

Estruturas principais:
- SingleCameraManager
- MultiCameraManager
"""

import threading
import time

from collections.abc import Callable
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Optional, Dict, List, Tuple

import cv2
import numpy as np


FrameCallback = Callable[[np.ndarray], None]
CameraDeviceInfo = Dict[str, Any]

_RESOLUTION_CANDIDATES: list[tuple[int, int]] = [
    (1920, 1080),
    (1280, 720),
    (640, 480),
]

# ── Backend cache ─────────────────────────────────────────────────────────────
# Keyed by camera index.  Populated on first successful open; consumed on
# reconnect to skip the full probe loop.  Cleared when the cached backend
# fails so the next start() retries the full ordered list.
_backend_cache: Dict[int, Tuple[str, int]] = {}


def _get_friendly_camera_names() -> List[str]:
    """
    Return OS-level camera names when Qt Multimedia is available.

    The video layer does not depend on UI widgets. Qt Multimedia is imported
    lazily and treated only as an optional device metadata provider.
    """
    try:
        from PyQt6.QtMultimedia import QMediaDevices
    except Exception:
        return []

    try:
        names: List[str] = []
        for device in QMediaDevices.videoInputs():
            try:
                names.append(device.description().strip())
            except Exception:
                names.append("")
        return names
    except Exception:
        return []


def discover_cameras(max_index: int = 10, verify_access: bool = True) -> List[CameraDeviceInfo]:
    """
    Discover available camera indexes using OpenCV.

    This function is intentionally UI-agnostic so any screen can reuse the same
    discovery path. The returned dictionaries keep the existing shape consumed
    by the UI and runtime managers:

    {
        "index": 0,
        "label": "Camera name",
        "display_label": "Camera name  (idx 0)"
    }
    """
    friendly_names = _get_friendly_camera_names()
    cameras: List[CameraDeviceInfo] = []

    if not verify_access:
        for index, friendly in enumerate(friendly_names[:max_index]):
            label = (friendly or "").strip() or f"Camera {index}"
            cameras.append({
                "index": index,
                "label": label,
                "display_label": f"{label}  (idx {index})",
            })
        return cameras

    probe_limit = min(max_index, max(len(friendly_names) + 2, 3))

    for index in range(probe_limit):
        cap = None
        try:
            # Evita DSHOW na descoberta inicial para reduzir warnings ruidosos
            # em maquinas que anunciam o backend mas falham ao abrir por indice.
            cap = cv2.VideoCapture(index, cv2.CAP_ANY)

            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(index, cv2.CAP_MSMF)

            if not cap.isOpened():
                cap.release()
                continue

            ok, _ = cap.read()
            if not ok:
                cap.release()
                continue

            friendly = ""
            if index < len(friendly_names):
                friendly = (friendly_names[index] or "").strip()

            if not friendly:
                friendly = f"Camera {index}"

            cameras.append({
                "index": index,
                "label": friendly,
                "display_label": f"{friendly}  (idx {index})",
            })

        except Exception:
            pass
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    label_totals = Counter(str(cam.get("label") or "") for cam in cameras)
    label_seen: defaultdict[str, int] = defaultdict(int)

    for cam in cameras:
        label = str(cam.get("label") or "")
        label_seen[label] += 1
        occurrence = label_seen[label]
        cam["label_occurrence"] = occurrence
        cam["identity_label"] = f"{label}#{occurrence}"

        if label_totals[label] > 1:
            cam["display_label"] = f"{label} #{occurrence}  (idx {cam['index']})"

    return cameras


def _apply_max_resolution(
    cap: cv2.VideoCapture,
    target: tuple[int, int] | None = None,
) -> tuple[int, int]:
    w, h = target or _RESOLUTION_CANDIDATES[0]
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    return (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

# =========================================================
# OPEN CAMERA
# =========================================================

def _open_camera(
    index: int,
    backend_order: Optional[List[tuple[str, int]]] = None,
) -> tuple[Optional[cv2.VideoCapture], str]:
    """
    Opens a camera with a cache-aware, DSHOW-first probe strategy.

    Performance decisions
    ─────────────────────
    • DSHOW is tried first: on Windows, DirectShow initialises in ~300–600 ms
      vs. MSMF's 2–4 s Media Foundation pipeline startup.
    • CAP_ANY is omitted from the default list because on Windows it resolves
      internally to MSMF — probing it first just adds a redundant slow attempt.
    • The module-level _backend_cache stores the winner per index so reconnects
      skip the probe loop entirely (cache hit path ~50–200 ms).
    """
    # ── Fast path: cached backend ────────────────────────────────────────────
    cached = _backend_cache.get(index)
    if cached is not None:
        cached_name, cached_id = cached
        cap: Optional[cv2.VideoCapture] = None
        try:
            cap = cv2.VideoCapture(index, cached_id)
            if cap and cap.isOpened():
                print(f"[CAM {index}] Cache hit → {cached_name}")
                return cap, cached_name
        except Exception:
            pass
        if cap:
            try:
                cap.release()
            except Exception:
                pass
        # Stale cache entry — clear and fall through to full probe
        del _backend_cache[index]
        print(f"[CAM {index}] Cache stale, re-probing")

    # ── Full probe: DSHOW first, then MSMF, then ANY as last resort ──────────
    attempts = backend_order or [
        ("ANY",   cv2.CAP_ANY),    # deixa o OS escolher — mais rápido na maioria das máquinas
        ("MSMF",  cv2.CAP_MSMF),   # Media Foundation — fallback Windows
        ("DSHOW", cv2.CAP_DSHOW),  # DirectShow — lento em algumas câmeras, último recurso
    ]

    for name, backend_id in attempts:
        cap = None
        try:
            cap = cv2.VideoCapture(index, backend_id)
            if cap and cap.isOpened():
                _backend_cache[index] = (name, backend_id)
                print(f"[CAM {index}] Opened via {name}")
                return cap, name
        except Exception:
            pass
        if cap:
            try:
                cap.release()
            except Exception:
                pass

    return None, "NONE"

# =========================================================
# MANAGER DE CAMERA UNITÁRIA
# =========================================================

class SingleCameraManager:
    """
    Gerencia uma camera fisica com captura e despacho desacoplados.

    A captura sempre drena o hardware e guarda apenas o ultimo frame. O limite
    de FPS, quando existe, e aplicado somente no despacho para callbacks.
    """

    def __init__(
        self,
        camera_index: int,
        fps_limit: Optional[float] = None,
        backend_order: Optional[List[tuple[str, int]]] = None,
        force_mjpg: bool = False,
        stabilize_seconds: float = 0.0,
        warmup_frames: int = 0,
        resolution: Optional[tuple[int, int]] = None,
    ):
        # -------------------------------------------------
        # Configuração base da câmera
        # -------------------------------------------------
        self._index = camera_index
        self._fps_limit = fps_limit
        self._backend_order = backend_order
        self._force_mjpg = force_mjpg
        self._stabilize_seconds = stabilize_seconds
        self._warmup_frames = warmup_frames
        self._resolution = resolution

        # -------------------------------------------------
        # Handle do OpenCV
        # -------------------------------------------------
        self._cap: Optional[cv2.VideoCapture] = None

        # -------------------------------------------------
        # Estado geral
        # -------------------------------------------------
        self._running = False
        self._stop_event = threading.Event()
        self._backend = "NONE"

        # -------------------------------------------------
        # Threads separadas:
        # 1) init hardware (non-blocking — spawned by start())
        # 2) captura frames do hardware
        # 3) despacha para callbacks
        # -------------------------------------------------
        self._init_thread:     Optional[threading.Thread] = None
        self._capture_thread:  Optional[threading.Thread] = None
        self._dispatch_thread: Optional[threading.Thread] = None

        # Signalled by _init_and_start() when the cap handle is ready
        # (or when init fails).  Capture / dispatch loops block on this
        # before touching self._cap.
        self._init_done = threading.Event()

        # -------------------------------------------------
        # Assinantes (callbacks)
        # -------------------------------------------------
        self._subscribers: set[FrameCallback] = set()
        self._lock = threading.Lock()

        # -------------------------------------------------
        # Pending camera properties
        #
        # set_property() calls that arrive before the hardware is ready
        # are queued here and flushed atomically when _cap is assigned.
        # -------------------------------------------------
        self._props_lock = threading.Lock()
        self._pending_props: List[Tuple[int, float]] = []

        # -------------------------------------------------
        # Buffer LIFO interno
        #
        # A thread de captura sobrescreve sempre o último frame.
        # A thread de despacho consome esse frame mais recente.
        # -------------------------------------------------
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._new_frame_event = threading.Event()

    # =====================================================
    # START  (non-blocking)
    # =====================================================

    def start(self):
        """
        Returns immediately.

        Hardware initialisation (backend probe, resolution negotiation,
        optional stabilise/warmup) runs in a background _init_thread.
        Capture and dispatch loops are started from that thread only after
        the cap handle is confirmed open, so callers never wait on slow
        driver startup.

        Callers that need to know when the camera is truly ready can wait
        on self._init_done (internal) or simply subscribe and wait for the
        first frame callback — the existing contract for all consumers.
        """
        print(f"[CAM {self._index}] Iniciando (async)...")

        self._stop_event.clear()
        self._init_done.clear()
        self._new_frame_event.clear()
        self._running = True   # optimistic — _init_and_start corrects on failure

        self._init_thread = threading.Thread(
            target=self._init_and_start,
            name=f"CameraInit-{self._index}",
            daemon=True,
        )
        self._init_thread.start()
        return self

    # =====================================================
    # INIT (background thread)
    # =====================================================

    def _init_and_start(self) -> None:
        """
        Heavy initialisation executed off the calling thread.

        Responsibilities
        ────────────────
        1. Probe / open the camera backend (cache-aware, DSHOW-first)
        2. Apply MJPG fourcc when requested
        3. Set minimum backend buffer (prevents stale-frame accumulation)
        4. Single-shot resolution negotiation
        5. Optional stabilise sleep + warmup reads
        6. Atomically assign self._cap and flush any pending set_property calls
        7. Start capture + dispatch threads
        8. Signal self._init_done regardless of success/failure so any waiter
           (e.g. a test that calls _init_done.wait()) is never deadlocked
        """
        cap: Optional[cv2.VideoCapture] = None
        try:
            # Guard: caller may have called stop() before we got here
            if self._stop_event.is_set():
                return

            cap, backend = _open_camera(self._index, self._backend_order)

            if cap is None or not cap.isOpened():
                print(f"[CAM {self._index}] Init failed: no backend responded")
                self._running = False
                return

            self._backend = backend

            # ── Apply FOURCC before first read ─────────────────────────────
            if self._force_mjpg:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                cap.set(cv2.CAP_PROP_FOURCC, fourcc)

            # ── Minimise internal backend buffer immediately ───────────────
            # Must be set before the first read() to take effect on DSHOW.
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            # ── Single-shot resolution request ────────────────────────────
            actual_w, actual_h = _apply_max_resolution(cap, self._resolution)

            # ── Optional stabilise / warmup ───────────────────────────────
            if self._stabilize_seconds > 0 and not self._stop_event.is_set():
                time.sleep(self._stabilize_seconds)

            if self._warmup_frames > 0 and not self._stop_event.is_set():
                for _ in range(self._warmup_frames):
                    cap.read()

            print(f"[CAM {self._index}] Pronta  {actual_w}×{actual_h}  [{self._backend}]")

            # ── Atomically publish cap + flush pending properties ──────────
            with self._props_lock:
                self._cap = cap
                for prop_id, value in self._pending_props:
                    cap.set(prop_id, value)
                self._pending_props.clear()

            # ── Launch capture + dispatch loops ───────────────────────────
            if self._stop_event.is_set():
                # stop() was called during init — release and abort
                with self._props_lock:
                    self._cap = None
                cap.release()
                self._running = False
                return

            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                name=f"CameraCapture-{self._index}",
                daemon=True,
            )
            self._dispatch_thread = threading.Thread(
                target=self._dispatch_loop,
                name=f"CameraDispatch-{self._index}",
                daemon=True,
            )
            self._capture_thread.start()
            self._dispatch_thread.start()

        except Exception as exc:
            print(f"[CAM {self._index}] Init error: {exc}")
            if cap:
                try:
                    cap.release()
                except Exception:
                    pass
            self._running = False

        finally:
            # Always unblock any waiter — success or failure
            self._init_done.set()

    # =====================================================
    # STOP
    # =====================================================

    def stop(self):
        print(f"[CAM {self._index}] Parando...")

        self._stop_event.set()
        self._new_frame_event.set()  # unblock dispatch loop if waiting
        self._init_done.set()        # unblock any waiter if init is mid-flight

        # Wait for the init thread before touching cap — it may be mid-open
        if self._init_thread and self._init_thread.is_alive():
            self._init_thread.join(timeout=3.0)

        if self._capture_thread:
            self._capture_thread.join(timeout=1.5)

        if self._dispatch_thread:
            self._dispatch_thread.join(timeout=1.5)

        with self._props_lock:
            cap = self._cap
            self._cap = None

        if cap:
            try:
                cap.release()
            except Exception:
                pass

        self._running = False

        with self._frame_lock:
            self._latest_frame = None

        self._init_thread    = None
        self._capture_thread = None
        self._dispatch_thread = None

    # =====================================================
    # SUBSCRIBERS
    # =====================================================

    def subscribe(self, callback: FrameCallback):
        with self._lock:
            self._subscribers.add(callback)

    def unsubscribe(self, callback: FrameCallback):
        with self._lock:
            self._subscribers.discard(callback)

    def set_property(self, prop_id: int, value: float) -> bool:
        """
        Sets a property on the underlying VideoCapture.

        If called before hardware init completes the call is queued and
        applied atomically when the cap handle is ready, so callers do not
        need to coordinate with the init timeline.
        """
        with self._props_lock:
            if self._cap is not None:
                return self._cap.set(prop_id, value)
            # Queue for application once _init_and_start assigns self._cap
            self._pending_props.append((prop_id, value))
            return True  # accepted / queued

    # =====================================================
    # LOOP 1 - CAPTURA
    # =====================================================

    def _capture_loop(self):
        """
        Lê frames da câmera na maior velocidade possível.

        Objetivo:
        - drenar o buffer do hardware/backend
        - evitar backlog
        - manter sempre o frame mais recente disponível
        """
        consecutive_failures = 0

        while not self._stop_event.is_set():
            ok, frame = self._cap.read() if self._cap else (False, None)

            if not ok or frame is None:
                consecutive_failures += 1

                if consecutive_failures > 10:
                    print(f"[CAM {self._index}] SEM SINAL")
                    break

                time.sleep(0.05)
                continue

            consecutive_failures = 0

            # LIFO: sempre sobrescreve o frame anterior
            with self._frame_lock:
                self._latest_frame = frame

            # Sinaliza que existe um frame novo pronto para despacho
            self._new_frame_event.set()

        self._running = False

    # =====================================================
    # LOOP 2 - DESPACHO
    # =====================================================

    def _dispatch_loop(self):
        """
        Entrega o frame mais recente para os callbacks.

        Essa separação é o ponto-chave:
        callback pesado não segura a leitura da câmera.
        """
        min_interval = 1.0 / self._fps_limit if self._fps_limit else 0.0
        last_ts = 0.0

        while not self._stop_event.is_set():
            # Espera até existir frame novo
            if not self._new_frame_event.wait(timeout=0.2):
                continue

            self._new_frame_event.clear()

            # Respeita o fps_limit apenas no despacho,
            # nunca na captura do hardware
            if min_interval:
                now = time.monotonic()
                if (now - last_ts) < min_interval:
                    continue
                last_ts = now

            with self._frame_lock:
                frame = self._latest_frame

            if frame is None:
                continue

            with self._lock:
                subscribers = list(self._subscribers)

            if not subscribers:
                continue

            for cb in subscribers:
                try:
                    cb(frame)
                except Exception:
                    pass

    # =====================================================
    # UTILITÁRIOS
    # =====================================================

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def resolution(self) -> Optional[tuple[int, int]]:
        """
        Retorna (width, height) lidos do driver, ou None se ainda não inicializado.

        Disponível apenas após _init_and_start completar; enquanto o init corre
        em background o valor é None — use fallback ou aguarde o primeiro frame.
        """
        with self._props_lock:
            cap = self._cap
        if cap is None or not cap.isOpened():
            return None
        try:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w > 0 and h > 0:
                return (w, h)
        except Exception:
            pass
        return None

@dataclass
class CameraRuntimeState:
    """Estado observavel por camera, usado pela UI para cabecalho e alertas."""

    camera_name: str
    index: Optional[int] = None
    is_configured: bool = False
    is_connected: bool = False
    has_received_first_frame: bool = False

    status_text: str = "○ Aguardando"
    status_kind: str = "idle"  # idle, loading, success, warning, danger

    opened_ts: float = 0.0
    last_frame_ts: float = 0.0

    reconnect_attempts: int = 0
    max_reconnect_attempts: int = 5
    is_reconnecting: bool = False

class MultiCameraManager:
    """
    Gerencia múltiplas câmeras com LIFO, watchdog e reconexão inteligente.

    Melhorias principais:
    - não confia só no índice da câmera
    - tenta reencontrar a câmera pelo nome amigável
    - evita roubar índice de outra câmera ativa
    - atualiza índice dinamicamente se a câmera voltar em outro index
    """

    WATCHDOG_INTERVAL_S = 0.5
    NO_FRAME_TIMEOUT_S = 3.0

    RECONNECT_BASE_DELAY_S = 5.0
    RECONNECT_MAX_DELAY_S = 25.0
    POST_STOP_COOLDOWN_S = 2.0

    def __init__(
        self,
        master_index: Optional[int],
        inner_left_index: Optional[int],
        inner_right_index: Optional[int],
    ):
        self._indexes = {
            "master": master_index,
            "inner_left": inner_left_index,
            "inner_right": inner_right_index,
        }

        self._display_names = {
            "master": "MASTER",
            "inner_left": "INNER ESQUERDA",
            "inner_right": "INNER DIREITA",
        }

        # Label da câmera escolhida originalmente.
        # Exemplo: "Logitech BRIO".
        # Isso ajuda quando o Windows muda o índice após reconectar USB.
        self._target_camera_labels: Dict[str, Optional[str]] = {
            "master": None,
            "inner_left": None,
            "inner_right": None,
        }
        self._target_camera_signatures: Dict[str, Optional[Tuple[str, int]]] = {
            "master": None,
            "inner_left": None,
            "inner_right": None,
        }

        self._managers: Dict[str, Optional[SingleCameraManager]] = {
            "master": None,
            "inner_left": None,
            "inner_right": None,
        }

        # Buffer LIFO: mantém apenas o frame mais recente.
        self._frames: Dict[str, Optional[np.ndarray]] = {
            "master": None,
            "inner_left": None,
            "inner_right": None,
        }

        self._states: Dict[str, CameraRuntimeState] = {
            name: CameraRuntimeState(
                camera_name=self._display_names[name],
                index=index,
                is_configured=index is not None,
                status_text="○ Aguardando configuração" if index is None else "◌ Aguardando stream",
                status_kind="warning" if index is None else "loading",
            )
            for name, index in self._indexes.items()
        }

        self._lock = threading.Lock()
        self._running = False
        self._watchdog_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # =====================================================
    # START / STOP
    # =====================================================

    def start(self) -> None:
        """Inicia cameras configuradas e o watchdog de saude do conjunto."""
        if self._running:
            return

        self._stop_event.clear()
        self._running = True

        # Captura o nome amigável das câmeras configuradas.
        self._capture_initial_camera_labels()

        for name in self._indexes.keys():
            state = self._states[name]
            state.index = self._indexes[name]
            state.is_configured = self._indexes[name] is not None
            state.reconnect_attempts = 0
            state.is_reconnecting = False
            state.has_received_first_frame = False
            state.is_connected = False
            state.opened_ts = 0.0
            state.last_frame_ts = 0.0

            if self._indexes[name] is None:
                self._set_status(name, "○ Aguardando configuração", "warning")
            else:
                self._set_status(name, "◌ Conectando...", "loading")
                self._start_camera_async(name)

        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            daemon=True,
            name="MultiCameraManager-Watchdog",
        )
        self._watchdog_thread.start()

    def stop(self) -> None:
        """Para watchdog, cameras ativas e limpa frames expostos ao runtime."""
        self._stop_event.set()

        if self._watchdog_thread:
            self._watchdog_thread.join(timeout=1.5)
            self._watchdog_thread = None

        for name, manager in self._managers.items():
            if manager:
                try:
                    manager.stop()
                except Exception:
                    pass

            self._managers[name] = None

        with self._lock:
            for name in self._frames:
                self._frames[name] = None

        for name, state in self._states.items():
            state.is_connected = False
            state.has_received_first_frame = False
            state.opened_ts = 0.0
            state.last_frame_ts = 0.0
            state.is_reconnecting = False

            if state.is_configured:
                self._set_status(name, "○ Aguardando", "idle")
            else:
                self._set_status(name, "○ Aguardando configuração", "warning")

        self._running = False

    # =====================================================
    # DISCOVERY / IDENTIDADE
    # =====================================================

    def _capture_initial_camera_labels(self) -> None:
        """
        Memoriza o nome amigável da câmera selecionada.

        Exemplo:
        - master estava no idx 1
        - discover diz que idx 1 é "Logitech BRIO"
        - salvamos "Logitech BRIO" como alvo da master
        """
        try:
            cameras = discover_cameras()
        except Exception as exc:
            print(f"[DISCOVERY] falha ao listar câmeras: {exc}")
            cameras = []

        by_index = {cam.get("index"): cam for cam in cameras}

        for name, index in self._indexes.items():
            if index is None:
                self._target_camera_labels[name] = None
                self._target_camera_signatures[name] = None
                continue

            cam = by_index.get(index)
            if cam:
                label = str(cam.get("label") or "")
                occurrence = int(cam.get("label_occurrence") or 1)
                self._target_camera_labels[name] = label
                self._target_camera_signatures[name] = (label, occurrence)
                print(f"[DISCOVERY] {name}: alvo '{label}' #{occurrence} no idx {index}")
            else:
                self._target_camera_labels[name] = None
                self._target_camera_signatures[name] = None
                print(f"[DISCOVERY] {name}: não encontrou label inicial para idx {index}")

    def _get_used_indexes_except(self, name: str) -> set[int]:
        """
        Retorna índices usados por outras câmeras ativas.

        Isso evita o cenário:
        - câmera master caiu
        - inner continua ativa no idx 2
        - master tenta assumir idx 2 por ter mesmo nome
        """
        used: set[int] = set()

        for other_name, manager in self._managers.items():
            if other_name == name:
                continue

            if manager is None:
                continue

            state = self._states[other_name]

            # Consideramos realmente ativa se abriu e já recebeu frame.
            if state.is_connected and state.has_received_first_frame:
                idx = self._indexes.get(other_name)
                if idx is not None:
                    used.add(idx)

        return used

    def _resolve_reconnect_index(self, name: str) -> Optional[int]:
        """
        Decide qual índice usar na reconexão.

        Prioridade:
        1. procurar mesma label sem roubar índice de câmera ativa
        2. tentar índice antigo, se estiver disponível e livre
        3. retornar None
        """
        try:
            cameras = discover_cameras()
        except Exception as exc:
            print(f"[DISCOVERY] {name}: erro no discover_cameras -> {exc}")
            return None

        if not cameras:
            print(f"[DISCOVERY] {name}: nenhuma câmera encontrada")
            return None

        target_label = self._target_camera_labels.get(name)
        target_signature = self._target_camera_signatures.get(name)
        old_index = self._indexes.get(name)
        used_indexes = self._get_used_indexes_except(name)

        if target_signature:
            target_name, target_occurrence = target_signature
            for cam in cameras:
                cam_index = cam.get("index")
                cam_label = str(cam.get("label") or "")
                cam_occurrence = int(cam.get("label_occurrence") or 1)

                if (
                    cam_label == target_name
                    and cam_occurrence == target_occurrence
                    and cam_index not in used_indexes
                ):
                    print(
                        f"[DISCOVERY] {name}: assinatura '{target_name}' "
                        f"#{target_occurrence} encontrada no idx {cam_index}"
                    )
                    return cam_index

        # 2. tenta indice antigo, desde que nao esteja sendo usado por outro slot ativo
        if old_index is not None and old_index not in used_indexes:
            for cam in cameras:
                if cam.get("index") == old_index:
                    print(f"[DISCOVERY] {name}: indice antigo {old_index} ainda disponivel")
                    return old_index

        # 3. fallback: tenta encontrar pela mesma label sem roubar indice ativo
        if target_label:
            for cam in cameras:
                cam_index = cam.get("index")
                cam_label = cam.get("label")

                if cam_label == target_label and cam_index not in used_indexes:
                    print(
                        f"[DISCOVERY] {name}: label '{target_label}' encontrada no idx {cam_index}"
                    )
                    return cam_index

        print(
            f"[DISCOVERY] {name}: nenhum índice compatível. "
            f"Usados por outras câmeras: {sorted(used_indexes)}"
        )
        return None

    # =====================================================
    # START ASYNC POR CÂMERA
    # =====================================================

    def _start_camera_async(self, name: str) -> None:
        threading.Thread(
            target=self._start_camera_safe,
            args=(name,),
            daemon=True,
            name=f"MultiCameraManager-Start-{name}",
        ).start()

    def _start_camera_safe(self, name: str) -> None:
        """
        Tenta abrir uma camera sem derrubar o restante do conjunto.

        Falhas viram status visual e agendamento de reconexao, pois cada slot
        (MASTER/INNER) deve sobreviver independentemente.
        """
        index = self._indexes[name]

        if index is None:
            print(f"[START_SAFE] {name}: índice nulo")
            return

        print(f"[START_SAFE] {name}: tentando abrir índice {index}")

        try:
            manager = SingleCameraManager(camera_index=index)
            manager.start()
            manager.subscribe(lambda frame, n=name: self._on_frame(n, frame))

            self._managers[name] = manager

            now = time.monotonic()
            state = self._states[name]
            state.index = index
            state.is_connected = True
            state.has_received_first_frame = False
            state.opened_ts = now
            state.last_frame_ts = now
            state.is_reconnecting = False

            self._set_status(name, "◌ Aberta, aguardando frames", "loading")
            print(f"[START_SAFE] {name}: iniciada com sucesso idx {index}")

        except Exception as exc:
            print(f"[START_SAFE] {name}: falha ao abrir idx {index} -> {exc}")

            self._managers[name] = None

            state = self._states[name]
            state.is_connected = False
            state.has_received_first_frame = False
            state.opened_ts = 0.0
            state.last_frame_ts = 0.0
            state.is_reconnecting = False

            with self._lock:
                self._frames[name] = None

            self._set_status(name, "✖ Falha ao abrir", "danger")
            self._schedule_reconnect(name)

    # =====================================================
    # CALLBACK
    # =====================================================

    def _on_frame(self, name: str, frame: np.ndarray) -> None:
        """Recebe callback do SingleCameraManager e publica ultimo frame LIFO."""
        with self._lock:
            self._frames[name] = frame.copy()

        now = time.monotonic()
        state = self._states[name]
        state.last_frame_ts = now

        if not state.has_received_first_frame:
            state.has_received_first_frame = True
            state.reconnect_attempts = 0
            state.is_reconnecting = False
            self._set_status(name, "● Recebendo frames", "success")

    # =====================================================
    # WATCHDOG / RECONNECT
    # =====================================================

    def _watchdog_loop(self) -> None:
        """
        Monitora cameras que abriram mas nao entregam frames ou perderam sinal.

        A reconexao e disparada pelo tempo sem frames, nao apenas por erro de
        abertura, porque algumas cameras ficam "abertas" no OpenCV sem imagem.
        """
        while not self._stop_event.is_set():
            now = time.monotonic()

            for name, state in self._states.items():
                if not state.is_configured:
                    continue

                if state.is_reconnecting:
                    continue

                manager = self._managers[name]
                if manager is None:
                    continue

                # abriu, mas nunca recebeu frame
                if state.opened_ts != 0.0 and not state.has_received_first_frame:
                    elapsed = now - state.opened_ts

                    if elapsed > self.NO_FRAME_TIMEOUT_S:
                        print(f"[WATCHDOG] {name}: sem frames iniciais há {elapsed:.2f}s")
                        self._set_status(name, "⚠ Sem frames iniciais", "warning")
                        self._dispose_camera(name)
                        self._schedule_reconnect(name)

                    continue

                # recebeu frame antes, mas parou de receber
                if state.has_received_first_frame:
                    elapsed = now - state.last_frame_ts

                    if elapsed > self.NO_FRAME_TIMEOUT_S:
                        print(f"[WATCHDOG] {name}: perda de sinal detectada ({elapsed:.2f}s)")
                        self._set_status(name, "⚠ Perda de sinal", "warning")
                        self._dispose_camera(name)
                        self._schedule_reconnect(name)

            time.sleep(self.WATCHDOG_INTERVAL_S)

    def _schedule_reconnect(self, name: str) -> None:
        """Agenda reconexao com backoff simples para nao martelar USB/backend."""
        state = self._states[name]

        if state.is_reconnecting:
            print(f"[RECONNECT] {name}: já reconectando")
            return

        if state.reconnect_attempts >= state.max_reconnect_attempts:
            print(f"[RECONNECT] {name}: limite atingido")
            self._set_status(name, "✖ Falha permanente", "danger")
            return

        state.is_reconnecting = True
        state.reconnect_attempts += 1

        delay = min(
            self.RECONNECT_BASE_DELAY_S * state.reconnect_attempts,
            self.RECONNECT_MAX_DELAY_S,
        )

        self._set_status(
            name,
            f"↻ Aguardando reconexão em {delay:.1f}s ({state.reconnect_attempts}/{state.max_reconnect_attempts})",
            "warning",
        )

        print(f"[RECONNECT] {name}: agendado em {delay:.1f}s")

        threading.Thread(
            target=self._reconnect_after_delay,
            args=(name, delay),
            daemon=True,
            name=f"MultiCameraManager-Reconnect-{name}",
        ).start()

    def _reconnect_after_delay(self, name: str, delay: float) -> None:
        """Reavalia o indice antes de reabrir para suportar troca de porta USB."""
        time.sleep(delay)

        if self._stop_event.is_set():
            print(f"[RECONNECT] {name}: cancelado por stop")
            return

        new_index = self._resolve_reconnect_index(name)

        if new_index is None:
            print(f"[RECONNECT] {name}: câmera ainda não reapareceu")
            self._set_status(name, "⌛ Aguardando retorno da câmera", "warning")

            state = self._states[name]
            state.is_reconnecting = False
            self._schedule_reconnect(name)
            return

        old_index = self._indexes[name]
        self._indexes[name] = new_index
        self._states[name].index = new_index

        if old_index != new_index:
            print(f"[RECONNECT] {name}: índice atualizado {old_index} -> {new_index}")

        self._set_status(name, "↻ Reabrindo câmera", "loading")
        self._start_camera_safe(name)

    def _dispose_camera(self, name: str) -> None:
        print(f"[DISPOSE] {name}: liberando câmera")

        manager = self._managers[name]
        if manager:
            try:
                manager.stop()
            except Exception as exc:
                print(f"[DISPOSE] {name}: erro ao parar -> {exc}")

        self._managers[name] = None

        with self._lock:
            self._frames[name] = None

        state = self._states[name]
        state.is_connected = False
        state.has_received_first_frame = False
        state.opened_ts = 0.0
        state.last_frame_ts = 0.0

        time.sleep(self.POST_STOP_COOLDOWN_S)

    # =====================================================
    # STATUS / ACCESS
    # =====================================================

    def _set_status(self, name: str, text: str, kind: str) -> None:
        state = self._states[name]
        state.status_text = text
        state.status_kind = kind

    def get_camera_states(self) -> Dict[str, CameraRuntimeState]:
        return {
            name: CameraRuntimeState(
                camera_name=state.camera_name,
                index=state.index,
                is_configured=state.is_configured,
                is_connected=state.is_connected,
                has_received_first_frame=state.has_received_first_frame,
                status_text=state.status_text,
                status_kind=state.status_kind,
                opened_ts=state.opened_ts,
                last_frame_ts=state.last_frame_ts,
                reconnect_attempts=state.reconnect_attempts,
                max_reconnect_attempts=state.max_reconnect_attempts,
                is_reconnecting=state.is_reconnecting,
            )
            for name, state in self._states.items()
        }

    def get_latest_frames(self) -> Dict[str, Optional[np.ndarray]]:
        """Retorna o ultimo frame conhecido de cada slot sem bloquear captura."""
        with self._lock:
            return {
                "master": self._frames["master"],
                "inner_left": self._frames["inner_left"],
                "inner_right": self._frames["inner_right"],
            }

    def get_active_cameras(self) -> Dict[str, bool]:
        return {
            "master": self._managers["master"] is not None,
            "inner_left": self._managers["inner_left"] is not None,
            "inner_right": self._managers["inner_right"] is not None,
        }

    @property
    def is_running(self) -> bool:
        return self._running
