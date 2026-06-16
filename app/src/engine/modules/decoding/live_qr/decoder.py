"""
Iris - QR Decoder
=================
Responsabilidade única: interpretar regiões detectadas pelo YOLO e retornar
os dados codificados nos QR Codes.

Pipeline de decodificação (em ordem de tentativa):
    1. cv2.QRCodeDetectorAruco  — mais robusto para ângulo/dano
    2. cv2.QRCodeDetector       — fallback clássico
    3. pyzbar                   — terceira opinião
    Cada decodificador tenta o crop original e depois cada variante
    gerada por enhance_for_qr(), parando no primeiro sucesso.

    Fallback final: tenta decodificar o frame completo sem crop.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import zxingcpp as _zxingcpp
    _ZXING_AVAILABLE = True
except ImportError:
    _zxingcpp = None  # type: ignore[assignment]
    _ZXING_AVAILABLE = False

try:
    from pyzbar.pyzbar import ZBarSymbol, decode as _pyzbar_decode
    _PYZBAR_AVAILABLE = True
except Exception:
    ZBarSymbol = None  # type: ignore[assignment]
    _PYZBAR_AVAILABLE = False

from app.src.engine.modules.decoding.live_qr.filter import crop_with_padding, enhance_for_qr

# Velocidade mínima (px/s escalada por VELOCITY_FACTOR) para ativar o duo-read.
# Abaixo disso o QR está essencialmente parado e o crop do YOLO já é suficiente.
_MIN_VEL_FOR_DUO_READ = 5.0


def _has_aruco_qr() -> bool:
    try:
        cv2.QRCodeDetectorAruco()
        return True
    except AttributeError:
        return False


_ARUCO_AVAILABLE = _has_aruco_qr()


class QrDecoder:
    """
    Decodificador de QR Codes a partir de polígonos de detecção.

    Cache espacial com TTL curto: falhas são reprocessadas quase imediatamente
    para maximizar hits em frames com variação de ângulo/luz.
    """

    def __init__(
        self,
        payload_validator: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self._detector_aruco = cv2.QRCodeDetectorAruco() if _ARUCO_AVAILABLE else None
        self._detector_cv2   = cv2.QRCodeDetector()
        self._payload_validator = payload_validator

        self._spatial_cache: List[Dict[str, Any]] = []
        # Raio menor: QR em movimento pode cruzar 50px e herdar texto de outro código.
        self._CACHE_RADIUS = 30.0
        # TTL curto: cena dinâmica muda rápido, cache longo mascara troca de código.
        self._SUCCESS_TTL  = 0.6
        self._FAIL_BASE = 0.0
        self._FAIL_MAX  = 0.0
        self._FAIL_KEEP = 1.5

    # ── API pública ────────────────────────────────────────────────────────────

    def decode(
        self,
        frame: np.ndarray,
        boxes: List[np.ndarray],
        hints: Optional[List[Optional[Tuple[float, float, float, float, float]]]] = None,
    ) -> List[Tuple[Optional[str], Optional[np.ndarray], int, int]]:
        """
        Decode QR codes from YOLO bounding boxes.

        hints : optional list aligned with `boxes`, one entry per detection:
            (pred1_cx, pred1_cy, bbox_w, bbox_h, vel_mag)
            When vel_mag > _MIN_VEL_FOR_DUO_READ and the detection has no VALID
            cache entry, the decoder tries both the YOLO crop AND a crop centred
            at the 1-frame-ahead predicted position (duo-read).  A successful
            predicted-crop decode is cached at the predicted centroid so the next
            YOLO frame gets an instant cache hit (pre-fetch).
        """
        now = time.monotonic()

        self._spatial_cache = [
            c for c in self._spatial_cache
            if (c["status"] == "VALID"  and now - c["ts"] < self._SUCCESS_TTL)
            or (c["status"] == "FAILED" and now - c["ts"] < self._FAIL_KEEP)
        ]

        results: List[Tuple[Optional[str], Optional[np.ndarray], int, int]] = []

        for idx, pts in enumerate(boxes):
            hint = hints[idx] if hints is not None and idx < len(hints) else None
            x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
            cx, cy = x + w / 2.0, y + h / 2.0

            entry = None
            best_dist = float("inf")
            for c in self._spatial_cache:
                dist = ((c["cx"] - cx) ** 2 + (c["cy"] - cy) ** 2) ** 0.5
                if dist < self._CACHE_RADIUS and dist < best_dist:
                    entry = c
                    best_dist = dist

            if entry is not None:
                entry["cx"], entry["cy"] = cx, cy

                if entry["status"] == "VALID":
                    entry["ts"] = now
                    results.append((entry["text"], None, 0, 0))
                    continue

                # FAILED — só reprocessa quando o backoff expirar
                backoff = min(self._FAIL_BASE * (2 ** entry["fails"]), self._FAIL_MAX)
                if now - entry["ts"] < backoff:
                    results.append((None, None, 0, 0))
                    continue

                text, patch, ox, oy = self._safe_decode(frame, pts)
                if text:
                    entry.update(status="VALID", text=text,
                                 ts=time.monotonic(), fails=0)
                else:
                    entry["fails"] += 1
                    entry["ts"] = time.monotonic()
                results.append((text, patch, ox, oy))
                continue

            # Região nova — primeira tentativa imediata
            text, patch, ox, oy = self._safe_decode(frame, pts)

            # Duo-read: QR em movimento e ainda não lido — tenta também o crop predito.
            # Se o crop predito decodificar, armazena no cache naquela posição
            # (pre-fetch): no próximo frame o YOLO vai detectar lá e terá cache hit.
            if not text and hint is not None:
                p1cx, p1cy, bbox_w, bbox_h, vel_mag = hint
                if vel_mag > _MIN_VEL_FOR_DUO_READ:
                    text, patch, ox, oy = self._decode_at(frame, p1cx, p1cy, bbox_w, bbox_h)
                    if text:
                        self._spatial_cache.append({
                            "cx": p1cx, "cy": p1cy,
                            "text": text, "status": "VALID",
                            "ts": time.monotonic(), "fails": 0,
                        })

            new_entry: Dict[str, Any] = {
                "cx": cx, "cy": cy,
                "text": text,
                "status": "VALID" if text else "FAILED",
                "ts": time.monotonic(),
                "fails": 0,
            }
            self._spatial_cache.append(new_entry)
            results.append((text, patch, ox, oy))

        return results

    def _decode_at(
        self,
        frame: np.ndarray,
        cx: float, cy: float,
        w: float, h: float,
    ) -> Tuple[Optional[str], Optional[np.ndarray], int, int]:
        """Build a bbox centred at (cx, cy) and decode it — used for duo-read."""
        fh, fw = frame.shape[:2]
        x1 = max(0, int(cx - w / 2))
        y1 = max(0, int(cy - h / 2))
        x2 = min(fw, int(cx + w / 2))
        y2 = min(fh, int(cy + h / 2))
        if x2 <= x1 or y2 <= y1:
            return None, None, 0, 0
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
        return self._safe_decode(frame, pts)

    def _safe_decode(
        self,
        frame: np.ndarray,
        pts: np.ndarray,
    ) -> Tuple[Optional[str], Optional[np.ndarray], int, int]:
        try:
            return self._decode_region(frame, pts)
        except Exception:
            return None, None, 0, 0

    # ── Pipeline de decodificação ──────────────────────────────────────────────

    def _decode_region(
        self,
        frame: np.ndarray,
        pts: np.ndarray,
    ) -> Tuple[Optional[str], Optional[np.ndarray], int, int]:
        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
        anchor = (x, y, w, h)

        # Crop normal (padding 24) e crop generoso (padding 48) para não cortar
        # finder patterns em detecções com bbox apertada
        crop_norm, ox, oy = crop_with_padding(frame, anchor, padding=24)
        crop_wide, _,  _  = crop_with_padding(frame, anchor, padding=48)

        if crop_norm.size == 0:
            return None, None, ox, oy

        # Tenta crop normal primeiro
        result = self._try_all_decoders(crop_norm)
        if result:
            return result, crop_norm, ox, oy

        # Tenta crop mais generoso antes de entrar no pipeline de preprocessing
        if crop_wide.size > 0 and crop_wide.shape != crop_norm.shape:
            result = self._try_all_decoders(crop_wide)
            if result:
                return result, crop_wide, ox, oy

        # Pipeline de preprocessamento (lazy — para no primeiro hit).
        # upscale=2.0: INTER_LINEAR 5× mais rápido que Lanczos sem perda real
        # para QRs em movimento — a qualidade do Otsu/adaptativo domina.
        last_img = crop_norm
        for _name, variant in enhance_for_qr(crop_norm, upscale=2.0):
            last_img = variant
            result = self._try_all_decoders(variant)
            if result:
                return result, variant, ox, oy

        return None, last_img, ox, oy

    def _try_all_decoders(self, img: np.ndarray) -> Optional[str]:
        """
        Tenta todos os decoders sobre a mesma imagem. Gray convertido uma vez.

        Ordem otimizada para leitura em movimento:
          1. zxing-cpp — GIL-releasing, batch API, melhor hit-rate (se disponível)
          2. pyzbar    — ZBar clássico, fallback quando zxing não instalado
          3. cv2       — fallback secundário
          4. Aruco     — lento, robusto para perspectiva/dano (último recurso)
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

        if _ZXING_AVAILABLE:
            result = self._run_zxing(gray)
            if result:
                return result
        else:
            result = self._run_pyzbar(gray)
            if result:
                return result

        result = self._run_cv2(gray)
        if result:
            return result

        if self._detector_aruco is not None:
            result = self._run_aruco(gray)
            if result:
                return result

        return None

    def _run_zxing(self, gray: np.ndarray) -> Optional[str]:
        try:
            results = _zxingcpp.read_barcodes(
                gray, formats=_zxingcpp.BarcodeFormat.QRCode
            )
            for r in results:
                if r.valid:
                    text = r.text.strip()
                    if self._accept_text(text):
                        return text
        except Exception:
            pass
        return None

    def _run_aruco(self, gray: np.ndarray) -> Optional[str]:
        try:
            data, _, _ = self._detector_aruco.detectAndDecode(gray)
            data = (data or "").strip()
            if self._accept_text(data):
                return data
        except Exception:
            pass
        return None

    def _run_cv2(self, gray: np.ndarray) -> Optional[str]:
        try:
            data, _, _ = self._detector_cv2.detectAndDecode(gray)
            data = (data or "").strip()
            if self._accept_text(data):
                return data
        except Exception:
            pass
        return None

    def _run_pyzbar(self, gray: np.ndarray) -> Optional[str]:
        if not _PYZBAR_AVAILABLE:
            return None
        try:
            kwargs = {"symbols": [ZBarSymbol.QRCODE]} if ZBarSymbol is not None else {}
            for result in _pyzbar_decode(gray, **kwargs):
                if not result.data:
                    continue
                text = result.data.decode("utf-8", errors="replace").strip()
                if self._accept_text(text):
                    return text
        except Exception:
            pass
        return None

    def _accept_text(self, text: Optional[str]) -> bool:
        text = (text or "").strip()
        if not text:
            return False
        if self._payload_validator is None:
            return True
        try:
            return bool(self._payload_validator(text))
        except Exception:
            return False
