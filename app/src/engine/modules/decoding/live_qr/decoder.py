"""
Iris - QR Decoder
=================
Responsabilidade unica: interpretar regioes detectadas pelo YOLO e retornar
os dados codificados nos QR Codes.

Pipeline interno:
    1. Recorte da regiao com padding de seguranca (filter.crop_with_padding)
    2. Tentativa direta sem aprimoramento
    3. Tentativas com variantes de aprimoramento (filter.enhance_for_qr)

Este modulo nao detecta QR Codes nem toma decisoes sobre o modelo YOLO.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from pyzbar.pyzbar import ZBarSymbol, decode as _pyzbar_decode
    _PYZBAR_AVAILABLE = True
except Exception:
    ZBarSymbol = None  # type: ignore[assignment]
    _PYZBAR_AVAILABLE = False

from app.src.engine.modules.decoding.live_qr.filter import crop_with_padding, enhance_for_qr


class QrDecoder:
    """
    Decodificador de QR Codes a partir de poligonos de deteccao.

    Mantem apenas o cache espacial simples que ja existia para evitar
    reprocessamento imediato de regioes estaveis.
    """

    def __init__(
        self,
        payload_validator: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self._cv2_detector = cv2.QRCodeDetector()
        self._payload_validator = payload_validator

        self._spatial_cache: List[Dict[str, Any]] = []
        self._CACHE_RADIUS = 60.0
        self._SUCCESS_TTL = 3.0
        self._FAIL_TTL = 0.5

    def decode(
        self,
        frame: np.ndarray,
        boxes: List[np.ndarray],
    ) -> List[Tuple[Optional[str], Optional[np.ndarray], int, int]]:
        now = time.monotonic()

        self._spatial_cache = [
            c for c in self._spatial_cache
            if c["status"] == "PROCESSING"
            or (c["status"] == "VALID" and now - c["ts"] < self._SUCCESS_TTL)
            or (c["status"] == "FAILED" and now - c["ts"] < self._FAIL_TTL)
        ]

        results: List[Tuple[Optional[str], Optional[np.ndarray], int, int]] = []

        for pts in boxes:
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

                if entry["status"] in ("PROCESSING", "FAILED"):
                    results.append((None, None, 0, 0))
                    continue

            new_entry: Dict[str, Any] = {
                "cx": cx,
                "cy": cy,
                "text": None,
                "status": "PROCESSING",
                "ts": now,
            }
            self._spatial_cache.append(new_entry)

            try:
                text, patch, ox, oy = self._decode_region(frame, pts)
            except Exception:
                text, patch, ox, oy = None, None, 0, 0

            if text:
                new_entry.update(status="VALID", text=text, ts=time.monotonic())
            else:
                new_entry.update(status="FAILED", text=None, ts=time.monotonic())

            results.append((text, patch, ox, oy))

        return results

    def _decode_region(
        self,
        frame: np.ndarray,
        pts: np.ndarray,
    ) -> Tuple[Optional[str], Optional[np.ndarray], int, int]:
        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
        anchor = (x, y, w, h)

        crop, ox, oy = crop_with_padding(frame, anchor, padding=12)
        if crop.size == 0:
            return None, None, ox, oy

        base_bgr = crop if crop.ndim == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

        result = self._run_qreader(base_bgr)
        if result:
            return result, base_bgr, ox, oy

        last_bgr = base_bgr
        for _name, variant in enhance_for_qr(crop):
            bgr = (
                cv2.cvtColor(variant, cv2.COLOR_GRAY2BGR)
                if variant.ndim == 2
                else variant
            )
            last_bgr = bgr

            result = self._run_qreader(bgr)
            if result:
                return result, bgr, ox, oy

        return None, last_bgr, ox, oy

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

    def _run_qreader(self, img: np.ndarray) -> Optional[str]:
        try:
            data, _, _ = self._cv2_detector.detectAndDecode(img)
            data = (data or "").strip()
            if self._accept_text(data):
                return data
        except Exception:
            pass

        if _PYZBAR_AVAILABLE:
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
                kwargs = {"symbols": [ZBarSymbol.QRCODE]} if ZBarSymbol is not None else {}
                results = _pyzbar_decode(gray, **kwargs)
                for result in results:
                    if not result.data:
                        continue
                    text = result.data.decode("utf-8", errors="replace").strip()
                    if self._accept_text(text):
                        return text
            except Exception:
                pass

        return None
