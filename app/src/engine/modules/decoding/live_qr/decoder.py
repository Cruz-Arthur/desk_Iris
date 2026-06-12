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
    from pyzbar.pyzbar import ZBarSymbol, decode as _pyzbar_decode
    _PYZBAR_AVAILABLE = True
except Exception:
    ZBarSymbol = None  # type: ignore[assignment]
    _PYZBAR_AVAILABLE = False

from app.src.engine.modules.decoding.live_qr.filter import crop_with_padding, enhance_for_qr


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
        self._CACHE_RADIUS = 60.0
        self._SUCCESS_TTL  = 3.0
        # Backoff exponencial de falhas: primeira retentativa quase imediata
        # (captura variação de ângulo/luz), depois dobra até o teto — uma
        # região permanentemente ilegível deixa de drenar CPU.
        self._FAIL_BASE = 0.08   # s — espera após a 1ª falha
        self._FAIL_MAX  = 1.0    # s — teto do backoff
        self._FAIL_KEEP = 3.0    # s — memória da região falhada

    # ── API pública ────────────────────────────────────────────────────────────

    def decode(
        self,
        frame: np.ndarray,
        boxes: List[np.ndarray],
    ) -> List[Tuple[Optional[str], Optional[np.ndarray], int, int]]:
        now = time.monotonic()

        self._spatial_cache = [
            c for c in self._spatial_cache
            if (c["status"] == "VALID"  and now - c["ts"] < self._SUCCESS_TTL)
            or (c["status"] == "FAILED" and now - c["ts"] < self._FAIL_KEEP)
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
        # As variantes em escala de cinza vão DIRETO aos decoders — todos
        # aceitam imagens 2D; converter gray→BGR→gray seria puro desperdício.
        last_img = crop_norm
        for _name, variant in enhance_for_qr(crop_norm):
            last_img = variant
            result = self._try_all_decoders(variant)
            if result:
                return result, variant, ox, oy

        return None, last_img, ox, oy

    def _try_all_decoders(self, img: np.ndarray) -> Optional[str]:
        """
        Tenta todos os decoders disponíveis sobre uma mesma imagem.

        A conversão para cinza acontece UMA única vez aqui — os três
        decoders aceitam entrada 2D, então nenhum reconverte internamente.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

        # 1. QRCodeDetectorAruco (mais robusto — trata perspectiva/dano)
        if self._detector_aruco is not None:
            result = self._run_aruco(gray)
            if result:
                return result

        # 2. QRCodeDetector clássico
        result = self._run_cv2(gray)
        if result:
            return result

        # 3. pyzbar
        result = self._run_pyzbar(gray)
        if result:
            return result

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
