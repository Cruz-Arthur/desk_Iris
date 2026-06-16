"""
Iris — Image Enhancement Pipeline
==================================
Responsabilidade única: gerar variantes de imagem que maximizam a chance
de leitura de QR Codes em condições adversas.

Ordem das variantes: custo crescente, probabilidade de hit decrescente.
O decoder para assim que uma variante produz uma leitura (lazy).

Redesign para leitura em movimento:
  - Tier 0: sem upscale, apenas binarizações baratas (~0.3ms total)
  - Tier 1: upscale 2× + binarização (~1ms) — cobre QRs pequenos
  - Tier 2: upscale + gamma/CLAHE (~1.5ms) — cobre subexposição
  - Tier 3: median + morfologia (~2ms) — último recurso

Removido: bilateral (10-20ms, inútil para blur de movimento),
          adaptativos sem upscale (falsa sensação de cobertura),
          yields de imagens não-binarizadas intermediárias (unsharp),
          3 upscales redundantes do Tier 4 original.
Total de variantes: 18 (era 32+).
"""

from __future__ import annotations

from typing import Generator, Tuple

import cv2
import numpy as np


# ── helpers internos ──────────────────────────────────────────────────────────

_GAMMA_LUTS: dict[float, np.ndarray] = {}
_CLAHE_OBJS: dict[tuple[float, int], "cv2.CLAHE"] = {}


def _gamma(img: np.ndarray, g: float) -> np.ndarray:
    lut = _GAMMA_LUTS.get(g)
    if lut is None:
        lut = (((np.arange(256, dtype=np.float32) / 255.0) ** g) * 255.0).astype(np.uint8)
        _GAMMA_LUTS[g] = lut
    return cv2.LUT(img, lut)


def _clahe(img: np.ndarray, clip: float, tile: int = 8) -> np.ndarray:
    key = (clip, tile)
    c = _CLAHE_OBJS.get(key)
    if c is None:
        c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        _CLAHE_OBJS[key] = c
    return c.apply(img)


def _otsu(img: np.ndarray, invert: bool = False) -> np.ndarray:
    flags = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    if invert:
        flags = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    _, out = cv2.threshold(img, 0, 255, flags)
    return out


def _adaptive(img: np.ndarray, block: int, c: int = 5) -> np.ndarray:
    return cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, c
    )


def _upscale(img: np.ndarray, factor: float) -> np.ndarray:
    return cv2.resize(img, None, fx=factor, fy=factor,
                      interpolation=cv2.INTER_LINEAR)


def _morph_close(img: np.ndarray, k: int = 3) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


# ── pipeline principal ────────────────────────────────────────────────────────

def enhance_for_qr(
    crop: np.ndarray,
    upscale: float = 2.0,
) -> Generator[Tuple[str, np.ndarray], None, None]:
    """
    Gera variantes em ordem crescente de custo. Para no primeiro hit.
    Entrada: crop BGR ou gray. Saída: sempre gray ou binário 2D.
    """
    # Gray direto — sem seleção de canal: QR é P&B, gray BT.601 é suficiente.
    # A seleção de canal fazia 4× np.std por crop sem ganho de hit-rate real.
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop

    # ── Tier 0: sem upscale, binarizações O(pixels) ───────────────────────────
    # Decoders fazem binarização interna — yield do gray já cobre condições ok.

    yield "gray", gray

    yield "otsu_raw", _otsu(gray)
    yield "otsu_inv", _otsu(gray, invert=True)

    # CLAHE moderado → Otsu (iluminação irregular leve)
    clahe_mod = _clahe(gray, clip=2.0)
    yield "clahe_otsu", _otsu(clahe_mod)

    # Gamma claro → Otsu (QR subexposto / ambiente escuro)
    yield "gamma_bright_otsu", _otsu(_gamma(gray, 0.45))

    # CLAHE agressivo → Otsu (sombras duras / luz lateral)
    clahe_hard = _clahe(gray, clip=6.0, tile=4)
    yield "clahe_hard_otsu", _otsu(clahe_hard)

    # ── Tier 1: upscale 2× — QRs pequenos na imagem (~60-100px) ─────────────
    # Gray upscalado UMA vez; todos os Tiers 1-3 reutilizam esta referência.

    up = _upscale(gray, upscale)

    yield "up_otsu", _otsu(up)
    yield "up_otsu_inv", _otsu(up, invert=True)

    # CLAHE moderado pós-upscale (base para Tier 2)
    up_clahe = _clahe(up, clip=2.0)
    yield "up_clahe_otsu", _otsu(up_clahe)
    yield "up_adapt_15",   _adaptive(up_clahe, 15, 4)
    yield "up_adapt_25",   _adaptive(up_clahe, 25, 6)

    # ── Tier 2: gamma + CLAHE agressivo pós-upscale ───────────────────────────
    # Upscale aplicado sobre gray original — não recalcula cv2.resize.

    up_bright = _upscale(_gamma(gray, 0.40), upscale)
    yield "up_bright_otsu",  _otsu(up_bright)
    yield "up_bright_adapt", _adaptive(up_bright, 21, 5)

    up_clahe_hard = _clahe(up, clip=8.0, tile=4)
    yield "up_clahe_hard_otsu",  _otsu(up_clahe_hard)
    yield "up_clahe_hard_adapt", _adaptive(up_clahe_hard, 21, 5)

    # ── Tier 3: median + morfologia (módulos partidos / ruído impresso) ───────
    # Só alcançado se todos os tiers anteriores falharam — custo já não importa.

    up_median = cv2.medianBlur(up_clahe, 3)
    yield "median_otsu", _otsu(up_median)

    otsu_bin = _otsu(up_clahe)
    yield "morph_close_3", _morph_close(otsu_bin, 3)
    yield "morph_close_5", _morph_close(otsu_bin, 5)


# ── crop auxiliar ─────────────────────────────────────────────────────────────

def crop_with_padding(
    frame:   np.ndarray,
    anchor:  Tuple[int, int, int, int],
    padding: int = 24,
) -> Tuple[np.ndarray, int, int]:
    ax, ay, aw, ah = anchor
    fh, fw = frame.shape[:2]
    x1 = max(0,  ax - padding)
    y1 = max(0,  ay - padding)
    x2 = min(fw, ax + aw + padding)
    y2 = min(fh, ay + ah + padding)
    return frame[y1:y2, x1:x2], x1, y1
