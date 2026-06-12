"""
Iris — Image Enhancement Pipeline
==================================
Responsabilidade única: gerar variantes de imagem que maximizam a chance
de leitura de QR Codes em condições adversas.

Ordem das variantes: custo crescente, retorno decrescente.
O decoder para assim que uma variante produz uma leitura.
"""

from __future__ import annotations

from typing import Generator, Tuple

import cv2
import numpy as np


# ── helpers internos ──────────────────────────────────────────────────────────

# Caches módulo-level: o pipeline usa um conjunto fixo e pequeno de gammas e
# configurações CLAHE — reconstruir LUT (256 pows em Python) e realocar o
# objeto C++ do CLAHE a cada variante era custo puro por região analisada.
_GAMMA_LUTS: dict[float, np.ndarray] = {}
_CLAHE_OBJS: dict[tuple[float, int], "cv2.CLAHE"] = {}


def _gamma(img: np.ndarray, g: float) -> np.ndarray:
    """Aplica correção de gamma. g < 1 clareia, g > 1 escurece."""
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
                      interpolation=cv2.INTER_LANCZOS4)


def _morph_close(img: np.ndarray, k: int = 3) -> np.ndarray:
    """Fecha buracos nos módulos do QR (ruído escuro interno)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


def _best_gray(crop: np.ndarray) -> np.ndarray:
    """Escolhe o canal (ou a conversão) com maior contraste para a imagem."""
    if crop.ndim == 2:
        return crop
    # testa gray padrão vs canal de maior desvio padrão
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(crop)
    best, best_std = gray, float(np.std(gray))
    for ch in (b, g, r):
        s = float(np.std(ch))
        if s > best_std:
            best, best_std = ch, s
    return best


# ── pipeline principal ────────────────────────────────────────────────────────

def enhance_for_qr(
    crop: np.ndarray,
    upscale: float = 3.0,
) -> Generator[Tuple[str, np.ndarray], None, None]:
    """
    Gera variantes de imagem em ordem crescente de custo computacional.
    Para no primeiro hit — use em conjunto com lógica lazy no caller.

    Estratégias cobertas:
      - Exposição normal, escura, clara (gamma)
      - Baixo contraste local (CLAHE moderado e agressivo)
      - Fundo colorido / canal de cor dominante
      - QR pequeno (upscale Lanczos)
      - Ruído / granularidade (mediana, bilateral)
      - Limiarização global e local (Otsu + adaptativo, múltiplos blocos)
      - QR invertido (fundo preto, módulos brancos)
      - Módulos partidos por ruído (morfologia)
    """
    gray = _best_gray(crop)

    # ── Tier 0: variantes baratas sem upscale ────────────────────────────────

    yield "gray", gray

    clahe_mod = _clahe(gray, clip=2.0)
    yield "clahe_mod", clahe_mod

    # QR escuro / subexposto
    bright = _gamma(clahe_mod, 0.45)
    yield "gamma_bright", bright

    # QR superexposto / reflexo
    dark = _gamma(clahe_mod, 2.2)
    yield "gamma_dark", dark

    # CLAHE agressivo (iluminação muito irregular)
    clahe_hard = _clahe(gray, clip=6.0, tile=4)
    yield "clahe_hard", clahe_hard

    # Otsu direto no gray original
    yield "otsu_raw", _otsu(gray)

    # Adaptativo bloco 11 (detalhes finos)
    yield "adapt_11", _adaptive(clahe_mod, 11, 4)

    # Adaptativo bloco 21 (iluminação suave)
    yield "adapt_21", _adaptive(clahe_mod, 21, 5)

    # Adaptativo bloco 31 (sombras amplas)
    yield "adapt_31", _adaptive(clahe_mod, 31, 7)

    # QR invertido (fundo escuro, módulos claros)
    yield "otsu_inv", _otsu(gray, invert=True)

    # ── Tier 1: upscale + variantes ──────────────────────────────────────────

    up = _upscale(clahe_mod, upscale)
    yield "lanczos", up

    # Unsharp masking pós-upscale
    blur = cv2.GaussianBlur(up, (0, 0), sigmaX=2.0)
    sharp = cv2.addWeighted(up, 1.5, blur, -0.5, 0)
    yield "unsharp", sharp

    # Limpeza mediana pós-sharp
    clean = cv2.medianBlur(sharp, 3)
    yield "median", clean

    # Otsu pós-pipeline completo
    yield "otsu_full", _otsu(clean)

    # Adaptativo pós-upscale (cobre mais variações de bloco)
    yield "adapt_up_15", _adaptive(clean, 15, 4)
    yield "adapt_up_25", _adaptive(clean, 25, 6)

    # Otsu invertido pós-upscale
    yield "otsu_full_inv", _otsu(clean, invert=True)

    # ── Tier 2: morfologia (fecha módulos partidos) ───────────────────────────

    otsu_bin = _otsu(clean)
    yield "morph_close_3", _morph_close(otsu_bin, 3)
    yield "morph_close_5", _morph_close(otsu_bin, 5)

    # Combinação morfologia + adaptativo
    adapt_clean = _adaptive(clean, 21, 5)
    yield "morph_adapt", _morph_close(adapt_clean, 3)

    # ── Tier 3: filtro bilateral (preserva bordas, remove granulação) ─────────

    bilat = cv2.bilateralFilter(up, d=7, sigmaColor=50, sigmaSpace=50)
    yield "bilateral", bilat
    yield "bilateral_otsu", _otsu(bilat)
    yield "bilateral_adapt", _adaptive(bilat, 21, 5)

    # ── Tier 4: gamma + pipeline completo ────────────────────────────────────

    up_bright = _upscale(_gamma(gray, 0.4), upscale)
    yield "gamma_up_bright_otsu", _otsu(up_bright)
    yield "gamma_up_bright_adapt", _adaptive(up_bright, 21, 5)

    up_dark = _upscale(_gamma(gray, 2.5), upscale)
    yield "gamma_up_dark_otsu", _otsu(up_dark)

    # CLAHE agressivo pós-upscale
    clahe_up_hard = _clahe(_upscale(gray, upscale), clip=8.0, tile=4)
    yield "clahe_hard_up", clahe_up_hard
    yield "clahe_hard_up_otsu", _otsu(clahe_up_hard)
    yield "clahe_hard_up_adapt", _adaptive(clahe_up_hard, 21, 5)


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
