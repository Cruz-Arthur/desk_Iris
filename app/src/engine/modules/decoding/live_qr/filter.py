"""
Iris — Image Enhancement Pipeline
==================================
Responsabilidade única: melhorar a qualidade de imagens para maximizar
a chance de leitura de QR Codes em condições adversas.

Este módulo NÃO detecta nem decodifica QR Codes.
Toda lógica de aprimoramento de imagem deve centralizar-se aqui.
"""

from __future__ import annotations

from typing import Tuple, Generator

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Aprimoramento principal
# ─────────────────────────────────────────────────────────────────────────────

def enhance_for_qr(
    crop:    np.ndarray,
    upscale: float = 3.0,
) -> Generator[Tuple[str, np.ndarray], None, None]:
    """
    Gera múltiplas variantes de imagem aprimoradas a partir de um crop.
    Implementa o pipeline sequencial calibrado para evitar distorção morfológica:
    Gray -> CLAHE -> Lanczos4 Upscale -> Gentle Unsharp -> Median Blur -> Thresholds (Otsu & Adaptativo).
    Executa processamento sob demanda (lazy evaluation) reduzindo consumo de CPU.

    Args:
        crop:    Região de interesse em BGR ou grayscale.
        upscale: Fator de zoom digital aplicado. Padrão 3.0.

    Yields:
        Tuplas (nome_variante, imagem_aprimorada) em ordem crescente
        de processamento computacional.
    """

    # ── 1. Tons de cinza (Baseline) ───────────────────────────────────────────
    gray = (
        cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if crop.ndim == 3
        else crop.copy()
    )
    yield "gray", gray

    # ── 2. CLAHE Moderado ─────────────────────────────────────────────────────
    # clipLimit reduzido para 2.0 para evitar superexposição de ruído de fundo
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(gray)
    yield "clahe", img_clahe

    # ── 3. Ampliação com Lanczos ──────────────────────────────────────────────
    if upscale > 1.0:
        img_ampliada = cv2.resize(
            img_clahe,
            None,
            fx=upscale,
            fy=upscale,
            interpolation=cv2.INTER_LANCZOS4
        )
    else:
        img_ampliada = img_clahe.copy()

    yield "lanczos_upscale", img_ampliada

    # ── 4. Unsharp Masking Suave ──────────────────────────────────────────────
    # Acentuação de bordas sem destruir a morfologia dos módulos do QR Code
    blur_pesado = cv2.GaussianBlur(img_ampliada, (0, 0), sigmaX=2.0)
    img_sharp = cv2.addWeighted(img_ampliada, 1.5, blur_pesado, -0.5, 0)
    yield "unsharp_masking", img_sharp

    # ── 5. Desfoque Mediano Leve (Limpeza) ────────────────────────────────────
    img_limpa = cv2.medianBlur(img_sharp, 3)
    yield "median_blur", img_limpa

    # ── 6. Limiarização de Otsu Global ────────────────────────────────────────
    _, final_otsu = cv2.threshold(img_limpa, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield "otsu_final", final_otsu

    # ── 7. Limiarização Adaptativa Gaussiana (Superior para Iluminação) ───────
    # Avalia a iluminação localmente (bloco de 21x21 pixels). Essencial para
    # QR Codes impressos em superfícies reflexivas ou com sombras curvas.
    adapt_thresh = cv2.adaptiveThreshold(
        img_limpa,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        5
    )
    yield "adaptive_thresh", adapt_thresh

    # ── 8. Otsu Invertido (Redundância de Segurança) ──────────────────────────
    # QR codes impressos em negativo (fundo preto, código branco)
    yield "otsu_invertido", cv2.bitwise_not(final_otsu)


# ─────────────────────────────────────────────────────────────────────────────
# Recorte auxiliar
# ─────────────────────────────────────────────────────────────────────────────

def crop_with_padding(
    frame:   np.ndarray,
    anchor:  Tuple[int, int, int, int],
    padding: int = 16,
) -> Tuple[np.ndarray, int, int]:
    """
    Extrai uma região do frame com padding de segurança, respeitando os
    limites da imagem.

    Args:
        frame:   Frame original (BGR ou grayscale).
        anchor:  Bounding box (x, y, w, h) em pixels.
        padding: Número de pixels extras a incluir em cada direção.

    Returns:
        (crop, x1, y1)
            crop — região recortada.
            x1   — origem X do crop no frame original.
            y1   — origem Y do crop no frame original.
            x1/y1 são necessárias para remapear coordenadas de detecção.
    """
    ax, ay, aw, ah = anchor
    fh, fw = frame.shape[:2]

    x1 = max(0,  ax - padding)
    y1 = max(0,  ay - padding)
    x2 = min(fw, ax + aw + padding)
    y2 = min(fh, ay + ah + padding)

    return frame[y1:y2, x1:x2], x1, y1