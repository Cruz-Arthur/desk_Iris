"""
infrastructure/video/enhance.py
--------------------------------
Realce de bordas via CLAHE no espaço LAB.

Uso:
    from app.src.infrastructure.video.enhance import EdgeEnhancer

    enhancer = EdgeEnhancer()          # cria CLAHE uma vez
    enhanced = enhancer.apply(frame)   # reutiliza a instância a cada frame
"""
from __future__ import annotations

import cv2
import numpy as np


class EdgeEnhancer:
    """
    Realce de bordas por CLAHE aplicado no canal L do espaço LAB.

    Parâmetros
    ----------
    clip_limit : float
        Limite de corte do histograma (padrão 2.5 — equilibra contraste e ruído).
    tile_size  : tuple[int, int]
        Grade de tiles para CLAHE local (padrão 8×8).

    Notas
    -----
    - A instância CLAHE é criada UMA vez no __init__, eliminando a alocação
      de objeto C++ a cada frame (~0.5 ms gratuitos por chamada).
    - Não thread-safe: cada thread deve usar sua própria instância.
    """

    def __init__(
        self,
        clip_limit: float = 2.5,
        tile_size: tuple[int, int] = (8, 8),
    ) -> None:
        self._clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_size,
        )

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Retorna uma cópia do frame com bordas realçadas.

        Parâmetros
        ----------
        frame : np.ndarray
            Frame BGR (uint8).

        Retorna
        -------
        np.ndarray
            Frame BGR realçado (uint8), mesma forma e dtype do input.
        """
        lab      = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b  = cv2.split(lab)
        l        = self._clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
