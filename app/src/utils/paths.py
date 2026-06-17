"""
utils/paths.py
--------------
Resolução de paths que funciona tanto em desenvolvimento quanto em app frozen
(cx_Freeze, Nuitka). Em modo frozen sys.frozen=True e sys.executable aponta
para o EXE — usamos ele como âncora no lugar de __file__.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _root() -> Path:
    if getattr(sys, "frozen", False):
        # Frozen: EXE está na raiz da pasta de distribuição
        return Path(sys.executable).resolve().parent
    # Dev: sobe de app/src/utils/ → app/src/ → app/ → repo root
    return Path(__file__).resolve().parents[3]


APP_ROOT   = _root()
ASSETS_DIR = APP_ROOT / "app" / "src" / "assets"
MODELS_DIR = APP_ROOT / "app" / "src" / "models"
DOCS_DIR   = APP_ROOT / "docs"
