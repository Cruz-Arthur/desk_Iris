"""
utils/paths.py
--------------
Resolução de paths que funciona em desenvolvimento e em app frozen
(PyInstaller, cx_Freeze, Nuitka).

Modos frozen:
  • PyInstaller onefile  — os dados (--add-data) são extraídos para uma pasta
    temporária em sys._MEIPASS; o EXE NÃO fica ao lado deles.
  • PyInstaller onedir / cx_Freeze / Nuitka — os dados ficam ao lado do EXE.

Por isso, em frozen, priorizamos sys._MEIPASS quando existir; caso contrário
usamos a pasta do executável.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _root() -> Path:
    if getattr(sys, "frozen", False):
        # PyInstaller onefile: dados extraídos para sys._MEIPASS
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        # cx_Freeze / Nuitka / PyInstaller onedir: dados ao lado do EXE
        return Path(sys.executable).resolve().parent
    # Dev: sobe de app/src/utils/ → app/src/ → app/ → repo root
    return Path(__file__).resolve().parents[3]


APP_ROOT   = _root()
ASSETS_DIR = APP_ROOT / "app" / "src" / "assets"
MODELS_DIR = APP_ROOT / "app" / "src" / "models"
DOCS_DIR   = APP_ROOT / "docs"
