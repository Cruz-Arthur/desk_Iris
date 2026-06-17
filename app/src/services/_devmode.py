"""
Modo de desenvolvimento do Iris — variável de ambiente IRIS_DEVMODE.

Quando IRIS_DEVMODE=1, todos os adaptadores de serviços externos (banco de
dados, APIs, PLC, impressora, SMB) suprimem os envios de dados e logam a
operação que seria executada.  Leituras do banco retornam None/defaults para
que a aplicação continue navegável sem rede.

Uso:
    # Windows
    set IRIS_DEVMODE=1 && python -m app.src.UIX.main

    # Ou via settings.local.json (já configurado):
    { "env": { "IRIS_DEVMODE": "1" } }
"""
from __future__ import annotations

import os

DEV_MODE: bool = os.getenv("IRIS_DEVMODE", "0").strip() == "1"

if DEV_MODE:
    pass
