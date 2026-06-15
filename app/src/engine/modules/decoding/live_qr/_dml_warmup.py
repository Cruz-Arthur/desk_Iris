"""
_dml_warmup.py — executado como subprocesso independente.

Objetivo: compilar shaders D3D12/DML e popular o cache em disco.
O processo principal NUNCA toca D3D12 durante esse trabalho.
Após este script terminar, o processo principal cria a InferenceSession
usando o cache de disco (rápido, sem compilação, sem freeze de UI).
"""

import sys


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(1)

    model_path = sys.argv[1]

    try:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        all_p = ort.get_available_providers()
        providers = (
            ["DmlExecutionProvider", "CPUExecutionProvider"]
            if "DmlExecutionProvider" in all_p
            else all_p
        )

        # Criação da sessão: compila shaders D3D12 e grava cache em disco.
        # Trabalho pesado fica 100% confinado a este subprocesso.
        session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        del session

    except Exception as exc:
        print(f"[DML-WARMUP] erro: {exc}", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
