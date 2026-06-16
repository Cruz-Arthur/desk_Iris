"""
tools/quantize_model.py
-----------------------
Standalone INT8 dynamic quantization for the Iris YOLO ONNX model.

Usage:
    python tools/quantize_model.py
    python tools/quantize_model.py --input path/to/best.onnx --output path/to/best_int8.onnx

The quantized model is a drop-in replacement: same inputs, same outputs, same
IrisDetector API — just load it with IrisDetector(model_path="...best_int8.onnx").

Expected speedup: 1.5–3× on inference-heavy CPU workloads (Intel/AMD x86-64 with
VNNI or AVX-512 VNNI). ARM NEON paths also benefit. Memory footprint drops ~4×.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_INPUT = Path(__file__).resolve().parents[1] / (
    "app/src/models/live_qr_yolo/train/weights/best.onnx"
)
_DEFAULT_OUTPUT = _DEFAULT_INPUT.parent / "best_int8.onnx"


def quantize(input_path: Path, output_path: Path) -> None:
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        logger.error(
            "onnxruntime.quantization not available. "
            "Install onnxruntime (not onnxruntime-directml) for quantization tools:\n"
            "  pip install onnxruntime"
        )
        sys.exit(1)

    if not input_path.exists():
        logger.error("Input model not found: %s", input_path)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Quantizing %s → %s", input_path.name, output_path.name)
    logger.info("Type: INT8 dynamic (no calibration dataset required)")

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        # Quantize all operators that support INT8 weights
        per_channel=False,
        reduce_range=False,
    )

    in_mb  = input_path.stat().st_size  / 1_048_576
    out_mb = output_path.stat().st_size / 1_048_576
    logger.info("Done. %.1f MB → %.1f MB (%.0f%% smaller)", in_mb, out_mb, (1 - out_mb / in_mb) * 100)
    logger.info("Load with: IrisDetector(model_path=%r)", str(output_path))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Iris ONNX INT8 quantizer")
    p.add_argument("--input",  "-i", type=Path, default=_DEFAULT_INPUT,
                   help=f"Input .onnx path (default: {_DEFAULT_INPUT})")
    p.add_argument("--output", "-o", type=Path, default=_DEFAULT_OUTPUT,
                   help=f"Output .onnx path (default: {_DEFAULT_OUTPUT})")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    quantize(args.input, args.output)
