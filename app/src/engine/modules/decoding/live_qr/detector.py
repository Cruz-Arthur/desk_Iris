"""
engine/detector.py
------------------
Motor de detecção ONNX para o projeto IRIS.

Recebe frames (np.ndarray BGR) ou caminhos de imagem,
executa inferência no modelo YOLOv8/ONNX e retorna uma
lista de detecções no formato:

    [Detection(x1, y1, x2, y2, confidence), ...]

Uso como módulo
---------------
    from engine.detector import IrisDetector, Detection

    detector = IrisDetector()  # carrega modelo padrão
    detections = detector.detect(frame)   # frame: np.ndarray BGR
    for d in detections:
        print(d.x1, d.y1, d.x2, d.y2, d.confidence)

Uso direto (teste com câmera ao vivo)
--------------------------------------
    python detector.py

Uso direto (teste com imagem estática)
---------------------------------------
    python detector.py --image caminho/para/imagem.jpg
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes padrão
# ---------------------------------------------------------------------------
_DEFAULT_IMG_SIZE   = 640
_DEFAULT_CONF_THRES = 0.1
_DEFAULT_IOU_THRES  = 0.45

_DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[4] / (
    "models/live_qr_yolo/train/weights/best.onnx"
)


# ---------------------------------------------------------------------------
# Estrutura de saída
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Detection:
    """Uma detecção retornada pelo modelo."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Retorna (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


# ---------------------------------------------------------------------------
# Motor principal
# ---------------------------------------------------------------------------
class IrisDetector:
    """
    Motor de detecção baseado em ONNX Runtime.

    Parâmetros
    ----------
    model_path : str | Path | None
        Caminho para o arquivo .onnx. Se None, usa o caminho padrão do projeto.
    img_size : int
        Tamanho de entrada do modelo (quadrado). Padrão: 640.
    conf_threshold : float
        Confiança mínima para aceitar uma detecção. Padrão: 0.4.
    iou_threshold : float
        Limiar de IoU para NMS. Padrão: 0.45.
    providers : list[str] | None
        Execution providers do ONNX Runtime. None = automático (GPU se disponível).
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        img_size: int = _DEFAULT_IMG_SIZE,
        conf_threshold: float = _DEFAULT_CONF_THRES,
        iou_threshold: float = _DEFAULT_IOU_THRES,
        providers: list[str] | None = None,
    ) -> None:
        self.img_size       = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold

        resolved = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        resolved = resolved.resolve()

        if not resolved.exists():
            raise FileNotFoundError(f"Modelo ONNX não encontrado: {resolved}")

        _providers = providers or ort.get_available_providers()

        self._session = ort.InferenceSession(str(resolved), providers=_providers)
        self._input_name = self._session.get_inputs()[0].name

        logger.info(
            "IrisDetector carregado — modelo: %s | providers: %s",
            resolved.name,
            self._session.get_providers(),
        )

    # ------------------------------------------------------------------
    # Pré e pós-processamento
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Redimensiona e normaliza o frame para entrada do modelo."""
        img = cv2.resize(frame, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))   # HWC → CHW
        img = np.expand_dims(img, axis=0)     # batch dim
        return img

    def _postprocess(
        self,
        raw_output: list[np.ndarray],
        original_shape: tuple[int, int],
    ) -> list[Detection]:
        """
        Converte a saída crua do modelo em lista de Detection.

        Formato esperado de saída do ONNX: (1, 6, 8400)
        Colunas: [cx, cy, w, h, conf, class_id]
        """
        h, w = original_shape

        output = np.squeeze(raw_output[0])

        if output.ndim == 1:
            output = np.expand_dims(output, axis=0)

        if output.ndim == 2 and output.shape[0] == 6:
            output = output.T

        raw_boxes: list[tuple[int, int, int, int, float]] = []

        for det in output:
            cx, cy, bw, bh, conf, _cls = det

            if conf < self.conf_threshold:
                continue

            # --- PONTO 1: Adicionar 10% de margem na largura e altura ---
            bw = bw * 1.15
            bh = bh * 1.15

            x1 = int((cx - bw / 2) * w / self.img_size)
            y1 = int((cy - bh / 2) * h / self.img_size)
            x2 = int((cx + bw / 2) * w / self.img_size)
            y2 = int((cy + bh / 2) * h / self.img_size)

            # --- PONTO 2: Travar limites para não vazar da imagem ---
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            raw_boxes.append((x1, y1, x2, y2, float(conf)))

        return self._apply_nms(raw_boxes)

    def _apply_nms(
        self,
        boxes: list[tuple[int, int, int, int, float]],
    ) -> list[Detection]:
        """Aplica Non-Maximum Suppression e retorna lista de Detection."""
        if not boxes:
            return []

        bboxes = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2, _ in boxes]
        scores = [conf for *_, conf in boxes]

        indices = cv2.dnn.NMSBoxes(
            bboxes,
            scores,
            self.conf_threshold,
            self.iou_threshold,
        )

        if len(indices) == 0:
            return []

        return [
            Detection(*boxes[i][:4], confidence=boxes[i][4])
            for i in indices.flatten()
        ]

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Executa detecção em um frame BGR.

        Parâmetros
        ----------
        frame : np.ndarray
            Imagem BGR (H, W, 3).

        Retorna
        -------
        list[Detection]
            Lista de detecções após NMS, ordenada por confiança decrescente.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Frame inválido ou vazio.")

        input_tensor = self._preprocess(frame)
        outputs = self._session.run(None, {self._input_name: input_tensor})
        detections = self._postprocess(outputs, frame.shape[:2])

        return sorted(detections, key=lambda d: d.confidence, reverse=True)

    def detect_from_path(self, image_path: str | Path) -> list[Detection]:
        """
        Executa detecção em uma imagem salva em disco.

        Parâmetros
        ----------
        image_path : str | Path
            Caminho para o arquivo de imagem.

        Retorna
        -------
        list[Detection]
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Imagem não encontrada: {path}")

        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Não foi possível ler a imagem: {path}")

        return self.detect(frame)


# ---------------------------------------------------------------------------
# Helpers de visualização (usados no modo CLI)
# ---------------------------------------------------------------------------

def _draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Desenha bounding boxes e confiança sobre o frame. Retorna cópia."""
    out = frame.copy()
    for d in detections:
        cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"{d.confidence:.2f}",
            (d.x1, d.y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    return out


# ---------------------------------------------------------------------------
# Entrypoint CLI
# ---------------------------------------------------------------------------

def _run_camera(detector: IrisDetector) -> None:
    """Loop ao vivo usando CameraManager de video.raw."""
    # Importação tardia — evita dependência circular em testes unitários
    try:
        from video.raw import CameraManager  # type: ignore[import]
    except ImportError:
        logger.warning(
            "video.raw não encontrado. Usando cv2.VideoCapture diretamente."
        )
        _run_camera_fallback(detector)
        return

    latest_frame: dict[str, np.ndarray | None] = {"frame": None}

    def _on_frame(frame: np.ndarray) -> None:
        latest_frame["frame"] = frame

    print("📷 Câmera iniciada via CameraManager (ESC para sair)\n")

    with CameraManager() as cam:
        cam.subscribe(_on_frame)

        while True:
            frame = latest_frame["frame"]
            if frame is None:
                time.sleep(0.01)
                continue

            start = time.perf_counter()
            detections = detector.detect(frame)
            fps = 1.0 / (time.perf_counter() - start)

            out = _draw_detections(frame, detections)
            cv2.putText(
                out,
                f"FPS: {fps:.1f}  Det: {len(detections)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.imshow("IRIS Detector — ao vivo", out)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cv2.destroyAllWindows()


def _run_camera_fallback(detector: IrisDetector) -> None:
    """Loop ao vivo com cv2.VideoCapture (sem CameraManager)."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir a câmera 0.")

    print("📷 Câmera iniciada (ESC para sair)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar frame.")
            break

        start = time.perf_counter()
        detections = detector.detect(frame)
        fps = 1.0 / (time.perf_counter() - start)

        out = _draw_detections(frame, detections)
        cv2.putText(
            out,
            f"FPS: {fps:.1f}  Det: {len(detections)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.imshow("IRIS Detector — ao vivo", out)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def _run_image(detector: IrisDetector, image_path: str) -> None:
    """Executa detecção em imagem estática e exibe resultado."""
    print(f"🖼  Imagem: {image_path}")
    detections = detector.detect_from_path(image_path)

    frame = cv2.imread(image_path)
    out   = _draw_detections(frame, detections)

    print(f"✅  {len(detections)} detecção(ões) encontrada(s):")
    for i, d in enumerate(detections, 1):
        print(f"   [{i}] bbox=({d.x1},{d.y1},{d.x2},{d.y2})  conf={d.confidence:.3f}")

    cv2.imshow("IRIS Detector — imagem", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IRIS Detector — teste rápido via CLI"
    )
    parser.add_argument(
        "--image", "-i",
        metavar="PATH",
        help="Caminho para imagem estática. Sem este argumento, abre a câmera.",
    )
    parser.add_argument(
        "--model", "-m",
        metavar="PATH",
        help="Caminho alternativo para o modelo .onnx.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=_DEFAULT_CONF_THRES,
        help=f"Threshold de confiança (padrão: {_DEFAULT_CONF_THRES}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    args = _parse_args()

    detector = IrisDetector(
        model_path=args.model,
        conf_threshold=args.conf,
    )

    if args.image:
        _run_image(detector, args.image)
    else:
        _run_camera(detector)
