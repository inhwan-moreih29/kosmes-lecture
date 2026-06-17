"""weights/*.pt -> ONNX export.

주의: 1교시 모델(MVTec 부품 결함)과 2교시 앱 모델(작업자 검출)은 도메인이 다르다.
이 스크립트는 1교시 모델을 onnx 로 내보낼 때 쓴다. 2교시 앱이 쓰는 사람 검출
모델은 02-app 쪽에서 별도로 준비/export 한다.

export 옵션은 02-app 의 후처리(NMS·좌표복원)를 단순화하도록 고정한다:
  - imgsz 고정, dynamic=False, simplify=True

사용:
  uv run export_onnx.py --weights weights/aug.pt
"""

import argparse
from pathlib import Path


def export(weights: str, imgsz: int = 640) -> None:
    # from ultralytics import YOLO
    # YOLO(weights).export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True, opset=12)
    raise NotImplementedError(f"onnx export 구현 예정: {weights} (imgsz={imgsz})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    print(f"[TODO] export {args.weights}")
