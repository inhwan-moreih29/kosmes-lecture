"""onnxruntime 추론 + YOLO 출력 디코딩.

onnxruntime 직접 추론(용량 최소) 선택에 따른 책임:
  - 전처리: letterbox 리사이즈 + 정규화 + (1,3,H,W) 텐서
  - 추론: ort.InferenceSession.run
  - 후처리: 출력 텐서 디코딩 + NMS + 원본 좌표 복원
    (강의노트 02 에 "출력 텐서 shape -> 박스 디코딩" 그림 1장 들어갈 자리)

속도 레버(프레임스킵·리사이즈)는 여기 입력 단에서 작동.

TODO(스캐폴드): InferenceSession 로드, letterbox, NMS 구현
"""

from dataclasses import dataclass


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls: int


class Detector:
    def __init__(self, onnx_path: str, imgsz: int = 640):
        # self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.imgsz = imgsz

    def infer(self, frame) -> list[Detection]:
        """BGR 프레임 -> Detection 리스트 (원본 좌표계)."""
        raise NotImplementedError("onnxruntime 추론 + NMS 구현 예정")
