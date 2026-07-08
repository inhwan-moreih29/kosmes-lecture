"""onnxruntime 추론 + YOLO 출력 디코딩.

onnxruntime 직접 추론(용량 최소) 선택에 따른 책임:
  - 전처리: letterbox 리사이즈 + 정규화 + (1,3,H,W) 텐서
  - 추론: ort.InferenceSession.run
  - 후처리: 출력 텐서 디코딩 + NMS + 원본 좌표 복원
    (강의노트 02 에 "출력 텐서 shape -> 박스 디코딩" 그림 1장 들어갈 자리)

속도 레버(프레임스킵·리사이즈)는 여기 입력 단에서 작동.

모델: YOLO11 detect (COCO). 입력 [1,3,640,640], 출력 [1,84,8400].
  84 = 4(bbox cxcywh, 입력 좌표계) + 80(클래스 점수, objectness 없음).
  작업자 검출이 목적이므로 person(class 0)만 남긴다.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

PERSON_CLASS = 0


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls: int


def _letterbox(frame, imgsz: int):
    h0, w0 = frame.shape[:2]
    ratio = min(imgsz / h0, imgsz / w0)
    new_w, new_h = int(round(w0 * ratio)), int(round(h0 * ratio))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    pad_w = (imgsz - new_w) // 2
    pad_h = (imgsz - new_h) // 2
    canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    return canvas, ratio, pad_w, pad_h


def _nms(boxes_cxcywh, scores, iou_thr: float):
    if len(boxes_cxcywh) == 0:
        return np.array([], dtype=np.int32)
    x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou < iou_thr]
    return np.array(keep, dtype=np.int32)


def _make_session(onnx_path: str) -> ort.InferenceSession:
    """가용 프로바이더를 감지해 세션 생성.

    우선순위: DirectML(GPU) > CPU. OS 분기 없이 런타임에 결정한다.
      - Mac(dev): onnxruntime CPU 패키지 -> DML 미가용 -> CPU 로 폴백.
      - Windows: onnxruntime-directml 패키지 -> DmlExecutionProvider 로 GPU 가속.
    프로바이더 리스트는 "선호 순서"이며, 앞의 것이 실패하면 뒤로 폴백한다.
    """
    available = ort.get_available_providers()
    opts = ort.SessionOptions()
    providers: list[str] = []
    if "DmlExecutionProvider" in available:
        # DirectML EP 는 메모리 패턴/병렬 실행을 지원하지 않는다(MS 권장 설정).
        opts.enable_mem_pattern = False
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        providers.append("DmlExecutionProvider")
    providers.append("CPUExecutionProvider")
    return ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)


class Detector:
    def __init__(self, onnx_path: str, imgsz: int = 640):
        self.sess = _make_session(onnx_path)
        # 실제 활성 프로바이더(폴백 결과 반영). UI/로그 표기용.
        self.provider = self.sess.get_providers()[0]
        self.imgsz = imgsz
        self._input_name = self.sess.get_inputs()[0].name
        self._output_name = self.sess.get_outputs()[0].name

    def infer(
        self,
        frame,
        conf_thr: float = 0.25,
        iou_thr: float = 0.45,
        imgsz: int | None = None,
    ) -> list[Detection]:
        """BGR 프레임 -> Detection 리스트 (원본 좌표계).

        imgsz: 이번 추론의 letterbox 목표 크기(32의 배수). 작을수록 빠르고 덜 정확.
               동적 입력 ONNX 라서 호출마다 다른 크기 가능(속도 레버).
        """
        h0, w0 = frame.shape[:2]
        img, ratio, pad_w, pad_h = _letterbox(frame, imgsz or self.imgsz)
        blob = (
            img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        )[np.newaxis]
        raw = self.sess.run([self._output_name], {self._input_name: blob})[0]
        preds = raw[0].T
        boxes_cxcywh = preds[:, :4]
        class_scores = preds[:, 4:]
        scores = class_scores.max(axis=1)
        cls_ids = class_scores.argmax(axis=1)
        mask = (scores >= conf_thr) & (cls_ids == PERSON_CLASS)
        boxes_cxcywh = boxes_cxcywh[mask]
        scores = scores[mask]
        cls_ids = cls_ids[mask]
        if len(scores) == 0:
            return []
        keep = _nms(boxes_cxcywh, scores, iou_thr)
        detections: list[Detection] = []
        for i in keep:
            cx, cy, bw, bh = boxes_cxcywh[i]
            x1 = max(0.0, min((cx - bw / 2 - pad_w) / ratio, float(w0)))
            y1 = max(0.0, min((cy - bh / 2 - pad_h) / ratio, float(h0)))
            x2 = max(0.0, min((cx + bw / 2 - pad_w) / ratio, float(w0)))
            y2 = max(0.0, min((cy + bh / 2 - pad_h) / ratio, float(h0)))
            detections.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2,
                score=float(scores[i]), cls=int(cls_ids[i])))
        return detections
