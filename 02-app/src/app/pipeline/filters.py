"""오탐 깎기 — 출력/크기/입력 레벨 노브.

  - confidence 임계값: score < thr 제거 (출력 레벨, 1교시 복습)
  - 최소 크기 필터: 박스 높이 < 기준(프레임 높이의 %) 제거 (크기 레벨)
  - 밝기/대비 보정(보너스): 분포 밖 실패 회수 (입력 레벨 전처리)

TODO(스캐폴드): 각 필터 함수 구현
"""

import cv2

from .infer import Detection


def by_confidence(dets: list[Detection], thr: float) -> list[Detection]:
    return [d for d in dets if d.score >= thr]


def by_min_height(dets: list[Detection], min_h: float) -> list[Detection]:
    """박스 높이(px) < min_h 제거. 사람은 거리에 따라 키가 줄어드므로
    '높이'가 면적보다 직관적인 '멀리/자잘한 오탐' 기준이 된다."""
    return [d for d in dets if (d.y2 - d.y1) >= min_h]


def adjust_brightness_contrast(frame, brightness: float = 0.0, contrast: float = 1.0):
    """보너스 노브. cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)."""
    return cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
