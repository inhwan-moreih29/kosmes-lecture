"""오탐 깎기 — 출력/크기/입력 레벨 노브.

  - confidence 임계값: score < thr 제거 (출력 레벨, 1교시 복습)
  - 최소 크기 필터: 박스 면적 < min_area 제거 (크기 레벨)
  - 밝기/대비 보정(보너스): 분포 밖 실패 회수 (입력 레벨 전처리)

TODO(스캐폴드): 각 필터 함수 구현
"""

from .infer import Detection


def by_confidence(dets: list[Detection], thr: float) -> list[Detection]:
    return [d for d in dets if d.score >= thr]


def by_min_size(dets: list[Detection], min_area: float) -> list[Detection]:
    return [d for d in dets if (d.x2 - d.x1) * (d.y2 - d.y1) >= min_area]


def adjust_brightness_contrast(frame, brightness: float = 0.0, contrast: float = 1.0):
    """보너스 노브. cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)."""
    raise NotImplementedError("밝기/대비 보정 구현 예정")
