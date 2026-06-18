"""위험구역(ROI) 침입 판정 — RISA 직결.

박스가 사용자가 그린 ROI 안으로 들어오면 경보. 디바운싱과 결합해
"헛경보 없이" 울리게 하는 게 핸즈온 미션의 핵심.

핸즈온 복합 규칙 예: "동시 3명 초과 OR 10초 체류" (tracking 과 함께).

TODO(스캐폴드): ROI 폴리곤 정의, 박스-ROI 교차 판정
"""

import cv2
import numpy as np

from .infer import Detection

# ROI = 화면 좌표 폴리곤 [(x, y), ...]


def overlap_ratio(det: Detection, roi: list[tuple[float, float]], grid: int = 7) -> float:
    """박스 면적 중 ROI 폴리곤 내부에 든 비율(0~1).

    박스를 grid×grid 격자점으로 샘플해 내부 점 비율로 근사. 폴리곤 클리핑 없이
    동적 ROI 에 바로 대응(교보재엔 충분). cv2.pointPolygonTest: 양수=내부.
    """
    if len(roi) < 3:
        return 0.0
    bw, bh = det.x2 - det.x1, det.y2 - det.y1
    if bw <= 0 or bh <= 0:
        return 0.0
    contour = np.array(roi, dtype=np.float32).reshape((-1, 1, 2))
    inside = 0
    for i in range(grid):
        px = det.x1 + bw * (i + 0.5) / grid
        for j in range(grid):
            py = det.y1 + bh * (j + 0.5) / grid
            if cv2.pointPolygonTest(contour, (float(px), float(py)), False) >= 0:
                inside += 1
    return inside / (grid * grid)


def intrudes(det: Detection, roi: list[tuple[float, float]], thr: float = 0.3) -> bool:
    """박스-ROI 겹침 면적 비율이 thr 이상이면 '진입'.

    thr=0 이면 조금이라도 닿으면(비율>0) 진입. 높일수록 깊이 들어와야 인정.
    """
    if len(roi) < 3:
        return False
    r = overlap_ratio(det, roi)
    return r > 0 and r >= thr
