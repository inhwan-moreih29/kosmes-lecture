"""위험구역(ROI) 침입 판정 — RISA 직결.

박스가 사용자가 그린 ROI 안으로 들어오면 경보. 디바운싱과 결합해
"헛경보 없이" 울리게 하는 게 핸즈온 미션의 핵심.

핸즈온 복합 규칙 예: "동시 3명 초과 OR 10초 체류" (tracking 과 함께).

TODO(스캐폴드): ROI 폴리곤 정의, 박스-ROI 교차 판정
"""

from .infer import Detection

# ROI = 화면 좌표 폴리곤 [(x, y), ...]


def intrudes(det: Detection, roi: list[tuple[float, float]]) -> bool:
    """박스가 ROI 와 겹치면 True (박스 중심 또는 하단변 기준 — 결정 필요)."""
    raise NotImplementedError("박스-ROI 교차 판정 구현 예정")
