"""디바운싱 — 시간 레벨 오탐 제거.

버튼 신호 안정화에서 온 개념. 프레임마다 깜빡이는 탐지를
"N프레임 연속일 때만 인정" -> 순간 오탐 제거. 대가는 반응 지연(트레이드오프).

TODO(스캐폴드): 상태별(예: ROI 침입, 경보) 연속 카운터 관리
"""


class Debouncer:
    def __init__(self, n_frames: int = 3):
        self.n_frames = n_frames
        self._streak = 0

    def update(self, active: bool) -> bool:
        """이번 프레임 active 여부를 받아 '확정 상태'를 반환."""
        self._streak = self._streak + 1 if active else 0
        return self._streak >= self.n_frames
