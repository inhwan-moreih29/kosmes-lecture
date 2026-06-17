"""추적 — ID 유지 위에 카운팅 / 체류시간.

detection 은 프레임마다 독립 -> 같은 객체에 ID 를 유지해야
"총 몇 명 / 얼마나 오래(체류시간)"가 정확해진다. 한 코너로 묶어 시연.

간단 IOU 기반 트래커로 충분(교보재 목적). 무거운 ReID 불필요.

TODO(스캐폴드): IOU 매칭 트래커, ID별 등장 프레임/시간 누적
"""

from .infer import Detection


class Track:
    def __init__(self, track_id: int):
        self.id = track_id
        self.frames_seen = 0      # 체류시간 = frames_seen / fps


class IouTracker:
    def __init__(self, iou_thr: float = 0.3):
        self.iou_thr = iou_thr
        self.tracks: dict[int, Track] = {}

    def update(self, dets: list[Detection]) -> dict[int, Detection]:
        """현재 프레임 탐지 -> {track_id: Detection} 매핑."""
        raise NotImplementedError("IOU 매칭 트래커 구현 예정")

    def count(self) -> int:
        return len(self.tracks)
