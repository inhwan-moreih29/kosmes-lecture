"""추적 — ID 유지 위에 카운팅 / 체류시간.

detection 은 프레임마다 독립 -> 같은 객체에 ID 를 유지해야
"총 몇 명 / 얼마나 오래(체류시간)"가 정확해진다. 한 코너로 묶어 시연.

간단 IOU 기반 트래커로 충분(교보재 목적). 무거운 ReID 불필요.
"""

from .infer import Detection


def _iou(a: Detection, b: Detection) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    return inter / (area_a + area_b - inter + 1e-9)


class Track:
    def __init__(self, track_id: int, det: Detection):
        self.id = track_id
        self.det = det
        self.frames_seen = 1     # 체류시간 = frames_seen / fps (누적)
        self.misses = 0          # 연속 미관측 프레임 수
        self.hit_streak = 1      # 연속 관측 프레임 수 (놓치면 0으로 리셋)
        self.confirmed = False   # hit_streak 가 min_hits 도달 시 True (이후 유지)


class IouTracker:
    def __init__(self, iou_thr: float = 0.3, max_misses: int = 10):
        self.iou_thr = iou_thr
        self.max_misses = max_misses    # 초과 미관측 시 트랙 폐기
        self.tracks: dict[int, Track] = {}
        self._next_id = 0

    def update(self, dets: list[Detection], min_hits: int = 1) -> dict[int, Detection]:
        """현재 프레임 탐지 -> {track_id: Detection}.

        그리디 IOU 매칭: (트랙, 탐지) 쌍을 IOU 내림차순으로 1:1 배정.
        미매칭 탐지는 신규 ID, 미매칭 트랙은 miss++ 후 한도 초과 시 제거.

        min_hits: '확정'에 필요한 연속 관측 프레임 수(디바운싱). hit_streak 가
            이 값에 도달하면 confirmed=True 가 되고 이후 유지된다(sticky). 도달 전
            한 프레임이라도 놓치면 hit_streak 가 0으로 리셋돼 처음부터 다시 쌓는다.
        """
        # 후보 쌍을 IOU 내림차순 정렬
        pairs = [
            (_iou(t.det, d), tid, di)
            for tid, t in self.tracks.items()
            for di, d in enumerate(dets)
        ]
        pairs.sort(reverse=True)

        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for iou, tid, di in pairs:
            if iou < self.iou_thr:
                break
            if tid in used_tracks or di in used_dets:
                continue
            t = self.tracks[tid]
            t.det = dets[di]
            t.frames_seen += 1
            t.misses = 0
            t.hit_streak += 1
            if t.hit_streak >= min_hits:
                t.confirmed = True
            used_tracks.add(tid)
            used_dets.add(di)

        # 미매칭 트랙 정리: 연속이 끊겼으므로 hit_streak 리셋(확정 전이면 처음부터)
        for tid in list(self.tracks):
            if tid not in used_tracks:
                t = self.tracks[tid]
                t.misses += 1
                t.hit_streak = 0
                if t.misses > self.max_misses:
                    del self.tracks[tid]

        # 미매칭 탐지 -> 신규 트랙 (min_hits==1 이면 즉시 확정)
        for di, d in enumerate(dets):
            if di not in used_dets:
                t = Track(self._next_id, d)
                if t.hit_streak >= min_hits:
                    t.confirmed = True
                self.tracks[self._next_id] = t
                self._next_id += 1

        return {tid: t.det for tid, t in self.tracks.items()}

    def count(self) -> int:
        """현재 살아있는 트랙 수 (= 현재 화면 내 인원 추정)."""
        return sum(1 for t in self.tracks.values() if t.misses == 0)

    def dwell_seconds(self, track_id: int, fps: float) -> float:
        t = self.tracks.get(track_id)
        return t.frames_seen / fps if t and fps > 0 else 0.0
