"""추론 워커 스레드 — UI 반응성과 추론을 분리.

CPU ONNX 추론은 프레임당 수십 ms 가 걸린다. 이를 GUI(메인) 스레드에서 돌리면
이벤트 루프가 막혀 슬라이더/버튼이 버벅인다. 그래서 캡처→전처리→추론→추적→
ROI/경보→그리기까지 모두 이 워커에서 수행하고, 완성 프레임과 상태만 시그널로
GUI 에 넘긴다. GUI 는 표시·상태 갱신만 → 추론과 무관하게 항상 반응.

스레드 간 전달:
  - GUI -> 워커: 노브값/ROI(불변 스냅샷), 재생 플래그(단순 대입, GIL 안전)
  - 워커 -> GUI: result_ready(FrameResult), opened(total_frames)
"""

import time
from dataclasses import dataclass

import cv2
import copy

import numpy as np
from PySide6.QtCore import QThread, Signal

from app.pipeline import filters, roi as roi_mod
from app.pipeline.tracking import IouTracker

BASE_CONF = 0.15   # 추론 1차 컷(노브 신뢰도는 후처리에서 다시 적용)


@dataclass
class FrameResult:
    frame: np.ndarray          # 오버레이까지 그려진 BGR 프레임
    pos: int                   # 현재 프레임 인덱스
    fps: float                 # 처리 FPS(워커 기준)
    count: int                 # 현재 인원
    roi_count: int             # ROI 내 인원
    max_dwell: float           # ROI 내 최대 체류(초)
    alarm: bool                # 디바운싱 적용 경보 상태
    alarm_count: int           # 누적 경보 횟수


class InferenceWorker(QThread):
    result_ready = Signal(object)   # FrameResult
    opened = Signal(int)            # total frames

    def __init__(self, path: str, detector, parent=None):
        super().__init__(parent)
        self.path = path
        self.detector = detector

        # GUI 가 갱신하는 공유 상태(단순 대입, GIL 안전)
        self._values = None
        self._roi: list[tuple[float, float]] = []
        self._speed = 1.0
        self._paused = False
        self._loop = True
        self._unlock = False                # 페이싱 해제(최대 속도)
        self._seek_to = -1
        self._running = True
        self._dirty = False                 # 일시정지 중 옵션 변경 → 재처리 요청
        self._last_frame = None             # 정지 미리보기용 직전 원본 프레임
        self._last_pos = 0

        # 워커 소유 파이프라인 상태
        self.tracker = IouTracker()
        self.roi_dwell: dict[int, int] = {}
        self.last_tracks: dict[int, object] = {}
        self.frame_idx = 0
        self.alarm_count = 0
        self.alarm_prev = False
        self.last_raw = []          # 직전 추론의 raw 검출(>= BASE_CONF)
        self.src_fps = 25.0
        # 처리 FPS = 1/작업시간EMA (read+process, 페이싱 sleep 제외). 매 프레임 갱신 →
        # 리사이즈/스킵으로 작업시간이 줄면 즉시 FPS 상승이 보인다(처리 능력 지표).
        self.fps_value = 0.0
        self.work_ema = 0.0

    # ---- GUI 스레드에서 호출하는 세터 ----
    def set_values(self, v): self._values = v; self._dirty = True
    def set_roi(self, roi): self._roi = list(roi); self._dirty = True
    def set_speed(self, s): self._speed = s
    def set_paused(self, p): self._paused = p
    def set_loop(self, on): self._loop = on
    def set_unlock(self, on): self._unlock = on
    def seek(self, frame_idx: int): self._seek_to = frame_idx
    def stop(self):
        self._running = False

    def _reset(self):
        self.tracker = IouTracker()
        self.roi_dwell.clear()
        self.last_tracks = {}
        self.last_raw = []
        self.frame_idx = 0
        self.alarm_prev = False

    # ---- 워커 루프 ----
    def run(self):
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            return
        self.src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.opened.emit(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        while self._running:
            if self._seek_to >= 0:
                pos = max(self._seek_to, 0)
                self._seek_to = -1
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                self._reset()
                # 일시정지 중에도 이동 위치를 한 프레임 미리보기로 보여준다
                if self._paused and self._values is not None:
                    ok, frame = cap.read()
                    if ok:
                        npos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        self._last_frame, self._last_pos = frame, npos
                        self.result_ready.emit(self._process(frame, npos))
                        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)  # 재개 시 pos 부터
                    self._dirty = False
                    continue

            if self._paused or self._values is None:
                # 일시정지 중 옵션 변경 → 정지된 1프레임만 재처리(영구 상태 불변)
                if self._paused and self._dirty and self._last_frame is not None:
                    self._dirty = False
                    self._preview()
                self.msleep(20)
                continue

            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                if self._loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._reset()
                    continue
                self._paused = True
                continue

            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            self._last_frame, self._last_pos = frame, pos
            result = self._process(frame, pos)

            # 처리 FPS: 작업시간(read+process, sleep 제외) EMA의 역수. 매 프레임 갱신 →
            # 리사이즈/스킵으로 작업시간이 줄면 즉시 FPS 상승이 보임(처리 능력 지표).
            work = time.time() - t0
            self.work_ema = work if self.work_ema == 0 else 0.9 * self.work_ema + 0.1 * work
            self.fps_value = 1.0 / self.work_ema if self.work_ema > 0 else 0.0
            result.fps = self.fps_value
            self.result_ready.emit(result)

            # 재생 속도에 맞춰 페이싱(처리가 더 느리면 즉시 다음 프레임).
            # '최대 속도' 모드면 페이싱을 건너뛰어 처리되는 대로 표시 → 리사이즈·스킵
            # 효과가 화면 속도로 바로 보인다(배속은 무시됨).
            if not self._unlock:
                target = 1.0 / (self.src_fps * self._speed)
                remain = target - work
                if remain > 0:
                    self.msleep(int(remain * 1000))

        cap.release()

    def _preview(self):
        """일시정지 중 옵션 변경 시 정지된 1프레임만 재처리해 표시.

        미리보기는 영구 상태(트래커·체류·경보 카운트)를 바꾸면 안 되므로
        스냅샷 후 복원한다. frame_idx=0 으로 강제 추론해 리사이즈/밝기까지 정확히 반영.
        """
        saved = (
            copy.deepcopy(self.tracker), dict(self.roi_dwell), self.frame_idx,
            self.alarm_prev, self.alarm_count, list(self.last_raw),
        )
        self.frame_idx = 0
        result = self._process(self._last_frame.copy(), self._last_pos)
        (self.tracker, self.roi_dwell, self.frame_idx, self.alarm_prev,
         self.alarm_count, self.last_raw) = saved
        self.result_ready.emit(result)

    # ---- 파이프라인 1프레임 ----
    def _process(self, frame, pos) -> FrameResult:
        v = self._values

        if v.brightness != 0 or abs(v.contrast - 1.0) > 1e-3:
            frame = filters.adjust_brightness_contrast(frame, v.brightness, v.contrast)

        # 프레임 스킵은 '추론'(가장 비싼 부분)만 건너뛴다. 신뢰도/크기 필터·추적·
        # 경보는 매 프레임 캐시된 raw 검출에 적용 → 스킵 중에도 슬라이더 즉시 반영,
        # 체류시간도 프레임당 정확히 누적.
        run_infer = (self.frame_idx % (v.frame_skip + 1) == 0)
        self.frame_idx += 1

        if run_infer:
            # 리사이즈 레버 = letterbox 목표 크기(imgsz, 32배수). 작을수록 빠르고 덜 정확.
            imgsz = max(int(round(640 * v.resize / 32)) * 32, 160)
            self.last_raw = self.detector.infer(frame, conf_thr=BASE_CONF, imgsz=imgsz)

        # 매 프레임: 후처리(필터 → 추적 → 확정 → 체류)
        dets = filters.by_confidence(self.last_raw, v.conf)
        if v.min_height > 0:
            # 최소 크기 = 프레임 높이 대비 %. 박스 높이가 이보다 작으면 버림(멀리·자잘).
            dets = filters.by_min_height(dets, frame.shape[0] * v.min_height / 100.0)
        self.tracker.update(dets, min_hits=v.debounce)
        # 디바운싱(탐지 레벨, 연속 프레임): N프레임 '연속' 검출돼야 '확정'(confirmed).
        # 확정 전 한 번이라도 놓치면 streak 리셋 → 처음부터(깜빡 오탐 제거). 확정은
        # sticky 라 이후 잠깐 놓쳐도 N 재대기 없이 복귀. 표시는 확정 & 이번 프레임 매칭만.
        self.last_tracks = {tid: t.det for tid, t in self.tracker.tracks.items()
                            if t.misses == 0 and t.confirmed}
        tracks = self.last_tracks
        roi = self._roi
        has_roi = len(roi) >= 3
        thr = v.roi_overlap / 100.0

        roi_ids = [tid for tid, d in tracks.items()
                   if roi_mod.intrudes(d, roi, thr)] if has_roi else []
        self._update_roi_dwell(set(roi_ids))
        roi_count = len(roi_ids)
        max_dwell = max((self.roi_dwell.get(t, 0) / self.src_fps for t in roi_ids), default=0.0)
        # 경보: 확정 트랙 기반 복합 규칙. OR(하나라도)/AND(둘 다)는 사용자가 선택.
        cond_people = roi_count >= v.alarm_people
        cond_dwell = max_dwell >= v.alarm_dwell
        alarm = (cond_people and cond_dwell) if v.alarm_and else (cond_people or cond_dwell)
        if alarm and not self.alarm_prev:
            self.alarm_count += 1
        self.alarm_prev = alarm

        disp = self._draw(frame.copy(), tracks, roi, roi_ids, v, alarm)

        return FrameResult(
            frame=disp, pos=pos, fps=self.fps_value, count=len(tracks),
            roi_count=roi_count, max_dwell=max_dwell, alarm=alarm,
            alarm_count=self.alarm_count,
        )

    def _update_roi_dwell(self, in_roi: set):
        # 증가는 'ROI 안 + 이번 프레임 보이는' 트랙만, 정리는 '아예 사라진' 트랙만
        # → 잠깐 놓친(misses>0) 트랙은 체류시간 유지(깜빡임 견딤).
        alive = set(self.tracker.tracks.keys())
        for tid in self.last_tracks:
            if tid in in_roi:
                self.roi_dwell[tid] = self.roi_dwell.get(tid, 0) + 1
            else:
                self.roi_dwell.pop(tid, None)
        for tid in list(self.roi_dwell):
            if tid not in alive:
                self.roi_dwell.pop(tid, None)

    def _draw(self, img, tracks, roi, roi_ids, v, alarm):
        h, w = img.shape[:2]
        if v.show_roi and len(roi) >= 2:
            pts = np.array(roi, dtype=np.int32).reshape((-1, 1, 2))
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], (40, 40, 200))
            cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
            cv2.polylines(img, [pts], len(roi) >= 3, (60, 60, 235), 2)
        for tid, d in tracks.items():
            in_roi = tid in roi_ids
            color = (60, 60, 235) if in_roi else (90, 200, 110)
            if v.show_boxes:
                cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), color, 2)
            labels = []
            if v.show_ids:
                labels.append(f"#{tid}")
            if v.show_conf:
                labels.append(f"{d.score:.2f}")
            if labels:
                cv2.putText(img, " ".join(labels), (int(d.x1), max(int(d.y1) - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        if alarm:
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), (40, 40, 230), 12)
            cv2.putText(img, "ALARM", (24, 56), cv2.FONT_HERSHEY_SIMPLEX,
                        1.6, (40, 40, 230), 4, cv2.LINE_AA)
        return img
