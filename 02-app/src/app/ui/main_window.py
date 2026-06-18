"""메인 윈도우 — 좌: 영상 뷰 / 우: 노브 패널 / 하단: FPS·경보 상태.

추론은 InferenceWorker(별도 스레드)가 수행하고, 여기서는 완성 프레임 표시와
재생 제어·상태 갱신만 한다(이벤트 루프가 추론에 막히지 않게). 노브값/ROI/재생
제어는 GUI→워커로 전달한다. 파이프라인 흐름(강의 지도):
  입력 → (밝기/대비·리사이즈·프레임스킵) → 추론 → (신뢰도·크기 필터) →
  추적 → (ROI 침입·복합규칙·디바운싱) → 시각화/경보
"""

import sys
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QButtonGroup, QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QScrollArea, QSlider, QStyle, QVBoxLayout, QWidget,
)

from app.pipeline.infer import Detector
from app.ui.worker import InferenceWorker

SPEEDS = [("0.5x", 0.5), ("1x", 1.0), ("2x", 2.0)]

# Material Design "repeat" 아이콘(미디어 플레이어 표준 — 두 화살표 수평 루프).
_REPEAT_PATH = "M7 7h10v3l4-4-4-4v3H5v6h2V7zm10 10H7v-3l-4 4 4 4v-3h12v-6h-2v4z"


def _loop_icon(color: str, size: int = 22) -> QIcon:
    """표준 repeat 아이콘을 SVG 로 렌더. ON/OFF 를 색으로 구분."""
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
           f'<path fill="{color}" d="{_REPEAT_PATH}"/></svg>')
    renderer = QSvgRenderer(bytearray(svg, "utf-8"))
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    renderer.render(p)
    p.end()
    return QIcon(pm)


def _model_path() -> str:
    # PyInstaller onefile: datas 가 _MEIPASS/models 에 풀린다. dev: 02-app/models.
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS)
    else:
        base = Path(__file__).resolve().parents[3]
    return str(base / "models" / "model.onnx")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KOSMES 비전 교보재 — 영상 추론 + 전후처리")
        self.resize(1280, 760)

        self.detector = Detector(_model_path())
        self.worker: InferenceWorker | None = None
        self.last_display = None
        self._last_alarm_count = 0
        self._was_playing = False           # 스크럽 시작 전 재생 상태

        self._build_ui()

    # ---------- UI 구성 ----------
    def _build_ui(self):
        from app.ui.video_view import VideoView
        from app.widgets.knob_panel import KnobPanel

        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)

        left = QVBoxLayout()
        self.view = VideoView()
        self.view.setText("영상을 열어주세요  (파일 열기)")
        left.addWidget(self.view, 1)
        left.addLayout(self._build_playbar())
        left.addLayout(self._build_statusbar())
        outer.addLayout(left, 3)

        self.panel = KnobPanel()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.panel)
        scroll.setFixedWidth(360)
        outer.addWidget(scroll)

        # 배선: 노브/ROI 변경 → 워커로 스냅샷 전달
        self.panel.changed.connect(self._push_values)
        self.view.roi_changed.connect(self._push_roi)
        self.view.roi_changed.connect(self._refresh_roi_status)
        self.panel.roi_draw_toggled.connect(self.view.set_draw_mode)
        self.panel.roi_draw_toggled.connect(self._refresh_roi_status)
        self.view.roi_completed.connect(lambda: self.panel.btn_draw.setChecked(False))
        self.view.roi_completed.connect(self._refresh_roi_status)
        self.panel.roi_clear_requested.connect(self.view.clear_roi)
        self.panel.min_size_changed.connect(self.view.set_min_ghost)
        self.panel.snapshot_requested.connect(self._save_snapshot)
        self._refresh_roi_status()

    def _refresh_roi_status(self):
        self.panel.set_roi_status(len(self.view.roi_points), self.view.draw_mode)

    def _icon(self, sp):
        return self.style().standardIcon(sp)

    def _icon_button(self, sp, tooltip, slot, checkable=False):
        btn = QPushButton()
        if sp is not None:
            btn.setIcon(self._icon(sp))
        btn.setIconSize(QSize(20, 20))
        btn.setToolTip(tooltip)
        btn.setFixedSize(40, 34)
        btn.setCheckable(checkable)
        (btn.toggled if checkable else btn.clicked).connect(slot)
        return btn

    def _build_playbar(self):
        SP = QStyle.StandardPixmap
        col = QVBoxLayout()
        col.setSpacing(6)

        # 행1: 진행바(전폭)
        self.seek = QSlider(Qt.Horizontal); self.seek.setRange(0, 0)
        self.seek.setToolTip("재생 위치 (드래그하면 실시간 미리보기)")
        self.seek.sliderPressed.connect(self._on_scrub_start)
        self.seek.sliderMoved.connect(self._on_scrub_move)
        self.seek.sliderReleased.connect(self._on_seek)
        row1 = QHBoxLayout()
        row1.setContentsMargins(8, 0, 8, 0)
        row1.addWidget(self.seek)
        col.addLayout(row1)

        # 구분선
        line = QFrame(); line.setFrameShape(QFrame.HLine); line.setFrameShadow(QFrame.Plain)
        line.setStyleSheet("color:#444;")
        col.addWidget(line)

        # 행2: 트랜스포트(좌) + 속도 토글(우)
        row2 = QHBoxLayout()
        self.btn_open = self._icon_button(SP.SP_DirOpenIcon, "파일 열기", self._open_file)
        self.btn_play = self._icon_button(SP.SP_MediaPlay, "재생 / 일시정지", self._toggle_play)
        self.btn_restart = self._icon_button(SP.SP_MediaSkipBackward, "처음으로", self._restart)
        self.btn_loop = self._icon_button(None, "반복 재생", self._on_loop_toggled, checkable=True)
        self.btn_loop.setChecked(True)
        self._update_loop_icon(True)
        for w in (self.btn_open, self.btn_play, self.btn_restart, self.btn_loop):
            row2.addWidget(w)

        row2.addStretch(1)
        row2.addWidget(QLabel("속도"))
        self.speed_group = QButtonGroup(self)
        self.speed_group.setExclusive(True)
        self._speed_val = 1.0
        for label, val in SPEEDS:
            b = QPushButton(label); b.setCheckable(True); b.setFixedSize(52, 34)
            b.clicked.connect(lambda _=False, v=val: self._set_speed(v))
            self.speed_group.addButton(b)
            row2.addWidget(b)
            if val == 1.0:
                b.setChecked(True)

        self.btn_unlock = QPushButton("⚡ 최대")
        self.btn_unlock.setCheckable(True)
        self.btn_unlock.setFixedSize(64, 34)
        self.btn_unlock.setToolTip(
            "페이싱 해제 — 원본 fps 에 맞추지 않고 처리되는 대로 표시. 리사이즈·스킵 "
            "효과가 화면 속도로 바로 보인다 (이 모드에선 배속 무시).")
        self.btn_unlock.toggled.connect(self._on_unlock)
        row2.addSpacing(8)
        row2.addWidget(self.btn_unlock)
        col.addLayout(row2)
        return col

    def _on_unlock(self, on: bool):
        if self.worker:
            self.worker.set_unlock(on)
        for b in self.speed_group.buttons():      # 언락 중엔 배속 무의미 → 비활성
            b.setEnabled(not on)

    def _on_loop_toggled(self, on: bool):
        self._update_loop_icon(on)
        if self.worker:
            self.worker.set_loop(on)

    def _update_loop_icon(self, on: bool):
        self.btn_loop.setIcon(_loop_icon("#6aa6ff" if on else "#888888"))

    def _set_speed(self, v: float):
        self._speed_val = v
        if self.worker:
            self.worker.set_speed(v)

    def _set_play_icon(self, playing: bool):
        sp = QStyle.StandardPixmap.SP_MediaPause if playing else QStyle.StandardPixmap.SP_MediaPlay
        self.btn_play.setIcon(self._icon(sp))

    def _build_statusbar(self):
        bar = QHBoxLayout()
        self.lbl_fps = QLabel("FPS —")
        self.lbl_count = QLabel("인원 —")
        self.lbl_roi = QLabel("ROI 인원 —")
        self.lbl_alarm = QLabel("경보 대기"); self.lbl_alarm.setObjectName("alarmOff")
        for w in (self.lbl_fps, self.lbl_count, self.lbl_roi):
            w.setObjectName("statusValue"); bar.addWidget(w)
        bar.addStretch(1); bar.addWidget(self.lbl_alarm)
        return bar

    # ---------- 워커 제어 ----------
    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "영상 파일 선택", "", "영상 (*.mp4 *.avi *.mov *.mkv);;모든 파일 (*)")
        if not path:
            return
        self._stop_worker()
        self.worker = InferenceWorker(path, self.detector)
        self.worker.result_ready.connect(self._on_result)
        self.worker.opened.connect(lambda total: self.seek.setRange(0, max(total - 1, 0)))
        # 초기 상태 주입
        self.worker.set_values(self.panel.values())
        self.worker.set_roi(self.view.roi_points)
        self.worker.set_loop(self.btn_loop.isChecked())
        self.worker.set_speed(self._speed_val)
        self.worker.set_unlock(self.btn_unlock.isChecked())
        self.worker.set_paused(False)
        self._last_alarm_count = 0
        self._set_play_icon(True)
        self.worker.start()

    def _stop_worker(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None

    def _push_values(self):
        if self.worker:
            self.worker.set_values(self.panel.values())

    def _push_roi(self):
        if self.worker:
            self.worker.set_roi(self.view.roi_points)

    def _toggle_play(self):
        if not self.worker:
            return
        paused = not self.worker._paused
        self.worker.set_paused(paused)
        self._set_play_icon(not paused)

    def _restart(self):
        if self.worker:
            self.worker.seek(0)

    def _on_scrub_start(self):
        # 드래그 시작: 재생 중이었으면 잠시 멈춰 위치 미리보기를 보여준다
        if not self.worker:
            return
        self._was_playing = not self.worker._paused
        if self._was_playing:
            self.worker.set_paused(True)
            self._set_play_icon(False)

    def _on_scrub_move(self, value):
        # 드래그 중: 그 위치 프레임을 실시간 미리보기(_seek_to 는 최신값으로 코얼레스)
        if self.worker:
            self.worker.seek(value)

    def _on_seek(self):
        # 드래그 놓음: 최종 위치로 이동, 드래그 전 재생 중이었으면 재개
        if not self.worker:
            return
        self.worker.seek(self.seek.value())
        if self._was_playing:
            self.worker.set_paused(False)
            self._set_play_icon(True)
            self._was_playing = False

    # ---------- 결과 표시 ----------
    def _on_result(self, r):
        self.last_display = r.frame
        self.view.show_frame(r.frame)

        self.lbl_fps.setText(f"처리 FPS {r.fps:4.1f}")
        self.lbl_count.setText(f"인원 {r.count}")
        self.lbl_roi.setText(f"ROI 인원 {r.roi_count} · 최대체류 {r.max_dwell:4.1f}s")
        if r.alarm:
            self.lbl_alarm.setObjectName("alarmOn")
            self.lbl_alarm.setText(f"⚠ 경보!  (누적 {r.alarm_count})")
        else:
            self.lbl_alarm.setObjectName("alarmOff")
            self.lbl_alarm.setText(f"정상  (누적 경보 {r.alarm_count})")
        self.lbl_alarm.style().unpolish(self.lbl_alarm)
        self.lbl_alarm.style().polish(self.lbl_alarm)

        # 경보 시 자동 일시정지(상승 에지 = 누적 경보 증가)
        if r.alarm_count > self._last_alarm_count:
            self._last_alarm_count = r.alarm_count
            if self.panel.values().auto_pause and not self.worker._paused:
                self._toggle_play()

        if not self.seek.isSliderDown():
            self.seek.blockSignals(True)
            self.seek.setValue(r.pos)
            self.seek.blockSignals(False)

    def _save_snapshot(self):
        if self.last_display is None:
            return
        path, sel = QFileDialog.getSaveFileName(
            self, "스냅샷 저장", "snapshot.png",
            "PNG 이미지 (*.png);;JPEG 이미지 (*.jpg)")
        if not path:
            return
        # 확장자 없이 저장하면 cv2 가 실패 → 선택 필터(없으면 .png)로 자동 보정
        if not Path(path).suffix:
            path += ".jpg" if "jpg" in sel.lower() else ".png"
        cv2.imwrite(path, self.last_display)

    def closeEvent(self, ev):
        self._stop_worker()
        super().closeEvent(ev)
