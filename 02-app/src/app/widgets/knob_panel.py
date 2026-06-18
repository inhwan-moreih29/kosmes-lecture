"""노브 패널 — 전후처리 노브 + 시각화 토글 + 리셋 + ROI/스냅샷.

각 노브가 pipeline 모듈 하나에 대응(패키지 구조 = 강의 코너 지도):
  confidence / min_size / brightness / contrast -> filters
  debounce N프레임                              -> debounce
  frame_skip / resize                           -> infer 입력 단(속도 레버)
  ROI 그리기                                     -> roi
  alarm_people / alarm_dwell                     -> roi + tracking (복합 규칙)
  시각화 토글(박스/ID/conf/ROI)

값은 MainWindow 가 매 프레임 values() 로 끌어가고(pull), 슬라이더 옆에 실수치를
실시간 표시한다. 리셋은 DEFAULTS 로 위젯 값을 일괄 복원한다.
슬라이더는 휠로 값이 바뀌지 않게(드래그 전용) — 휠은 패널 스크롤에 쓰인다.
"""

from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSlider,
    QStyle, QVBoxLayout, QWidget,
)

# 기본값 = 리셋 대상
DEFAULTS = {
    "conf": 30, "debounce": 3, "min_height": 0, "frame_skip": 0,
    "resize": 100, "brightness": 0, "contrast": 100,
    "roi_overlap": 30, "alarm_people": 3, "alarm_dwell": 10,
}


class NoWheelSlider(QSlider):
    """휠 이벤트를 무시(상위 스크롤로 전달) → 값은 드래그로만 변경."""

    def wheelEvent(self, e):
        e.ignore()


@dataclass
class KnobValues:
    conf: float
    debounce: int
    min_height: float
    frame_skip: int
    resize: float
    brightness: int
    contrast: float
    roi_overlap: int
    alarm_people: int
    alarm_dwell: float
    alarm_and: bool
    show_boxes: bool
    show_ids: bool
    show_conf: bool
    show_roi: bool
    auto_pause: bool


class _Knob(QWidget):
    """라벨 + 슬라이더 + 실수치 표시 한 줄. raw 정수값을 fmt 로 표기."""

    changed = Signal()

    def __init__(self, title: str, lo: int, hi: int, init: int, fmt, tooltip: str):
        super().__init__()
        self._fmt = fmt
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        head = QHBoxLayout()
        name = QLabel(title)
        name.setToolTip(tooltip)
        self.value_lbl = QLabel()
        self.value_lbl.setObjectName("knobValue")
        head.addWidget(name)
        head.addStretch(1)
        head.addWidget(self.value_lbl)
        self.slider = NoWheelSlider(Qt.Horizontal)
        self.slider.setRange(lo, hi)
        self.slider.setValue(init)
        self.slider.setToolTip(tooltip)
        self.slider.valueChanged.connect(self._on_change)
        lay.addLayout(head)
        lay.addWidget(self.slider)
        self._refresh()

    def _on_change(self):
        self._refresh()
        self.changed.emit()

    def _refresh(self):
        self.value_lbl.setText(self._fmt(self.slider.value()))

    def raw(self) -> int:
        return self.slider.value()

    def set_raw(self, v: int):
        self.slider.setValue(v)


class KnobPanel(QWidget):
    changed = Signal()
    roi_draw_toggled = Signal(bool)
    roi_clear_requested = Signal()
    snapshot_requested = Signal()
    min_size_changed = Signal(int)   # 최소크기 조절 → 영상에 기준 박스 잠깐 표시

    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self)
        root.setSpacing(8)

        # --- 오탐 3종 (출력/시간/크기) ---
        g1 = QGroupBox("오탐 깎기 — 임계값·시간·크기")
        l1 = QVBoxLayout(g1)
        self.k_conf = _Knob("신뢰도 임계값", 1, 95, DEFAULTS["conf"],
            lambda v: f"{v/100:.2f}",
            "출력 레벨. 점수가 이 값보다 낮은 탐지를 버린다 (1교시 복습).")
        self.k_debounce = _Knob("디바운싱(연속 프레임)", 1, 30, DEFAULTS["debounce"],
            lambda v: f"{v} f",
            "시간 레벨. 탐지가 N프레임 연속돼야 '확정'으로 인정 → 박스·카운트·경보에 반영. "
            "깜빡이는 순간 오탐 제거, 대가는 반응 지연.")
        self.k_min = _Knob("최소 크기 필터", 0, 60, DEFAULTS["min_height"],
            lambda v: f"키 {v}%" if v else "off",
            "크기 레벨. 박스 높이가 화면 높이의 이 비율보다 작으면 버린다 → 멀리·자잘한 "
            "오탐 제거. 화면 좌하단 노란 기준 박스보다 작은 사람은 무시된다.")
        for k in (self.k_conf, self.k_debounce, self.k_min):
            k.changed.connect(self.changed)
            l1.addWidget(k)
        self.k_min.changed.connect(lambda: self.min_size_changed.emit(self.k_min.raw()))
        root.addWidget(g1)

        # --- 속도 레버 ---
        g2 = QGroupBox("속도 레버 — FPS 트레이드오프")
        l2 = QVBoxLayout(g2)
        self.k_skip = _Knob("프레임 스킵", 0, 5, DEFAULTS["frame_skip"],
            lambda v: "off" if v == 0 else f"1/{v+1}",
            "N프레임마다 1번만 추론 → FPS↑, 반응성↓ (1교시 '큰 모델은 느리다' 회수).")
        self.k_resize = _Knob("입력 리사이즈", 25, 100, DEFAULTS["resize"],
            lambda v: f"{max(round(640 * v / 100 / 32) * 32, 160)}px",
            "추론 입력 크기(letterbox). 작을수록 FPS↑·정확도↓ (동적 모델).")
        for k in (self.k_skip, self.k_resize):
            k.changed.connect(self.changed)
            l2.addWidget(k)
        root.addWidget(g2)

        # --- 보너스: 밝기/대비 ---
        g3 = QGroupBox("입력 보정 (보너스)")
        l3 = QVBoxLayout(g3)
        self.k_bright = _Knob("밝기", -100, 100, DEFAULTS["brightness"],
            lambda v: f"{v:+d}",
            "입력 레벨 전처리. 어두운/밝은 분포 밖 영상 회수 (1교시 실패편).")
        self.k_contrast = _Knob("대비", 50, 200, DEFAULTS["contrast"],
            lambda v: f"{v/100:.2f}x",
            "입력 레벨 전처리. 대비를 키워 흐린 영상의 검출을 돕는다.")
        for k in (self.k_bright, self.k_contrast):
            k.changed.connect(self.changed)
            l3.addWidget(k)
        root.addWidget(g3)

        # --- 위험구역 + 복합 규칙 ---
        # --- 위험구역(ROI): 그리기 + 진입 기준 ---
        g4 = QGroupBox("위험구역(ROI)")
        l4 = QVBoxLayout(g4)
        roi_btns = QHBoxLayout()
        self.btn_draw = QPushButton(" ROI 그리기")
        self.btn_draw.setCheckable(True)
        self.btn_draw.toggled.connect(self.roi_draw_toggled)
        self.btn_draw.toggled.connect(
            lambda on: self.btn_draw.setText(
                " 그리는 중… (첫 점=완성)" if on else " ROI 그리기"))
        self.btn_clear = QPushButton(" 지우기")
        trash = getattr(QStyle.StandardPixmap, "SP_TrashIcon",
                        QStyle.StandardPixmap.SP_DialogDiscardButton)
        self.btn_clear.setIcon(self.style().standardIcon(trash))
        self.btn_clear.setEnabled(False)
        self.btn_clear.clicked.connect(self.roi_clear_requested)
        roi_btns.addWidget(self.btn_draw, 1)
        roi_btns.addWidget(self.btn_clear)
        l4.addLayout(roi_btns)
        self.lbl_roi_status = QLabel("구역 없음")
        self.lbl_roi_status.setStyleSheet("color:#9aa;")
        l4.addWidget(self.lbl_roi_status)
        self.k_overlap = _Knob("진입 기준(겹침)", 0, 90, DEFAULTS["roi_overlap"],
            lambda v: "닿으면(any)" if v == 0 else f"{v}%",
            "사람 박스가 구역과 겹치는 면적 비율이 이 값 이상이어야 '진입'으로 판정. "
            "0이면 조금만 닿아도, 높일수록 더 깊이 들어와야 인정.")
        self.k_overlap.changed.connect(self.changed)
        l4.addWidget(self.k_overlap)
        root.addWidget(g4)

        # --- 경보 규칙: 인원/체류 두 규칙 + 결합(OR/AND) ---
        g5b = QGroupBox("경보 규칙")
        l4b = QVBoxLayout(g5b)
        self.k_people = _Knob("인원 기준(명)", 1, 10, DEFAULTS["alarm_people"],
            lambda v: f"≥ {v} 명",
            "ROI 안 인원이 이 값 이상이면 경보 (추적 카운팅 기반).")
        self.k_dwell = _Knob("체류시간 기준(초)", 1, 30, DEFAULTS["alarm_dwell"],
            lambda v: f"≥ {v} 초",
            "ROI 안에서 이 시간 이상 머문 사람이 있으면 경보 (추적 ID 기반).")
        for k in (self.k_people, self.k_dwell):
            k.changed.connect(self.changed)
            l4b.addWidget(k)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("두 규칙 결합"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["OR — 인원·체류 하나라도", "AND — 인원·체류 둘 다"])
        self.combo_mode.setToolTip("위 두 규칙(인원·체류)을 어떻게 결합할지. "
                                   "OR=둘 중 하나, AND=둘 다 만족해야 경보.")
        self.combo_mode.currentIndexChanged.connect(self.changed)
        mode_row.addWidget(self.combo_mode, 1)
        l4b.addLayout(mode_row)
        root.addWidget(g5b)

        # --- 시각화 토글 + 옵션 ---
        g5 = QGroupBox("표시 / 옵션")
        l5 = QVBoxLayout(g5)
        self.cb_boxes = QCheckBox("박스"); self.cb_boxes.setChecked(True)
        self.cb_ids = QCheckBox("ID"); self.cb_ids.setChecked(True)
        self.cb_conf = QCheckBox("신뢰도"); self.cb_conf.setChecked(True)
        self.cb_roi = QCheckBox("ROI"); self.cb_roi.setChecked(True)
        self.cb_pause = QCheckBox("경보 시 자동 일시정지")
        toggles = QHBoxLayout()
        for cb in (self.cb_boxes, self.cb_ids, self.cb_conf, self.cb_roi):
            cb.toggled.connect(self.changed)
            toggles.addWidget(cb)
        l5.addLayout(toggles)
        self.cb_pause.toggled.connect(self.changed)
        l5.addWidget(self.cb_pause)
        actions = QHBoxLayout()
        btn_snap = QPushButton(" 스냅샷 저장")
        btn_snap.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        btn_snap.clicked.connect(self.snapshot_requested)
        btn_reset = QPushButton("리셋")
        btn_reset.setToolTip("모든 노브를 기본값으로 되돌린다")
        btn_reset.clicked.connect(lambda: self.apply_preset(DEFAULTS))
        actions.addWidget(btn_snap, 1)
        actions.addWidget(btn_reset)
        l5.addLayout(actions)
        root.addWidget(g5)
        root.addStretch(1)

    # 모든 슬라이더 노브 한 곳에서 관리
    def _knobs(self):
        return {
            "conf": self.k_conf, "debounce": self.k_debounce, "min_height": self.k_min,
            "frame_skip": self.k_skip, "resize": self.k_resize,
            "brightness": self.k_bright, "contrast": self.k_contrast,
            "roi_overlap": self.k_overlap,
            "alarm_people": self.k_people, "alarm_dwell": self.k_dwell,
        }

    def set_roi_status(self, n_points: int, drawing: bool):
        """MainWindow 가 ROI 상태 변할 때 호출 → 상태 라벨 + 지우기 버튼 갱신."""
        if drawing:
            self.lbl_roi_status.setText(
                f"그리는 중… {n_points}점  (3점 이상 → 첫 점 클릭으로 완성)")
        elif n_points >= 3:
            self.lbl_roi_status.setText(f"구역 설정됨 · {n_points}각형")
        else:
            self.lbl_roi_status.setText("구역 없음")
        self.btn_clear.setEnabled(n_points > 0)

    def apply_preset(self, preset: dict):
        for key, knob in self._knobs().items():
            if key in preset:
                knob.set_raw(preset[key])
        self.changed.emit()

    def values(self) -> KnobValues:
        return KnobValues(
            conf=self.k_conf.raw() / 100,
            debounce=self.k_debounce.raw(),
            min_height=self.k_min.raw(),
            frame_skip=self.k_skip.raw(),
            resize=self.k_resize.raw() / 100,
            brightness=self.k_bright.raw(),
            contrast=self.k_contrast.raw() / 100,
            roi_overlap=self.k_overlap.raw(),
            alarm_people=self.k_people.raw(),
            alarm_dwell=self.k_dwell.raw(),
            alarm_and=self.combo_mode.currentIndex() == 1,
            show_boxes=self.cb_boxes.isChecked(),
            show_ids=self.cb_ids.isChecked(),
            show_conf=self.cb_conf.isChecked(),
            show_roi=self.cb_roi.isChecked(),
            auto_pause=self.cb_pause.isChecked(),
        )
