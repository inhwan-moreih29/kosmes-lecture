"""영상 표시 위젯 + ROI 폴리곤 드로잉.

BGR 프레임을 받아 종횡비 유지로 표시하고, ROI 그리기 모드에서 마우스 클릭을
받아 **원본 프레임 좌표계**의 폴리곤 점을 모은다(표시 크기와 무관하게 유효).
오버레이(박스/ROI 등)는 MainWindow 가 cv2 로 프레임에 직접 그린 뒤 넘긴다 —
여기서는 좌표 변환과 마우스 입력만 책임진다.
"""

import cv2
import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel

_ROI_RGB = (235, 70, 70)        # ROI 선/점 색 (Qt 는 RGB)
_FIRST_RGB = (70, 160, 255)     # 첫 꼭짓점 강조색
_CLOSE_DIST = 14                # 첫 점 근처 클릭 = 폴리곤 닫기 (위젯 px)


class VideoView(QLabel):
    roi_changed = Signal()
    roi_completed = Signal()    # 폴리곤 완성(그리기 종료) → 버튼 해제용

    def __init__(self):
        super().__init__()
        self.setObjectName("videoView")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setMouseTracking(True)
        self.roi_points: list[tuple[float, float]] = []
        self.draw_mode = False
        self._cursor_pt: tuple[float, float] | None = None   # 위젯 좌표(고무줄선용)
        self._min_ghost_pct = 0                              # 최소크기 기준 미리보기(%)
        self._ghost_timer = QTimer(self)                     # 조절 후 잠시 뒤 자동 숨김
        self._ghost_timer.setSingleShot(True)
        self._ghost_timer.timeout.connect(self._hide_min_ghost)
        self._frame_w = 0
        self._frame_h = 0
        # 표시된 픽스맵 기하 (마우스->프레임 좌표 변환용)
        self._disp_scale = 1.0
        self._disp_off_x = 0
        self._disp_off_y = 0

    def set_draw_mode(self, on: bool):
        self.draw_mode = on
        self._cursor_pt = None
        if on:
            # 새로 그리기: 기존 폴리곤을 비워 처음부터(닫히기 전까진 열린 폴리라인).
            self.roi_points = []
            self.roi_changed.emit()
        self.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)
        self.update()

    def clear_roi(self):
        self.roi_points = []
        self.roi_changed.emit()

    def set_min_ghost(self, pct: int):
        """최소크기 슬라이더 조절 시 호출 → 기준 박스를 잠깐 표시 후 자동 숨김."""
        self._min_ghost_pct = pct
        if pct > 0:
            self._ghost_timer.start(1500)
        else:
            self._ghost_timer.stop()
        self.update()

    def _hide_min_ghost(self):
        self._min_ghost_pct = 0
        self.update()

    def show_frame(self, frame_bgr: np.ndarray):
        """BGR 프레임을 위젯에 종횡비 유지로 표시."""
        self._frame_h, self._frame_w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        scaled = pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # 표시 기하 저장
        self._disp_scale = scaled.width() / w if w else 1.0
        self._disp_off_x = (self.width() - scaled.width()) / 2
        self._disp_off_y = (self.height() - scaled.height()) / 2
        self.setPixmap(scaled)

    def _to_frame_coords(self, x: float, y: float):
        if self._disp_scale <= 0:
            return None
        fx = (x - self._disp_off_x) / self._disp_scale
        fy = (y - self._disp_off_y) / self._disp_scale
        if 0 <= fx <= self._frame_w and 0 <= fy <= self._frame_h:
            return (float(fx), float(fy))
        return None

    def _to_widget_coords(self, fx: float, fy: float):
        return (fx * self._disp_scale + self._disp_off_x,
                fy * self._disp_scale + self._disp_off_y)

    def _near_first(self, x: float, y: float) -> bool:
        if len(self.roi_points) < 3:
            return False
        wx, wy = self._to_widget_coords(*self.roi_points[0])
        return (wx - x) ** 2 + (wy - y) ** 2 <= _CLOSE_DIST ** 2

    def _finish_roi(self):
        """폴리곤 완성: 그리기 모드 종료(점은 유지) + 신호로 버튼 해제."""
        self.draw_mode = False
        self._cursor_pt = None
        self.setCursor(Qt.ArrowCursor)
        self.update()
        self.roi_completed.emit()

    def mousePressEvent(self, ev):
        if not self.draw_mode:
            return
        if ev.button() == Qt.LeftButton:
            # 첫 점 근처(3점 이상)면 새 점 대신 폴리곤 닫기
            if self._near_first(ev.position().x(), ev.position().y()):
                self._finish_roi()
                return
            pt = self._to_frame_coords(ev.position().x(), ev.position().y())
            if pt is not None:
                self.roi_points.append(pt)
                self.roi_changed.emit()
        elif ev.button() == Qt.RightButton and self.roi_points:
            self.roi_points.pop()
            self.roi_changed.emit()
        self.update()

    def mouseDoubleClickEvent(self, ev):
        # 더블클릭 = 어디서든 완성. 첫 클릭의 press 가 이미 마지막 점을 찍은 상태.
        if self.draw_mode and ev.button() == Qt.LeftButton and len(self.roi_points) >= 3:
            self._finish_roi()

    def mouseMoveEvent(self, ev):
        if self.draw_mode:
            self._cursor_pt = (ev.position().x(), ev.position().y())
            self.update()

    def leaveEvent(self, ev):
        self._cursor_pt = None
        self.update()

    def _draw_min_ghost(self):
        if self._min_ghost_pct <= 0 or self._frame_h <= 0 or self._disp_scale <= 0:
            return
        scaled_h = self._frame_h * self._disp_scale
        mh = scaled_h * self._min_ghost_pct / 100.0
        mw = max(mh * 0.4, 6)
        x0 = self._disp_off_x + 14
        y_bot = self._disp_off_y + scaled_h - 14
        col = QColor(255, 200, 60)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(QPen(col, 2))
        p.setBrush(Qt.NoBrush)
        p.drawRect(QRectF(x0, y_bot - mh, mw, mh))
        p.drawText(QPointF(x0, max(y_bot - mh - 6, 14)), f"최소 {self._min_ghost_pct}%")
        p.end()

    def paintEvent(self, ev):
        super().paintEvent(ev)   # 픽스맵(영상 + cv2 오버레이) 먼저
        self._draw_min_ghost()
        if not self.draw_mode or not self.roi_points:
            return
        pts = [self._to_widget_coords(x, y) for (x, y) in self.roi_points]
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        edge = QColor(*_ROI_RGB)
        near = self._cursor_pt is not None and self._near_first(*self._cursor_pt)

        # 확정된 꼭짓점 연결선
        p.setPen(QPen(edge, 2))
        for a, b in zip(pts, pts[1:]):
            p.drawLine(QPointF(*a), QPointF(*b))

        # 커서까지 고무줄선 + 닫힘 미리보기. 첫 점 근처면 그 점으로 스냅(실선).
        if self._cursor_pt is not None:
            end = pts[0] if near else self._cursor_pt
            p.setPen(QPen(edge, 2))
            p.drawLine(QPointF(*pts[-1]), QPointF(*end))
            if len(pts) >= 2 and not near:
                p.setPen(QPen(QColor(235, 70, 70, 140), 1, Qt.DashLine))
                p.drawLine(QPointF(*self._cursor_pt), QPointF(*pts[0]))

        # 꼭짓점(첫 점은 크게·강조 → 닫는 지점). 근처에 오면 링으로 "여기서 닫힘".
        for i, (wx, wy) in enumerate(pts):
            p.setPen(QPen(QColor(255, 255, 255), 1))
            p.setBrush(QBrush(QColor(*_FIRST_RGB) if i == 0 else edge))
            p.drawEllipse(QPointF(wx, wy), 6 if i == 0 else 4, 6 if i == 0 else 4)
        if near:
            wx, wy = pts[0]
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(*_FIRST_RGB), 2))
            p.drawEllipse(QPointF(wx, wy), 11, 11)
        p.end()
