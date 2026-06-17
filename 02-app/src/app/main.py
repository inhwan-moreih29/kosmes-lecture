"""앱 진입점.

레이아웃 의도(docs/lecture-notes.md 71~78줄):
  - 좌: 영상 뷰 (박스/마스크/ID/confidence 토글)
  - 우: 노브 패널 (widgets/) — 임계값·디바운싱·최소크기·프레임스킵·리사이즈·ROI·추적
  - 하단: FPS / 경보 상태 표시

데이터: ../data/mendeley_samples 의 작업자 영상 클립.

TODO(스캐폴드): QApplication + MainWindow 구성, pipeline 연결, 타이머 기반 프레임 루프
"""

import sys


def main() -> int:
    # from PySide6.QtWidgets import QApplication
    # from app.ui.main_window import MainWindow
    # app = QApplication(sys.argv)
    # win = MainWindow(); win.show()
    # return app.exec()
    print("[TODO] PySide6 앱 부팅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
