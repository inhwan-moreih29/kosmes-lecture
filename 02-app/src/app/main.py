"""앱 진입점.

레이아웃(docs/lecture-notes.md 71~78줄):
  - 좌: 영상 뷰 (박스/ID/confidence/ROI 토글) + 재생 컨트롤 + FPS/경보 상태
  - 우: 노브 패널 (widgets/) — 임계값·디바운싱·최소크기·프레임스킵·리사이즈·밝기/대비·ROI·경보규칙

데이터: 사용자가 OS 파일 다이얼로그로 임의 영상 선택 (예: ../data/mendeley_samples).
"""

import sys
from pathlib import Path

import qdarktheme
from PySide6.QtWidgets import QApplication


def main() -> int:
    app = QApplication(sys.argv)

    # qdarktheme 다크 베이스 + 앱 고유 오버레이(theme.qss)
    qss_path = Path(__file__).resolve().parent / "ui" / "theme.qss"
    overlay = qss_path.read_text(encoding="utf-8") if qss_path.exists() else ""
    qdarktheme.setup_theme("dark", additional_qss=overlay)

    from app.ui.main_window import MainWindow
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
