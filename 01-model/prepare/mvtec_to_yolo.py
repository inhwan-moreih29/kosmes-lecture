"""MVTec AD -> YOLO 포맷 변환.

MVTec 는 이상탐지(anomaly detection) 포맷이라 Ultralytics 검출/분할로 바로 못 쓴다.
구조:  data/<category>/{train/good, test/<defect>, ground_truth/<defect>/*_mask.png}

이 스크립트가 하는 일:
  - test 이미지 + ground_truth 마스크 -> YOLO 라벨(.txt) 생성
  - 마스크에서 외곽선 추출 -> bbox(검출) 또는 polygon(분할)
  - images/{train,val} + labels/{train,val} + data.yaml 출력

강의 메모(docs/lecture-notes.md): 추천 카테고리 screw / metal_nut / transistor.
1교시 데모는 "데이터셋 구축"이 아니라 "체험"이 목적이므로 소규모로 빠르게.

TODO(스캐폴드): 실제 변환 로직 구현
  - cv2.findContours 로 마스크 -> 폴리곤/바운딩박스
  - 정상(good) 이미지는 객체 없음(빈 라벨) 처리 방식 결정
  - train/val split 비율 인자화
"""

from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"  # kosmes-lecture/data


def convert(category: str, task: str = "detect") -> None:
    """category(예: 'screw')를 YOLO 포맷으로 변환. task: detect | segment."""
    src = DATA_ROOT / category
    raise NotImplementedError("마스크->YOLO 변환 구현 예정")


if __name__ == "__main__":
    # 예시: python prepare/mvtec_to_yolo.py
    for cat in ("screw", "metal_nut", "transistor"):
        print(f"[TODO] convert {cat}")
