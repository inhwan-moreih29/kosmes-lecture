"""1교시 비교 모델 학습.

configs/comparisons.yaml 의 각 비교군을 학습해 weights/<name>.pt 로 저장한다.
허블 업로드 또는 추론 비교 데모에 사용.

사용:
  uv run train.py                 # 전체 비교군 학습
  uv run train.py --only aug n100 # 특정 비교군만

TODO(스캐폴드): 실제 학습 루프 연결
  - max_train_images 가 있으면 데이터 서브샘플링
  - augment 플래그 -> Ultralytics 증강 하이퍼파라미터 on/off
  - results(runs/) 의 학습곡선은 보너스 슬라이드 소스로 보존
"""

import argparse
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "weights"


def load_groups(only: list[str] | None):
    cfg = yaml.safe_load((HERE / "configs" / "comparisons.yaml").read_text())
    base, groups = cfg["base"], cfg["groups"]
    if only:
        groups = [g for g in groups if g["name"] in only]
    return base, groups


def train_one(name: str, params: dict) -> None:
    # from ultralytics import YOLO
    # model = YOLO(params["model"])
    # model.train(... , project=str(HERE / "runs"), name=name)
    # shutil.copy(best_pt, WEIGHTS / f"{name}.pt")
    raise NotImplementedError(f"학습 구현 예정: {name} <- {params}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="학습할 비교군 이름 (생략 시 전체)")
    args = ap.parse_args()

    base, groups = load_groups(args.only)
    for g in groups:
        params = {**base, **g}
        print(f"[TODO] train {g['name']}: {params}")


if __name__ == "__main__":
    main()
