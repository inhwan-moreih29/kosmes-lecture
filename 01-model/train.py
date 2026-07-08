"""1교시 비교 모델 학습.

configs/comparisons.yaml 의 각 비교군을 학습해 weights/<name>.pt 로 저장한다.
허블 업로드 또는 추론 비교 데모에 사용.

설계 포인트:
  - seed 고정: 데이터 수 비교(n10 vs n90)의 변수 통제
  - augment on/off: 단순 플래그가 아니라 "통제된 하이퍼파라미터 셋"으로 정의
    (회전/밝기/뒤집기 등 -> OOD '일부러 틀리기'를 증강 모델이 고치는 인과를 재현 가능하게)
  - max_train_images: train 이미지 리스트(.txt) 서브샘플로 구현. 같은 풀에서 뽑아 분포 유지
  - results(runs/)의 학습곡선은 보너스 슬라이드 소스로 보존

사용:
  uv run train.py                 # 전체 비교군 학습
  uv run train.py --only aug noaug
"""

import argparse
import random
import shutil
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "weights"
RUNS = HERE.parent / "outputs" / "model" / "runs"
DATASETS = HERE / "datasets"

# 증강 없음: 모든 증강 하이퍼파라미터 0 (순수 baseline)
NOAUG = dict(
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.0,
    mosaic=0.0, mixup=0.0, copy_paste=0.0, erasing=0.0, auto_augment=None,
)
# 증강 있음: 회전/밝기/뒤집기/스케일 — OOD 데모(회전·밝기·반전)와 짝지은 통제 셋
AUG = dict(
    hsv_h=0.015, hsv_s=0.4, hsv_v=0.5,   # 밝기/채도 (조명 변화 대응)
    degrees=30.0, translate=0.1, scale=0.4, shear=0.0, perspective=0.0,
    flipud=0.5, fliplr=0.5,              # 상하/좌우 반전 (각도 변화 대응)
    mosaic=0.0, mixup=0.0, copy_paste=0.0, erasing=0.0, auto_augment=None,
)


def load_groups(only: list[str] | None):
    cfg = yaml.safe_load((HERE / "configs" / "comparisons.yaml").read_text())
    base, groups = cfg["base"], cfg["groups"]
    if only:
        groups = [g for g in groups if g["name"] in only]
    return base, groups


def resolve_data(category: str, task: str) -> Path:
    yml = DATASETS / f"{category}_{task}" / "data.yaml"
    if not yml.exists():
        raise FileNotFoundError(
            f"{yml} 없음. 먼저 변환: uv run prepare/mvtec_to_yolo.py "
            f"--category {category} --task {task}"
        )
    return yml


def subsample_data(data_yaml: Path, n: int, seed: int, name: str) -> Path:
    """train 이미지를 n장으로 서브샘플한 임시 data.yaml 생성."""
    cfg = yaml.safe_load(data_yaml.read_text())
    base = Path(cfg["path"])
    train_dir = base / cfg["train"]
    imgs = sorted(train_dir.glob("*.png"))
    random.Random(seed).shuffle(imgs)
    picked = imgs[:n]
    list_path = base / f"train_{name}.txt"
    list_path.write_text("\n".join(str(p.resolve()) for p in picked))
    new_cfg = dict(cfg)
    new_cfg["train"] = str(list_path.resolve())
    out = base / f"data_{name}.yaml"
    out.write_text(yaml.safe_dump(new_cfg, allow_unicode=True, sort_keys=False))
    print(f"  subsample: {len(imgs)} -> {len(picked)} train imgs ({name})")
    return out


def train_one(name: str, params: dict, task: str, seed: int, epochs_override: int | None, device: int = 0) -> dict:
    from ultralytics import YOLO

    category = params["data"]
    data_yaml = resolve_data(category, task)

    n = params.get("max_train_images")
    if n:
        data_yaml = subsample_data(data_yaml, n, seed, name)

    aug = AUG if params.get("augment") else NOAUG
    epochs = epochs_override or params["epochs"]

    model = YOLO(params["model"])
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=params["imgsz"],
        seed=seed,
        deterministic=True,
        project=str(RUNS),
        name=name,
        exist_ok=True,
        device=device,
        workers=2,    # 소규모 데이터셋(n10 등)에서 워커 과다 -> 데드락 방지
        verbose=False,
        plots=True,   # 학습곡선 png 보존 (보너스 슬라이드)
        **aug,
    )
    # best.pt -> weights/<name>.pt
    best = Path(results.save_dir) / "weights" / "best.pt"
    WEIGHTS.mkdir(exist_ok=True)
    dst = WEIGHTS / f"{name}.pt"
    if best.exists():
        shutil.copy(best, dst)

    # 검증 지표 회수
    metrics = {}
    try:
        box = results.box  # type: ignore[attr-defined]
        metrics = {
            "map50": float(box.map50),
            "map5095": float(box.map),
            "precision": float(box.mp),
            "recall": float(box.mr),
        }
    except Exception:
        pass
    print(f"[done] {name}: {metrics} -> {dst}")
    return {"name": name, "weights": str(dst), "save_dir": str(results.save_dir), **metrics}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="학습할 비교군 이름 (생략 시 전체)")
    ap.add_argument("--task", default="detect", choices=["detect", "segment"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None, help="config epochs 덮어쓰기")
    ap.add_argument("--device", type=int, default=0, help="GPU index")
    args = ap.parse_args()

    base, groups = load_groups(args.only)
    summary = []
    for g in groups:
        params = {**base, **g}
        print(f"\n=== train {g['name']}: model={params['model']} "
              f"aug={params.get('augment', False)} n={params.get('max_train_images', 'all')} ===")
        summary.append(train_one(g["name"], params, args.task, args.seed, args.epochs, args.device))

    print("\n=== SUMMARY ===")
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
