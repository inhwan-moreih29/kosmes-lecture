"""다회·다지점 실험 공용 라이브러리.

train_eval(): 한 번 학습 -> best mAP50 스칼라 반환 (가중치는 디스크 절약 위해 삭제).
AUG_PRESETS: 증강 기법별 하이퍼파라미터 셋 (Phase2 용).
subsample(): train 이미지 리스트를 n장으로 서브샘플 (Phase1/2 용).

설계 의도:
  - 학습 변수는 (model, data, n, seed, aug) 다섯으로 한정 -> 효과 분리.
  - seed 가 서브샘플 + 초기화를 동시에 흔들어 '지점당 분포'를 만든다(박스플롯).
  - val 은 데이터셋 고정 -> 같은 데이터셋 내 비교는 공정.
"""

import shutil
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # 01-model
RUNS = ROOT.parent / "outputs" / "experiments" / "runs"
RESULTS = ROOT.parent / "outputs" / "experiments" / "results"

# 모든 증강 0 (순수 baseline)
NOAUG = dict(
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.0,
    mosaic=0.0, mixup=0.0, copy_paste=0.0, erasing=0.0, auto_augment=None,
)


def _preset(**over):
    p = dict(NOAUG)
    p.update(over)
    return p


# 증강 기법별 셋 (한 기법씩 분리해 효과를 격리; all 은 통합)
AUG_PRESETS = {
    "none":   _preset(),
    "flip":   _preset(flipud=0.5, fliplr=0.5),
    "rotate": _preset(degrees=30.0, translate=0.1),
    "hsv":    _preset(hsv_h=0.015, hsv_s=0.7, hsv_v=0.5),
    "scale":  _preset(scale=0.5),
    "mosaic": _preset(mosaic=1.0),
    "all":    _preset(hsv_h=0.015, hsv_s=0.4, hsv_v=0.5, degrees=30.0,
                      translate=0.1, scale=0.4, flipud=0.5, fliplr=0.5),
}


def subsample(data_yaml: Path, n: int, seed: int, tag: str) -> Path:
    """train 이미지를 n장으로 서브샘플한 임시 data.yaml 생성 (seed 로 셔플)."""
    import random
    cfg = yaml.safe_load(data_yaml.read_text())
    base = Path(cfg["path"])
    train_dir = base / cfg["train"] if not str(cfg["train"]).endswith(".txt") else None
    imgs = sorted(p for p in train_dir.iterdir()
                  if p.suffix.lower() in (".png", ".jpg", ".jpeg")) if train_dir else []
    random.Random(seed).shuffle(imgs)
    picked = imgs[:n]
    list_path = base / f"train_{tag}.txt"
    list_path.write_text("\n".join(str(p.resolve()) for p in picked))
    new_cfg = dict(cfg)
    new_cfg["train"] = str(list_path.resolve())
    out = base / f"data_{tag}.yaml"
    out.write_text(yaml.safe_dump(new_cfg, allow_unicode=True, sort_keys=False))
    return out


def train_eval(model: str, data_yaml: Path, seed: int, aug: dict,
               n: int | None = None, epochs: int = 100, imgsz: int = 640,
               device: int = 0, tag: str = "exp", keep_weights: bool = False) -> float:
    """한 조건 학습 -> best mAP50 반환. 디스크 절약 위해 가중치 삭제(기본)."""
    from ultralytics import YOLO

    run_name = f"{tag}_s{seed}"
    if n:
        data_yaml = subsample(data_yaml, n, seed, run_name)

    m = YOLO(model)
    r = m.train(
        data=str(data_yaml), epochs=epochs, imgsz=imgsz,
        seed=seed, deterministic=True,
        project=str(RUNS), name=run_name, exist_ok=True,
        device=device, workers=2, verbose=False, plots=False,
        **aug,
    )
    try:
        map50 = float(r.box.map50)
    except Exception:
        map50 = float("nan")
    if not keep_weights:
        shutil.rmtree(Path(r.save_dir) / "weights", ignore_errors=True)
    return map50
