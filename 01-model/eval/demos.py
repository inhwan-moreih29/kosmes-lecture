"""1교시 데모용 시각자료 생성 (학습 완료 후 실행).

생성물(eval/out/ 아래):
  - threshold/   : 같은 모델·같은 이미지에 conf 임계값 스윕 -> 박스 나타남/사라짐
  - datasize/    : n10 vs n100 모델 같은 val 이미지 비교 추론
  - speed.json   : small/base/large 추론 속도(ms) + 파라미터/FLOPs (속도 vs 정확도)
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WEIGHTS = ROOT / "weights"
VAL = ROOT / "datasets" / "screw_detect" / "images" / "val"
OUT = HERE / "out"


def draw(img, result, color=(0, 200, 0), title=""):
    out = img.copy()
    for b in (result.boxes or []):
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        c = float(b.conf[0])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{c:.2f}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if title:
        cv2.putText(out, title, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return out


def pick_high_spread(model, candidates, thresholds, k=5):
    """임계값에 따라 탐지수 변화가 큰 이미지 k장 선택 (임계값 데모 효과 극대화)."""
    scored = []
    for p in candidates:
        im = cv2.imread(str(p))
        counts = [len(model.predict(im, conf=t, verbose=False)[0].boxes) for t in thresholds]
        scored.append((counts[0] - counts[-1], p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:k]]


def threshold_demo(model_path, images, thresholds=(0.05, 0.25, 0.5, 0.75)):
    from ultralytics import YOLO
    m = YOLO(model_path)
    d = OUT / "threshold"; d.mkdir(parents=True, exist_ok=True)
    images = pick_high_spread(m, images, thresholds, k=5)
    for img_path in images:
        img = cv2.imread(str(img_path))
        panels = []
        for th in thresholds:
            r = m.predict(img, conf=th, verbose=False)[0]
            panels.append(draw(img, r, (0, 0, 255), f"conf>={th}  ({len(r.boxes)} det)"))
        cv2.imwrite(str(d / f"{img_path.stem}.png"), np.hstack(panels))
    print(f"threshold demo -> {d}")


def datasize_demo(images, conf=0.25):
    from ultralytics import YOLO
    steps = [("n10", "10장", (0, 0, 255)),
             ("n50", "50장", (0, 140, 255)),
             ("n100", "100장", (0, 200, 0))]
    models = [(YOLO(WEIGHTS / f"{tag}.pt"), label, color) for tag, label, color in steps]
    d = OUT / "datasize"; d.mkdir(parents=True, exist_ok=True)
    for img_path in images:
        img = cv2.imread(str(img_path))
        panels = [draw(img, m.predict(img, conf=conf, verbose=False)[0], color, label)
                  for m, label, color in models]
        cv2.imwrite(str(d / f"{img_path.stem}.png"), np.hstack(panels))
    print(f"datasize demo -> {d}")


def speed_bench(conf=0.25):
    from ultralytics import YOLO
    import time
    val_imgs = sorted(VAL.glob("*.png"))[:30]
    imgs = [cv2.imread(str(p)) for p in val_imgs]
    rows = []
    # 모델 크기 n/s/m/l — 's' 지점은 noaug(yolo11s) 재사용
    size_map = [("size_n", "yolo11n"), ("noaug", "yolo11s"),
                ("size_m", "yolo11m"), ("size_l", "yolo11l")]
    for tag, disp in size_map:
        p = WEIGHTS / f"{tag}.pt"
        if not p.exists():
            continue
        m = YOLO(p)
        m.predict(imgs[0], conf=conf, verbose=False)  # warmup
        t = time.time()
        for im in imgs:
            m.predict(im, conf=conf, verbose=False)
        ms = (time.time() - t) / len(imgs) * 1000
        params_m = sum(p.numel() for p in m.model.parameters()) / 1e6
        rows.append({
            "name": disp,
            "tag": tag,
            "ms_per_img": round(ms, 2),
            "fps": round(1000 / ms, 1),
            "params_M": round(params_m, 2),
        })
    (OUT / "speed.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print("speed:", rows)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    all_defects = sorted(p for p in VAL.glob("def_*.png") if "good" not in p.name)
    threshold_demo(WEIGHTS / "aug.pt", all_defects)   # 전체에서 변화폭 큰 것 자동 선택
    datasize_demo(all_defects[: args.n])
    speed_bench()
