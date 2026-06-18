"""OOD(분포 밖) 변형 생성 + 모델 비교 추론.

목적(1교시 "일부러 틀리기 -> 고치기"):
  분포 밖 이미지를 의도적으로 만들어 증강 없는 모델이 틀리는지 보고,
  증강 모델이 그걸 고치는지 확인한다. 통제된 변형(회전/밝기/반전/블러/노이즈)을
  쓰므로 "어떤 변형에 약하고, 어떤 증강이 고치는지" 인과가 또렷하다.

산출: eval/out/ood/ 아래 비교 이미지 + ood_results.json (변형별 탐지수/평균 conf)
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "out" / "ood"


# ---------- OOD 변형들 (원본 분포에 없던 변화) ----------
def t_rotate(img, deg):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=(114, 114, 114))


def t_bright(img, factor):
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def t_blur(img, k):
    return cv2.GaussianBlur(img, (k, k), 0)


def t_flip(img):
    return cv2.flip(img, 1)


def t_noise(img, sigma):
    n = np.random.default_rng(0).normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


TRANSFORMS = {
    "original": lambda im: im,
    "rotate45": lambda im: t_rotate(im, 45),
    "rotate90": lambda im: t_rotate(im, 90),
    "dark": lambda im: t_bright(im, 0.4),
    "bright": lambda im: t_bright(im, 1.7),
    "flip": t_flip,
    "blur": lambda im: t_blur(im, 9),
    "noise": lambda im: t_noise(im, 25),
}


def annotate(img, result, color, label):
    out = img.copy()
    boxes = result.boxes
    confs = []
    if boxes is not None and len(boxes):
        for b in boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            c = float(b.conf[0])
            confs.append(c)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f"{c:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(out, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return out, confs


def run(models: dict, images: list[Path], conf=0.25):
    from ultralytics import YOLO
    loaded = {name: YOLO(p) for name, p in models.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    results = []

    for img_path in images:
        base = cv2.imread(str(img_path))
        for tname, tfn in TRANSFORMS.items():
            timg = tfn(base)
            panels = []
            row = {"image": img_path.name, "transform": tname}
            for mname, model in loaded.items():
                r = model.predict(timg, conf=conf, verbose=False)[0]
                color = (0, 0, 255) if mname == "noaug" else (0, 200, 0)
                ann, _ = annotate(timg, r, color, f"{mname}: {len(r.boxes)} det")
                panels.append(ann)
                row[f"{mname}_ndet"] = int(len(r.boxes))
                row[f"{mname}_maxconf"] = round(max([float(b.conf[0]) for b in r.boxes], default=0.0), 3)
            combo = np.hstack(panels)
            cv2.imwrite(str(OUT / f"{img_path.stem}_{tname}.png"), combo)
            results.append(row)

    (OUT / "ood_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--noaug", default=str(ROOT / "weights" / "noaug.pt"))
    ap.add_argument("--aug", default=str(ROOT / "weights" / "aug.pt"))
    ap.add_argument("--n", type=int, default=6, help="사용할 결함 이미지 수")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    # val 결함 이미지에서 추출 (good=빈라벨 제외)
    val_dir = ROOT / "datasets" / "screw_detect" / "images" / "val"
    imgs = sorted(p for p in val_dir.glob("def_*.png") if "good" not in p.name)[: args.n]
    print(f"OOD 대상 {len(imgs)}장")
    res = run({"noaug": args.noaug, "aug": args.aug}, imgs, conf=args.conf)
    # 요약: 변형별로 noaug 가 놓치고 aug 가 잡은 케이스
    print("\n변형별 평균 탐지수 (noaug -> aug):")
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0, 0])
    for r in res:
        a = agg[r["transform"]]
        a[0] += r["noaug_ndet"]; a[1] += r["aug_ndet"]; a[2] += 1
    for t, (na, au, c) in agg.items():
        print(f"  {t:10s}: {na/c:.2f} -> {au/c:.2f}")
