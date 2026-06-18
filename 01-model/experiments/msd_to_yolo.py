"""MSD(휴대폰 표면 결함) -> YOLO 3-클래스 detection 변환.

데이터: data/MSD/{oil,scratch,stain}/*.jpg (각 400) + good/*.png (20)
마스크: ground_truth_1/<stem>.png (Scr_/Sta_), ground_truth_2/<stem>.png (Oil_)
        전경 픽셀값이 클래스 코드: 38=oil, 113=scratch, 75=stain (한 마스크에 복수값 가능)

출력: 01-model/datasets/msd_detect/{images,labels}/{train,val} + data.yaml
  names: {0: oil, 1: scratch, 2: stain}
  good 이미지 -> 빈 라벨(정상 샘플)
  val 은 고정(서브샘플은 train 만 건드림) -> 규모 비교 공정.
"""

import random
import shutil
from pathlib import Path

import cv2
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent              # 01-model
DATA = ROOT.parent / "data" / "MSD"
OUT = ROOT / "datasets" / "msd_detect"
MIN_AREA = 60                   # 1920x1080 기준 잡티 제거

# 클래스 정의 + 마스크 픽셀값 -> 클래스 id
NAMES = {0: "oil", 1: "scratch", 2: "stain"}
VAL2CLS = {38: 0, 113: 1, 75: 2}
FOLDERS = [("oil", "ground_truth_2"), ("scratch", "ground_truth_1"), ("stain", "ground_truth_1")]


def boxes_from_mask(mask_path: Path):
    """마스크 -> (cls, cx, cy, w, h) 정규화 bbox 목록. 픽셀값별로 클래스 분리."""
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return []
    h, w = m.shape
    out = []
    for val, cls in VAL2CLS.items():
        binary = ((m == val).astype("uint8")) * 255
        if binary.sum() == 0:
            continue
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < MIN_AREA:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            out.append((cls, (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h))
    return out


def collect():
    """(이미지경로, 라벨텍스트) 목록. defect + good."""
    items = []
    for folder, gt in FOLDERS:
        for img in sorted((DATA / folder).glob("*.jpg")):
            boxes = boxes_from_mask(DATA / gt / f"{img.stem}.png")
            label = "\n".join(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                              for c, cx, cy, bw, bh in boxes)
            items.append((img, label))
    goods = [(img, "") for img in sorted((DATA / "good").glob("*.png"))]
    return items, goods


def write(items, split):
    img_out, lbl_out = OUT / "images" / split, OUT / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for img_path, label in items:
        stem = img_path.stem
        shutil.copy(img_path, img_out / f"{stem}{img_path.suffix}")
        (lbl_out / f"{stem}.txt").write_text(label)
    return len(items)


def build(val_ratio=0.2, seed=0):
    defects, goods = collect()
    n_box = sum(1 for _, l in defects if l)
    rng = random.Random(seed)
    rng.shuffle(defects)
    rng.shuffle(goods)

    def sp(lst):
        nv = max(1, int(len(lst) * val_ratio)) if lst else 0
        return lst[nv:], lst[:nv]

    d_tr, d_va = sp(defects)
    g_tr, g_va = sp(goods)
    if OUT.exists():
        shutil.rmtree(OUT)
    nt = write(d_tr + g_tr, "train")
    nv = write(d_va + g_va, "val")
    cfg = {"path": str(OUT.resolve()), "train": "images/train",
           "val": "images/val", "names": NAMES}
    (OUT / "data.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    print(f"train={nt} val={nv} | defect={len(defects)}(빈마스크 {len(defects)-n_box}) "
          f"good={len(goods)} -> {OUT}")
    return OUT / "data.yaml"


if __name__ == "__main__":
    build()
