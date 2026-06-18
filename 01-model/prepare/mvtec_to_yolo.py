"""MVTec AD -> YOLO 포맷 변환.

MVTec 는 이상탐지(anomaly detection) 포맷이라 Ultralytics 검출/분할로 바로 못 쓴다.
구조:  data/<category>/{train/good, test/<defect>, ground_truth/<defect>/*_mask.png}

핵심 제약(직접 확인): 결함(라벨 가능) 이미지는 test/ 에만 있다. train/good 은 결함이 없다.
따라서 검출/분할 학습은 test 의 결함 이미지를 train/val 로 재분할해서 쓴다.

이 스크립트가 하는 일:
  - test/<defect> 이미지 + ground_truth/<defect>/NNN_mask.png -> YOLO 라벨(.txt)
  - 이진 마스크에서 cv2.findContours -> bbox(검출) 또는 polygon(분할)
  - 단일 클래스 "defect" (1교시는 "결함을 찾는다" 한 가지 메시지로 통일)
  - good 이미지 일부를 빈 라벨(배경)로 섞어 오탐 억제
  - images/{train,val} + labels/{train,val} + data.yaml 출력 (절대경로)
  - seed 고정 split (데이터 수 비교 실험의 변수 통제)

강의 메모(docs/lecture-notes.md): 추천 카테고리 metal_nut(결함 큼, 안정) / screw(어려운 예시).
1교시 데모는 "데이터셋 구축"이 아니라 "체험"이 목적이므로 소규모로 빠르게.

사용:
  uv run prepare/mvtec_to_yolo.py                  # metal_nut, screw 둘 다 detect
  uv run prepare/mvtec_to_yolo.py --category metal_nut --task detect
  uv run prepare/mvtec_to_yolo.py --category metal_nut --task segment
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # 01-model
DATA_ROOT = ROOT.parent / "data"  # kosmes-lecture/data
OUT_ROOT = ROOT / "datasets"  # 01-model/datasets

MIN_AREA = 80  # 이보다 작은 노이즈 윤곽선은 버림 (마스크 잡티 제거)


def mask_to_boxes(mask: np.ndarray) -> list[tuple[float, float, float, float]]:
    """이진 마스크 -> YOLO bbox 리스트 (정규화된 cx,cy,w,h)."""
    h, w = mask.shape
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        if cv2.contourArea(c) < MIN_AREA:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        boxes.append(((x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h))
    return boxes


def mask_to_polys(mask: np.ndarray) -> list[list[float]]:
    """이진 마스크 -> YOLO polygon 리스트 (정규화된 x1,y1,x2,y2,...)."""
    h, w = mask.shape
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < MIN_AREA:
            continue
        # 윤곽선 단순화 (점 수 절감)
        eps = 0.002 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        poly = []
        for x, y in approx:
            poly.extend([float(x) / w, float(y) / h])
        polys.append(poly)
    return polys


def label_for(img_path: Path, mask_path: Path | None, task: str) -> str:
    """이미지 한 장의 YOLO 라벨 텍스트 생성. good(마스크 없음)이면 빈 문자열."""
    if mask_path is None or not mask_path.exists():
        return ""  # good = 배경, 빈 라벨
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return ""
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    lines = []
    if task == "detect":
        for cx, cy, bw, bh in mask_to_boxes(mask):
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    else:  # segment
        for poly in mask_to_polys(mask):
            coords = " ".join(f"{v:.6f}" for v in poly)
            lines.append(f"0 {coords}")
    return "\n".join(lines)


def collect(category: str) -> tuple[list[tuple[Path, Path | None]], list[tuple[Path, Path | None]]]:
    """(결함 이미지, good 이미지) 각각 (img, mask|None) 리스트로 수집."""
    src = DATA_ROOT / category
    test_dir = src / "test"
    gt_dir = src / "ground_truth"
    defects, goods = [], []
    for defect_dir in sorted(test_dir.iterdir()):
        if not defect_dir.is_dir():
            continue
        dname = defect_dir.name
        for img in sorted(defect_dir.glob("*.png")):
            if dname == "good":
                goods.append((img, None))
            else:
                mask = gt_dir / dname / f"{img.stem}_mask.png"
                defects.append((img, mask))
    return defects, goods


def write_split(
    items: list[tuple[Path, Path | None]],
    split: str,
    out: Path,
    task: str,
    tag_prefix: str,
) -> int:
    """items 를 out/images/<split>, out/labels/<split> 로 기록."""
    img_out = out / "images" / split
    lbl_out = out / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    n = 0
    for img_path, mask_path in items:
        # 파일명 충돌 방지: <defect>_<stem> 형태로 통일
        defect = img_path.parent.name
        stem = f"{tag_prefix}_{defect}_{img_path.stem}"
        shutil.copy(img_path, img_out / f"{stem}.png")
        (lbl_out / f"{stem}.txt").write_text(label_for(img_path, mask_path, task))
        n += 1
    return n


def convert(
    category: str,
    task: str = "detect",
    val_ratio: float = 0.2,
    seed: int = 0,
    good_ratio: float = 0.3,
) -> Path:
    """category 를 YOLO 포맷으로 변환. task: detect | segment.

    good_ratio: 결함 이미지 수 대비 섞을 good(배경) 이미지 비율.
    반환: 생성된 data.yaml 경로.
    """
    assert task in ("detect", "segment")
    defects, goods = collect(category)
    rng = random.Random(seed)
    rng.shuffle(defects)
    rng.shuffle(goods)

    # good 은 결함 수의 good_ratio 만큼만 사용 (배경 과다 방지)
    n_good = int(len(defects) * good_ratio)
    goods = goods[:n_good]

    def split(lst):
        n_val = max(1, int(len(lst) * val_ratio)) if lst else 0
        return lst[n_val:], lst[:n_val]

    d_train, d_val = split(defects)
    g_train, g_val = split(goods)

    out = OUT_ROOT / f"{category}_{task}"
    if out.exists():
        shutil.rmtree(out)

    nt = write_split(d_train + g_train, "train", out, task, "def") + 0
    # good 도 같은 split 함수로 분리해 train/val 에 합류 (라벨은 빈 파일)
    # (write_split 은 폴더명을 prefix 로 쓰므로 good/defect 구분 자동)
    nv = write_split(d_val + g_val, "val", out, task, "def")

    data = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "defect"},
    }
    yaml_path = out / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))

    print(
        f"[{category}/{task}] train={nt} val={nv} "
        f"(defect {len(defects)} + good {len(goods)}) -> {out}"
    )
    return yaml_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default=None, help="단일 카테고리 (생략 시 기본 세트)")
    ap.add_argument("--task", default="detect", choices=["detect", "segment"])
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cats = [args.category] if args.category else ["metal_nut", "screw"]
    for cat in cats:
        convert(cat, task=args.task, val_ratio=args.val_ratio, seed=args.seed)
