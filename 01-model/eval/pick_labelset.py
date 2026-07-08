# -*- coding: utf-8 -*-
"""실습 라벨링용 너트 이미지 유형별 5장 선정.

- 각 유형(bent/color/flip/scratch)에서 결함이 또렷해 라벨링하기 좋은 5장을 aug 모델
  검출 신뢰도로 고른다(단, 시나리오 이미지는 강제 포함).
- good(정상)도 5장: none 모델이 과검하는 대표 이미지(과검→복구 데모용) 포함.
- 산출: out/lab_labelset/<type>/ 이미지 복사 + 유형별 컨택트시트 PNG(육안 검증용).

시나리오 강제 포함:
  color: def_color_004 (임계값 경계)
  bent : def_bent_019  (증강 후 미검 위험)
  good : def_good_021  (none 과검 → all 정상)
"""
from pathlib import Path
import shutil
import cv2
import numpy as np
from ultralytics import YOLO

R = Path("/home/kih/workspaces/resources/kosmes-lecture/01-model")
DS = R / "datasets" / "metal_nut_detect"
OUT = R.parent / "outputs" / "model" / "eval" / "lab_labelset"
NONE = YOLO(str(R.parent / "outputs/model/runs/lab_nut_none/weights/best.pt"))
ALL = YOLO(str(R.parent / "outputs/model/runs/lab_nut_all/weights/best.pt"))

TYPES = ["bent", "color", "flip", "scratch", "good"]
FORCE = {"color": ["def_color_004"], "bent": ["def_bent_019"], "good": ["def_good_021"]}
N = 5


def all_imgs(t):
    out = []
    for split in ("train", "val"):
        out += sorted((DS / "images" / split).glob(f"def_{t}_*.png"))
    return out


def maxconf(model, p):
    r = model.predict(str(p), conf=0.01, imgsz=640, verbose=False, device="cpu")[0]
    b = r.boxes
    return 0.0 if (b is None or len(b) == 0) else float(max(b.conf.tolist()))


def pick_for_type(t):
    imgs = all_imgs(t)
    scored = [(p, maxconf(ALL, p), maxconf(NONE, p)) for p in imgs]
    forced = [s for s in scored if s[0].stem in FORCE.get(t, [])]
    rest = [s for s in scored if s[0].stem not in FORCE.get(t, [])]
    if t == "good":
        # 정상인데 none이 강하게 오검하는 순 → 과검 데모에 좋은 부정 샘플
        rest.sort(key=lambda s: s[2], reverse=True)
    else:
        # 결함이 또렷(=aug conf 높음)한 순 → 라벨링 명확
        rest.sort(key=lambda s: s[1], reverse=True)
    sel = forced + rest
    return sel[:N]


def contact_sheet(t, sel):
    tiles = []
    for p, ca, cn in sel:
        im = cv2.imread(str(p))
        im = cv2.resize(im, (300, 300))
        star = " *" if p.stem in FORCE.get(t, []) else ""
        cv2.putText(im, p.stem + star, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.putText(im, f"all={ca:.2f} none={cn:.2f}", (6, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        tiles.append(im)
    pad = np.full((300, 6, 3), 255, np.uint8)
    row = tiles[0]
    for tl in tiles[1:]:
        row = np.hstack([row, pad, tl])
    cv2.imwrite(str(OUT / f"sheet_{t}.png"), row)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    summary = {}
    for t in TYPES:
        sel = pick_for_type(t)
        d = OUT / t
        d.mkdir()
        for p, ca, cn in sel:
            shutil.copy(p, d / p.name)
            lbl = DS / "labels" / p.parent.name / (p.stem + ".txt")
            if lbl.exists():
                shutil.copy(lbl, d / (p.stem + ".txt"))
        contact_sheet(t, sel)
        summary[t] = [(p.name, round(ca, 2), round(cn, 2)) for p, ca, cn in sel]
    print("=== 선정 결과 (파일, all-conf, none-conf) ===")
    for t in TYPES:
        print(f"\n[{t}]")
        for name, ca, cn in summary[t]:
            mark = " ★시나리오" if name.rsplit(".", 1)[0] in sum(FORCE.values(), []) else ""
            print(f"  {name:20} all={ca:<5} none={cn:<5}{mark}")
    print("\n저장:", OUT)


if __name__ == "__main__":
    main()
