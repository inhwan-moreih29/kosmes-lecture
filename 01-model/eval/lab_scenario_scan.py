# -*- coding: utf-8 -*-
"""너트 실습 3요소 검증 — none 모델 vs all(증강) 모델을 val 전체에 추론.

1) 임계값 경계: 증강 모델에서 conf가 중간대(0.25~0.6)라 임계값을 밀면 검출/미검이
   뒤집히는 이미지 + 정상(good)에서 뜨는 오검(과검) 후보.
2) 분포밖 실패: none 모델이 놓치거나 매우 낮은 conf인 이미지(특히 flip/scale).
3) 증강 복구: 2)의 이미지에서 all 모델이 제대로 검출로 복구.

산출: out/lab_scan/summary 표(stdout) + 후보 이미지 side-by-side PNG.
"""
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VAL = ROOT / "datasets" / "metal_nut_detect" / "images" / "val"
NONE_W = ROOT.parent / "outputs" / "model" / "runs" / "lab_nut_none" / "weights" / "best.pt"
ALL_W = ROOT.parent / "outputs" / "model" / "runs" / "lab_nut_all" / "weights" / "best.pt"
OUT = ROOT.parent / "outputs" / "model" / "eval" / "lab_scan"


def maxconf(model, img):
    r = model.predict(str(img), conf=0.01, imgsz=640, verbose=False, device="cpu")[0]
    b = r.boxes
    if b is None or len(b) == 0:
        return 0.0, r
    return float(max(b.conf.tolist())), r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mn, ma = YOLO(str(NONE_W)), YOLO(str(ALL_W))
    imgs = sorted(VAL.glob("*.png"))
    rows = []
    for p in imgs:
        cn, _ = maxconf(mn, p)
        ca, _ = maxconf(ma, p)
        rows.append((p.name, cn, ca))

    print(f"{'image':22}{'none':>8}{'all':>8}  note")
    for name, cn, ca in rows:
        is_good = "good" in name
        note = ""
        if is_good:
            note = "정상샘플(여기 검출=과검)"
        elif cn < 0.25 <= ca:
            note = "★복구(none 미검→all 검출)"
        elif 0.25 <= ca <= 0.6 and not is_good:
            note = "임계값 경계 후보"
        print(f"{name:22}{cn:8.3f}{ca:8.3f}  {note}")

    # 시나리오별 대표 이미지 자동 선정
    defects = [r for r in rows if "good" not in r[0]]
    goods = [r for r in rows if "good" in r[0]]
    recover = sorted([r for r in defects if r[1] < 0.25 <= r[2]], key=lambda r: r[2] - r[1], reverse=True)
    boundary = sorted([r for r in defects if 0.25 <= r[2] <= 0.6], key=lambda r: abs(r[2] - 0.4))
    overkill = sorted(goods, key=lambda r: r[2], reverse=True)  # all 모델이 정상에 얼마나 오검하나

    print("\n=== 시나리오 대표 ===")
    print("복구(none미검→all검출) top:", [(r[0], round(r[1], 2), round(r[2], 2)) for r in recover[:5]])
    print("임계값 경계(all conf 0.25~0.6):", [(r[0], round(r[2], 2)) for r in boundary[:5]])
    print("정상샘플 all conf(과검 위험 순):", [(r[0], round(r[2], 2)) for r in overkill[:5]])

    # side-by-side 저장 (복구 top3 + 경계 top2)
    def save_pair(name, tag):
        img = VAL / name
        rn = mn.predict(str(img), conf=0.25, imgsz=640, verbose=False, device="cpu")[0]
        ra = ma.predict(str(img), conf=0.25, imgsz=640, verbose=False, device="cpu")[0]
        a = rn.plot(); b = ra.plot()
        h = max(a.shape[0], b.shape[0])
        pad = np.full((h, 12, 3), 255, np.uint8)
        canvas = np.hstack([a, pad, b])
        cv2.imwrite(str(OUT / f"{tag}_{name}"), canvas)

    for i, r in enumerate(recover[:3]):
        save_pair(r[0], f"recover{i+1}")
    for i, r in enumerate(boundary[:2]):
        save_pair(r[0], f"boundary{i+1}")
    print("\n저장 위치:", OUT)


if __name__ == "__main__":
    main()
