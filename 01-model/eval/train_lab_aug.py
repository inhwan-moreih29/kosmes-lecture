# -*- coding: utf-8 -*-
"""실습용 너트 모델 학습 — 증강 없음(none) vs 증강(all).

p4 실험과 동일: 같은 19장 학습셋, 증강만 on/off. 실습 3요소
(임계값 조절 / 분포밖 실패 / 증강 복구)를 실제 추론으로 검증하기 위한 모델을 만든다.

사용: .venv/bin/python eval/train_lab_aug.py <none|all> <device>
"""
import sys
from pathlib import Path
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "datasets" / "metal_nut_detect" / "data_p4_metal_nut_none_s0.yaml"  # 학습셋 동일

PRESETS = {
    "none": dict(degrees=0.0, translate=0.0, scale=0.0, flipud=0.0, fliplr=0.0, mosaic=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0),
    "all":  dict(degrees=30.0, translate=0.1, scale=0.4, flipud=0.5, fliplr=0.5, mosaic=0.0),
}

def main():
    preset = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "0"
    m = YOLO("yolo11s.pt")
    m.train(
        data=str(DATA), epochs=100, batch=16, imgsz=640, device=device,
        pretrained=True, project=str(ROOT.parent / "outputs" / "model" / "runs"), name=f"lab_nut_{preset}",
        exist_ok=True, verbose=False, plots=False, workers=4, cache=True,
        **PRESETS[preset],
    )
    print(f"DONE {preset}")

if __name__ == "__main__":
    main()
