# -*- coding: utf-8 -*-
"""에폭 진행에 따른 내부 필터 변화 시연용 학습.

metal_nut 검출 데이터 30장으로 yolo11s를 **랜덤 초기화(from scratch)** 100에폭 학습.
save_period=10 으로 epoch{10,20,...}.pt 중간 체크포인트를 남겨, 이후
gen_featprog_epochs.py 가 에폭별 특징맵/판정을 뽑는다.

랜덤 초기화 이유: COCO 사전학습에서 시작하면 저수준 필터가 이미 형성돼 있어
'학습으로 필터가 만들어지는 과정'이 안 보인다. 잡음->구조 변화를 보이려면 scratch.

실행: .venv/bin/python eval/train_epochprog_nut.py
"""
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DS = ROOT / "datasets" / "metal_nut_detect"
POOL = DS / "images" / "train"
N = 30
SEED = 0

def make_subset():
    imgs = sorted(p for p in POOL.glob("*.png"))
    random.Random(SEED).shuffle(imgs)
    picked = sorted(imgs[:N])
    txt = DS / f"train_n{N}.txt"
    txt.write_text("\n".join(str(p) for p in picked) + "\n", encoding="utf-8")
    yaml = DS / f"data_n{N}.yaml"
    yaml.write_text(
        f"path: {DS}\n"
        f"train: {txt}\n"
        f"val: images/val\n"
        f"names:\n  0: defect\n",
        encoding="utf-8",
    )
    print(f"subset: {len(picked)} imgs -> {txt}")
    return yaml

def main():
    from ultralytics import YOLO
    data_yaml = make_subset()
    model = YOLO("yolo11s.yaml")  # 랜덤 초기화 (from scratch)
    model.train(
        data=str(data_yaml),
        epochs=100,
        imgsz=640,
        seed=SEED,
        deterministic=True,
        pretrained=False,
        project=str(ROOT / "runs"),
        name="nut_epochprog",
        exist_ok=True,
        device=0,
        workers=2,
        save_period=10,   # epoch10,20,...,100 체크포인트 보존
        verbose=False,
        plots=True,
    )
    wdir = ROOT / "runs" / "nut_epochprog" / "weights"
    print("saved checkpoints:", sorted(p.name for p in wdir.glob("*.pt")))

if __name__ == "__main__":
    main()
