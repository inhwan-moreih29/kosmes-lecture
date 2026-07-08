# -*- coding: utf-8 -*-
"""에폭 진행별(학습 전 → epoch10 → ... → epoch100) 실제 CNN 특징맵 + 판정 변화.

train_epochprog_nut.py 가 남긴 runs/nut_epochprog/weights/epoch*.pt 를 사용.
동일 너트 결함 이미지에 대해 얕은/중간/깊은 3개 레이어 활성화를 forward hook으로 뽑아
3x3 채널 몽타주(PNG)로 저장하고, 같은 단계의 실제 판정(conf)도 기록한다.
합성/가짜 없음 — 랜덤 초기화 + 실제 학습 체크포인트만.

실행: .venv/bin/python eval/gen_featprog_epochs.py
"""
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WDIR = ROOT.parent / "outputs" / "model" / "runs" / "nut_epochprog" / "weights"
OUT = ROOT.parent / "outputs" / "model" / "eval" / "lesson_epochprog"
VAL = ROOT / "datasets" / "metal_nut_detect" / "images" / "val"

# 보여줄 에폭 단계 (파일 존재하는 것만 사용). e0 = 랜덤 초기화(학습 전).
EPOCH_STAGES = [0, 50, 70, 100]

LAYER_SPECS = [("shallow", 2), ("mid", 6), ("deep", 9)]
TILE_PX, GAP_PX = 78, 2
GRID_COLOR = (15, 15, 20)
CMAP = matplotlib.colormaps["Blues"]


def ckpt_for(ep: int) -> Path | None:
    if ep == 100:
        for cand in ["epoch100.pt", "last.pt", "best.pt"]:
            if (WDIR / cand).exists():
                return WDIR / cand
        return None
    p = WDIR / f"epoch{ep}.pt"
    return p if p.exists() else None


def build_stage(ep: int):
    """(tag, YOLO) 반환. e0 은 랜덤 초기화."""
    if ep == 0:
        base = ckpt_for(100) or next(WDIR.glob("*.pt"))
        m = YOLO(str(base))
        torch.manual_seed(0)
        fresh = DetectionModel(cfg="yolo11s.yaml", nc=1, verbose=False)
        fresh.names = {0: "defect"}
        m.model = fresh
        return ("e0", m)
    ck = ckpt_for(ep)
    if ck is None:
        return None
    return (f"e{ep}", YOLO(str(ck)))


def register_hooks(model):
    mm = model.model.model
    captured = {}
    def make_hook(idx):
        def hook(_m, _i, out):
            o = out[0] if isinstance(out, (list, tuple)) else out
            captured[idx] = o.detach().clone()
        return hook
    handles = [mm[idx].register_forward_hook(make_hook(idx)) for _, idx in LAYER_SPECS]
    return captured, handles


def top_variance_channels(act, k=9):
    c = act.shape[0]
    if c < k:
        return list(range(c))
    return torch.topk(act.var(dim=(1, 2)), k=k, largest=True).indices.tolist()


def make_montage(act, channels):
    grid = 3
    size = grid * TILE_PX + (grid + 1) * GAP_PX
    canvas = np.full((size, size, 3), GRID_COLOR, dtype=np.uint8)
    for i, ch in enumerate(channels):
        arr = act[ch].float().cpu().numpy()
        # 백분위(2~98%) 정규화: min-max는 이상치 1개가 스케일을 독점해 나머지를
        # 흰색으로 뭉갠다. 백분위 클립으로 실제 활성 패턴(구조)을 드러낸다.
        lo, hi = float(np.percentile(arr, 2)), float(np.percentile(arr, 98))
        norm = np.clip((arr - lo) / (hi - lo + 1e-12), 0.0, 1.0)
        norm = cv2.resize(norm.astype(np.float32), (TILE_PX, TILE_PX), interpolation=cv2.INTER_LINEAR)
        rgb = (CMAP(norm)[:, :, :3] * 255).astype(np.uint8)
        row, col = divmod(i, grid)
        y0 = GAP_PX + row * (TILE_PX + GAP_PX)
        x0 = GAP_PX + col * (TILE_PX + GAP_PX)
        canvas[y0:y0 + TILE_PX, x0:x0 + TILE_PX] = rgb
    return canvas


def describe(result, names):
    b = result.boxes
    n = 0 if b is None else len(b)
    if n == 0:
        return "검출 없음", 0.0
    confs = b.conf.tolist()
    return f"{names[int(b.cls.tolist()[max(range(n), key=lambda i: confs[i])])]}", max(confs)


def pick_image(final_model) -> Path:
    """최종 모델이 가장 확신하는 val 결함 이미지 선택 (판정 진행이 또렷하도록)."""
    cands = [p for p in sorted(VAL.glob("def_*.png")) if "good" not in p.name]
    best, best_conf = cands[0], -1.0
    for p in cands[:20]:
        r = final_model.predict(str(p), conf=0.25, imgsz=640, verbose=False, device="cpu")[0]
        _, c = describe(r, final_model.names)
        if c > best_conf:
            best, best_conf = p, c
    print(f"선택 이미지: {best.name} (최종 conf={best_conf:.3f})")
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    avail = sorted(p.name for p in WDIR.glob("*.pt"))
    print("가용 체크포인트:", avail)

    final = build_stage(100)
    assert final, "epoch100/last 체크포인트 없음 — 학습 완료 후 실행"
    img = pick_image(final[1])

    stages = [s for s in (build_stage(ep) for ep in EPOCH_STAGES) if s]
    verdicts = {}
    shapes = {}
    for tag, model in stages:
        captured, handles = register_hooks(model)
        # conf=0.02: 판정 기준(25%) 아래 신뢰도까지 캡처해 '확신도 상승' 곡선을 만든다
        # (특징맵은 forward hook이라 conf 임계값과 무관).
        res = model.predict(str(img), conf=0.02, imgsz=640, verbose=False, device="cpu")[0]
        for h in handles:
            h.remove()
        cls, conf = describe(res, model.names)
        verdicts[tag] = (cls, conf)
        for depth_name, idx in LAYER_SPECS:
            act = captured[idx][0]
            shapes.setdefault(depth_name, (idx, tuple(act.shape)))
            ch = top_variance_channels(act, 9)
            mont = make_montage(act, ch)
            cv2.imwrite(str(OUT / f"{tag}_{depth_name}.png"), cv2.cvtColor(mont, cv2.COLOR_RGB2BGR))
        print(f"{tag}: 판정={cls} conf={conf:.3f}")

    print("\n=== 이미지 ===", img)
    print("=== shape ===", shapes)
    print("=== 판정 진행 ===")
    for tag, _ in stages:
        print(f"  {tag}: {verdicts[tag][0]} {verdicts[tag][1]:.3f}")


if __name__ == "__main__":
    main()
