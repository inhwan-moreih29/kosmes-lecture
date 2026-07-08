# -*- coding: utf-8 -*-
"""학습 진행 단계별(S0=학습전 -> S1=조금학습 -> S2=많이학습) 실제 CNN 특징맵 비교.

동일 이미지에 대해 얕은/중간/깊은 3개 레이어의 실제 활성화를 forward hook으로 뽑아
3x3 채널 몽타주(PNG)로 저장하고, 같은 단계에서 실제 model.predict() 판정
(박스 개수 / 최상위 클래스+conf 또는 검출없음)도 함께 기록한다.
합성/가짜 데이터 없이, 실제 학습된 가중치(n10.pt / n50.pt / n100.pt)와 랜덤 초기화
가중치만 사용.

사용 모델:
  - S0 "학습 전": 랜덤 초기화 YOLO11s(nc=1, torch.manual_seed(0) 고정).
    (주의) 원 지시는 YOLO('yolov8n.yaml') 였으나, weights/*.pt의 실제 체크포인트
    train_args를 확인한 결과 실 아키텍처는 yolov8n이 아니라 yolo11s.yaml이었다
    (동일 레이어 인덱스로 n10/n100과 직접 비교하려면 아키텍처가 같아야 하므로,
    yolov8n.yaml 대신 실제와 같은 yolo11s.yaml을 사용해 랜덤 초기화했다).
  - S1 "조금 학습": weights/n50.pt (n10은 이 이미지에서 검출0 → 판정 진행이 밋밋해
    n50으로 교체: 없음→0.84→0.93 상승을 실측으로 보여줌)
  - S2 "많이 학습": weights/n100.pt
    (주의) n10/n50/n100은 체크포인트 train_args.data 확인 결과 metal_nut이 아니라
    datasets/screw_detect (screw 데이터셋)로 학습되었다. 그래서 metal_nut 테스트
    이미지 93장 전체(bent/color/flip/scratch, conf=0.25 및 0.05)에서 n10/n50/n100
    모두 박스 0개/최대conf 0.0 이었다 (모델이 약한 게 아니라 도메인 불일치).
    실제로 screw_detect val 이미지에서는 n10=검출없음 -> n50=0.84 -> n100=0.93로
    또렷한 학습 진행이 나타난다. 판정 진행을 실측으로 보여주기 위해 샘플 이미지를
    datasets/screw_detect/images/val/def_scratch_neck_021.png 로 교체했다
    (조율자 확인 후 확정된 결정 — 상세 근거는 완료 보고 참고).

실행:
  .venv/bin/python eval/gen_featprog.py
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
WEIGHTS = ROOT / "weights"
OUT = ROOT.parent / "outputs" / "model" / "eval" / "lesson_featprog"

# 동일 이미지를 모든 단계에서 사용.
# n10/n50/n100은 실제로 datasets/screw_detect로 학습되어 metal_nut 이미지에서는
# 어떤 이미지를 골라도 검출 진행을 보여줄 수 없음을 93장 전수 스윕으로 확인했다
# (완료 보고 참고). 그래서 실제 학습 도메인(screw_detect val)에서 n10->n50->n100
# 검출 신뢰도가 뚜렷이 오르는 이미지로 교체했다.
IMG_CANDIDATE = (
    ROOT / "datasets" / "screw_detect" / "images" / "val" / "def_scratch_neck_021.png"
)
SCRATCH_DIR = ROOT / "datasets" / "screw_detect" / "images" / "val"

# 얕은/중간/깊은 레이어 인덱스 (m.model.model = nn.Sequential, 사전 hook 탐색으로 확정)
#   idx 2  : C3k2  -> (1, 128, 160, 160)  얕음  (공간 해상도 큼)
#   idx 6  : C3k2  -> (1, 256,  40,  40)  중간
#   idx 9  : SPPF  -> (1, 512,  20,  20)  깊음  (공간 해상도 작음)
LAYER_SPECS = [("shallow", 2), ("mid", 6), ("deep", 9)]

TILE_PX = 78
GAP_PX = 2
GRID_COLOR = (15, 15, 20)  # 타일 사이 얇은 어두운 격자선 (RGB)
CMAP = matplotlib.colormaps["Blues"]


def resolve_image() -> Path:
    if IMG_CANDIDATE.exists():
        return IMG_CANDIDATE
    alt = sorted(SCRATCH_DIR.glob("*.png"))
    if not alt:
        raise FileNotFoundError(f"scratch 이미지가 없습니다: {SCRATCH_DIR}")
    return alt[0]


def build_stage_models():
    """S0(랜덤 초기화)/S1(n10)/S2(n100) YOLO 래퍼 3개를 반환."""
    # S1/S2: 실제 학습된 체크포인트 그대로 로드
    # S1은 n50 사용 — 이 이미지에서 n10=검출없음(0.0)이라 판정 진행이 밋밋하지만,
    # n50=0.842로 '없음→0.84→0.93'의 또렷한 판정 상승을 보여줄 수 있어 교체 (조율자 확정).
    m_s1 = YOLO(WEIGHTS / "n50.pt")
    m_s2 = YOLO(WEIGHTS / "n100.pt")

    # S0: n10 래퍼(예측 파이프라인/이름 그대로 재사용)에 랜덤 초기화 가중치만 주입
    m_s0 = YOLO(WEIGHTS / "n10.pt")
    torch.manual_seed(0)
    fresh = DetectionModel(cfg="yolo11s.yaml", nc=1, verbose=False)
    fresh.names = {0: "defect"}
    m_s0.model = fresh

    return [("s0", m_s0), ("s1", m_s1), ("s2", m_s2)]


def register_hooks(model):
    """m.model.model의 LAYER_SPECS 인덱스에 forward hook을 걸고 캡처 dict를 반환."""
    mm = model.model.model
    captured = {}

    def make_hook(idx):
        def hook(_module, _inp, out):
            o = out[0] if isinstance(out, (list, tuple)) else out
            captured[idx] = o.detach().clone()
        return hook

    handles = [mm[idx].register_forward_hook(make_hook(idx)) for _, idx in LAYER_SPECS]
    return captured, handles


def top_variance_channels(act: torch.Tensor, k: int = 9) -> list[int]:
    """(C,H,W) 활성화에서 채널별 분산 상위 k개 인덱스. 채널이 부족하면 앞 k개로 대체."""
    c = act.shape[0]
    if c < k:
        return list(range(c))
    var = act.var(dim=(1, 2))
    top = torch.topk(var, k=k, largest=True).indices.tolist()
    return top


def make_montage(act: torch.Tensor, channels: list[int]) -> np.ndarray:
    """채널별 [0,1] 정규화 + Blues 컬러맵 3x3 몽타주 (RGB uint8)."""
    grid = 3
    canvas_size = grid * TILE_PX + (grid + 1) * GAP_PX
    canvas = np.full((canvas_size, canvas_size, 3), GRID_COLOR, dtype=np.uint8)

    for i, ch in enumerate(channels):
        arr = act[ch].float().cpu().numpy()
        amin, amax = float(arr.min()), float(arr.max())
        norm = (arr - amin) / (amax - amin + 1e-12)
        norm = cv2.resize(norm.astype(np.float32), (TILE_PX, TILE_PX), interpolation=cv2.INTER_LINEAR)
        rgb = (CMAP(norm)[:, :, :3] * 255).astype(np.uint8)

        row, col = divmod(i, grid)
        y0 = GAP_PX + row * (TILE_PX + GAP_PX)
        x0 = GAP_PX + col * (TILE_PX + GAP_PX)
        canvas[y0:y0 + TILE_PX, x0:x0 + TILE_PX] = rgb

    return canvas


def describe_detection(result, names) -> str:
    boxes = result.boxes
    n = 0 if boxes is None else len(boxes)
    if n == 0:
        return f"검출 없음 (박스 0개)"
    confs = boxes.conf.tolist()
    clss = boxes.cls.tolist()
    top_i = max(range(n), key=lambda i: confs[i])
    return f"{names[int(clss[top_i])]} conf={confs[top_i]:.3f} (박스 {n}개)"


def main():
    img_path = resolve_image()
    OUT.mkdir(parents=True, exist_ok=True)

    report = {"image": str(img_path), "layers": {name: idx for name, idx in LAYER_SPECS}, "stages": {}}
    shapes_reported = {}

    for stage_tag, model in build_stage_models():
        captured, handles = register_hooks(model)
        results = model.predict(str(img_path), conf=0.25, imgsz=640, verbose=False, device="cpu")
        for h in handles:
            h.remove()

        det_str = describe_detection(results[0], model.names)
        report["stages"][stage_tag] = {"detection": det_str, "names": dict(model.names)}

        for depth_name, idx in LAYER_SPECS:
            act = captured[idx][0]  # (C,H,W), batch=1
            shapes_reported.setdefault(depth_name, (idx, tuple(act.shape)))
            channels = top_variance_channels(act, k=9)
            montage = make_montage(act, channels)
            out_path = OUT / f"{stage_tag}_{depth_name}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
            print(f"saved {out_path}  channels={channels}")

    print("\n=== 이미지 ===")
    print(img_path)
    print("\n=== 레이어 인덱스 / shape ===")
    for depth_name, (idx, shape) in shapes_reported.items():
        print(f"{depth_name}: idx={idx} shape={shape}")
    print("\n=== 판정 결과 ===")
    for stage_tag, info in report["stages"].items():
        print(f"{stage_tag}: {info['detection']}")
    print("\n=== model.names (s1 기준) ===")
    print(report["stages"]["s1"]["names"])


if __name__ == "__main__":
    main()
