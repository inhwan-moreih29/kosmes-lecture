"""다회 실험 실행기 (GPU 샤딩 병렬).

전체 job 리스트를 결정적으로 만들고, idx %% nshards == shard 인 job 만 실행.
-> 4개 프로세스를 device 0~3 / shard 0~3 로 띄우면 4 GPU 병렬.

결과: experiments/results/phase<p>.d<device>.jsonl (한 줄 = 한 job 결과)

사용:
  uv run experiments/run.py --phase 1 --device 0 --nshards 4 --shard 0   (×4 병렬)
  uv run experiments/run.py --phase 2 --points 20,100 --device 0 ...
  uv run experiments/run.py --phase 3 --device 0 ...
"""

import argparse
import json
from pathlib import Path

from lib import AUG_PRESETS, RESULTS, train_eval

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCREW = ROOT / "datasets" / "screw_detect" / "data.yaml"
MSD = ROOT / "datasets" / "msd_detect" / "data.yaml"
MERGED = HERE / "datasets"
SIZE_MODEL = {"n": "yolo11n.pt", "s": "yolo11s.pt", "m": "yolo11m.pt", "l": "yolo11l.pt"}


def jobs_phase1(seeds):
    out = []
    for n in range(10, 101, 10):
        for s in range(seeds):
            out.append({"phase": 1, "n": n, "seed": s})
    return out


def jobs_phase2(points, seeds):
    out = []
    for pt in points:
        for preset in AUG_PRESETS:
            for s in range(seeds):
                out.append({"phase": 2, "n": pt, "preset": preset, "seed": s})
    return out


def jobs_phase3(scales, seeds):
    out = []
    for sc in scales:
        for size in ("n", "s", "m", "l"):
            for s in range(seeds):
                out.append({"phase": 3, "scale": sc, "size": size, "seed": s})
    return out


def jobs_phase4(cats, seeds):
    """데이터 종류(형태)별 증강 효능: n=20 고정, 카테고리 × 증강기법 × 시드."""
    out = []
    for c in cats:
        for preset in AUG_PRESETS:
            for s in range(seeds):
                out.append({"phase": 4, "cat": c, "preset": preset, "seed": s})
    return out


def jobs_phase5(points, seeds):
    """매칭 가설(실데이터): MSD 3-클래스에서 데이터규모 × 모델크기. 탐색용 단일런 기본."""
    out = []
    for n in points:
        for size in ("n", "s", "m", "l"):
            for s in range(seeds):
                out.append({"phase": 5, "n": n, "size": size, "seed": s})
    return out


def run_job(j, device, aug="none"):
    if j["phase"] == 1:
        tag = f"p1_n{j['n']}"
        return train_eval("yolo11s.pt", SCREW, j["seed"], AUG_PRESETS["none"],
                          n=j["n"], device=device, tag=tag)
    if j["phase"] == 2:
        tag = f"p2_n{j['n']}_{j['preset']}"
        return train_eval("yolo11s.pt", SCREW, j["seed"], AUG_PRESETS[j["preset"]],
                          n=j["n"], device=device, tag=tag)
    if j["phase"] == 3:
        data = MERGED / f"merged_{j['scale']}" / "data.yaml"
        # aug!="none" -> Phase3b (증강 ON 재검증). 별도 tag 로 run 디렉터리 분리.
        prefix = "p3b" if aug != "none" else "p3"
        tag = f"{prefix}_{j['scale']}_{j['size']}"
        return train_eval(SIZE_MODEL[j["size"]], data, j["seed"], AUG_PRESETS[aug],
                          n=None, device=device, tag=tag)
    if j["phase"] == 4:
        data = ROOT / "datasets" / f"{j['cat']}_detect" / "data.yaml"
        tag = f"p4_{j['cat']}_{j['preset']}"
        return train_eval("yolo11s.pt", data, j["seed"], AUG_PRESETS[j["preset"]],
                          n=20, device=device, tag=tag)
    if j["phase"] == 5:
        # MSD 3-클래스. n=full 이면 서브샘플 없이 전체 train.
        n = None if j["n"] >= 976 else j["n"]
        tag = f"p5_n{j['n']}_{j['size']}"
        data = j.get("data", MSD)  # GPU별 사본 경로 (캐시 레이스 회피)
        return train_eval(SIZE_MODEL[j["size"]], data, j["seed"], AUG_PRESETS[aug],
                          n=n, device=device, tag=tag)
    raise ValueError(j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=4)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--points", default="", help="phase2 데이터 지점 (예: 20,100)")
    ap.add_argument("--scales", default="S,M,L")
    ap.add_argument("--cats", default="screw,metal_nut,pill,carpet", help="phase4 카테고리")
    ap.add_argument("--aug", default="none", help="phase3 증강 프리셋 (none=원본, all=Phase3b)")
    ap.add_argument("--datadir", default="", help="phase5 데이터셋 디렉터리 오버라이드 (GPU별 사본)")
    args = ap.parse_args()

    if args.phase == 1:
        jobs = jobs_phase1(args.seeds or 5)
    elif args.phase == 2:
        pts = [int(x) for x in args.points.split(",") if x]
        jobs = jobs_phase2(pts, args.seeds or 3)
    elif args.phase == 3:
        jobs = jobs_phase3(args.scales.split(","), args.seeds or 3)
    elif args.phase == 4:
        jobs = jobs_phase4(args.cats.split(","), args.seeds or 3)
    else:
        pts = [int(x) for x in args.points.split(",") if x] or [100, 300, 600, 976]
        jobs = jobs_phase5(pts, args.seeds or 1)
        if args.datadir:
            data_path = Path(args.datadir) / "data.yaml"
            for j in jobs:
                j["data"] = data_path

    mine = [j for i, j in enumerate(jobs) if i % args.nshards == args.shard]
    RESULTS.mkdir(parents=True, exist_ok=True)
    # phase3 + 증강 ON -> phase3b 로 분리 저장 (기존 no-aug 결과 보존)
    label = "3b" if (args.phase == 3 and args.aug != "none") else str(args.phase)
    outf = RESULTS / f"phase{label}.d{args.device}.jsonl"
    print(f"phase{label} shard {args.shard}/{args.nshards} device{args.device} aug={args.aug}: "
          f"{len(mine)}/{len(jobs)} jobs -> {outf}")

    for k, j in enumerate(mine):
        map50 = run_job(j, args.device, aug=args.aug)
        # 'data'(PosixPath)는 직렬화 불가 -> 기록에서 제외
        jrec = {key: val for key, val in j.items() if key != "data"}
        rec = {**jrec, "aug": args.aug, "map50": round(map50, 4), "device": args.device}
        with outf.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  [{k+1}/{len(mine)}] {rec}")


if __name__ == "__main__":
    main()
