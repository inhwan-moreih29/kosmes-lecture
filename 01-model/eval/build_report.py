"""실험 보고서 HTML 생성 (모든 실험 종료 후 실행).

수집원:
  - runs/<name>/results.csv  -> 비교군별 최종 지표
  - runs/<name>/results.png  -> 학습곡선
  - eval/out/ood/*           -> OOD 비교 이미지 + ood_results.json
  - eval/out/threshold/*     -> 임계값 스윕 이미지
  - eval/out/datasize/*      -> n10 vs n100 비교 이미지
  - eval/out/speed.json      -> 속도 벤치

이미지는 base64 임베드 -> 단일 HTML 로 포터블.
출력: docs/lesson1-experiment-report.html
"""

import base64
import csv
import json
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = ROOT.parent / "outputs" / "model" / "runs"
OUT = ROOT.parent / "outputs" / "model" / "eval"
REPORT = ROOT.parent / "docs" / "lesson1-experiment-report.html"

GROUP_LABEL = {
    "noaug": "증강 없음 (yolo11s)",
    "aug": "증강 있음 (yolo11s)",
    "n10": "데이터 10장 (yolo11s)",
    "n50": "데이터 50장 (yolo11s)",
    "n100": "데이터 100장 (yolo11s)",
    "size_n": "yolo11n (가장 작음)",
    "size_m": "yolo11m (중간)",
    "size_l": "yolo11l (가장 큼)",
}

# OOD 변형 한글 설명
OOD_DESC = {
    "original": "원본",
    "dark": "어둡게 (밝기 0.4배)",
    "bright": "밝게 (밝기 1.7배)",
    "flip": "좌우 반전",
    "rotate45": "45도 회전",
    "rotate90": "90도 회전",
    "blur": "블러 (가우시안)",
    "noise": "노이즈 추가",
}


def b64_jpeg(path: Path, max_w: int = 1600, quality: int = 85) -> tuple[str, str]:
    """이미지를 max_w 폭으로 다운스케일 + JPEG 인코딩 -> (mime, base64). 용량 절감."""
    img = cv2.imread(str(path))
    if img is None:
        return "image/png", base64.b64encode(path.read_bytes()).decode()
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return "image/png", base64.b64encode(path.read_bytes()).decode()
    return "image/jpeg", base64.b64encode(buf.tobytes()).decode()


def img_tag(path: Path, width="100%") -> str:
    if not path.exists():
        return f"<p class='missing'>[이미지 없음: {path.name}]</p>"
    mime, data = b64_jpeg(path)
    return f"<img style='width:{width}' src='data:{mime};base64,{data}'/>"


def read_metrics(name: str) -> dict | None:
    csv_path = RUNS / name / "results.csv"
    if not csv_path.exists():
        return None
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    last = rows[-1]

    def g(*keys):
        for k in keys:
            for col in last:
                if col.strip() == k:
                    try:
                        return float(last[col])
                    except ValueError:
                        return None
        return None

    return {
        "epochs": len(rows),
        "map50": g("metrics/mAP50(B)"),
        "map5095": g("metrics/mAP50-95(B)"),
        "precision": g("metrics/precision(B)"),
        "recall": g("metrics/recall(B)"),
        "train_box": g("train/box_loss"),
        "val_box": g("val/box_loss"),
    }


def metrics_table() -> str:
    head = ("<tr><th>비교군</th><th>설명</th><th>mAP50</th><th>mAP50-95</th>"
            "<th>Precision</th><th>Recall</th>"
            "<th>train box loss</th><th>val box loss</th></tr>")
    body = ""
    for name, label in GROUP_LABEL.items():
        m = read_metrics(name)
        if not m:
            body += f"<tr><td>{name}</td><td>{label}</td><td colspan=6 class='missing'>미완료</td></tr>"
            continue
        body += (
            f"<tr><td><b>{name}</b></td><td>{label}</td>"
            f"<td>{m['map50']:.3f}</td><td>{m['map5095']:.3f}</td>"
            f"<td>{m['precision']:.3f}</td><td>{m['recall']:.3f}</td>"
            f"<td>{m['train_box']:.3f}</td><td>{m['val_box']:.3f}</td></tr>"
        )
    return (f"<table>{head}{body}</table>"
            "<p style='font-size:14px;color:#555'>※ train box loss는 낮은데 val box loss가 높으면 "
            "<b>과적합</b>(학습 데이터만 외움). size_l 행을 보라.</p>")


def gallery(folder: Path, caption_fn=None) -> str:
    if not folder.exists():
        return f"<p class='missing'>[{folder.name} 없음]</p>"
    out = ""
    for p in sorted(folder.glob("*.png")):
        cap = caption_fn(p) if caption_fn else p.stem
        out += f"<figure>{img_tag(p)}<figcaption>{cap}</figcaption></figure>"
    return out


def ood_curated_gallery() -> str:
    """변형별로 가장 선명한 케이스 1장씩 선택해 보여준다.
    회복형(dark/bright/flip/rotate45): noaug < aug 가 가장 큰 이미지.
    한계형(noise/blur): aug 오탐(또는 미회복)이 가장 큰 이미지.
    """
    jf = OUT / "ood" / "ood_results.json"
    if not jf.exists():
        return "<p class='missing'>[OOD 결과 없음]</p>"
    rows = json.loads(jf.read_text())
    picks = []  # (transform, filename, caption)

    def best(transform, key, reverse):
        cand = [r for r in rows if r["transform"] == transform]
        if not cand:
            return None
        return sorted(cand, key=key, reverse=reverse)[0]

    # 회복형: aug_ndet - noaug_ndet 최대 (증강이 놓친 걸 잡음)
    for t in ["dark", "bright", "flip", "rotate45"]:
        r = best(t, lambda r: r["aug_ndet"] - r["noaug_ndet"], True)
        if r and r["aug_ndet"] > r["noaug_ndet"]:
            picks.append((t, f"{Path(r['image']).stem}_{t}.png",
                          f"{OOD_DESC[t]}: 증강없음 {r['noaug_ndet']}개 탐지 → 증강있음 {r['aug_ndet']}개 (회복)"))
    # 한계형: noise 오탐 최대
    r = best("noise", lambda r: r["aug_ndet"], True)
    if r:
        picks.append(("noise", f"{Path(r['image']).stem}_noise.png",
                      f"{OOD_DESC['noise']}: 증강있음이 {r['aug_ndet']}개 오탐 — 학습에 없던 변형엔 증강도 무력(헛것을 봄)"))

    out = ""
    for _, fname, cap in picks:
        out += f"<figure>{img_tag(OUT / 'ood' / fname)}<figcaption>{cap}</figcaption></figure>"
    return out or "<p class='missing'>[큐레이션 실패]</p>"


def ood_summary_table() -> str:
    jf = OUT / "ood" / "ood_results.json"
    if not jf.exists():
        return "<p class='missing'>[OOD 결과 없음]</p>"
    rows = json.loads(jf.read_text())
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0, 0, 0])  # noaug_det, aug_det, noaug_conf, n
    for r in rows:
        a = agg[r["transform"]]
        a[0] += r["noaug_ndet"]; a[1] += r["aug_ndet"]; a[3] += 1
    head = "<tr><th>변형(OOD)</th><th>noaug 평균 탐지수</th><th>aug 평균 탐지수</th><th>해석</th></tr>"
    body = ""
    for t, (na, au, _, n) in agg.items():
        na_avg, au_avg = na / n, au / n
        verdict = "✅ 증강이 회복" if au_avg > na_avg + 0.2 else ("— 비슷" if abs(au_avg - na_avg) <= 0.2 else "⚠️ 증강이 더 못함")
        body += f"<tr><td>{t}</td><td>{na_avg:.2f}</td><td>{au_avg:.2f}</td><td>{verdict}</td></tr>"
    return f"<table>{head}{body}</table>"


def speed_table() -> str:
    jf = OUT / "speed.json"
    if not jf.exists():
        return "<p class='missing'>[속도 결과 없음]</p>"
    rows = json.loads(jf.read_text())
    head = "<tr><th>모델</th><th>파라미터(M)</th><th>추론(ms/img)</th><th>FPS</th><th>mAP50</th></tr>"
    body = ""
    for r in rows:
        # speed.json 의 tag 가 곧 runs/<tag> 디렉터리명 ('s'는 noaug 재사용)
        m = read_metrics(r["tag"]) or {}
        map50 = f"{m.get('map50'):.3f}" if m.get("map50") is not None else "—"
        body += (f"<tr><td>{r['name']}</td><td>{r.get('params_M','—')}</td>"
                 f"<td>{r['ms_per_img']}</td><td>{r['fps']}</td><td>{map50}</td></tr>")
    return f"<table>{head}{body}</table>"


def size_takeaway() -> str:
    """모델 크기 실험의 해석을 실제 지표에서 자동 생성 (과적합 여부 판정 포함)."""
    sizes = [("size_n", "yolo11n"), ("noaug", "yolo11s"),
             ("size_m", "yolo11m"), ("size_l", "yolo11l")]
    rows = []
    for tag, disp in sizes:
        m = read_metrics(tag)
        if m and m.get("map50") is not None:
            rows.append((disp, m))
    if not rows:
        return "<div class=takeaway>[모델 크기 지표 없음]</div>"

    best = max(rows, key=lambda x: x[1]["map50"])
    worst = min(rows, key=lambda x: x[1]["map50"])
    largest = rows[-1]  # yolo11l (있으면)

    # 과적합 신호: train box loss는 낮은데 val box loss가 상대적으로 높음
    def overfit_gap(m):
        tb, vb = m.get("train_box"), m.get("val_box")
        return (vb - tb) if (tb is not None and vb is not None) else None

    lines = [
        f"가장 정확한 모델은 <b>{best[0]}</b> (mAP50 {best[1]['map50']:.3f}), "
        f"가장 부정확한 모델은 <b>{worst[0]}</b> (mAP50 {worst[1]['map50']:.3f})."
    ]
    g_best, g_largest = overfit_gap(best[1]), overfit_gap(largest[1])
    if best[0] != "yolo11l" and g_largest is not None and g_best is not None and g_largest > g_best + 0.3:
        lines.append(
            f"가장 큰 {largest[0]}이 최고 성능이 아니다 — train/val box loss 격차가 "
            f"{g_largest:.2f}로 커 <b>과적합</b>(작은 데이터를 암기) 신호. "
            f"<b>데이터가 모델 크기에 비해 적으면 더 큰 모델이 손해</b>를 본다."
        )
    else:
        lines.append(
            "크기를 키운 만큼 정확도가 또렷이 오르지는 않는다 — "
            "소규모 데이터에서는 <b>정확도-속도 균형점</b>을 직접 재봐야 한다."
        )
    lines.append(
        "※ 속도 격차는 GPU(RTX 4090)에선 작아 보여도 CPU·엣지 장비에선 크게 벌어진다 → "
        "2교시 실시간 영상에서 체감."
    )
    return "<div class=takeaway><b>해석.</b> " + "<br><br>".join(lines) + "</div>"


def curves_gallery() -> str:
    out = ""
    for name, label in GROUP_LABEL.items():
        p = RUNS / name / "results.png"
        if p.exists():
            out += f"<figure>{img_tag(p)}<figcaption>{label} ({name}) 학습곡선</figcaption></figure>"
    return out or "<p class='missing'>[학습곡선 없음]</p>"


HTML = """<!doctype html><html lang=ko><head><meta charset=utf-8>
<title>1교시 비전 AI 실험 보고서</title>
<style>
 /* 다크모드에서도 항상 밝은 배경 + 진한 글씨로 강제 (이전: 배경 미지정 -> 다크모드에서 글씨 안 보임) */
 :root{{color-scheme:light}}
 html{{background:#eef1f6}}
 body{{font-family:-apple-system,'Segoe UI',Roboto,'Noto Sans KR',sans-serif;max-width:1080px;margin:0 auto;padding:40px;
       background:#ffffff;color:#15181d;line-height:1.7;font-size:17px;
       box-shadow:0 0 24px rgba(0,0,0,.08)}}
 h1{{font-size:30px;border-bottom:4px solid #2d6cdf;padding-bottom:10px;color:#10243f}}
 h2{{font-size:24px;margin-top:52px;border-left:6px solid #2d6cdf;padding:4px 0 4px 14px;color:#10243f}}
 h3{{font-size:19px;margin-top:30px;color:#1d4ed8}}
 p,li{{color:#222831}}
 b{{color:#0c1320}}
 table{{border-collapse:collapse;width:100%;margin:18px 0;font-size:16px}}
 th,td{{border:1px solid #c4ccd8;padding:10px 12px;text-align:center;color:#15181d}}
 th{{background:#234b8a;color:#ffffff;font-weight:700}}
 tr:nth-child(even) td{{background:#eef3fb}}
 tr:nth-child(odd) td{{background:#ffffff}}
 figure{{margin:18px 0;border:1px solid #d4d9e2;border-radius:8px;padding:10px;background:#f4f6fa}}
 figcaption{{font-size:15px;color:#2a2f37;margin-top:8px;text-align:center;font-weight:600}}
 img{{display:block;border:1px solid #c4ccd8;border-radius:4px;background:#000}}
 .missing{{color:#c0152b;font-style:italic}}
 .takeaway{{background:#e7eefc;border-left:6px solid #2d6cdf;padding:18px 22px;border-radius:6px;margin:18px 0;color:#15233f}}
 .key{{background:#fff3da;border-left:6px solid #e0911c;padding:16px 22px;border-radius:6px;color:#3d2a05}}
 .key b,.takeaway b{{color:inherit}}
 code{{background:#e7e9ee;color:#9b1d4a;padding:2px 6px;border-radius:4px;font-size:90%}}
 .grid{{display:grid;grid-template-columns:1fr;gap:10px}}
</style></head><body>

<h1>1교시 비전 AI — 실험 보고서</h1>
<p>중소 제조업 AI 실습 강의 / 4종 비전 AI "실사용" 데모를 위한 비교 모델 준비.
데이터: <b>MVTec AD · screw</b> (나사 결함, 불량 119장), 태스크: <b>검출(detection)</b>, 단일 클래스 <code>defect</code>.
GPU: RTX 4090, Ultralytics YOLO11.</p>

<div class=key><b>1교시 핵심 메시지 (이 실험으로 증명하려는 것)</b>
<ol>
<li>판정선(임계값)은 AI가 아니라 <b>내가</b> 정한다 — confidence 슬라이더로 오탐/미탐 트레이드오프 조절</li>
<li>AI가 틀리는 건 멍청해서가 아니라 <b>데이터 분포 문제</b> — 분포 밖(OOD) 이미지에서 무너짐</li>
<li>그건 <b>증강·데이터</b>로 고친다 — 증강 모델은 OOD를 회복, 데이터 양은 성능을 가른다</li>
</ol></div>

<h2>1. 비교 실험 한눈에 보기</h2>
<p>같은 데이터·같은 seed(=0)로 8개 비교군을 학습. 변수 하나씩만 바꿔 효과를 분리.</p>
{metrics}

<h2>2. 증강 없음 vs 있음 + "일부러 틀리기"(OOD)</h2>
<p>핵심 데모. 원본 분포에 없던 변형(회전·밝기·반전·블러·노이즈)을 가해 모델을 일부러 틀리게 만든 뒤,
증강 모델이 회복하는지 확인. 증강은 회전/밝기/반전을 학습에 포함(통제된 셋).</p>
<h3>변형별 회복 요약</h3>
{ood_table}
<h3>비교 추론 이미지 (왼쪽 빨강=증강없음, 오른쪽 초록=증강있음)</h3>
<div class=grid>{ood_gallery}</div>
<div class=takeaway><b>해석.</b> 학습 증강에 넣은 변형(밝기·반전·회전)은 증강 모델이 회복한다.
하지만 넣지 않은 변형(노이즈)은 회복은커녕 헛것을 본다 →
<b>"증강은 만능이 아니라, 내가 학습에 넣은 변형에만 강하다."</b> 현장에서 마주칠 변형을 미리 알아야 한다.</div>

<h2>3. 데이터 양: 10 → 50 → 100장</h2>
<p>"라벨이 모델을 만든다" — 같은 설정에서 학습 데이터 수만 10/50/100으로 늘려가며 비교.
왼쪽 빨강(10장) → 주황(50장) → 초록(100장).</p>
<div class=grid>{datasize_gallery}</div>

<h2>4. 임계값 만지기 (추론 시점 노브)</h2>
<p>모델은 하나, confidence 임계값만 바꿈. 임계값이 낮으면 다 잡지만 오탐↑, 높으면 놓침↑.
"불량 놓칠래 vs 정상 버릴래"는 <b>사람의 의사결정</b>.</p>
<div class=grid>{threshold_gallery}</div>

<h2>5. 모델 크기: 큰 게 정말 더 좋은가? (속도 vs 정확도)</h2>
<p>2교시(영상 실시간) 복선. yolo11 <b>n / s / m / l</b> 네 크기를 같은 데이터(screw 풀데이터, 증강 없음)로 학습해
파라미터 수·추론 속도·정확도를 한자리에서 비교.</p>
{speed_table}
{size_takeaway}

<h2>6. 학습곡선 (보너스)</h2>
<div class=grid>{curves}</div>

<div class=takeaway><b>정리.</b> 1교시는 "모델을 만드는 법"이 아니라 "모델이 왜 그렇게 행동하는지"를 체득하는 시간.
임계값은 내가 정하고, 실패는 데이터 분포에서 오며, 증강·데이터로 고친다. 다음(2교시): 이 모델을 영상에 실시간으로 돌리면?</div>

</body></html>"""


def main():
    html = HTML.format(
        metrics=metrics_table(),
        ood_table=ood_summary_table(),
        ood_gallery=ood_curated_gallery(),
        datasize_gallery=gallery(OUT / "datasize"),
        threshold_gallery=gallery(OUT / "threshold"),
        speed_table=speed_table(),
        size_takeaway=size_takeaway(),
        curves=curves_gallery(),
    )
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(html, encoding="utf-8")
    print(f"보고서 -> {REPORT} ({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
