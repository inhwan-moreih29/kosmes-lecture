# -*- coding: utf-8 -*-
"""1교시 강의 시나리오 -> docs/lesson1-scenario.html (내부 논의용).

기존 v3의 40분 핸즈온 골격(3종), 섹션별 블록 + 각 섹션 시각자료 인라인.
허블 화면 대체 자료: 3종 출력(task3), 라벨링 임시화면(labeling),
임계값 스윕, OOD 실패, 증강 없음→증강 고침(0→7), Phase4/5/1.
강사용 단서(단일런/품질 vs 양·특성/'이득 없음') 명시. 흰 배경 고정.
"""
import base64
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # 01-model
PLOTS = ROOT.parent / "outputs" / "experiments" / "plots"
EVAL = ROOT.parent / "outputs" / "model" / "eval"
OUT = ROOT.parent / "docs" / "lesson1-scenario.html"
MAXW = 1100                              # 임베드 최대 폭(다운스케일)


def embed(path: Path):
    if not path.exists():
        return f"<p style='color:#b00'>[그림 없음: {path.name}]</p>"
    im = cv2.imread(str(path))
    if im is None:                       # 못 읽으면 원본 그대로
        b64 = base64.b64encode(path.read_bytes()).decode()
        return f'<img src="data:image/png;base64,{b64}" alt="{path.name}">'
    h, w = im.shape[:2]
    if w > MAXW:
        im = cv2.resize(im, (MAXW, int(h * MAXW / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 88])
    b64 = base64.b64encode(buf.tobytes()).decode()
    return f'<img src="data:image/jpeg;base64,{b64}" alt="{path.name}">'


def fig(path: Path, cap):
    return f'<div class="fig">{embed(path)}<div class="cap">{cap}</div></div>'


# 섹션: (시간, 제목, 태그, 본문 HTML)
SECTIONS = [
    ("0–5", "오프닝 + 3종 감 잡기", "",
     "<p>같은 제품 이미지 1장에 <b>분류 · 검출 · 분할</b>의 실제 출력을 겹쳐 비교. "
     "\"내 현장 문제는 이 중 무엇인가\" + \"왜 내 AI는 별로였나\"로 문제 제기.</p>"
     "<p class='talk'>멘트: \"정밀할수록 라벨링 비용도 커진다 — 분류 &lt; 검출 &lt; 분할. 과제는 <u>필요한 만큼만</u> 고른다.\"</p>"
     + fig(EVAL / "lesson" / "task3.png",
           "같은 이미지 → ① 분류(라벨 1개) ② 검출(박스 위치) ③ 분할(픽셀 경계). ※ 예시는 스마트폰 표면(MSD) — 실제 강의는 허블 화면.")),

    ("5–15", "허블에서 3종 라벨링 + 추론 시각화", "실습",
     "<p>각 과제 소수 라벨링 후 출력 형태(라벨/박스/마스크) 비교. "
     "<b>라벨링 실습이 곧 '데이터 품질'을 체감하는 지점</b> — 기준이 사람마다 흔들리면 모델도 흔들린다.</p>"
     + fig(EVAL / "lesson" / "labeling.png",
           "라벨링 화면(예시) — 사람이 결함에 박스를 그리고 클래스를 단다. 이 '정답'의 일관성이 데이터 품질. ※ 허블 캡처로 교체 예정.")),

    ("15–21", "임계값(신뢰도) 만지기", "실습",
     "<p>신뢰도 슬라이더를 밀며 검출이 늘고 주는 것 체감 → \"불량 놓칠래 vs 정상 버릴래\". <b>판정선은 내가 정한다.</b></p>"
     + fig(EVAL / "threshold" / "def_manipulated_front_016.png",
           "같은 모델·같은 이미지, 임계값만 0.05→0.75. 낮추면 다 잡지만 헛검출↑, 높이면 깔끔하지만 놓침↑ — 정답은 공정 비용이 정한다.")),

    ("21–27", "일부러 틀리기 — 분포 밖 실패", "실습",
     "<p>학습 때 못 본 변화를 일부러 만들어(=분포 밖) 모델이 <b>틀리는 것</b>을 관찰. \"틀림은 분포 문제\".</p>"
     "<p><b>트리는 법(예시)</b>: 회전 · 저조도/과노출 · 블러 · 노이즈 · 좌우반전 · <i>처음 보는 형태의 불량</i>. "
     "→ 다음 '고치기' 구간의 복선.</p>"
     + fig(EVAL / "ood" / "def_scratch_head_005_noise.png",
           "분포 밖 예시(노이즈 주입). 증강 없는 모델(빨강)은 결함을 0개로 놓친다 — 깨끗한 사진만 보던 모델의 한계.")),

    ("27–35", "증강·데이터로 고치기", "우리 실험 근거",
     "<p>'틀린 걸 고치는' 해결편. 핵심 히어로 예시 — <b>같은 회전 이미지에서 증강 없는 모델은 0개(놓침) → 증강 모델은 7개(잡음)</b>.</p>"
     + fig(EVAL / "ood" / "def_manipulated_front_016_rotate90.png",
           "회전(분포 밖). 좌=증강 없음 0 det(놓침) / 우=증강 7 det(conf 0.54로 잡음). \"증강이 분포를 넓혀 고친다.\"")
     + "<p style='margin-top:14px'>여기서 핵심 하나를 못박는다 (미리 돌려둔 우리 실험):</p>"
     + fig(PLOTS / "phase4_trafficlight.png",
           "증강은 데이터 '종류'에 맞춰 — 🟢켜면 도움 / ⚪미미 / 🔴끄는 게 나음. 색(hsv) 증강은 알약·너트엔 🔴주의, 원단 텍스처엔 🟢도움. \"체크리스트가 아니라 내 제품에 맞춰.\"")),

    ("35–40", "핵심 정리 (3줄) + 다음 타임 복선", "",
     "<ul>"
     "<li><b>데이터가 성능을 만든다</b> — 양·특성 (아래 자료)</li>"
     "<li><b>판정선은 내가 정한다</b> — 임계값은 공정 비용으로</li>"
     "<li><b>틀림은 분포 문제</b> — 분포 밖을 데이터·증강으로 좁힌다</li>"
     "</ul>"
     + fig(PLOTS / "phase1_datasize.png",
           "데이터 양 ↑ → 성능 ↑ 후 포화. \"얼마나 모아야 하나\"의 감 — 무한정이 아니라 충분량이 있다. (여유 시)")),
]

CAVEATS = """
<li><b>모델 크기 비교는 강의에서 제외</b>(데이터량에 따라 결과가 달라져 비전문가에겐 오해 소지). 질문 들어올 때만: "큰 모델 키운다고 답 아니고, 데이터가 우선." (정량 근거는 내부 실험 스크립트로 재생성 가능)</li>
<li>우리 실험이 직접 만진 변수는 데이터 <b>양·종류</b>와 증강·모델크기다. <b>"라벨 품질"</b>은 정량 실험으로 보이지 않았다 — 품질은 <b>라벨링 실습(5–15분)의 체감</b>으로, 정량 근거는 양·특성에 한정.</li>
<li>큰 모델이 진 것을 "큰 모델이 나쁘다"로 과장 금지. 정확히는 <b>"이 과제·이 규모에선 이득이 없다"</b>(+ 동일 epoch 고정이라 큰 모델은 덜 수렴했을 수 있음).</li>
<li>모든 결함 이미지(나사·스마트폰)는 공개 데이터(MVTec/MSD)로 만든 <b>예시</b>다 — 실제 강의는 허블 + 샘플 부품으로 교체.</li>
"""

DISCUSS = """
<li><b>키포인트 제외 확정</b> → 1교시는 <b>3종</b>. 강의/교시 제목이 "<b>4종</b> 비전 AI"라면 제목·홍보 문구도 3종으로 맞출지 결정 필요.</li>
<li><b>40분 압축 현실성</b>: 라벨링+임계값+실패+고치기 4개 핸즈온을 35분 안에? 실습 장수·시연 비중 조절.</li>
<li><b>허블 화면 캡처</b> 어느 단계가 필요한가(3종 출력 / 라벨링 UI / 임계값 슬라이더) — task3·labeling 그림을 실제 캡처로 교체.</li>
<li><b>실험 자료 노출 범위</b>: 27–35분에 그림 2장(OOD 대비 + 신호등)이 적절한지.</li>
"""

sec_html = ""
for t, title, tag, body in SECTIONS:
    badge = ""
    if tag == "실습":
        badge = '<span class="hb">실습</span>'
    elif tag:
        badge = f'<span class="ev">{tag}</span>'
    sec_html += (f'<section><h2><span class="tm">{t}</span> {title} {badge}</h2>{body}</section>')

HTML = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light only">
<title>1교시 강의 시나리오 — 내부 논의용</title>
<style>
:root{{--ink:#10161f;--mut:#475066;--line:#d7deea;--accent:#1746c4;--soft:#eef3fb;}}
*{{box-sizing:border-box;}}
html{{background:#fff;}}
body{{font-family:-apple-system,'Segoe UI','Apple SD Gothic Neo','Noto Sans KR',sans-serif;
 color:var(--ink);background:#fff;max-width:900px;margin:0 auto;padding:32px 22px 80px;
 line-height:1.7;font-size:16px;}}
h1{{font-size:26px;margin:0 0 4px;color:#0b1118;}}
.sub{{color:var(--mut);margin:0 0 20px;font-size:15px;}}
.meta{{display:flex;gap:16px;flex-wrap:wrap;background:var(--soft);border:1px solid var(--line);
 border-radius:10px;padding:13px 16px;margin:0 0 24px;font-size:14.5px;}}
.meta b{{color:var(--accent);}}
.flow{{display:flex;align-items:center;gap:7px;flex-wrap:wrap;margin:0 0 26px;font-size:13.5px;color:var(--mut);}}
.flow .step{{background:var(--soft);border:1px solid var(--line);border-radius:7px;padding:4px 9px;font-weight:600;color:var(--ink);}}
.flow .arr{{color:var(--mut);font-weight:800;}}
section{{margin:0 0 8px;}}
h2{{font-size:18px;border-bottom:2px solid var(--line);padding-bottom:7px;margin:30px 0 12px;
 color:#0b1118;display:flex;align-items:center;gap:9px;flex-wrap:wrap;}}
.tm{{display:inline-flex;align-items:center;justify-content:center;background:var(--accent);color:#fff;
 border-radius:7px;padding:2px 9px;font-size:14px;font-weight:700;min-width:62px;}}
.hb{{display:inline-block;background:#e7eefc;color:var(--accent);font-size:12px;font-weight:700;padding:2px 8px;border-radius:5px;}}
.ev{{display:inline-block;background:#e9faef;color:#136c34;font-size:12px;font-weight:700;padding:2px 8px;border-radius:5px;}}
p{{margin:8px 0;}}
.talk{{background:#e7eefc;border-left:4px solid var(--accent);padding:9px 13px;border-radius:0 6px 6px 0;font-size:15px;}}
.fig{{text-align:center;margin:14px 0;}}
.fig img{{max-width:100%;border:1px solid var(--line);border-radius:8px;background:#fff;}}
.cap{{color:var(--mut);font-size:13.5px;margin-top:6px;text-align:left;}}
ul,ol{{margin:8px 0 8px 2px;padding-left:22px;}}li{{margin:6px 0;}}
.caveat{{background:#fff5e6;border:1px solid #e0a45c;border-radius:10px;padding:13px 17px;margin:24px 0 18px;font-size:14.5px;}}
.caveat .h{{font-weight:800;color:#9a4d00;display:block;margin-bottom:4px;}}.caveat b{{color:#9a4d00;}}
.discuss{{background:#e9faef;border:1px solid #8cc9a0;border-radius:10px;padding:15px 19px;margin:24px 0;}}
.discuss h2{{border:none;margin:0 0 8px;color:#136c34;}}
footer{{color:var(--mut);font-size:12.5px;margin-top:40px;border-top:1px solid var(--line);padding-top:12px;}}
</style></head><body>

<h1>1교시 강의 시나리오 <span style="font-size:14px;color:#475066;font-weight:normal;">(내부 논의용 초안)</span></h1>
<p class="sub">중소제조업 비전 AI 실습 · "모델을 직접 만져보고, 데이터가 답임을 체득" · 허블 핸즈온</p>

<div class="meta">
<span><b>대상</b> 중소제조업 실무자</span>
<span><b>시간</b> 40분 (3타임 중 1타임)</span>
<span><b>과제</b> 3종 — 분류·검출·분할</span>
<span><b>핵심</b> "직접 만지면 보인다 + 모델보다 데이터(양·특성)"</span>
</div>

<div class="flow">
<span class="step">3종 비교</span><span class="arr">→</span>
<span class="step">라벨링·추론</span><span class="arr">→</span>
<span class="step">임계값</span><span class="arr">→</span>
<span class="step">일부러 틀리기</span><span class="arr">→</span>
<span class="step">증강·데이터로 고치기</span><span class="arr">→</span>
<span class="step">정리</span>
</div>

{sec_html}

<div class="caveat">
<span class="h">⚠ 강사용 단서 (슬라이드엔 올리지 말 것)</span>
<ul>{CAVEATS}</ul>
</div>

<div class="discuss">
<h2>내부 논의 포인트</h2>
<ul>{DISCUSS}</ul>
</div>

<footer>1교시 전용 초안 · 기존 v3의 40분 핸즈온 골격 + 자체 실험 근거(MVTec/MSD) 주입 · 결함 이미지는 공개데이터 예시(허블 캡처로 교체 예정)</footer>
</body></html>"""

OUT.write_text(HTML, encoding="utf-8")
print("wrote", OUT, f"({len(HTML)//1024} KB)")
