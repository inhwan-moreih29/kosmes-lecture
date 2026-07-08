# -*- coding: utf-8 -*-
"""슬라이드 11(HIERARCHY)을 '에폭 진행 인터랙티브 슬라이더'로 재구성.

10페이지(스스로 학습)처럼 ▶재생/슬라이더로 학습 진행을 끌면, 같은 너트 결함을 처리하는
신경망 내부 특징맵(얕은/중간/깊은)이 실제 체크포인트로 스냅되어 바뀌고, 결함 확신도 바가
0→7→33→69%로 오른다. gen_featprog_epochs.py 산출물 12장을 JS 배열에 base64 임베드.
"""
import base64
from pathlib import Path

DOCS = Path(__file__).resolve().parent
DECK = DOCS / "lesson1-theory-vision.html"
FEAT = DOCS.parent / "outputs" / "model" / "eval" / "lesson_epochprog"

def b64(name):
    return "data:image/png;base64," + base64.b64encode((FEAT / name).read_bytes()).decode()

# (slider위치, 에폭라벨, 확신도%, 판정클래스, 판정텍스트, 파일태그)
STAGES = [
    (0,   "학습 전",  "0에폭 · 랜덤",  0,  "bad",  "검출 없음", "e0"),
    (33,  "학습 초반", "50에폭",        7,  "bad",  "아직 통과", "e50"),
    (66,  "학습 중반", "70에폭",        33, "mid",  "검출!",     "e70"),
    (100, "학습 완료", "100에폭",       69, "good", "확신",      "e100"),
]

# JS EP 배열
js_items = []
for pos, lab, sub, conf, cls, vtxt, tag in STAGES:
    js_items.append(
        "{pos:%d,lab:%r,sub:%r,conf:%d,cls:%r,v:%r,"
        "sh:'%s',mid:'%s',dp:'%s'}" % (
            pos, lab, sub, conf, cls, vtxt,
            b64(f"{tag}_shallow.png"), b64(f"{tag}_mid.png"), b64(f"{tag}_deep.png"))
    )
EP_JS = "const EP=[\n  " + ",\n  ".join(js_items) + "\n];"

section = '''<!-- 8 HIERARCHY (epoch progression · interactive slider) -->
<section class="slide">
 <span class="tag live"><span class="dot"></span>라이브 데모 · 학습 진행</span>
 <h2>실제 신경망 속 — <span class="em">학습할수록 필터가 또렷</span>해진다</h2>
 <p class="lead" style="margin:6px 0 8px">룰 기반은 사람이 필터를 고정했지만, 딥러닝은 <b>데이터로 스스로</b> 만듭니다. <b>▶ 재생</b>하거나 슬라이더를 끌어, 같은 너트 결함을 처리하는 신경망 내부가 학습에 따라 어떻게 변하는지 보세요. <span class="mut">(합성 없이 실제 학습 체크포인트·특징맵)</span></p>
 <div class="row" style="gap:16px;align-items:center;margin-bottom:10px">
  <button class="btn sel" id="epPlay" style="font-size:16px;padding:10px 20px">▶ 재생</button>
  <div class="col"><div class="label" style="margin-bottom:3px">학습 진행 = <b id="epLab" style="color:var(--accent-d)">학습 전 · 0에폭</b></div><input id="epP" type="range" min="0" max="100" value="0"></div>
 </div>
 <div class="row fill" style="align-items:stretch;gap:16px">
  <div class="col"><div class="demo" style="height:100%">
   <div class="label">신경망 내부 특징맵 (같은 결함 이미지)</div>
   <div class="row" style="gap:14px;margin-top:10px;justify-content:center">
    <div style="text-align:center"><div class="label">얕은 층<span style="display:block;font-weight:700;color:var(--mut)">엣지·윤곽</span></div><img id="epShallow" class="epimg"></div>
    <div style="text-align:center"><div class="label">중간 층<span style="display:block;font-weight:700;color:var(--mut)">부분 패턴</span></div><img id="epMid" class="epimg"></div>
    <div style="text-align:center"><div class="label">깊은 층<span style="display:block;font-weight:700;color:var(--mut)">집중 위치</span></div><img id="epDeep" class="epimg"></div>
   </div>
   <p class="hint" style="margin-top:12px;text-align:center"><b>얕은 층</b>은 처음부터 엣지를 얼추 잡지만, <b>깊은 층</b>은 학습이 쌓일수록 흐릿한 반응이 <b>결함 위치로 집중</b>됩니다.</p>
  </div></div>
  <div class="col" style="flex:0 0 330px"><div class="demo" style="height:100%">
   <div class="label">판정 — 결함 확신도</div>
   <div id="epVerBig" style="font-size:56px;font-weight:900;line-height:1.05;margin:8px 0 2px">0%</div>
   <div id="epVerTxt" class="big" style="font-weight:800;margin-bottom:12px">검출 없음</div>
   <div class="meter" style="height:26px;position:relative"><i id="epBarI" style="width:0%;background:linear-gradient(90deg,#0e9f6e,#5fcf9e)"></i>
     <span style="position:absolute;left:25%;top:-4px;bottom:-4px;width:2px;background:#0e1622"></span></div>
   <div class="hint" style="margin-top:4px">↑ 세로선 = 판정 기준 25% (넘으면 '불량 검출')</div>
   <div class="term" style="margin-top:14px">사람이 도장을 짜지 않아도 <b>데이터가 필터를 만든다</b> — 그게 '학습'입니다. 룰 기반의 고정 도장과 결정적으로 다른 지점.</div>
  </div></div>
 </div>
</section>
'''

css_add = """.epmx .fpimg{width:120px;height:120px;}
.epimg{width:150px;height:150px;border-radius:8px;object-fit:cover;background:#0e1622;box-shadow:0 2px 8px rgba(20,40,90,.12);display:block;margin-top:4px;}
"""

demo_js = '''
/* DEMO epoch progression — scrub learning, feature maps snap to real checkpoints */
(function(){
 const P=document.getElementById('epP');if(!P)return;
 __EP__
 const play=document.getElementById('epPlay'),lab=document.getElementById('epLab'),
   sh=document.getElementById('epShallow'),mid=document.getElementById('epMid'),dp=document.getElementById('epDeep'),
   barI=document.getElementById('epBarI'),verBig=document.getElementById('epVerBig'),verTxt=document.getElementById('epVerTxt');
 const COL={bad:'#c0392b',mid:'var(--warn)',good:'var(--good)'};
 let curIdx=-1;
 function nearest(pos){let bi=0,bd=1e9;EP.forEach((s,i)=>{const d=Math.abs(s.pos-pos);if(d<bd){bd=d;bi=i;}});return bi;}
 function lerpConf(pos){for(let i=0;i<EP.length-1;i++){const a=EP[i],b=EP[i+1];
   if(pos>=a.pos&&pos<=b.pos){const t=(pos-a.pos)/(b.pos-a.pos);return a.conf+(b.conf-a.conf)*t;}}
   return EP[EP.length-1].conf;}
 function draw(){const pos=+P.value,idx=nearest(pos),s=EP[idx];
  if(idx!==curIdx){curIdx=idx;sh.src=s.sh;mid.src=s.mid;dp.src=s.dp;
    lab.textContent=s.lab+' · '+s.sub;verTxt.textContent=s.v;verTxt.style.color=COL[s.cls];verBig.style.color=COL[s.cls];}
  const c=lerpConf(pos);barI.style.width=c.toFixed(0)+'%';verBig.textContent=Math.round(c)+'%';}
 let tm=null;
 function stop(){if(tm){clearInterval(tm);tm=null;}play.textContent='▶ 재생';play.classList.add('sel');}
 function go(){if(tm){stop();return;}if(+P.value>=100)P.value=0;
   play.textContent='⏸ 재생 중…';play.classList.remove('sel');
   tm=setInterval(()=>{let v=+P.value+2;if(v>=100){P.value=100;draw();stop();return;}P.value=v;draw();},55);}
 play.onclick=go;P.oninput=()=>{stop();draw();};
 draw();
})();
'''.replace("__EP__", EP_JS)

html = DECK.read_text(encoding="utf-8")

# CSS: .epimg 없으면 추가 (matrix용 .epmx .fpimg 라인 뒤)
if ".epimg{" not in html:
    html = html.replace(".epmx .fpimg{width:120px;height:120px;}\n", css_add, 1)

# 섹션 교체
start = -1
for m in ["<!-- 8 HIERARCHY (epoch progression · interactive slider) -->",
          "<!-- 8 HIERARCHY (real feature maps · epoch progression, nut) -->",
          "<!-- 8 HIERARCHY (real feature maps · training progression) -->",
          "<!-- 8 HIERARCHY (real feature maps) -->"]:
    start = html.find(m)
    if start != -1:
        break
assert start != -1, "HIERARCHY 마커 없음"
end = html.find("<!-- 9 DIVIDER 3/4 -->", start)
assert end != -1
html = html[:start] + section + "\n" + html[end:]

# JS 데모 추가 (self-learning IIFE 뒤, </script> 앞) — 중복 방지
if "DEMO epoch progression" not in html:
    html = html.replace("})();\n</script>", "})();\n" + demo_js + "</script>", 1)

DECK.write_text(html, encoding="utf-8")
print("slide 11 → interactive epoch slider. deck:", len(html), "bytes")
