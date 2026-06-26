#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""한빛정밀 가상 데이터 생성기 -> seed.sql (stdout).
기간: 2025-01-01 ~ 2026-06-30 (18개월). 데이터 기준시점(오늘)=2026-06-30. 시드 고정.

심어둔 발견거리(시연 핵심):
  정밀가공 3라인의 노후 CNC(EQ id5, 2016설치)가 2025 하반기부터 다운타임 우상향(전조),
  2026-04-15 임계 돌파 -> 다운타임/치수공차이탈/안전경보 동시 급증.
  라인3은 최대고객(대성모터스) 납품 SH 샤프트 주력 -> 품질문제가 핵심 매출 위협.
  미끼: 2026-04 신규 공급사(대한특수강)로 SCM440 전환. 라인1에도 갔으나 무해 -> 자재 무죄(설비가 원인).

깊이:
  - 계절성+성장추세(월별 수요 배수, 설/휴가 저점), YoY 비교 가능. 한국 공휴일 휴무.
  - 노후설비 열화곡선: 정상 -> 2025-07 전조 -> 2026-01 악화 -> 2026-04-15 급성.
  - 단가 이력(price_list), order_items=주문시점 단가.
  - 자재 수지: 제품별 투입중량(material_input_g) -> 로트 qty_kg를 실제 소비 기반 산정(입고>=소비, 잔량>=0).
  - 재고 이력(inventory_snapshots, 월말) — 최종월 = 현재 inventory 와 일치.

정합성:
  - 물량수지: 출고=실수주합, 재고=기초+생산-출고. 자재 입고<=생산, 합격 로트만 투입.
  - 품질: 공차 상/하한, 검사 샘플링(전 런 1:1 아님)·무결점 다수·result 판정·측정기(eq8) 연결.
  - 급성기 라인3 가동저하로 런 감소. 미래데이터 없음(<=오늘). 시각 분/초 분산, 2교대(06~22시).
"""
import random, sys
from datetime import date, timedelta, datetime
from collections import defaultdict

random.seed(20250101)

START     = date(2025, 1, 1)
END       = date(2026, 6, 30)
TODAY     = date(2026, 6, 30)   # 기준시점 = 반기 마감(미래 데이터 없음)
TROUBLE   = date(2026, 4, 15)
LOT_NEW   = date(2026, 4, 10)
PRICE_REV = date(2026, 1, 1)
DEGRADE   = date(2025, 7, 1)
WORSEN    = date(2026, 1, 1)

HOLIDAYS = set()
for _y,_m,_d in [
    (2025,1,1),(2025,1,28),(2025,1,29),(2025,1,30),(2025,3,1),(2025,5,5),(2025,5,6),
    (2025,6,6),(2025,8,15),(2025,10,3),(2025,10,6),(2025,10,7),(2025,10,8),(2025,10,9),(2025,12,25),
    (2026,1,1),(2026,2,16),(2026,2,17),(2026,2,18),(2026,3,1),(2026,5,5),(2026,5,25),
    (2026,6,6),(2026,8,15),(2026,9,24),(2026,9,25),(2026,9,26),(2026,10,3),(2026,10,9),(2026,12,25)]:
    HOLIDAYS.add(date(_y,_m,_d))

def days(a, b):
    d = a
    while d <= b:
        yield d; d += timedelta(days=1)
def workdays(a, b):
    return [d for d in days(a, b) if d.weekday() != 6 and d not in HOLIDAYS]
def month_end(y, m):
    nxt = date(y+1,1,1) if m == 12 else date(y,m+1,1)
    return min(nxt - timedelta(days=1), END)
def q(s):
    return "'" + str(s).replace("'", "''") + "'"

OUT = []
def emit(sql): OUT.append(sql)
def insert(table, cols, rows):
    if not rows: return
    emit("INSERT INTO %s (%s) VALUES" % (table, ", ".join(cols)))
    vals = []
    for r in rows:
        cells = []
        for v in r:
            if v is None: cells.append("NULL")
            elif isinstance(v, float): cells.append(repr(round(v, 3)))
            elif isinstance(v, int): cells.append(str(v))
            else: cells.append(q(v))
        vals.append("(" + ", ".join(cells) + ")")
    emit(",\n".join(vals) + ";\n")

SEASON = {1:1.00,2:0.85,3:1.05,4:1.08,5:1.10,6:1.05,7:1.00,8:0.80,9:1.05,10:1.10,11:1.08,12:0.95}
def midx(d):  return (d.year-2025)*12 + (d.month-1)
def demand_factor(d):  return SEASON[d.month] * (1 + 0.006*midx(d))

# ── 기준정보 ────────────────────────────────────────────────
families = [
    (1,"SH","정밀 샤프트","탄소강 SCM440"),(2,"BU","부싱","청동 CAC406"),
    (3,"PN","커넥터 핀","스테인리스 SUS303"),(4,"GR","스퍼 기어","합금강 SCM415"),
    (5,"VB","밸브 바디","알루미늄 AL6061")]
insert("product_families", ["family_id","family_code","name","base_material"], families)

IT_HALF = {"IT5":0.006,"IT6":0.011,"IT7":0.018}
# (id, part_no, family, dia, len, tol, input_g, price2026)
prod_base = [
    (1,"SH-08-040-IT6",1, 8.0, 40.0,"IT6", 30,1840),(2,"SH-08-060-IT6",1, 8.0, 60.0,"IT6", 42,2090),
    (3,"SH-10-060-IT5",1,10.0, 60.0,"IT5", 58,2630),(4,"SH-10-080-IT5",1,10.0, 80.0,"IT5", 74,2970),
    (5,"SH-12-080-IT6",1,12.0, 80.0,"IT6", 96,3110),(6,"SH-12-100-IT6",1,12.0,100.0,"IT6",118,3440),
    (7,"BU-12-08-IT7", 2,12.0,  8.0,"IT7", 22, 970),(8,"BU-16-10-IT7", 2,16.0, 10.0,"IT7", 38,1170),
    (9,"PN-03-15-IT6", 3, 3.0, 15.0,"IT6",  4, 610),(10,"PN-05-20-IT6",3, 5.0, 20.0,"IT6",  8, 730),
    (11,"GR-20-M1-IT6",4,20.0,  0.0,"IT6", 85,3280),(12,"GR-30-M15-IT6",4,30.0, 0.0,"IT6",185,4130),
    (13,"VB-25-00-IT7",5,25.0,  0.0,"IT7",230,5180),(14,"VB-40-00-IT7",5,40.0, 0.0,"IT7",495,6770)]
products = []
DIA, PRICE26, PRICE25, FAMILY, TOL_U, TOL_L, INPUT_G = {},{},{},{},{},{},{}
for pid,pn,fam,dia,ln,tol,ing,price in prod_base:
    t = IT_HALF[tol]; tl,tu = round(dia-t,3), round(dia+t,3)
    products.append((pid,pn,fam,dia,ln,tol,tl,tu,ing,price))
    DIA[pid],PRICE26[pid],FAMILY[pid],INPUT_G[pid] = dia,price,fam,ing
    PRICE25[pid] = int(round(price*0.95/10.0))*10
    TOL_U[pid],TOL_L[pid] = tu,tl
insert("products", ["product_id","part_no","family_id","diameter_mm","length_mm",
                    "tolerance_class","tol_lower","tol_upper","material_input_g","unit_price"], products)

price_rows, pr_id = [], 0
for pid in sorted(PRICE26):
    pr_id+=1; price_rows.append((pr_id,pid,"2025-01-01",PRICE25[pid]))
    pr_id+=1; price_rows.append((pr_id,pid,"2026-01-01",PRICE26[pid]))
insert("price_list", ["price_id","product_id","effective_from","unit_price"], price_rows)
def price_at(pid, d):  return PRICE26[pid] if d >= PRICE_REV else PRICE25[pid]

insert("customers", ["customer_id","name","industry","country"], [
    (1,"대성모터스","자동차부품","대한민국"),(2,"한라공조","공조시스템","대한민국"),
    (3,"유진정밀기계","산업기계","대한민국"),(4,"세종일렉트로","전자부품","베트남"),
    (5,"광성기어","감속기","대한민국")])
insert("suppliers", ["supplier_id","name","material_category","country"], [
    (1,"신성특수강","탄소강/합금강","대한민국"),(2,"대한특수강","탄소강","대한민국"),
    (3,"동방비철","비철금속","대한민국"),(4,"대원스틸","스테인리스","대한민국")])
insert("production_lines", ["line_id","name","location","commissioned_date"], [
    (1,"정밀가공 1라인","1공장 A동","2019-03-11"),(2,"정밀가공 2라인","1공장 B동","2020-07-02"),
    (3,"정밀가공 3라인","1공장 C동","2016-05-20"),(4,"정밀가공 4라인","2공장","2021-09-15")])
insert("equipment", ["equipment_id","line_id","name","model","install_date","last_maintenance_date"], [
    (1,1,"CNC 선반 1호","DMG NLX2500","2019-03-01","2026-05-12"),
    (2,1,"원통 연삭기 1호","OKAMOTO GP","2019-03-01","2026-05-12"),
    (3,2,"CNC 선반 2호","DMG NLX2500","2020-06-20","2026-05-20"),
    (4,2,"원통 연삭기 2호","OKAMOTO GP","2020-06-20","2026-05-20"),
    (5,3,"CNC 선반 3호","MORI SL-25 (노후)","2016-05-10","2026-02-18"),
    (6,3,"원통 연삭기 3호","OKAMOTO GP","2016-05-10","2026-03-02"),
    (7,4,"CNC 머시닝센터 4호","DOOSAN DNM","2021-09-01","2026-05-25"),
    (8,4,"3D 측정기","ZEISS CONTURA","2021-09-01","2026-05-25")])
CULPRIT_EQ = 5
LINE_PRODUCTS = {1:[1,2,7,8], 2:[9,10,11,12], 3:[3,4,5,6], 4:[13,14]}
FAMILY_GRADE = {1:"SCM440",2:"CAC406",3:"SUS303",4:"SCM415",5:"AL6061"}

# ── 자재 로트 스케줄 (qty 는 소비 확정 후 산정) ──────────────
lots_info = {}                       # lot_id -> [supplier, grade, recv_date, status]
lots_by_grade = {g:[] for g in ["SCM440","SCM415","CAC406","AL6061","SUS303"]}
lot_id = 0
QC_CHOICES = ["합격"]*8 + ["보류","재검"]
def add_lot(supplier, grade, recv, status=None):
    global lot_id
    lot_id += 1
    if status is None: status = random.choice(QC_CHOICES)
    lots_info[lot_id] = [supplier, grade, recv, status]
    lots_by_grade[grade].append((lot_id, supplier, recv, status))
    return lot_id

d = date(2024,12,20)                 # SCM440 기존(신성) — 신규 전환 전까지
while d < LOT_NEW:
    add_lot(1,"SCM440",d); d += timedelta(days=15)
d = LOT_NEW                          # SCM440 신규(대한) 전환 — 첫 로트 보류
while d <= END:
    add_lot(2,"SCM440",d,"보류" if d==LOT_NEW else "합격"); d += timedelta(days=18)
d = date(2024,12,22)
while d <= END: add_lot(1,"SCM415",d); d += timedelta(days=24)
d = date(2024,12,21)
while d <= END: add_lot(3,"CAC406",d); d += timedelta(days=20)
d = date(2024,12,23)
while d <= END: add_lot(3,"AL6061",d); d += timedelta(days=22)
d = date(2024,12,24)
while d <= END: add_lot(4,"SUS303",d); d += timedelta(days=20)

def pick_lot(grade, run_date):
    cand = [(lid,sup,r) for (lid,sup,r,st) in lots_by_grade[grade] if r <= run_date and st=="합격"]
    if not cand:
        passed = [c for c in lots_by_grade[grade] if c[3]=="합격"]
        cand = [passed[0][:3]] if passed else [lots_by_grade[grade][0][:3]]
    if grade=="SCM440" and run_date>=LOT_NEW:
        new = [c for c in cand if c[1]==2]
        if new: return max(new, key=lambda c:c[2])[0]
    return max(cand, key=lambda c:c[2])[0]

# ── 생산 실적 + 품질 검사 + 자재 소비 ───────────────────────
runs, inspections = [], []
run_id = insp_id = 0
produced_by_pid = {p[0]:0 for p in prod_base}
consumed_by_lot = defaultdict(float)   # kg
BASE_QTY = {1:900,2:850,3:620,4:560,5:540,6:480,7:1500,8:1300,9:2200,10:2000,11:700,12:560,13:300,14:240}
COSMETIC = ["스크래치","표면조도불량","버(burr)"]

for d in workdays(START, END):
    f = demand_factor(d)
    for line_id, prods in LINE_PRODUCTS.items():
        for pid in prods:
            acute = (line_id==3 and d>=TROUBLE)
            if random.random() > (0.62 if acute else 0.82): continue
            run_id += 1
            planned = int(BASE_QTY[pid]*f*random.uniform(0.9,1.1))
            produced = int(planned*random.uniform(0.93,1.0))
            if acute: produced = int(produced*random.uniform(0.6,0.85))
            produced_by_pid[pid] += produced
            lot = pick_lot(FAMILY_GRADE[FAMILY[pid]], d)
            consumed_by_lot[lot] += produced * INPUT_G[pid] / 1000.0
            runs.append((run_id, line_id, pid, lot, d.isoformat(), planned, produced))

            if not (acute or random.random() < 0.88): continue   # 검사 샘플링
            insp_id += 1
            tgt = DIA[pid]
            if acute:
                prog = (d-TROUBLE).days / max(1,(END-TROUBLE).days)
                rate = max(0.0, 0.04 + 0.09*prog + random.uniform(-0.01,0.01))
                dtype = "치수공차이탈"
                measured = round(TOL_U[pid] + 0.005 + 0.04*prog + random.uniform(0,0.01), 3)
                defect_qty, result = int(round(produced*rate)), "불합격"
            else:
                tt = (TOL_U[pid]-TOL_L[pid])/2.0
                measured = round(tgt + random.uniform(-0.4*tt, 0.4*tt), 3)
                if random.random() < 0.45:
                    dtype, defect_qty = None, 0
                else:
                    dtype = random.choice(COSMETIC)
                    defect_qty = max(1, int(round(produced*random.uniform(0.006,0.022))))
                result = "합격"
            insp_day = min(d + timedelta(days=random.randint(0,1)), END)
            its = datetime(insp_day.year,insp_day.month,insp_day.day,
                           random.randint(9,18),random.randint(0,59),random.randint(0,59))
            inspections.append((insp_id, run_id, its.isoformat()+"+09", 8, result, dtype, defect_qty, measured, tgt))

# ── 자재 로트 qty_kg 산정 (소비 + 여유분 -> 입고>=소비). FK 위해 production_runs 보다 먼저 emit ──
lots = []
for lid in sorted(lots_info):
    sup, grade, recv, status = lots_info[lid]
    used = consumed_by_lot.get(lid, 0.0)
    qty = round(used*random.uniform(1.05,1.25),1) if used > 0 else round(random.uniform(800,2200),1)
    lots.append((lid, sup, grade, recv.isoformat(), qty, status))
insert("material_lots", ["lot_id","supplier_id","material_grade","received_date","qty_kg","qc_status"], lots)

insert("production_runs", ["run_id","line_id","product_id","lot_id","run_date","planned_qty","produced_qty"], runs)
insert("quality_inspections", ["inspection_id","run_id","inspected_at","equipment_id","result",
                               "defect_type","defect_qty","measured_value","spec_target"], inspections)

# ── 설비 정지 이력 (eq5 열화곡선) ───────────────────────────
downs = []
dt_id = 0
def add_down(eq, day, hour, minutes, reason):
    global dt_id
    dt_id += 1
    s = datetime(day.year,day.month,day.day,hour,random.randint(0,59),random.randint(0,59))
    e = s + timedelta(minutes=minutes)
    downs.append((dt_id, eq, s.isoformat()+"+09", e.isoformat()+"+09", minutes, reason))
for eq in range(1,9):
    d = START + timedelta(days=random.randint(8,20))
    while d <= END:
        add_down(eq, d, random.choice([7,9,13,15]), random.choice([60,90,120]), "정기 점검")
        d += timedelta(days=random.randint(28,38))
for _ in range(120):
    eq = random.randint(1,8)
    d = START + timedelta(days=random.randint(0,(END-START).days))
    add_down(eq, d, random.randint(6,21), random.choice([20,30,45]),
             random.choice(["공구 교체","칩 배출 막힘","센서 오류"]))
d = DEGRADE
while d < WORSEN:
    add_down(CULPRIT_EQ, d, random.randint(6,21), random.randint(30,70),
             random.choice(["스핀들 진동 경미","예방 정비","윤활 보충"])); d += timedelta(days=random.randint(20,30))
d = WORSEN
while d < date(2026,4,14):
    add_down(CULPRIT_EQ, d, random.randint(6,21), random.randint(50,110),
             random.choice(["스핀들 진동 경고","주축 베어링 마모 점검","치수 보정"])); d += timedelta(days=random.randint(9,14))
d = date(2026,4,14)
while d <= END:
    prog = (d-date(2026,4,14)).days / max(1,(END-date(2026,4,14)).days)
    add_down(CULPRIT_EQ, d, random.randint(6,21), max(20,int(40+220*prog+random.uniform(-20,30))),
             random.choice(["스핀들 진동 경고","주축 베어링 마모","치수 보정 정지","비상 정지"])); d += timedelta(days=random.randint(2,4))
insert("equipment_downtime", ["downtime_id","equipment_id","start_ts","end_ts","downtime_minutes","reason"], downs)

# ── 안전 경보 로그 (RISA) ───────────────────────────────────
sevs = []
CAM = {1:"CAM-L1-01",2:"CAM-L2-01",3:"CAM-L3-01",4:"CAM-L4-01"}
SAFE_TYPES = ["위험구역침입","PPE미착용","체류초과"]
def add_safe(line, day, etype, sev, dur):
    h=random.randint(6,22); m=random.randint(0,59); sec=random.randint(0,59)
    sevs.append((line, CAM[line], etype, sev,
                 datetime(day.year,day.month,day.day,h,m,sec).isoformat()+"+09", dur))
for d in workdays(START, END):
    for line in (1,2,3,4):
        if random.random() < 0.18:
            add_safe(line, d, random.choice(SAFE_TYPES), random.choice(["하","하","중"]), random.randint(3,12))
d = TROUBLE
while d <= END:
    if d.weekday()!=6:
        prog = (d-TROUBLE).days / max(1,(END-TROUBLE).days)
        for _ in range(1 + int(round(random.uniform(0,1)+2.5*prog))):
            add_safe(3, d, "위험구역침입", random.choice(["중","중","상"]), random.randint(15,70))
    d += timedelta(days=1)
sevs.sort(key=lambda r:r[4])
sevs = [(i+1,)+r for i,r in enumerate(sevs)]
insert("safety_events", ["event_id","line_id","camera_id","event_type","severity","detected_at","duration_sec"], sevs)

# ── 영업(수주/품목) — 출고를 "생산량 비례 월별 배분" -> 누적출고<=누적생산 보장 ──
PRODUCT_BUYERS = {1:[1],2:[1],3:[1],4:[1],5:[1],6:[1],7:[2],8:[3,5],9:[4],10:[4],11:[5,3],12:[5,3],13:[2],14:[2]}
SHIP_RATIO = {p[0]: random.uniform(0.975,0.995) for p in prod_base}
SHIP_RATIO[8] = 0.978
shipped = {pid:int(produced_by_pid[pid]*SHIP_RATIO[pid]) for pid in produced_by_pid}
cust_prod_demand = {}
for pid, buyers in PRODUCT_BUYERS.items():
    each = shipped[pid] // len(buyers)
    for i,c in enumerate(buyers):
        cust_prod_demand[(c,pid)] = each if i<len(buyers)-1 else shipped[pid]-each*(len(buyers)-1)

# 월별 생산량(배분 가중치) + 월별 영업일
months = [(y,m) for y in (2025,2026) for m in range(1,13) if START <= date(y,m,1) <= END]
yms = ["%04d-%02d" % (y,m) for (y,m) in months]
prodM = defaultdict(int)
for (rid,lid,pid,lot,rd,pl,pq) in runs: prodM[(rd[:7],pid)] += pq
wd_by_ym = defaultdict(list)
for d in workdays(START, END): wd_by_ym[d.isoformat()[:7]].append(d)

# (고객,제품) 수요를 생산월 비례로 쪼갬 -> (고객,월) 단위 수주로 묶음
by_cym = defaultdict(list)   # (c, ym) -> [(pid, qty)]
for (c,pid), dem in cust_prod_demand.items():
    if dem <= 0: continue
    mp = [(ym, prodM[(ym,pid)]) for ym in yms if prodM[(ym,pid)] > 0]
    tot = sum(s for _,s in mp); alloc = 0
    for j,(ym,share) in enumerate(mp):
        if j == len(mp)-1:
            qv = dem - alloc                          # 마지막 달이 잔량 흡수
        else:
            jf = 1.0 if pid == 8 else random.uniform(0.65, 1.35)   # 미끼 pid8은 지터 없이(순감 음수 방지)
            qv = min(dem-alloc, max(1, int(dem*share/tot*jf)))     # 월별 지터
        alloc += qv
        if qv > 0: by_cym[(c,ym)].append((pid, qv))

orders, items = [], []
o_id = oi_id = 0
o_date = {}
for (c,ym) in sorted(by_cym.keys(), key=lambda k:(k[1],k[0])):
    dt = random.choice(wd_by_ym[ym])              # 그 달 영업일 중 하루
    o_id += 1; o_date[o_id] = dt
    due = dt + timedelta(days=random.randint(20,45))
    if due <= TODAY:
        status = "지연" if (c==1 and dt>=date(2026,5,1) and random.random()<0.40) else "완료"
    else:
        status = "진행중"
    orders.append((o_id, c, dt.isoformat(), due.isoformat(), status))
    for (pid,qv) in by_cym[(c,ym)]:
        oi_id += 1; items.append((oi_id, o_id, pid, qv, price_at(pid, dt)))
insert("sales_orders", ["order_id","customer_id","order_date","due_date","status"], orders)
insert("order_items", ["order_item_id","order_id","product_id","quantity","unit_price"], items)

# ── 재고 (현재 스냅샷) + 월말 재고 이력 ─────────────────────
ordered_by_pid = defaultdict(int)
ordM = defaultdict(int)
for (_oi,oo,pid,qty,_p) in items:
    ordered_by_pid[pid] += qty
    ordM[(o_date[oo].isoformat()[:7], pid)] += qty
# 월별 누적 net(생산-출고) 시퀀스 -> 기초재고가 최저점 흡수 (on_hand 항상 양수)
net_seq = {}
for pid in sorted(produced_by_pid):
    cp = co = 0; seq = []
    for (y,m) in months:
        ym = "%04d-%02d" % (y,m); cp += prodM[(ym,pid)]; co += ordM[(ym,pid)]; seq.append(cp-co)
    net_seq[pid] = seq
opening, reorder = {}, {}
for pid in sorted(produced_by_pid):
    monthly = ordered_by_pid[pid]/18.0
    reorder[pid] = max(500, int(round(monthly*0.6/100.0))*100)
    base = random.randint(500,1500) if pid==8 else int(reorder[pid]*random.uniform(1.5,2.2))
    opening[pid] = base - min(0, min(net_seq[pid]))     # 최저 누적순감을 메워 음수 방지(미끼 pid8은 얇게)
inv = [(pid, opening[pid]+net_seq[pid][-1], reorder[pid], "2026-06-30")
       for pid in sorted(produced_by_pid)]
insert("inventory", ["product_id","on_hand_qty","reorder_point","updated_at"], inv)

snaps, sid = [], 0
for pid in sorted(produced_by_pid):
    for i,(y,m) in enumerate(months):
        sid += 1
        snaps.append((sid, month_end(y,m).isoformat(), pid, opening[pid]+net_seq[pid][i]))
insert("inventory_snapshots", ["snapshot_id","snapshot_date","product_id","on_hand_qty"], snaps)

print("\n".join(OUT))
print("-- rows: runs=%d insp=%d downtime=%d safety=%d orders=%d items=%d lots=%d price=%d snaps=%d"
      % (len(runs),len(inspections),len(downs),len(sevs),len(orders),len(items),len(lots),len(price_rows),len(snaps)), file=sys.stderr)
