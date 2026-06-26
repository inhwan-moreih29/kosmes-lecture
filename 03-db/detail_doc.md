# 한빛정밀 사내 DB 안내 (읽기전용)

정밀 절삭부품 제조사. 4개 라인에서 샤프트·부싱·핀·기어·밸브바디를 가공·검사·출하한다.
PostgreSQL, **읽기전용(SELECT만)**. 조회 기준일(현재)은 **2026-06-30**. 데이터는 2025-01-01 ~ 2026-06-30.
시각 컬럼은 `timestamptz`(KST, +09). 날짜 컬럼은 `date`.

## 기준정보
- **product_families**(family_id, family_code, name, base_material) — 제품군. 5개: SH 샤프트, BU 부싱, PN 핀, GR 기어, VB 밸브바디.
- **products**(product_id, part_no, family_id→families, diameter_mm, length_mm, tolerance_class, tol_lower, tol_upper, material_input_g, unit_price)
  - 정밀부품이라 같은 제품군도 치수(지름·길이)마다 part_no가 다르다(=별도 SKU). 14개 품목.
  - `tol_lower/tol_upper` = 핵심치수(지름) 규격 하·상한(mm). `material_input_g` = 피스당 투입 원자재(g, 손실 포함). `unit_price` = 현재 단가(원).
- **customers**(customer_id, name, industry, country) — 고객사 5곳.
- **suppliers**(supplier_id, name, material_category, country) — 원자재 공급사 4곳.
- **production_lines**(line_id, name, location, commissioned_date) — 라인 4개.
- **equipment**(equipment_id, line_id→lines, name, model, install_date, last_maintenance_date) — 설비. 라인별 CNC/연삭기 + 4라인에 3D 측정기.

## 운영 트랜잭션
- **material_lots**(lot_id, supplier_id→suppliers, material_grade, received_date, qty_kg, qc_status) — 원자재 입고 로트. `qc_status` ∈ {합격, 보류, 재검}. 생산 투입은 합격 로트만.
- **production_runs**(run_id, line_id→lines, product_id→products, lot_id→material_lots, run_date, planned_qty, produced_qty) — 일별 생산 실적. 한 run = 특정 라인에서 특정 품목을 특정 로트 자재로 가공.
- **quality_inspections**(inspection_id, run_id→production_runs, inspected_at, equipment_id→equipment, result, defect_type, defect_qty, measured_value, spec_target) — 생산 run에 대한 품질검사. `result` ∈ {합격, 불합격}. `defect_type` ∈ {치수공차이탈, 스크래치, 표면조도불량, 버(burr)}. `measured_value`=측정 핵심치수, `spec_target`=목표치수. 검사는 표본추출이라 모든 run에 검사가 있진 않다.
- **equipment_downtime**(downtime_id, equipment_id→equipment, start_ts, end_ts, downtime_minutes, reason) — 설비 비가동 이력. `reason`은 정비/고장/점검 사유 텍스트.
- **safety_events**(event_id, line_id→lines, camera_id, event_type, severity, detected_at, duration_sec) — 라인 CCTV 안전 이벤트. `event_type` ∈ {위험구역침입, 체류초과, PPE미착용}, `severity` ∈ {상,중,하}.

## 영업 / 재고
- **sales_orders**(order_id, customer_id→customers, order_date, due_date, status) — 수주 헤더. `status` ∈ {완료, 진행중, 지연}.
- **order_items**(order_item_id, order_id→sales_orders, product_id→products, quantity, unit_price) — 수주 품목. `unit_price`=주문 시점 단가 스냅샷(매출 = quantity×unit_price).
- **inventory**(product_id→products PK, on_hand_qty, reorder_point, updated_at) — 품목별 **현재** 재고와 재주문점.
- **price_list**(price_id, product_id→products, effective_from, unit_price) — 단가 개정 이력(연 단위). products.unit_price는 최신값.
- **inventory_snapshots**(snapshot_id, snapshot_date, product_id→products, on_hand_qty) — 월말 재고 스냅샷(추이 분석용).

## 조인 경로 요약
- 라인별 생산: production_runs.line_id → production_lines
- 라인별 불량률: production_runs ↔ quality_inspections(run_id), 라인은 runs.line_id. 불량률 = 불합격(또는 defect_qty)/생산 기준.
- 불량 ↔ 설비: production_runs.line_id → equipment.line_id, 또는 inspections.equipment_id(측정 설비). 설비 상태는 equipment_downtime.
- 불량 ↔ 자재: production_runs.lot_id → material_lots → suppliers.
- 매출: order_items × sales_orders × customers (매출액 = quantity×unit_price).
- 재고 추이: inventory_snapshots(월말) vs inventory(현재) vs reorder_point.
