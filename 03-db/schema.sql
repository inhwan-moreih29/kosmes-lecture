-- 한빛정밀 (가상 정밀부품 제조사) — 3교시 M.AX 시연용 스키마
-- PostgreSQL. 읽기전용 데모용. PK는 명시적으로 적재하므로 plain integer 사용.

DROP TABLE IF EXISTS price_list, inventory_snapshots, order_items, sales_orders, inventory, safety_events,
  equipment_downtime, quality_inspections, production_runs, material_lots,
  equipment, production_lines, suppliers, customers, products, product_families CASCADE;

-- ── 묶음 ① 기준정보 ──────────────────────────────────────────
CREATE TABLE product_families (
  family_id     int PRIMARY KEY,
  family_code   text NOT NULL,
  name          text NOT NULL,
  base_material text
);

CREATE TABLE products (
  product_id      int PRIMARY KEY,
  part_no         text NOT NULL UNIQUE,
  family_id       int NOT NULL REFERENCES product_families,
  diameter_mm     numeric(6,2),
  length_mm       numeric(6,2),
  tolerance_class text,               -- IT 등급 라벨
  tol_lower       numeric(8,3),       -- 규격 하한(mm, 핵심치수 기준)
  tol_upper       numeric(8,3),       -- 규격 상한(mm)
  material_input_g numeric(8,1),      -- 피스당 투입 원자재 중량(g, 절삭손실 포함)
  unit_price      numeric(10,2)
);

CREATE TABLE customers (
  customer_id int PRIMARY KEY,
  name        text NOT NULL,
  industry    text,
  country     text
);

CREATE TABLE suppliers (
  supplier_id       int PRIMARY KEY,
  name              text NOT NULL,
  material_category text,
  country           text
);

CREATE TABLE production_lines (
  line_id           int PRIMARY KEY,
  name              text NOT NULL,
  location          text,
  commissioned_date date
);

CREATE TABLE equipment (
  equipment_id          int PRIMARY KEY,
  line_id               int NOT NULL REFERENCES production_lines,
  name                  text NOT NULL,
  model                 text,
  install_date          date,
  last_maintenance_date date
);

-- ── 묶음 ② 운영 트랜잭션 ─────────────────────────────────────
CREATE TABLE material_lots (
  lot_id        int PRIMARY KEY,
  supplier_id   int NOT NULL REFERENCES suppliers,
  material_grade text,
  received_date date,
  qty_kg        numeric(10,1),
  qc_status     text
);

CREATE TABLE production_runs (
  run_id       int PRIMARY KEY,
  line_id      int NOT NULL REFERENCES production_lines,
  product_id   int NOT NULL REFERENCES products,
  lot_id       int REFERENCES material_lots,
  run_date     date NOT NULL,
  planned_qty  int,
  produced_qty int
);

CREATE TABLE quality_inspections (
  inspection_id int PRIMARY KEY,
  run_id        int NOT NULL REFERENCES production_runs,
  inspected_at  timestamptz,                 -- 검사 시각(생산 이후)
  equipment_id  int REFERENCES equipment,    -- 측정 설비(3D 측정기 등)
  result        text,                        -- 합격 / 불합격(치수 규격 이탈)
  defect_type   text,
  defect_qty    int,
  measured_value numeric(8,3),
  spec_target   numeric(8,3)
);

CREATE TABLE equipment_downtime (
  downtime_id      int PRIMARY KEY,
  equipment_id     int NOT NULL REFERENCES equipment,
  start_ts         timestamptz,
  end_ts           timestamptz,
  downtime_minutes int,
  reason           text
);

CREATE TABLE safety_events (
  event_id     int PRIMARY KEY,
  line_id      int NOT NULL REFERENCES production_lines,
  camera_id    text,
  event_type   text,
  severity     text,
  detected_at  timestamptz,
  duration_sec int
);

-- ── 묶음 ③ 영업 / 재고 ──────────────────────────────────────
CREATE TABLE sales_orders (
  order_id    int PRIMARY KEY,
  customer_id int NOT NULL REFERENCES customers,
  order_date  date,
  due_date    date,
  status      text
);

CREATE TABLE order_items (
  order_item_id int PRIMARY KEY,
  order_id      int NOT NULL REFERENCES sales_orders,
  product_id    int NOT NULL REFERENCES products,
  quantity      int,
  unit_price    numeric(10,2)
);

CREATE TABLE inventory (
  product_id    int PRIMARY KEY REFERENCES products,
  on_hand_qty   int,
  reorder_point int,
  updated_at    date
);

-- 단가 이력 (연 단위 개정). order_items.unit_price 는 주문 시점 단가 스냅샷,
-- products.unit_price 는 현재(최신) 단가.
CREATE TABLE price_list (
  price_id       int PRIMARY KEY,
  product_id     int NOT NULL REFERENCES products,
  effective_from date NOT NULL,
  unit_price     numeric(10,2)
);

-- 월말 재고 이력 (현재 inventory 는 최신 스냅샷, 이건 추이 분석용)
CREATE TABLE inventory_snapshots (
  snapshot_id   int PRIMARY KEY,
  snapshot_date date NOT NULL,
  product_id    int NOT NULL REFERENCES products,
  on_hand_qty   int
);

-- ── 인덱스 (조인/필터 키) ────────────────────────────────────
CREATE INDEX idx_runs_line     ON production_runs(line_id);
CREATE INDEX idx_runs_product  ON production_runs(product_id);
CREATE INDEX idx_runs_lot      ON production_runs(lot_id);
CREATE INDEX idx_runs_date     ON production_runs(run_date);
CREATE INDEX idx_insp_run      ON quality_inspections(run_id);
CREATE INDEX idx_down_eq       ON equipment_downtime(equipment_id);
CREATE INDEX idx_down_start    ON equipment_downtime(start_ts);
CREATE INDEX idx_safety_line   ON safety_events(line_id);
CREATE INDEX idx_safety_at     ON safety_events(detected_at);
CREATE INDEX idx_oi_order      ON order_items(order_id);
CREATE INDEX idx_oi_product    ON order_items(product_id);
CREATE INDEX idx_orders_cust   ON sales_orders(customer_id);
CREATE INDEX idx_lots_supplier ON material_lots(supplier_id);
CREATE INDEX idx_price_product  ON price_list(product_id, effective_from);
CREATE INDEX idx_snap_product   ON inventory_snapshots(product_id, snapshot_date);
