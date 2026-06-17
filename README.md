# kosmes-lecture — 중소 제조업 AI 실습 강의 (강사 작업 repo)

> 이 저장소는 **강사 준비용**이다. 수강생에게 그대로 배포하지 않는다.
> 강의 계획은 `docs/lecture-notes.md`, 시나리오는 `docs/lecture-scenario.html`.

## 구성

| 경로 | 용도 |
|---|---|
| `docs/lecture-notes.md` | 3교시 기획 메모 (확정) |
| `docs/lecture-scenario.html` | 진행 시나리오 |
| `data/` | 공유 데이터셋 (MVTec AD = 1교시, Mendeley 영상 = 2교시). git 추적 안 함 |
| `01-model/` | **1교시** 허블 실습용 비교 모델 준비 (Ultralytics). 독립 uv 프로젝트 |
| `02-app/` | **2교시** 영상 추론 교보재 앱 (PySide6 + onnxruntime). 독립 uv 프로젝트 |

3교시(M.AX)는 별도 프로젝트 `~/workspaces/projects/max-agent` — 이 repo 밖.

## 두 코드 프로젝트는 독립 (venv 분리)

`01-model` 은 torch 로 무겁고, `02-app` 은 onnxruntime 로 가벼워야 패키징이 작다.
**venv 를 공유하지 않는다.** 1교시 모델(부품 결함)과 2교시 모델(작업자 검출)은
도메인이 달라 서로 인계 관계가 아니다.

```bash
# 1교시 모델 준비
cd 01-model && uv sync
uv run prepare/mvtec_to_yolo.py   # MVTec -> YOLO 변환
uv run train.py                   # configs/comparisons.yaml 의 비교군 학습
uv run export_onnx.py --weights weights/aug.pt

# 2교시 앱
cd 02-app && uv sync
uv run kosmes-app                 # 앱 실행
uv run --group dev pyinstaller app.spec   # 배포본 빌드
```

## 현재 상태

전부 **스캐폴드**(골격 + TODO). 실제 로직은 미구현.
