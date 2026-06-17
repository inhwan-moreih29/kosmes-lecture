# PyInstaller 스펙 — 2교시 배포본(백업/수강생 배포용).
# torch 없이 onnxruntime 만 포함 -> 경량.
# 빌드:  uv run --group dev pyinstaller app.spec
#
# 주의: models/model.onnx 와 샘플 클립을 datas 로 동봉할지,
#       외부 경로로 둘지 결정. 동봉하면 단일 실행파일이 커진다.

# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ["src/app/main.py"],
    pathex=["src"],
    binaries=[],
    datas=[
        ("models/model.onnx", "models"),
        # ("samples", "samples"),  # 샘플 동봉 시 주석 해제
    ],
    hiddenimports=[],
    hookspath=[],
    excludes=["torch", "torchvision", "ultralytics"],  # 혹시라도 끌려오면 제외
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name="kosmes-vision-tool",
    console=False,
    onefile=True,
)
