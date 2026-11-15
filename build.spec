# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller one-dir spec：TAA_TailsAnnouncerArknights

.build/venv/Scripts/python.exe -m PyInstaller build.spec --noconfirm
产物：dist/TAA_TailsAnnouncerArknights/TAA_TailsAnnouncerArknights.exe
packs/ 不打进 _internal，构建后放在 exe 同目录（见 build.bat）。
"""

from pathlib import Path

ROOT = Path(SPECPATH).resolve()
APP = ROOT / "app"
PROTO = ROOT / "proto"

a = Analysis(
    [str(APP / "gui.py")],
    pathex=[str(APP), str(PROTO)],
    binaries=[],
    datas=[
        # 冻结后 proto 模块的 __file__ 落在 _internal 根，模板/模型也要在根
        (str(PROTO / "models"), "models"),
        (str(PROTO / "templates"), "templates"),
    ],
    hiddenimports=["onnxruntime", "onnxruntime.capi._pybind_state"],
    hookspath=[],
    runtime_hooks=[],
    excludes=["setuptools", "pip", "pydoc_data", "unittest", "lib2to3",
              "idlelib", "turtle", "turtledemo", "distutils", "curses"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TAA_TailsAnnouncerArknights",
    debug=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="TAA_TailsAnnouncerArknights",
)