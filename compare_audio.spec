# PyInstaller spec: builds dist/CompareAudio/ (a folder with CompareAudio.exe).
# Build on the target OS (PyInstaller cannot cross-compile):  build_windows.bat
# ruff: noqa

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

datas = collect_data_files("pyqtgraph")
binaries = collect_dynamic_libs("sounddevice") + collect_dynamic_libs("soundfile")
datas += collect_data_files("_sounddevice_data") + collect_data_files("_soundfile_data")

a = Analysis(
    ["compare_audio/__main__.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=["soxr", "_sounddevice_data", "_soundfile_data"],
    excludes=[
        "matplotlib",
        "tkinter",
        "pytest",
        "tools",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.Qt3DCore",
        "PySide6.QtQuick",
        "PySide6.QtQml",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="CompareAudio",
    console=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="CompareAudio")
