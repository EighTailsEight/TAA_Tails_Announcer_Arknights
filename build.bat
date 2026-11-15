@echo off
setlocal
cd /d "%~dp0"

if not exist ".build\venv\Scripts\python.exe" (
    echo [1/3] creating build venv...
    python -m venv .build\venv || goto fail
    .build\venv\Scripts\python.exe -m pip install --disable-pip-version-check -q --upgrade pip || goto fail
    .build\venv\Scripts\python.exe -m pip install --disable-pip-version-check -q -r requirements.txt -r requirements-dev.txt || goto fail
)

echo [2/3] pyinstaller one-dir...
.build\venv\Scripts\python.exe -m PyInstaller build.spec --noconfirm --distpath dist --workpath .build\work || goto fail

echo [3/3] packs + launcher...
if not exist "dist\TAA_TailsAnnouncerArknights\packs" mkdir "dist\TAA_TailsAnnouncerArknights\packs"
if exist "packs\*.zip" xcopy /e /i /y /q "packs\*.zip" "dist\TAA_TailsAnnouncerArknights\packs" >nul || goto fail
copy /y "dist_assets\start.bat" "dist\TAA_TailsAnnouncerArknights\start.bat" >nul || goto fail

echo done: dist\TAA_TailsAnnouncerArknights
exit /b 0

:fail
echo BUILD FAILED
exit /b 1
