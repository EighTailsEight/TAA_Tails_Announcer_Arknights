@echo off
setlocal
cd /d "%~dp0"
net session >nul 2>&1
if not errorlevel 1 goto run
echo Requesting administrator rights (same as OBS admin mode)...
powershell -NoProfile -Command "Start-Process -FilePath '%~dp0TAA_TailsAnnouncerArknights.exe' -WorkingDirectory '%~dp0' -Verb RunAs"
exit /b

:run
start "" "%~dp0TAA_TailsAnnouncerArknights.exe"