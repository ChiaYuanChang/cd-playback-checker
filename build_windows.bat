@echo off
REM Build the Windows app (run this on Windows, from the project folder).
REM Needs uv: https://docs.astral.sh/uv/  (or use: pip install -e . pyinstaller)
setlocal
uv sync --group build || goto :error
uv run pyinstaller --noconfirm compare_audio.spec || goto :error
echo.
echo Done: dist\CompareAudio\CompareAudio.exe
echo Copy the whole dist\CompareAudio folder to the test PC.
goto :eof
:error
echo Build failed.
exit /b 1
