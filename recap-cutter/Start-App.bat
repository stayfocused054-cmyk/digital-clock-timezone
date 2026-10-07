@echo off
title recapcut
cd /d "%~dp0"

set PY=python
where python >nul 2>nul || set PY=py
%PY% --version >nul 2>nul || (
  echo [!] Python nahi mila. https://www.python.org/downloads/ se install karo ^("Add to PATH" tick karna^).
  pause
  exit /b 1
)
where ffmpeg >nul 2>nul || echo [!] ffmpeg nahi mila. Install karo: winget install Gyan.FFmpeg  ^(phir ye window band karke dobara kholo^)

if not exist ".installed" (
  echo Pehli baar: packages install ho rahe hain, thoda time lagega...
  %PY% -m pip install -r requirements-app.txt || (pause & exit /b 1)
  %PY% -m pip install -r requirements-ai.txt || echo [!] AI packages install nahi hue - app basic mode me chalega.
  echo ok> .installed
)

echo.
echo App khul raha hai: http://127.0.0.1:7860   ^(band karne ke liye ye window band karo^)
%PY% -m recapcut app
pause
