@echo off
chcp 65001 >nul
echo ========================================
echo  Build lingjing_explore.dll (ARC-AGI-3)
echo ========================================

call "D:\Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 (
  echo [FAIL] vcvars64 not found
  exit /b 1
)

cd /d "%~dp0"
if not exist bin mkdir bin

cl /nologo /O2 /EHsc /std:c++17 /LD /DLJ_EXPLORE_EXPORTS /I include ^
  /Febin\lingjing_explore.dll src\lingjing_explore.cpp /link /DLL

if errorlevel 1 (
  echo [FAIL] compile error
  exit /b 1
)

del /q lingjing_explore.obj 2>nul
del /q bin\lingjing_explore.exp 2>nul
del /q bin\lingjing_explore.lib 2>nul 2>nul

echo.
echo [OK] bin\lingjing_explore.dll
echo Test: python scripts\bench_hash.py
exit /b 0
