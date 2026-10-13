@echo off
call "D:\Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
cd /d "%~dp0"
if not exist bin mkdir bin
cl /nologo /O2 /EHsc /std:c++17 /LD /DLJ_EXPLORE_EXPORTS /I"%~dp0include" /Fe"%~dp0bin\lingjing_explore.dll" "%~dp0src\lingjing_explore.cpp" /link /DLL
if errorlevel 1 exit /b 1
del /q "%~dp0lingjing_explore.obj" 2>nul
del /q "%~dp0bin\lingjing_explore.exp" 2>nul
echo BUILD_OK
dir "%~dp0bin\lingjing_explore.dll"
