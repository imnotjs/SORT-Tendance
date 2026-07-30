@echo off
REM ============================================================================
REM SORT-tendance :: main.py only launcher (Windows)
REM ============================================================================
REM Launches ONLY main.py (no dashboard). For running both, use:
REM   start_sortendance.bat
REM ============================================================================

cd /d "%~dp0"
python start_sortendance.py main %*
pause
