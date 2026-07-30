@echo off
REM ============================================================================
REM SORT-tendance :: dashboard.py only launcher (Windows)
REM ============================================================================
REM Launches ONLY dashboard.py (no main). For running both, use:
REM   start_sortendance.bat
REM ============================================================================

cd /d "%~dp0"
python start_sortendance.py dashboard %*
pause
