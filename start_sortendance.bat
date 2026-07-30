@echo off
REM ============================================================================
REM SORT-tendance auto-start wrapper (Windows)
REM ============================================================================
REM Launches BOTH main.py and dashboard.py via start_sortendance.py.
REM
REM AUTO-START ON BOOT:
REM   1. Press Win+R, type:  shell:startup   -> Enter
REM   2. Copy this file (or a shortcut to it) into that folder
REM   3. (Optional) Right-click the shortcut -> Properties -> Run: Minimized
REM      if you don't want a console window on every boot
REM
REM To stop: press Ctrl+C in the console window.
REM ============================================================================

cd /d "%~dp0"
python start_sortendance.py
pause
