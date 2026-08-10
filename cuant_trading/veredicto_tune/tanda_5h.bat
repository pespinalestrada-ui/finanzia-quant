@echo off
REM Una tanda de 10 configuraciones del Veredicto. Lo lanza el programador de
REM tareas de Windows cada 5 horas. Registro en data\tune_veredicto.log
REM Quitar la tarea:  schtasks /Delete /TN "FinanzIA - busqueda Veredicto" /F
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
"C:\Users\pablo\.conda\envs\ehu_ml\python.exe" tanda_programada.py
