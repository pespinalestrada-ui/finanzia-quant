@echo off
REM Escanea la watchlist y registra avisos en alerts_log.csv
REM Requiere Python con yfinance/pandas/numpy en el PATH (o ajusta la ruta).
cd /d "%~dp0"
python alerts.py SAB.MC BBVA.MC IBE.MC AAPL MSFT NVDA
echo.
pause
