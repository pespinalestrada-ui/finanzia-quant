# alerts

Vigilancia de una watchlist: dispara avisos cuando algo cambia. Autónomo y ligero
(no carga modelos pesados) → pensado para ejecutarse periódicamente y registrar los
avisos en `alerts_log.csv`.

## Reglas
- **RSI(14)** < 30 (sobreventa) / > 70 (sobrecompra).
- **Pico de volatilidad**: vol realizada 5 d con z-score > 2 vs su media 60 d.
- **Movimiento brusco**: |retorno de hoy| > 2·σ diaria.
- **Cruce de medias**: el precio cruza hoy la SMA50 (al alza / a la baja).
- **Proximidad a extremos**: a < 2 % del máximo / mínimo de 52 semanas.

## Uso
```bash
python alerts.py AAPL MSFT SAB.MC
python alerts.py --file watchlist.txt          # un ticker por línea
python alerts.py AAPL MSFT --no-log            # sin escribir el log
```

## Ejecución periódica (Windows)
Usa `alertas.bat` (doble clic) o el Programador de tareas de Windows para lanzarlo
cada X horas en horario de mercado. Cada ejecución añade los avisos a
`alerts_log.csv` con marca de tiempo.

> No es asesoramiento de inversión.
