# signal_scanner

Escanea una watchlist y devuelve **señales accionables** con sesgo BUY/SELL/NEUTRAL.

## Detecta
- **Golden / Death cross** — SMA50 cruza SMA200 en las últimas 5 sesiones.
- **RSI** — sobreventa (<30, compra) o sobrecompra (>70, venta).
- **Cruce MACD** — línea cruza su señal (reciente).
- **Breakout Bollinger** — cierre fuera de la banda (20,2).

## Uso
```bash
python signal_scanner.py AAPL MSFT NVDA SAB.MC BBVA.MC
python signal_scanner.py --file watchlist.txt --only-actionable
```

## Salida
Tabla ordenada por fuerza absoluta de señal:
- **Señales** — lista de las disparadas hoy.
- **Sesgo** — BUY si la suma neta es de compra, SELL si de venta, NEUTRAL si nada/empate.
- **Fuerza** — suma de señales (+compra, −venta).

`--only-actionable` oculta los NEUTRAL.

> Señales técnicas, no recomendación. Confírmalas con `indicators` y `backtester`.
