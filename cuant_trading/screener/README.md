# screener

Escanea una watchlist y la rankea por un score de momentum + tendencia.

## Uso
```bash
python screener.py AAPL MSFT NVDA SAB.MC BBVA.MC
python screener.py --file watchlist.txt --sort momentum
```
`watchlist.txt`: un ticker por línea (`#` para comentarios). Sin argumentos usa una lista por defecto.

## Columnas
- **Mom1m / Mom3m** — momentum (rentabilidad %) a 1 y 3 meses.
- **RSI** — RSI(14) actual.
- **VolAnual** — volatilidad anualizada %.
- **vsSMA50** — distancia % al precio respecto de su media de 50 sesiones.
- **Tend** — ALC/BAJ según SMA50 vs SMA200.
- **Score** — momentum medio + bonus tendencia − penalización por RSI extremo.

## Orden
`--sort score` (defecto) · `momentum` · `rsi`.
