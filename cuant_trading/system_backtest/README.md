# system_backtest

Backtest de la **estrategia completa** del sistema (event-driven), no de una sola
regla. En cada rebalanceo calcula el score técnico del Veredicto (point-in-time, sin
leakage) para todo el universo, compra el **top-N** (equiponderado, long-only),
mantiene hasta el siguiente rebalanceo y **resta costes de rotación**. Compara la
curva de equity vs **SPY** buy & hold.

Métricas: retorno total, CAGR, Sharpe, máx drawdown, y si bate a SPY.

## Uso
```bash
python system_backtest.py AAPL MSFT NVDA GOOGL AMZN META JPM XOM KO WMT
python system_backtest.py --file watchlist.txt --top 3 --rebal 21 --coste-bps 10
```
En el dashboard: pestaña **📊 Backtest Sistema**.

## Aviso honesto (importante)
Un backtest que bate a SPY en UN universo y UN periodo **no prueba un edge**: puede
ser selección de tickers, el régimen de ese periodo, o suerte. El `veredicto_backtest`
mostró que el score **no supera el multiple-testing** (Deflated Sharpe bajo). Trata
este resultado como **medición**, no como promesa. Compáralo siempre con SPY (con
costes) y vigila el drawdown. No es recomendación de inversión.
