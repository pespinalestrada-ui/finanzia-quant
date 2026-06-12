# backtester

Backtest vectorizado de estrategias simples, comparado contra buy & hold.

## Estrategias
- **sma** — cruce de medias (compra cuando rápida > lenta).
- **rsi** — compra RSI<30, vende RSI>70.
- **bb** — compra bajo banda inferior de Bollinger, vende sobre la superior.

## Uso
```bash
python backtester.py AAPL --strategy sma --fast 50 --slow 200
python backtester.py SAB.MC --strategy rsi --period 5y --save
python backtester.py NVDA --strategy bb --cost 0.001 --save
```

## Métricas
Retorno total, CAGR, Sharpe anualizado, max drawdown, win rate, nº operaciones —
todo junto a la columna **Buy&Hold** para ver si la estrategia aporta.

## Detalles
- Sin look-ahead: la señal de hoy entra mañana (`shift(1)`).
- `--cost` aplica coste por operación (0.001 = 0.1 %).
- `--save` guarda la curva de equity (1 € inicial).

> Backtest ≠ futuro. Un buen backtest es condición necesaria, no suficiente.
