# signal_engine

Generador de **señales del sistema** (qué operar hoy). Corre el score técnico del
Veredicto —el mismo que valida `veredicto_backtest`, point-in-time— sobre una
watchlist y devuelve un ranking **COMPRAR / MANTENER / VENDER**.

Es la **entrada del bucle de ejecución** (sizing → Alpaca paper → diario). Opción
`--factor` añade la nota de factores como segunda capa de convicción.

## Uso
```bash
python signal_engine.py AAPL MSFT NVDA GOOGL AMZN JPM XOM KO
python signal_engine.py --file watchlist.txt --umbral 0.35 --factor
```

## Honestidad
El backtest mostró que el núcleo técnico **no supera el multiple-testing** en
large-caps líquidas. Trata estas señales como un **marco de PAPER** para medir tu
expectancy real, no como alfa garantizado.

> No es recomendación de inversión.
