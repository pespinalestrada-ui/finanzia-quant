# intraday

Análisis y **backtest intradía con costes** — para DESARROLLAR y VALIDAR un método
intradía gratis, antes de arriesgar nada. Intradía es otra disciplina: barras de
minutos, VWAP, rango de apertura, y sobre todo **costes** (comisión + spread +
slippage) que se comen el edge.

## Qué hace
- **VWAP por sesión** (referencia institucional intradía) + **rango de apertura** (ORB)
  + **ATR intradía** + RSI.
- **`snapshot(ticker)`** — foto del estado de hoy: precio vs VWAP, dentro/fuera del
  rango de apertura, sesgo intradía.
- **`backtest_orb()`** — Opening Range Breakout con **MODELO DE COSTES**: muestra la
  expectancy **bruta vs neta**. Sin coste, un backtest intradía miente.

## Uso
```bash
python intraday.py AAPL
python intraday.py SAB.MC --interval 15m --or-min 30 --coste-bps 8
```

## Límite de los datos (importante)
yfinance intradía es **gratis pero con retraso ~15 min** y poco histórico
(1m → 7 días, 5m/15m → 60 días). Sirve para **backtestear y desarrollar**, NO para
ejecutar en vivo. Para tiempo real + órdenes simuladas, el siguiente paso es
**Alpaca paper trading** (gratis, API; las órdenes las disparas tú).

## La verdad intradía
El retail pierde sobre todo por **costes y velocidad** (HFT/creadores de mercado
dominan). Ejemplo medido: ORB sobre AAPL 15m, 50 operaciones → expectancy bruta ~0%,
**neta −0.07%/op** tras costes → sin edge. Reportarlo es lo correcto: no operes real
hasta que el backtest CON costes (y luego el paper) den expectancy positiva.

> No es recomendación de inversión.
