# cuant_trading — Suite de herramientas para trading algorítmico

Herramientas de finanzas cuantitativas para el día a día. Cada una en su carpeta,
independiente, ejecutable desde terminal. Todo **local y gratis** (yfinance +
pandas/numpy/matplotlib/scipy). Datos de Yahoo Finance (retardo ~15 min, fin de día).

> ⚠️ **Aviso:** estas herramientas son de análisis y educación. No ejecutan
> órdenes ni mueven dinero. No son recomendación de inversión.

## Herramientas

| Carpeta | Qué hace | Ejemplo rápido |
|---------|----------|----------------|
| [`indicators/`](indicators/) | RSI, MACD, Bollinger, ATR, SMA + señal actual | `python indicators.py AAPL --save` |
| [`screener/`](screener/) | Rankea una watchlist por momentum/tendencia | `python screener.py AAPL MSFT NVDA SAB.MC` |
| [`backtester/`](backtester/) | Backtest SMA/RSI/Bollinger vs buy&hold | `python backtester.py AAPL --strategy sma --save` |
| [`position_sizer/`](position_sizer/) | Tamaño de posición por riesgo + Kelly | `python position_sizer.py --capital 10000 --risk 1 --entry 50 --stop 47` |
| [`portfolio_optimizer/`](portfolio_optimizer/) | Markowitz: máx Sharpe + frontera eficiente | `python portfolio_optimizer.py AAPL MSFT NVDA --save` |
| [`correlation/`](correlation/) | Matriz de correlación + diversificación | `python correlation.py AAPL TLT GLD --save` |
| [`signal_scanner/`](signal_scanner/) | Señales accionables BUY/SELL (cruces, RSI, breakouts) | `python signal_scanner.py AAPL SAB.MC --only-actionable` |
| [`sentiment/`](sentiment/) | Sentimiento FinBERT + NER, **noticias multi-fuente sin clave** (Yahoo + Google News + yfinance) con decaimiento temporal | `python sentiment_news.py AAPL` · `python news_feeds.py SAB.MC` |
| [`market_context/`](market_context/) | Fear & Greed (bolsa+cripto), VIX, fundamentales | `python market_context.py SAB.MC` |
| [`journal/`](journal/) | Diario de paper-trading para medir expectancy real | `python journal.py` |
| [`autogluon_forecast/`](autogluon_forecast/) | Forecast AutoML (AutoGluon/Chronos) | `python autogluon_forecast.py SAB.MC` |
| [`lstm_forecast/`](lstm_forecast/) | Forecast con red LSTM (PyTorch), bandas OOS | `python lstm_forecast.py SAB.MC` |
| [`neuralprophet_forecast/`](neuralprophet_forecast/) | Forecast NeuralProphet (AR-Net) | `python neuralprophet_forecast.py SAB.MC` |
| [`backtest_forecast/`](backtest_forecast/) | Mide si el forecast bate al azar (DA, Theil U2, cobertura) | `python backtest_forecast.py AAPL SAB.MC` |
| [`alpha_forecast/`](alpha_forecast/) | Dirección corto plazo (ML+Hurst) + volatilidad GARCH, con test de significancia | `python alpha_forecast.py MSFT` |
| [`conformal_forecast/`](conformal_forecast/) | Bandas **calibradas** (split conformal) con cobertura medida | `python conformal_forecast.py SAB.MC` |
| [`risk_metrics/`](risk_metrics/) | VaR/CVaR/drawdown/correlación de cartera | `python risk_metrics.py SAB.MC BBVA.MC IBE.MC` |
| [`factor_scorer/`](factor_scorer/) | **Modelo multi-factor (smart beta)**: rankea un universo por value/momentum/quality/low-vol | `python factor_scorer.py AAPL MSFT NVDA JPM XOM` |
| [`intraday/`](intraday/) | **Intradía**: VWAP + rango apertura + backtest Opening Range Breakout **con costes** (bruto vs neto) | `python intraday.py AAPL --interval 15m` |
| [`alpaca_paper/`](alpaca_paper/) | **Paper trading en vivo** (Alpaca, dinero ficticio, real-time IEX). Órdenes que disparas tú | `python alpaca_paper.py cuenta` |
| [`veredicto_backtest/`](veredicto_backtest/) | **Valida el Veredicto**: IC + quintiles + **Deflated Sharpe** (multiple-testing) | `python veredicto_backtest.py AAPL MSFT NVDA ...` |
| [`signal_engine/`](signal_engine/) | **Señales del sistema**: ranking COMPRAR/MANTENER/VENDER del Veredicto | `python signal_engine.py AAPL MSFT NVDA ...` |
| [`risk_manager/`](risk_manager/) | Señales → **plan de órdenes** (vol targeting + stop ATR + tope de riesgo) | `python risk_manager.py AAPL MSFT ... --capital 10000` |
| [`orchestrator/`](orchestrator/) | **El sistema completo**: señales→plan→paper→diario (ejecución que disparas tú) | `python orchestrator.py AAPL MSFT ... --capital 10000` |
| [`performance/`](performance/) | **¿Bates al mercado?** Equity + métricas + comparación **vs SPY** | `python performance.py` |
| [`pairs_trading/`](pairs_trading/) | **Arbitraje por cointegración** (Engle-Granger + half-life OU), market-neutral | `python pairs_trading.py KO PEP XOM CVX V MA` |
| [`hrp_portfolio/`](hrp_portfolio/) | **Cartera robusta** HRP + Ledoit-Wolf vs Markowitz (OOS) | `python hrp_portfolio.py AAPL MSFT NVDA ... GLD TLT` |
| [`evt_risk/`](evt_risk/) | **Riesgo de cola (crash)** con EVT/GPD: VaR/ES extremos vs normal | `python evt_risk.py SPY` |
| [`montecarlo/`](montecarlo/) | **Monte Carlo**: trayectorias de precio + robustez del sistema (prob. ruina) | `python montecarlo.py precio AAPL` |
| [`system_backtest/`](system_backtest/) | **Backtest de la estrategia completa** (top-N + rebal + costes) vs SPY | `python system_backtest.py AAPL MSFT NVDA ...` |
| [`alerts/`](alerts/) | Vigilancia de watchlist (RSI, vol, cruces, extremos) + log | `python alerts.py SAB.MC AAPL MSFT` |
| [`dashboard/`](dashboard/) | **UI web única con todas las funciones** (16 pestañas, incl. ★ Veredicto y 🪙 Veredicto Cripto) | `python dashboard.py` → http://127.0.0.1:7862 |

**Atajo Windows:** doble clic en `Lanzar_Dashboard.bat` (raíz del proyecto) — si el
dashboard ya corre solo abre el navegador; si no, lo arranca.

## Instalación

Usa el mismo entorno del proyecto (`ehu_ml`). Si arrancas de cero:

```bash
pip install yfinance pandas numpy matplotlib scipy
```

## Flujo de uso típico (día a día)

1. **`screener`** — barres tu watchlist por la mañana, sacas los 3-5 con mejor momentum.
2. **`indicators`** — miras en detalle los candidatos (RSI, MACD, posición vs Bollinger).
3. **`backtester`** — compruebas si tu regla de entrada/salida tiene ventaja histórica.
4. **`position_sizer`** — calculas cuántas acciones comprar para arriesgar solo el 1 %.
5. **`correlation`** / **`portfolio_optimizer`** — antes de añadir una posición, verificas
   que no duplica riesgo y ajustas pesos de la cartera.

## Notas

- Tickers de Yahoo: acciones `AAPL`, bolsa española `SAB.MC`/`BBVA.MC`, índices `^IBEX`,
  ETF `SPY`/`TLT`/`GLD`, cripto `BTC-EUR`.
- Las opciones `--save` guardan PNG en la carpeta de la herramienta.
- `--period`: `1y`, `2y`, `3y`, `5y`, `10y`, `max`.
