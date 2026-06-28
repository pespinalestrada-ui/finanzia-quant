# risk_manager

El puente entre **qué operar** (`signal_engine`) y **ejecutar** (`alpaca_paper`):
convierte las señales en un **plan de órdenes con control de riesgo**.

Para cada señal accionable calcula CUÁNTO, con reglas de riesgo de verdad:
- **Volatility targeting** — peso ∝ vol_objetivo / vol_activo (las más volátiles pesan menos).
- **Máximo de posiciones** — solo las N señales de mayor convicción (|score|).
- **Tope de exposición** — la suma de pesos no pasa del 100 % (sin apalancar).
- **Stop por ATR** — stop = precio ∓ k·ATR → riesgo en € por posición.
- **Tope de riesgo diario** — avisa si el riesgo total planificado supera tu límite.

Vol GARCH(1,1) y ATR vienen de `position_sizer`.

## Uso
```bash
python risk_manager.py AAPL MSFT NVDA GOOGL AMZN JPM XOM KO --capital 10000
python risk_manager.py --file watchlist.txt --capital 20000 --target-vol 0.12 --max-pos 4
```

## Salida
Tabla con ticker, lado, precio, **acciones**, coste, peso %, vol anual, **stop** y
**riesgo €** por posición; más exposición total y riesgo total vs tu límite.
Es la **entrada del bucle de ejecución** (Alpaca paper). Las órdenes las disparas tú.

> No es recomendación de inversión.
