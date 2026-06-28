# orchestrator

El **sistema completo** en un sitio: señales → plan/riesgo → ejecución paper → diario.

Une las 4 piezas del flujo algorítmico:
1. `signal_engine` → **qué** operar (ranking del Veredicto).
2. `risk_manager` → **cuánto** (vol targeting + stop ATR + tope de riesgo).
3. `alpaca_paper` → **ejecuta** el plan en PAPER (dinero ficticio, real-time).
4. `journal` → **registra** cada orden con su nota de factores → expectancy real.

## Seguridad
`plan_de_hoy()` es solo lectura. `ejecutar()` manda órdenes (de papel) y por eso la
dispara **el usuario** (botón del dashboard, o CLI con `--ejecutar`). El asistente
nunca opera por su cuenta.

## Uso
```bash
python orchestrator.py AAPL MSFT NVDA GOOGL AMZN JPM XOM KO --capital 10000
python orchestrator.py --file watchlist.txt --capital 10000 --ejecutar   # manda paper
```
En el dashboard: pestaña **🤖 Sistema** (genera el plan, y con confirmación lo ejecuta).

## Recordatorio honesto
El backtest mostró que el Veredicto no supera el multiple-testing en líquidas. Esto
es un **marco de paper** para medir tu expectancy real con disciplina, no una máquina
de hacer dinero. Opera real solo si el paper confirma expectancy positiva.

> Dinero ficticio. No es recomendación de inversión.
