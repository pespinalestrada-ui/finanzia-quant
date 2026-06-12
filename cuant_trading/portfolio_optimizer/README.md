# portfolio_optimizer

Optimización media-varianza de Markowitz sobre una cesta de activos.

## Uso
```bash
python portfolio_optimizer.py AAPL MSFT NVDA GOOGL
python portfolio_optimizer.py SAB.MC BBVA.MC ITX.MC REP.MC --period 3y --rf 0.03 --save
```

## Qué devuelve
- **Cartera de máximo Sharpe** — mejor retorno por unidad de riesgo.
- **Cartera de mínima volatilidad** — la más defensiva.
- Pesos %, retorno esperado anual, volatilidad y Sharpe de cada una.
- Con `--save`: **frontera eficiente** (4000 carteras aleatorias coloreadas por Sharpe + las óptimas marcadas).

## Parámetros
- `--rf` tasa libre de riesgo anual (ej. `0.03` = 3 %).
- `--period` histórico para estimar retornos/covarianzas.

> Sin posiciones cortas (pesos 0–100 %). Optimiza sobre el pasado: úsalo como
> guía de asignación, no como verdad absoluta.
