# pairs_trading

Arbitraje estadístico por **cointegración** (market-neutral) — de lo poco con edge
persistente. Dos activos cointegrados tienen un spread que revierte a la media:
largo el barato / corto el caro, cierras al revertir.

Matemática:
- **Engle-Granger** (`statsmodels.coint`): p<0.05 → cointegrados.
- **Hedge ratio β** por OLS: spread = B − β·A.
- **Half-life** de reversión (Ornstein-Uhlenbeck): `ln(2)/θ`, estimado regresando
  ΔX sobre X_lag. Corto = revierte rápido = tradeable.
- **z-score** del spread: z<−2 → largo spread; z>+2 → corto.

## Uso
```bash
python pairs_trading.py KO PEP XOM CVX V MA AAPL MSFT
python pairs_trading.py --par KO PEP
```
En el dashboard: pestaña **🔗 Pairs (cointegración)**.

> Opera los de p<0.05 cuando |z|>2. No es recomendación de inversión.
