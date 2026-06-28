# hrp_portfolio

Asignación de cartera **robusta**: arregla la inestabilidad de Markowitz (la
covarianza muestral es ruidosa y al invertirla amplifica el error).

- **Ledoit-Wolf shrinkage**: Σ* = δ·F + (1−δ)·S, con δ óptimo cerrado → min-variance estable.
- **Hierarchical Risk Parity (López de Prado)**: clustering jerárquico de la
  correlación + bisección recursiva con inverse-variance. **No invierte la matriz**
  → robusto. Bate a Markowitz fuera de muestra.

Compara HRP vs Min-Var (Ledoit-Wolf) vs equiponderada **fuera de muestra** (pesos con
la 1ª mitad, vol/Sharpe medidos en la 2ª).

## Uso
```bash
python hrp_portfolio.py AAPL MSFT NVDA GOOGL AMZN JPM XOM KO GLD TLT
```
En el dashboard: pestaña **🧮 HRP Cartera**.

## Resultado típico
HRP y Min-Var baten a la equiponderada en Sharpe OOS con bastante menos volatilidad
(p.ej. Sharpe 1.87 vs 1.33, vol 10% vs 15%). No es recomendación de inversión.
