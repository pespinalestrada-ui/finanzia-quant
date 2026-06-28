# evt_risk

Riesgo de **cola** (crash) con Teoría de Valores Extremos (EVT). El VaR/CVaR
histórico o normal **subestima los crashes**: las colas de los retornos son más
gordas que una normal.

EVT modela solo la cola con la **Pareto Generalizada (GPD)**, método
Peaks-Over-Threshold (POT):
- **ξ** (índice de cola): >0 = colas gordas (típico en bolsa).
- **VaR_q** = u + (β/ξ)·[((n/Nu)(1−q))^(−ξ) − 1]
- **ES_q** = (VaR_q + β − ξu)/(1 − ξ) — pérdida media SI superas el VaR.

Compara EVT vs histórico vs normal a 99% y 99.5%.

## Uso
```bash
python evt_risk.py SPY
python evt_risk.py AAPL --period 8y --umbral 0.95
```
En el dashboard: pestaña **📉 EVT Colas**.

## Resultado típico
SPY: ξ≈+0.23 (colas gordas). A 99.5%, VaR EVT ≈4.2% / ES ≈6% vs normal ≈2.8% / 3.2%
→ **la normal subestima el crash a la mitad**. No es recomendación de inversión.
