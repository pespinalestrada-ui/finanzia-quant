# kalman_hedge

Hedge ratio **dinámico** para pairs trading con **filtro de Kalman** (procesamiento
de señales). En `pairs_trading` el β es fijo (OLS), pero la relación entre dos activos
**deriva en el tiempo**. Kalman estima β y α día a día como un estado oculto que sigue
un paseo aleatorio:

    estado_t = estado_{t-1} + w_t          (β, α evolucionan)
    y_t = β_t·x_t + α_t + e_t               (e_t = spread)

Operas el z-score del spread (la innovación del filtro). Mejora directa sobre el β
estático. Implementación en numpy, sin dependencias nuevas.

## Uso
```bash
python kalman_hedge.py KO PEP
python kalman_hedge.py EWA EWC --period 5y
```
En el dashboard: pestaña **🛰️ Kalman (pairs)**.

## Por qué importa
Ejemplo: KO/PEP a 5 años da β OLS ≈ −0.03 (basura, la relación cambió de nivel),
mientras el β de Kalman ≈ 1.70 sigue la relación reciente. El dinámico no se queda
anclado al pasado. No es recomendación de inversión.
