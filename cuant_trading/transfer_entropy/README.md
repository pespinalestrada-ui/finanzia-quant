# transfer_entropy

Flujo de información y **lead-lag NO lineal** con **entropía de transferencia**
(Schreiber, 2000 — teoría de la información). La correlación de Pearson solo ve
dependencia lineal y es simétrica. La entropía de transferencia mide cuánta
información aporta el pasado de X sobre el futuro de Y, más allá del pasado de Y:

    TE(X→Y) = Σ p(y_{t+1},y_t,x_t)·log2[ p(y_{t+1}|y_t,x_t) / p(y_{t+1}|y_t) ]

Si TE(X→Y) > TE(Y→X), **X lidera a Y**. Detecta qué activo mueve a cuál (lo que la
correlación no distingue) y sirve para selección de features. Estimador por binning
(símbolos por cuantiles). Sin dependencias nuevas.

## Uso
```bash
python transfer_entropy.py SPY QQQ TLT GLD HYG XLF XLE
python transfer_entropy.py AAPL MSFT NVDA --bins 4 --period 3y
```
En el dashboard: pestaña **📡 Entropía (lead-lag)**.

## Salida
Matriz TE(X→Y) (heatmap) + ranking de **líderes** (emiten info) vs **seguidores**
(la reciben), por flujo neto en bits. No es recomendación de inversión.
