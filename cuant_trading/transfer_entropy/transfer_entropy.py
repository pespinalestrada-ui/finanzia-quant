"""
transfer_entropy — flujo de información y lead-lag NO LINEAL (teoría de la información).

La correlación de Pearson solo ve dependencia LINEAL y es simétrica. La entropía de
transferencia (Schreiber, 2000) mide cuánta información aporta el PASADO de X sobre el
FUTURO de Y, más allá del propio pasado de Y → dependencia direccional y no lineal:

    TE(X→Y) = Σ p(y_{t+1}, y_t, x_t) · log2[ p(y_{t+1}|y_t, x_t) / p(y_{t+1}|y_t) ]

Si TE(X→Y) > TE(Y→X), X LIDERA a Y (su información fluye hacia Y). Útil para detectar
qué activo mueve a cuál (lo que la correlación no distingue) y para selección de
features. Estimador por binning (símbolos por cuantiles). Sin deps nuevas.

No es recomendación de inversión.

Uso:
    python transfer_entropy.py SPY QQQ TLT GLD HYG XLF XLE
    python transfer_entropy.py AAPL MSFT NVDA --bins 4 --period 3y
"""
import argparse
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf


def _retornos(tickers, period="3y"):
    series = {}
    for tk in tickers:
        h = yf.Ticker(tk).history(period=period, auto_adjust=True)
        if h.empty:
            continue
        s = h["Close"].astype(float)
        s.index = pd.to_datetime(s.index).tz_localize(None)
        series[tk] = s
    return pd.DataFrame(series).dropna().pct_change().dropna()


def _simbolizar(serie, k):
    return pd.qcut(serie.rank(method="first"), k, labels=False).values


def transfer_entropy(x, y, k=3):
    """TE(X→Y) en bits. x,y arrays de símbolos (0..k-1) alineados."""
    yn, yp, xp = y[1:], y[:-1], x[:-1]
    n = len(yn)
    P = np.zeros((k, k, k))                    # (y_{t+1}, y_t, x_t)
    for a, b, c in zip(yn, yp, xp):
        P[a, b, c] += 1
    P /= n
    P_ypxp = P.sum(0)                          # p(y_t, x_t)
    P_ynyp = P.sum(2)                          # p(y_{t+1}, y_t)
    P_yp = P_ypxp.sum(1)                       # p(y_t)
    te = 0.0
    for a in range(k):
        for b in range(k):
            for c in range(k):
                pj = P[a, b, c]
                if pj <= 0 or P_ypxp[b, c] <= 0 or P_yp[b] <= 0 or P_ynyp[a, b] <= 0:
                    continue
                num = pj / P_ypxp[b, c]         # p(y_{t+1}|y_t,x_t)
                den = P_ynyp[a, b] / P_yp[b]    # p(y_{t+1}|y_t)
                te += pj * np.log2(num / den)
    return max(0.0, float(te))


def matriz(tickers, period="3y", k=3):
    ret = _retornos(tickers, period)
    cols = ret.columns.tolist()
    sym = {c: _simbolizar(ret[c], k) for c in cols}
    n = len(cols)
    M = pd.DataFrame(0.0, index=cols, columns=cols)      # M[X,Y] = TE(X→Y)
    for i in cols:
        for j in cols:
            if i == j:
                continue
            M.loc[i, j] = transfer_entropy(sym[i], sym[j], k)
    # flujo neto: emite − recibe
    neto = (M.sum(1) - M.sum(0)).sort_values(ascending=False)
    return M, neto, ret


def _plot(M):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(M.values, cmap="viridis")
    ax.set_xticks(range(len(M))); ax.set_yticks(range(len(M)))
    ax.set_xticklabels(M.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(M.index, fontsize=8)
    ax.set_xlabel("→ hacia (Y)"); ax.set_ylabel("desde (X)")
    ax.set_title("Entropía de transferencia TE(X→Y) [bits] · fila lidera a columna")
    for i in range(len(M)):
        for j in range(len(M)):
            if i != j:
                ax.text(j, i, f"{M.values[i,j]:.2f}", ha="center", va="center", color="w", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def informe(M, neto):
    L = ["=== Flujo de información (entropía de transferencia) ===\n",
         "  LÍDERES de información (emiten más de lo que reciben):"]
    for tk, v in neto.head(3).items():
        L.append(f"    {tk:6} flujo neto {v:+.3f} bits")
    L.append("  SEGUIDORES (reciben más):")
    for tk, v in neto.tail(3).items():
        L.append(f"    {tk:6} flujo neto {v:+.3f} bits")
    L.append("\n> TE(X→Y)>TE(Y→X) ⇒ X lidera a Y (dependencia direccional NO lineal que la")
    L.append("> correlación no ve). Útil para lead-lag y selección de features. No es recomendación.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Entropía de transferencia (lead-lag no lineal).")
    ap.add_argument("tickers", nargs="*", default=["SPY", "QQQ", "TLT", "GLD", "HYG", "XLF", "XLE"])
    ap.add_argument("--bins", type=int, default=3)
    ap.add_argument("--period", default="3y")
    a = ap.parse_args()
    print(f"\nCalculando flujo de información entre {len(a.tickers)} activos...")
    M, neto, _ = matriz(a.tickers, a.period, a.bins)
    print("\n" + informe(M, neto) + "\n")


if __name__ == "__main__":
    main()
