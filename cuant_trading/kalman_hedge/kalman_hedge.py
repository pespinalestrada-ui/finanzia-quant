"""
kalman_hedge — hedge ratio DINÁMICO para pairs trading con filtro de Kalman.

En `pairs_trading` el hedge ratio β es fijo (OLS sobre toda la muestra). Pero la
relación entre dos activos DERIVA en el tiempo. El filtro de Kalman (procesamiento de
señales) estima un β y un α que cambian día a día, tratándolos como un estado oculto
que sigue un paseo aleatorio:

    estado_t = estado_{t-1} + w_t          (β y α evolucionan)
    y_t = β_t·x_t + α_t + e_t               (observación)

El error de observación e_t ES el spread (ya descontado el β dinámico). Operas su
z-score. Mejora directa sobre el β estático. Implementación en numpy, sin deps nuevas.

No es recomendación de inversión.

Uso:
    python kalman_hedge.py KO PEP
    python kalman_hedge.py EWA EWC --period 5y --delta 1e-4
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


def _serie(ticker, period):
    h = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if h.empty:
        raise ValueError(f"'{ticker}' sin datos.")
    s = h["Close"].astype(float)
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def kalman_hedge(x, y, delta=1e-4, Ve=1e-3):
    """
    Filtro de Kalman para [β, α] con x,y arrays alineados.
    delta controla cuánto deja moverse el estado (mayor = más adaptable).
    Devuelve (betas, alphas, spread) como arrays.
    """
    n = len(y)
    Vw = delta / (1 - delta) * np.eye(2)
    state = np.array([1.0, 0.0])              # β, α
    P = np.zeros((2, 2))
    betas, alphas, spread = np.zeros(n), np.zeros(n), np.zeros(n)
    for t in range(n):
        if t > 0:
            P = P + Vw                        # predict (estado = paseo aleatorio)
        H = np.array([x[t], 1.0])             # observación: y = β·x + α
        yhat = H @ state
        e = y[t] - yhat                       # innovación = spread
        S = H @ P @ H + Ve
        K = (P @ H) / S                       # ganancia de Kalman
        state = state + K * e
        P = P - np.outer(K, H) @ P
        betas[t], alphas[t], spread[t] = state[0], state[1], e
    return betas, alphas, spread


def analizar(a, b, period="5y", delta=1e-4):
    sa, sb = _serie(a, period), _serie(b, period)
    df = pd.DataFrame({"x": sa, "y": sb}).dropna()
    x, y = df["x"].values, df["y"].values
    betas, alphas, spread = kalman_hedge(x, y, delta)

    # z-score del spread (descarta el burn-in inicial del filtro)
    burn = min(60, len(spread) // 5)
    sp = pd.Series(spread, index=df.index)
    mu, sd = sp.iloc[burn:].mean(), sp.iloc[burn:].std()
    z = float((sp.iloc[-1] - mu) / sd) if sd > 0 else 0.0
    # β estático OLS para comparar
    import numpy as _np
    beta_ols = float(_np.polyfit(x, y, 1)[0])
    señal = ("🟢 LARGO spread (y barato vs x)" if z < -2 else
             "🔴 CORTO spread (y caro vs x)" if z > 2 else "🟡 fuera (|z|<2)")
    return {"a": a.upper(), "b": b.upper(), "fechas": df.index, "betas": betas,
            "spread": sp, "z": z, "beta_hoy": float(betas[-1]), "beta_ols": beta_ols,
            "mu": mu, "sd": sd, "señal": señal}


def _plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.plot(res["fechas"], res["betas"], color="tab:blue", lw=1.2, label="β dinámico (Kalman)")
    ax1.axhline(res["beta_ols"], color="gray", ls="--", lw=1, label=f"β estático OLS = {res['beta_ols']:.2f}")
    ax1.set_ylabel("Hedge ratio β"); ax1.legend(loc="best")
    ax1.set_title(f"{res['a']} / {res['b']} · hedge ratio dinámico vs estático")
    z = (res["spread"] - res["mu"]) / res["sd"]
    ax2.plot(res["fechas"], z, color="tab:purple", lw=1.0)
    for lvl, c in [(2, "tab:red"), (-2, "tab:green"), (0, "gray")]:
        ax2.axhline(lvl, color=c, ls="--", lw=0.8)
    ax2.set_ylabel("z-score del spread"); ax2.set_xlabel("Fecha")
    fig.tight_layout()
    return fig


def informe(res):
    return "\n".join([
        f"=== Kalman hedge · {res['a']} / {res['b']} ===\n",
        f"  β dinámico HOY   : {res['beta_hoy']:.3f}   (estático OLS: {res['beta_ols']:.3f})",
        f"  z-score spread   : {res['z']:+.2f}σ → {res['señal']}",
        "\n> El β se adapta solo a la deriva de la relación. Operas el z-score del spread",
        "> (innovación del filtro). Mejora sobre el β fijo de pairs_trading. No es recomendación."])


def main():
    ap = argparse.ArgumentParser(description="Hedge ratio dinámico (Kalman) para pairs trading.")
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--period", default="5y")
    ap.add_argument("--delta", type=float, default=1e-4)
    a = ap.parse_args()
    res = analizar(a.a, a.b, a.period, a.delta)
    print("\n" + informe(res) + "\n")


if __name__ == "__main__":
    main()
