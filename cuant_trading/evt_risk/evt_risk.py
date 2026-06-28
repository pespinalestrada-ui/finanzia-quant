"""
evt_risk — riesgo de COLA con Teoría de Valores Extremos (EVT).

El VaR/CVaR histórico o normal SUBESTIMA los crashes (las colas de los retornos son
más gordas que una normal). EVT modela solo la cola con la distribución Pareto
Generalizada (GPD), método Peaks-Over-Threshold (POT):

  Para pérdidas que exceden un umbral u:  P(X>u+y | X>u) ≈ (1 + ξy/β)^(−1/ξ)

  - ξ (índice de cola): >0 colas gordas (típico en finanzas).
  - VaR_q  = u + (β/ξ)·[ ((n/Nu)(1−q))^(−ξ) − 1 ]
  - ES_q   = (VaR_q + β − ξu) / (1 − ξ)     (pérdida media SI superas el VaR)

Compara EVT vs histórico vs normal a 99% y 99.5% → se ve cuánto miente la normal en
la cola. No es recomendación de inversión.

Uso:
    python evt_risk.py SPY
    python evt_risk.py AAPL --period 8y --umbral 0.95
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


def _retornos(ticker, period="8y"):
    h = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if h.empty:
        raise ValueError(f"'{ticker}' sin datos.")
    return h["Close"].astype(float).pct_change().dropna()


def evt(retornos, u_q=0.95, niveles=(0.99, 0.995)):
    """Ajusta GPD a la cola de pérdidas (POT) y devuelve VaR/ES por método."""
    from scipy.stats import genpareto, norm
    perdidas = -retornos.values                 # pérdida positiva
    n = len(perdidas)
    u = np.quantile(perdidas, u_q)
    exc = perdidas[perdidas > u] - u
    Nu = len(exc)
    if Nu < 30:
        return None
    xi, _, beta = genpareto.fit(exc, floc=0)    # forma ξ, escala β
    mu, sd = perdidas.mean(), perdidas.std()

    filas = []
    for q in niveles:
        # EVT (POT)
        var_evt = u + (beta / xi) * (((n / Nu) * (1 - q)) ** (-xi) - 1) if xi != 0 else \
                  u + beta * np.log((n / Nu) / (1 - q))
        es_evt = (var_evt + beta - xi * u) / (1 - xi) if xi < 1 else np.nan
        # histórico
        var_hist = np.quantile(perdidas, q)
        es_hist = perdidas[perdidas >= var_hist].mean()
        # normal
        var_norm = mu + sd * norm.ppf(q)
        es_norm = mu + sd * norm.pdf(norm.ppf(q)) / (1 - q)
        filas.append({"Nivel": f"{q*100:.1f}%",
                      "VaR EVT": f"{var_evt*100:.2f}%", "ES EVT": f"{es_evt*100:.2f}%",
                      "VaR hist": f"{var_hist*100:.2f}%", "ES hist": f"{es_hist*100:.2f}%",
                      "VaR normal": f"{var_norm*100:.2f}%", "ES normal": f"{es_norm*100:.2f}%"})
    return {"xi": float(xi), "beta": float(beta), "u_pct": float(u*100),
            "Nu": Nu, "n": n, "tabla": pd.DataFrame(filas)}


def informe(ticker, res):
    if res is None:
        return f"{ticker}: pocas observaciones en la cola para ajustar GPD."
    cola = ("MUY gordas (riesgo de crash alto)" if res["xi"] > 0.3 else
            "gordas (típico en bolsa)" if res["xi"] > 0.1 else
            "moderadas")
    L = [f"=== EVT · {ticker} · {res['n']} días · cola sobre umbral {res['u_pct']:.2f}% ({res['Nu']} excesos) ===\n"]
    L.append(f"  Índice de cola ξ : {res['xi']:+.3f} → colas {cola}")
    L.append(f"  Escala β         : {res['beta']:.4f}\n")
    L.append(res["tabla"].to_string(index=False))
    L.append("\n> Compara las 3 columnas: la NORMAL casi siempre da el VaR/ES de cola más PEQUEÑO")
    L.append("> = subestima el crash. EVT y el histórico capturan mejor el riesgo extremo.")
    L.append("> ES = pérdida media SI superas el VaR (lo que de verdad duele). No es recomendación.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Riesgo de cola con EVT (GPD / Peaks-Over-Threshold).")
    ap.add_argument("ticker", nargs="?", default="SPY")
    ap.add_argument("--period", default="8y")
    ap.add_argument("--umbral", type=float, default=0.95, help="Cuantil de umbral u (0.95 = 5% peores).")
    a = ap.parse_args()
    r = _retornos(a.ticker, a.period)
    res = evt(r, a.umbral)
    print("\n" + informe(a.ticker.upper(), res) + "\n")


if __name__ == "__main__":
    main()
