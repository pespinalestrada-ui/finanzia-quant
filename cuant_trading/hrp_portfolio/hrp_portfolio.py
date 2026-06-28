"""
hrp_portfolio — asignación robusta de cartera: HRP + Ledoit-Wolf.

Arregla la inestabilidad de Markowitz (la covarianza muestral es ruidosa y al
invertirla amplifica el error). Dos técnicas con base sólida:

  - Ledoit-Wolf shrinkage: Σ* = δ·F + (1−δ)·S, con δ óptimo cerrado. Encoge la
    covarianza hacia una estructura → min-variance estable.
  - Hierarchical Risk Parity (López de Prado): clustering jerárquico de la
    correlación + bisección recursiva con inverse-variance. NO invierte la matriz
    → robusto a singularidad. Bate a Markowitz fuera de muestra.

Compara HRP vs min-variance (Ledoit-Wolf) vs equiponderada FUERA DE MUESTRA
(pesos con la 1ª mitad, vol/Sharpe medidos en la 2ª). No es recomendación.

Uso:
    python hrp_portfolio.py AAPL MSFT NVDA GOOGL AMZN JPM XOM KO GLD TLT
"""
import argparse
import sys
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf


def _retornos(tickers, period="4y"):
    series = {}
    for tk in tickers:
        h = yf.Ticker(tk).history(period=period, auto_adjust=True)
        if h.empty:
            continue
        s = h["Close"].astype(float)
        s.index = pd.to_datetime(s.index).tz_localize(None)
        series[tk] = s
    px = pd.DataFrame(series).dropna()
    return px.pct_change().dropna()


# --- HRP (López de Prado) ----------------------------------------------------
def _ivp(cov):
    iv = 1.0 / np.diag(cov)
    return iv / iv.sum()


def _cluster_var(cov, items):
    c = cov.loc[items, items].values
    w = _ivp(c).reshape(-1, 1)
    return float((w.T @ c @ w)[0, 0])


def _quasi_diag(link):
    from scipy.cluster.hierarchy import to_tree
    link = link.astype(int)
    root = link[-1, 0], link[-1, 1]
    orden = []

    def recurse(node, n):
        if node < n:
            orden.append(node)
        else:
            izq, der = link[node - n, 0], link[node - n, 1]
            recurse(izq, n); recurse(der, n)
    n = link.shape[0] + 1
    recurse(n + link.shape[0] - 1, n)
    return orden


def hrp(retornos):
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    cov = retornos.cov()
    corr = retornos.corr()
    dist = np.sqrt(0.5 * (1 - corr))
    link = linkage(squareform(dist.values, checks=False), method="single")
    orden_idx = _quasi_diag(link)
    cols = corr.columns[orden_idx].tolist()
    w = pd.Series(1.0, index=cols)
    clusters = [cols]
    while clusters:
        nuevos = []
        for cl in clusters:
            if len(cl) <= 1:
                continue
            mit = len(cl) // 2
            izq, der = cl[:mit], cl[mit:]
            v_izq, v_der = _cluster_var(cov, izq), _cluster_var(cov, der)
            alpha = 1 - v_izq / (v_izq + v_der)
            w[izq] *= alpha
            w[der] *= (1 - alpha)
            nuevos += [izq, der]
        clusters = nuevos
    return w.reindex(retornos.columns).fillna(0.0)


def min_var_ledoit(retornos):
    """Min-variance con covarianza Ledoit-Wolf (estable)."""
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(retornos.values)
    cov = lw.covariance_
    inv = np.linalg.pinv(cov)
    ones = np.ones(cov.shape[0])
    w = inv @ ones / (ones @ inv @ ones)
    w = np.clip(w, 0, None)                  # sin cortos
    w = w / w.sum() if w.sum() > 0 else np.repeat(1/len(w), len(w))
    return pd.Series(w, index=retornos.columns)


def _metricas_oos(retornos_oos, w):
    r = retornos_oos.values @ w.reindex(retornos_oos.columns).fillna(0).values
    vol = float(np.std(r) * np.sqrt(252))
    sharpe = float(np.mean(r) / np.std(r) * np.sqrt(252)) if np.std(r) > 0 else float("nan")
    return vol, sharpe


def comparar(tickers, period="4y"):
    ret = _retornos(tickers, period)
    if ret.shape[1] < 3 or len(ret) < 252:
        return None
    n = len(ret)
    tr, te = ret.iloc[:n//2], ret.iloc[n//2:]
    metodos = {
        "HRP": hrp(tr),
        "Min-Var (Ledoit-Wolf)": min_var_ledoit(tr),
        "Equiponderada": pd.Series(np.repeat(1/ret.shape[1], ret.shape[1]), index=ret.columns),
    }
    filas, pesos = [], {}
    for nombre, w in metodos.items():
        vol, sharpe = _metricas_oos(te, w)
        filas.append({"Método": nombre, "Vol OOS (anual)": f"{vol*100:.1f}%",
                      "Sharpe OOS": round(sharpe, 2),
                      "Máx peso": f"{w.max()*100:.0f}%", "Nº efectivo": round(1/(w**2).sum(), 1)})
        pesos[nombre] = w
    return pd.DataFrame(filas), pesos, ret.columns.tolist()


def main():
    ap = argparse.ArgumentParser(description="Cartera robusta: HRP + Ledoit-Wolf vs equiponderada.")
    ap.add_argument("tickers", nargs="*",
                    default=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "JPM", "XOM", "KO", "GLD", "TLT"])
    ap.add_argument("--period", default="4y")
    a = ap.parse_args()

    print(f"\nComparando asignación sobre {len(a.tickers)} activos (out-of-sample)...")
    out = comparar(a.tickers, a.period)
    if out is None:
        print("Datos insuficientes."); return
    tabla, pesos, cols = out
    print("\n" + tabla.to_string(index=False))
    print("\nPesos HRP:")
    print((pesos["HRP"] * 100).round(1).to_string())
    print("\n> Vol/Sharpe medidos FUERA de muestra (pesos con 1ª mitad, evaluados en 2ª).")
    print("> HRP no invierte la covarianza → robusto. 'Nº efectivo' = diversificación real.")
    print("> No es recomendación de inversión.\n")


if __name__ == "__main__":
    main()
