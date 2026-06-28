"""
rmt_clean — limpieza de la matriz de correlación con Random Matrix Theory (econofísica).

La correlación de N activos estimada con T datos está dominada por RUIDO. La teoría de
matrices aleatorias (Marchenko-Pastur; Bouchaud & Potters) dice que, si no hubiera
señal, los autovalores caerían en el "bulk" [λ−, λ+] con:

    λ± = (1 ± √(N/T))²      (para correlación, σ²=1)

Los autovalores POR ENCIMA de λ+ son señal real (modos de mercado/sector); el resto
es ruido. Se "limpian" reemplazando los autovalores del bulk por su media (preservando
la traza) → matriz de correlación mucho más estable para construir cartera.

Compara min-variance con correlación CRUDA vs LIMPIA (RMT) vs Ledoit-Wolf, fuera de
muestra. Sin dependencias nuevas. No es recomendación de inversión.

Uso:
    python rmt_clean.py AAPL MSFT NVDA GOOGL AMZN META JPM XOM KO WMT GLD TLT
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
    return pd.DataFrame(series).dropna().pct_change().dropna()


def mp_bounds(N, T):
    q = N / T
    return (1 - np.sqrt(q)) ** 2, (1 + np.sqrt(q)) ** 2


def limpiar_correlacion(corr, T):
    """Eigenvalue clipping (Marchenko-Pastur). Devuelve (corr_limpia, n_señal, λ+)."""
    N = corr.shape[0]
    vals, vecs = np.linalg.eigh(corr)
    _, lam_plus = mp_bounds(N, T)
    ruido = vals <= lam_plus
    vals_c = vals.copy()
    if ruido.sum() > 0:
        vals_c[ruido] = vals[ruido].mean()          # aplana el bulk a su media
    C = vecs @ np.diag(vals_c) @ vecs.T
    d = np.sqrt(np.clip(np.diag(C), 1e-12, None))
    C = C / np.outer(d, d)                            # re-normaliza diagonal a 1
    return C, int((~ruido).sum()), float(lam_plus), vals


def _min_var(cov):
    inv = np.linalg.pinv(cov)
    ones = np.ones(cov.shape[0])
    w = inv @ ones / (ones @ inv @ ones)
    w = np.clip(w, 0, None)
    return w / w.sum() if w.sum() > 0 else np.repeat(1/len(w), len(w))


def _oos(ret_te, w, cols):
    r = ret_te[cols].values @ w
    vol = float(np.std(r) * np.sqrt(252))
    sh = float(np.mean(r) / np.std(r) * np.sqrt(252)) if np.std(r) > 0 else float("nan")
    return vol, sh


def comparar(tickers, period="4y"):
    ret = _retornos(tickers, period)
    if ret.shape[1] < 4 or len(ret) < 252:
        return None
    n = len(ret)
    tr, te = ret.iloc[:n//2], ret.iloc[n//2:]
    cols = ret.columns.tolist()
    std = tr.std().values
    corr = tr.corr().values
    T = len(tr)
    corr_clean, n_sig, lam_plus, vals = limpiar_correlacion(corr, T)

    def cov_de(c):
        return np.outer(std, std) * c
    from sklearn.covariance import LedoitWolf
    cov_lw = LedoitWolf().fit(tr.values).covariance_

    metodos = {
        "Cruda (sample)": _min_var(cov_de(corr)),
        "RMT (limpia)": _min_var(cov_de(corr_clean)),
        "Ledoit-Wolf": _min_var(cov_lw),
        "Equiponderada": np.repeat(1/len(cols), len(cols)),
    }
    filas = []
    for nombre, w in metodos.items():
        vol, sh = _oos(te, w, cols)
        filas.append({"Método": nombre, "Vol OOS": f"{vol*100:.1f}%", "Sharpe OOS": round(sh, 2),
                      "Máx peso": f"{w.max()*100:.0f}%"})
    meta = {"N": len(cols), "T": T, "n_señal": n_sig, "lam_plus": lam_plus,
            "vals": vals, "tabla": pd.DataFrame(filas)}
    return meta


def _plot(meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    N, T = meta["N"], meta["T"]
    q = N / T
    lam_m, lam_p = mp_bounds(N, T)
    x = np.linspace(max(1e-3, lam_m), lam_p, 200)
    rho = T / N * np.sqrt(np.clip((lam_p - x) * (x - lam_m), 0, None)) / (2 * np.pi * x)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(meta["vals"], bins=max(10, N), density=True, color="tab:blue", alpha=0.5, label="Autovalores reales")
    ax.plot(x, rho, color="black", lw=2, label="Marchenko-Pastur (ruido)")
    ax.axvline(lam_p, color="tab:red", ls="--", label=f"λ+ = {lam_p:.2f} (señal por encima)")
    ax.set_title(f"Espectro de autovalores · {N} activos · {meta['n_señal']} de señal (resto = ruido)")
    ax.set_xlabel("Autovalor"); ax.set_ylabel("Densidad"); ax.legend(loc="best")
    fig.tight_layout()
    return fig


def informe(meta):
    if meta is None:
        return "Datos insuficientes (≥4 activos)."
    L = [f"=== Limpieza RMT de la correlación · {meta['N']} activos · T={meta['T']} ===\n",
         f"  Autovalores de SEÑAL : {meta['n_señal']} de {meta['N']}  (resto = ruido bajo λ+={meta['lam_plus']:.2f})\n",
         meta["tabla"].to_string(index=False),
         "\n> La min-variance con correlación LIMPIA (RMT) suele dar menor vol OOS que la cruda,",
         "> porque no optimiza sobre ruido. Econofísica (Marchenko-Pastur). No es recomendación."]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Limpieza de correlación con Random Matrix Theory.")
    ap.add_argument("tickers", nargs="*",
                    default=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "XOM",
                             "KO", "WMT", "GLD", "TLT", "XLE", "XLF"])
    ap.add_argument("--period", default="4y")
    a = ap.parse_args()
    print(f"\nLimpiando la correlación de {len(a.tickers)} activos con RMT...")
    meta = comparar(a.tickers, a.period)
    print("\n" + informe(meta) + "\n")


if __name__ == "__main__":
    main()
