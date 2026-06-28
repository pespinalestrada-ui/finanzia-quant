"""
pairs_trading — arbitraje estadístico por COINTEGRACIÓN (market-neutral).

De lo poco con edge persistente: dos activos cointegrados tienen un spread que
revierte a la media. Largo el barato / corto el caro, cierras al revertir.

Matemática:
  - Cointegración: test de Engle-Granger (statsmodels.coint). p<0.05 → cointegrados.
  - Hedge ratio β por OLS: spread = B − β·A.
  - Reversión = proceso de Ornstein-Uhlenbeck: dX = θ(μ−X)dt + σdW.
    Half-life = ln(2)/θ, estimado regresando ΔX sobre X_lag (λ = pendiente,
    half-life = −ln(2)/λ). Half-life corto = revierte rápido = tradeable.
  - Señal: z-score del spread. z < −entrada → largo spread; z > +entrada → corto.

Incluye backtest del z-score (entrar a ±2σ, salir a 0) para ver si el par habría
funcionado. No es recomendación de inversión.

Uso:
    python pairs_trading.py KO PEP XOM CVX V MA AAPL MSFT      # busca pares cointegrados
    python pairs_trading.py --par KO PEP                       # analiza un par concreto
"""
import argparse
import sys
from itertools import combinations
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


def _precios(tickers, period="3y"):
    series = {}
    for tk in tickers:
        h = yf.Ticker(tk).history(period=period, auto_adjust=True)
        if h.empty:
            continue
        s = h["Close"].astype(float)
        s.index = pd.to_datetime(s.index).tz_localize(None)
        series[tk] = s
    return pd.DataFrame(series).dropna()


def _hedge_spread(a, b):
    """OLS b = β·a + c → spread = b − β·a (sin look-ahead: usa toda la muestra para β)."""
    import statsmodels.api as sm
    X = sm.add_constant(a.values)
    beta = sm.OLS(b.values, X).fit().params[1]
    spread = b - beta * a
    return float(beta), spread


def half_life(spread):
    """Vida media de reversión (OU). Pequeña = revierte rápido."""
    s = spread.dropna()
    lag = s.shift(1).dropna()
    delta = s.diff().dropna()
    lag = lag.loc[delta.index]
    if len(lag) < 30 or lag.std() == 0:
        return np.nan
    import statsmodels.api as sm
    lam = sm.OLS(delta.values, sm.add_constant(lag.values)).fit().params[1]
    if lam >= 0:
        return np.nan                       # no revierte
    return float(-np.log(2) / lam)


def evaluar_par(a, b, sa, sb):
    """Devuelve dict con p-valor de cointegración, β, half-life y z-score actual."""
    from statsmodels.tsa.stattools import coint
    try:
        _, pval, _ = coint(sa, sb)
    except Exception:
        return None
    beta, spread = _hedge_spread(sa, sb)
    hl = half_life(spread)
    mu, sd = spread.mean(), spread.std()
    z = float((spread.iloc[-1] - mu) / sd) if sd > 0 else 0.0
    return {"A": a, "B": b, "pval": float(pval), "beta": beta, "half_life": hl,
            "z": z, "spread": spread, "mu": mu, "sd": sd}


def backtest_par(spread, entrada=2.0, salida=0.0):
    """Backtest del z-score: entra a ±entrada·σ, sale a salida·σ. Retorno por trade del spread."""
    z = (spread - spread.mean()) / spread.std()
    pos, entry_z, trades = 0, 0.0, []
    for zt in z:
        if pos == 0:
            if zt <= -entrada:
                pos, entry_z = 1, zt           # largo spread (espera que suba)
            elif zt >= entrada:
                pos, entry_z = -1, zt           # corto spread
        elif (pos == 1 and zt >= -salida) or (pos == -1 and zt <= salida):
            trades.append(pos * (zt - entry_z) * -1)   # ganancia en unidades de z (revertir)
            pos = 0
    if not trades:
        return {"n": 0, "win": None, "media_z": None}
    t = np.array(trades)
    return {"n": len(t), "win": round(float((t > 0).mean()) * 100, 1), "media_z": round(float(t.mean()), 2)}


def buscar(tickers, period="3y", max_pares=12):
    px = _precios(tickers, period)
    if px.shape[1] < 2 or len(px) < 200:
        return pd.DataFrame()
    filas = []
    for a, b in combinations(px.columns, 2):
        r = evaluar_par(a, b, px[a], px[b])
        if r is None or np.isnan(r["half_life"]):
            continue
        bt = backtest_par(r["spread"])
        filas.append({"Par": f"{a}/{b}", "p-coint": round(r["pval"], 4),
                      "β": round(r["beta"], 3), "Half-life (d)": round(r["half_life"], 1),
                      "z actual": round(r["z"], 2), "Trades": bt["n"], "Win %": bt["win"],
                      "Señal": _senal(r["z"])})
    df = pd.DataFrame(filas)
    if df.empty:
        return df
    # rankea por cointegración fuerte + reversión rápida
    df = df[df["p-coint"] < 0.10].sort_values(["p-coint", "Half-life (d)"]).head(max_pares).reset_index(drop=True)
    return df


def _senal(z, entrada=2.0):
    if z <= -entrada:
        return "🟢 LARGO spread (largo A·? / según β)"
    if z >= entrada:
        return "🔴 CORTO spread"
    return "🟡 fuera (z dentro de ±2σ)"


def main():
    ap = argparse.ArgumentParser(description="Arbitraje estadístico por cointegración (pairs trading).")
    ap.add_argument("tickers", nargs="*",
                    default=["KO", "PEP", "XOM", "CVX", "V", "MA", "AAPL", "MSFT", "JPM", "BAC"])
    ap.add_argument("--par", nargs=2, metavar=("A", "B"), help="Analiza un par concreto.")
    ap.add_argument("--period", default="3y")
    a = ap.parse_args()

    if a.par:
        px = _precios(a.par, a.period)
        if px.shape[1] < 2:
            print("Sin datos para el par."); return
        r = evaluar_par(a.par[0], a.par[1], px[a.par[0]], px[a.par[1]])
        bt = backtest_par(r["spread"])
        coint_ok = "SÍ" if r["pval"] < 0.05 else "no"
        print(f"\n=== {a.par[0]} / {a.par[1]} ===")
        print(f"  Cointegrados (p={r['pval']:.4f}) : {coint_ok}")
        print(f"  Hedge ratio β               : {r['beta']:.3f}  (spread = {a.par[1]} − β·{a.par[0]})")
        print(f"  Half-life reversión         : {r['half_life']:.1f} días")
        print(f"  z-score actual              : {r['z']:+.2f}σ → {_senal(r['z'])}")
        print(f"  Backtest z (±2σ→0)          : {bt['n']} trades · win {bt['win']}% · media {bt['media_z']}σ")
        print("\n> Si p<0.05 y half-life corto, el par revierte y es tradeable. No es recomendación.\n")
        return

    print(f"\nBuscando pares cointegrados entre {len(a.tickers)} tickers...")
    df = buscar(a.tickers, a.period)
    if df.empty:
        print("Ningún par cointegrado (p<0.10) con reversión clara.")
        return
    print("\n" + df.to_string(index=False))
    print("\n> Rankeado por cointegración fuerte (p bajo) + reversión rápida (half-life corto).")
    print("> Opera los de p<0.05 cuando |z|>2. Market-neutral. No es recomendación.\n")


if __name__ == "__main__":
    main()
