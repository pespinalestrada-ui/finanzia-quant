"""
correlation — matriz de correlación de una cesta de activos.

Calcula la correlación de los retornos diarios entre varios tickers, la
correlación media por activo (para ver diversificación) y guarda un heatmap.
Útil para no llevar 5 posiciones que en realidad son la misma apuesta.

Uso:
    python correlation.py AAPL MSFT NVDA TLT GLD
    python correlation.py SAB.MC BBVA.MC ITX.MC REP.MC --period 2y --save
"""
import argparse
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import pandas as pd
import yfinance as yf


def main():
    ap = argparse.ArgumentParser(description="Matriz de correlación.")
    ap.add_argument("tickers", nargs="+")
    ap.add_argument("--period", default="2y")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()

    data = {}
    for t in a.tickers:
        h = yf.Ticker(t).history(period=a.period, auto_adjust=True)
        if h.empty:
            print(f"  (sin datos: {t})"); continue
        s = h["Close"]; s.index = pd.to_datetime(s.index).tz_localize(None)
        data[t.upper()] = s
    if len(data) < 2:
        raise SystemExit("Necesito al menos 2 tickers.")

    px = pd.DataFrame(data).dropna()
    corr = px.pct_change().dropna().corr()

    print(f"\n=== Matriz de correlación ({a.period}, {len(px)} sesiones) ===\n")
    print(corr.round(2).to_string())

    media = (corr.sum() - 1) / (len(corr) - 1)
    print("\n=== Correlación media por activo (menor = más diversificador) ===")
    for nm, v in media.sort_values().items():
        marca = "  ← buen diversificador" if v < 0.3 else ("  ← muy correlacionado" if v > 0.7 else "")
        print(f"  {nm:<10} {v:.2f}{marca}")

    # par más y menos correlacionado
    c = corr.copy(); np.fill_diagonal(c.values, np.nan)
    amax = c.stack().idxmax(); amin = c.stack().idxmin()
    print(f"\n  Par MÁS correlacionado : {amax[0]}–{amax[1]} ({c.loc[amax]:.2f})")
    print(f"  Par MENOS correlacionado: {amin[0]}–{amin[1]} ({c.loc[amin]:.2f})\n")

    if a.save:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(1.2*len(corr)+2, 1.0*len(corr)+1.5))
        im = ax.imshow(corr, cmap="RdYlGn_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index)
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(j, i, f"{corr.values[i, j]:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if abs(corr.values[i, j]) > 0.6 else "black")
        plt.colorbar(im, label="correlación")
        ax.set_title(f"Correlación de retornos ({a.period})")
        fig.tight_layout()
        out = "correlacion_heatmap.png"; fig.savefig(out, dpi=110)
        print(f"Heatmap guardado: {out}\n")


if __name__ == "__main__":
    main()
