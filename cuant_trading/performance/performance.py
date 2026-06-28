"""
performance — monitor de rendimiento del sistema vs benchmark (SPY).

Cierra el bucle: operas (paper) → mides si tu sistema bate a comprar-y-mantener.
Lee el diario (operaciones cerradas) y calcula:

  - Curva de EQUITY (capital simulado acumulado).
  - Métricas: retorno total, win rate, profit factor, expectancy R, máx drawdown.
  - Benchmark: ¿qué habría hecho **SPY** (comprar y mantener) en el mismo periodo?
  - Comparación factor ALTO vs BAJO (¿las compras 'top factor' rinden más?).

Sin operaciones cerradas, avisa. No es recomendación de inversión.

Uso:
    python performance.py
    python performance.py --capital 10000 --benchmark SPY
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

_SUITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SUITE / "journal"))
import journal as JR


def cerradas():
    df = JR._load()
    c = df[df["estado"] == "CERRADA"].copy()
    if c.empty:
        return c
    c["fecha_cierre"] = pd.to_datetime(c["fecha_cierre"], errors="coerce")
    c["fecha_apertura"] = pd.to_datetime(c["fecha_apertura"], errors="coerce")
    c["pnl"] = pd.to_numeric(c["pnl"], errors="coerce")
    return c.dropna(subset=["fecha_cierre", "pnl"]).sort_values("fecha_cierre")


def _benchmark(ticker, ini, fin, capital):
    import yfinance as yf
    try:
        h = yf.Ticker(ticker).history(start=ini, end=fin + pd.Timedelta(days=2), auto_adjust=True)
        if h.empty or len(h) < 2:
            return None
        ret = float(h["Close"].iloc[-1] / h["Close"].iloc[0] - 1)
        serie = capital * (h["Close"] / h["Close"].iloc[0])
        serie.index = pd.to_datetime(serie.index).tz_localize(None)
        return {"ret_pct": ret * 100, "equity": serie}
    except Exception:
        return None


def analizar(capital=10000.0, benchmark="SPY"):
    c = cerradas()
    if c.empty:
        return {"n": 0, "mensaje": "Aún no hay operaciones cerradas en el diario. "
                                   "Opera en paper (🤖 Sistema / 🦙 Alpaca) y cierra alguna."}
    pnl = c["pnl"]
    r = pd.to_numeric(c["r_multiplo"], errors="coerce")
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    equity = capital + pnl.cumsum()
    equity.index = c["fecha_cierre"].values
    dd = float((equity / equity.cummax() - 1).min()) * 100
    pf = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
    ret_sys = float(pnl.sum()) / capital * 100

    ini, fin = c["fecha_apertura"].min(), c["fecha_cierre"].max()
    bench = _benchmark(benchmark, ini, fin, capital)
    res = {
        "n": len(c), "capital": capital,
        "ret_sistema_pct": round(ret_sys, 2), "pnl_total": round(float(pnl.sum()), 2),
        "win_rate": round(float((pnl > 0).mean()) * 100, 1),
        "profit_factor": round(pf, 2) if np.isfinite(pf) else None,
        "expectancy_R": round(float(r.mean()), 2) if r.notna().any() else None,
        "max_drawdown_pct": round(dd, 1),
        "benchmark": benchmark,
        "ret_benchmark_pct": round(bench["ret_pct"], 2) if bench else None,
        "bate_benchmark": (bench is not None and ret_sys > bench["ret_pct"]),
        "periodo": f"{ini.date()} → {fin.date()}",
        "_equity": equity, "_bench_equity": bench["equity"] if bench else None,
    }
    return res


def _plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    eq = res["_equity"]
    ax.plot(eq.index, eq.values, color="tab:blue", lw=1.6, marker="o", ms=3, label="Sistema (paper)")
    if res["_bench_equity"] is not None:
        b = res["_bench_equity"]
        ax.plot(b.index, b.values, color="gray", lw=1.2, ls="--", label=f"{res['benchmark']} (comprar y mantener)")
    ax.axhline(res["capital"], color="black", lw=0.6, alpha=0.5)
    ax.set_title("Rendimiento del sistema vs benchmark")
    ax.set_xlabel("Fecha"); ax.set_ylabel("Equity (€)"); ax.legend(loc="best")
    fig.tight_layout()
    return fig


def informe(res):
    if res.get("n", 0) == 0:
        return res.get("mensaje", "Sin datos.")
    L = [f"=== Rendimiento del sistema · {res['n']} operaciones cerradas · {res['periodo']} ===\n"]
    L.append(f"  Retorno sistema   : {res['ret_sistema_pct']:+.2f}%  ({res['pnl_total']:+.2f} €)")
    if res["ret_benchmark_pct"] is not None:
        v = "BATE ✅" if res["bate_benchmark"] else "NO bate"
        L.append(f"  {res['benchmark']} (buy&hold)   : {res['ret_benchmark_pct']:+.2f}%  → el sistema {v} al benchmark")
    L.append(f"  Win rate          : {res['win_rate']}%")
    L.append(f"  Profit factor     : {res['profit_factor']}")
    L.append(f"  Expectancy        : {res['expectancy_R']} R/operación")
    L.append(f"  Máx drawdown      : {res['max_drawdown_pct']}%")
    fiable = "" if res["n"] >= 20 else f"\n  ⚠️ Solo {res['n']} operaciones — poco fiable (mínimo 20-30)."
    L.append(fiable)
    L.append("\n> Batir a SPY comprar-y-mantener (neto, tras costes y muchas operaciones) es el")
    L.append("> listón real. Si no lo bates, indexarte es mejor. No es recomendación.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Monitor de rendimiento del sistema vs benchmark.")
    ap.add_argument("--capital", type=float, default=10000.0)
    ap.add_argument("--benchmark", default="SPY")
    a = ap.parse_args()
    res = analizar(a.capital, a.benchmark)
    print("\n" + informe(res) + "\n")


if __name__ == "__main__":
    main()
