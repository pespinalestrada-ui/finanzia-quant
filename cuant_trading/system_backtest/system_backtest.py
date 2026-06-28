"""
system_backtest — backtest de la ESTRATEGIA completa del sistema (event-driven).

Corre el sistema entero sobre el histórico, no una sola regla: en cada rebalanceo
calcula el score técnico del Veredicto (point-in-time, sin leakage) para todo el
universo, compra el top-N (equiponderado, long-only), mantiene hasta el siguiente
rebalanceo y resta COSTES de rotación. Compara la curva de equity vs SPY buy & hold.

Métricas: retorno total, CAGR, Sharpe, máx drawdown, y si bate a SPY.
El backtest SIN costes miente; aquí se restan. No es recomendación de inversión.

Uso:
    python system_backtest.py AAPL MSFT NVDA GOOGL AMZN META JPM XOM KO WMT
    python system_backtest.py --file watchlist.txt --top 3 --rebal 21 --coste-bps 10
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

_SUITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SUITE / "veredicto_backtest"))
import veredicto_backtest as VB


def _metricas(equity, capital):
    r = equity.pct_change().dropna()
    años = len(r) / 252 if len(r) else 1
    tot = float(equity.iloc[-1] / capital - 1)
    cagr = float((equity.iloc[-1] / capital) ** (1 / max(años, 1e-9)) - 1)
    sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan")
    dd = float((equity / equity.cummax() - 1).min())
    return {"ret_total": tot * 100, "cagr": cagr * 100, "sharpe": sharpe, "max_dd": dd * 100}


def backtest(tickers, period="5y", top_n=3, rebal=21, coste_bps=10.0, capital=10000.0):
    scores, rets = {}, {}
    for tk in tickers:
        df = VB.descargar(tk, period)
        if df is None or len(df) < 260:
            continue
        scores[tk] = VB.score_historico(df)
        rets[tk] = df["Close"].pct_change()
    if len(scores) < 2:
        return None
    S = pd.DataFrame(scores).dropna(how="all")
    R = pd.DataFrame(rets).reindex(S.index)
    dates = S.index
    rebal_dates = set(dates[::rebal])

    weights = pd.DataFrame(0.0, index=dates, columns=S.columns)
    coste = pd.Series(0.0, index=dates)
    cur = pd.Series(0.0, index=S.columns)
    for d in dates:
        if d in rebal_dates:
            row = S.loc[d].dropna()
            if len(row) >= 1:
                top = row.nlargest(min(top_n, len(row))).index
                neww = pd.Series(0.0, index=S.columns)
                neww[top] = 1.0 / len(top)
                coste[d] = float((neww - cur).abs().sum()) * coste_bps / 10000.0
                cur = neww
        weights.loc[d] = cur.values

    port_ret = (weights.shift(1).fillna(0.0) * R).sum(axis=1) - coste
    equity = (1 + port_ret).cumprod() * capital

    # benchmark SPY
    try:
        spy = yf.Ticker("SPY").history(start=dates.min(), end=dates.max() + pd.Timedelta(days=2),
                                       auto_adjust=True)["Close"]
        spy.index = pd.to_datetime(spy.index).tz_localize(None)
        spy_eq = capital * (spy / spy.iloc[0]).reindex(equity.index, method="ffill")
    except Exception:
        spy_eq = None

    m = _metricas(equity, capital)
    m_spy = _metricas(spy_eq.dropna(), capital) if spy_eq is not None else None
    res = {"equity": equity, "spy_eq": spy_eq, "metricas": m, "spy": m_spy,
           "n_tickers": len(scores), "periodo": f"{dates.min().date()} → {dates.max().date()}",
           "top_n": top_n, "rebal": rebal, "coste_bps": coste_bps,
           "bate_spy": (m_spy is not None and m["ret_total"] > m_spy["ret_total"])}
    return res


def _plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    eq = res["equity"]
    ax.plot(eq.index, eq.values, color="tab:blue", lw=1.6, label="Sistema (con costes)")
    if res["spy_eq"] is not None:
        ax.plot(res["spy_eq"].index, res["spy_eq"].values, color="gray", lw=1.2, ls="--",
                label="SPY (buy & hold)")
    ax.set_title(f"Backtest del sistema · top-{res['top_n']} · rebal {res['rebal']}d · coste {res['coste_bps']}bps")
    ax.set_xlabel("Fecha"); ax.set_ylabel("Equity (€)"); ax.legend(loc="best")
    fig.tight_layout()
    return fig


def informe(res):
    if res is None:
        return "Datos insuficientes (mete ≥2 tickers con histórico)."
    m, s = res["metricas"], res["spy"]
    L = [f"=== Backtest del sistema · {res['n_tickers']} tickers · {res['periodo']} ===",
         f"    top-{res['top_n']} · rebalanceo {res['rebal']}d · coste {res['coste_bps']}bps/rotación\n",
         f"  {'Métrica':<16}{'SISTEMA':>12}{'SPY (B&H)':>12}",
         f"  {'Retorno total':<16}{m['ret_total']:>11.1f}%{(str(round(s['ret_total'],1))+'%') if s else 'n/d':>12}",
         f"  {'CAGR':<16}{m['cagr']:>11.1f}%{(str(round(s['cagr'],1))+'%') if s else 'n/d':>12}",
         f"  {'Sharpe':<16}{m['sharpe']:>12.2f}{(round(s['sharpe'],2)) if s else 'n/d':>12}",
         f"  {'Máx drawdown':<16}{m['max_dd']:>11.1f}%{(str(round(s['max_dd'],1))+'%') if s else 'n/d':>12}"]
    if s:
        v = "BATE ✅" if res["bate_spy"] else "NO bate"
        L.append(f"\n  → El sistema {v} a SPY comprar-y-mantener (con costes).")
    L.append("\n> Long-only top-N por score del Veredicto, rebalanceado, con costes de rotación.")
    L.append("> Recuerda: el score no superó el multiple-testing → esto mide, no promete. No es recomendación.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Backtest de la estrategia completa del sistema vs SPY.")
    ap.add_argument("tickers", nargs="*",
                    default=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "XOM", "KO", "WMT"])
    ap.add_argument("--file")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--rebal", type=int, default=21, help="Días entre rebalanceos.")
    ap.add_argument("--coste-bps", type=float, default=10.0)
    ap.add_argument("--period", default="5y")
    ap.add_argument("--capital", type=float, default=10000.0)
    a = ap.parse_args()
    tickers = a.tickers or []
    if a.file and Path(a.file).exists():
        tickers = [l.strip() for l in Path(a.file).read_text().splitlines() if l.strip()]

    print(f"\nBacktesteando el sistema sobre {len(tickers)} tickers...")
    res = backtest(tickers, a.period, a.top, a.rebal, a.coste_bps, a.capital)
    print("\n" + informe(res) + "\n")


if __name__ == "__main__":
    main()
