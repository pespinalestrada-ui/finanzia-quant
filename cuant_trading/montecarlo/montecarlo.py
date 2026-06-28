"""
montecarlo — simulación de Monte Carlo: precio futuro y robustez del sistema.

Dos usos:

  1. PRECIO: simula miles de trayectorias futuras del precio (bootstrap de retornos
     históricos o GBM) → distribución del precio a 'horizon' días, percentiles,
     probabilidad de subir/bajar, VaR del camino. Honesto: NO predice, da el ABANICO
     de lo posible bajo la volatilidad histórica.

  2. SISTEMA: re-muestrea con reemplazo los resultados de tus operaciones (del diario,
     o de win-rate/payoff dados) → miles de curvas de equity → distribución del
     resultado final, distribución del MÁXIMO DRAWDOWN, probabilidad de RUINA y de
     acabar en positivo. Es el modo correcto de medir si un sistema es robusto o tuvo
     suerte.

No es recomendación de inversión.

Uso:
    python montecarlo.py precio AAPL --horizon 90 --sims 3000
    python montecarlo.py sistema --winrate 0.55 --payoff 1.5 --trades 50 --sims 5000
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
sys.path.insert(0, str(_SUITE / "journal"))
np.random.seed(42)


# --- 1. Monte Carlo de PRECIO ------------------------------------------------
def simular_precio(ticker, horizon=90, n_sims=3000, metodo="bootstrap", period="3y"):
    h = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if h.empty:
        raise ValueError(f"'{ticker}' sin datos.")
    c = h["Close"].astype(float)
    px0 = float(c.iloc[-1])
    r = np.log(c / c.shift(1)).dropna().values
    if metodo == "gbm":
        mu, sd = r.mean(), r.std()
        Z = np.random.standard_normal((n_sims, horizon))
        paths = px0 * np.exp(np.cumsum((mu - 0.5 * sd ** 2) + sd * Z, axis=1))
    else:  # bootstrap (captura colas reales)
        idx = np.random.randint(0, len(r), (n_sims, horizon))
        paths = px0 * np.exp(np.cumsum(r[idx], axis=1))
    fin = paths[:, -1]
    pct = {p: float(np.percentile(fin, p)) for p in (5, 25, 50, 75, 95)}
    meta = {"px0": px0, "horizon": horizon, "n_sims": n_sims, "metodo": metodo,
            "pct": pct, "prob_subir": float((fin > px0).mean()),
            "var95_pct": float((np.percentile(fin, 5) / px0 - 1) * 100),
            "esperado_pct": float((pct[50] / px0 - 1) * 100)}
    return paths, fin, meta


# --- 2. Monte Carlo del SISTEMA (bootstrap de operaciones) -------------------
def _retornos_trade_diario():
    """R-múltiplos de las operaciones cerradas del diario (fracción sobre riesgo)."""
    try:
        import journal as JR
        df = JR._load()
        c = df[df["estado"] == "CERRADA"]
        r = pd.to_numeric(c["r_multiplo"], errors="coerce").dropna().values
        return r if len(r) >= 5 else None
    except Exception:
        return None


def simular_sistema(retornos=None, winrate=0.55, payoff=1.5, riesgo_pct=0.01,
                    n_trades=50, n_sims=5000, capital=10000.0):
    """
    Bootstrap de resultados por operación → distribución de equity final, drawdown,
    prob. de ruina. Si 'retornos' (R-múltiplos) es None, los sintetiza de winrate/payoff.
    Cada operación arriesga riesgo_pct del capital; un trade de +R suma R·riesgo.
    """
    if retornos is None or len(retornos) < 5:
        n = 100000
        gana = np.random.random(n) < winrate
        retornos = np.where(gana, payoff, -1.0)        # R-múltiplos sintéticos
        fuente = f"sintético (win {winrate*100:.0f}% · payoff {payoff})"
    else:
        retornos = np.asarray(retornos, dtype=float)
        fuente = f"diario ({len(retornos)} operaciones reales)"

    sims = np.random.choice(retornos, size=(n_sims, n_trades), replace=True)
    # equity multiplicativo: cada trade cambia el capital en R·riesgo_pct
    factores = 1.0 + sims * riesgo_pct
    equity = capital * np.cumprod(factores, axis=1)
    equity = np.hstack([np.full((n_sims, 1), capital), equity])
    final = equity[:, -1]
    picos = np.maximum.accumulate(equity, axis=1)
    dd = ((equity - picos) / picos).min(axis=1)        # max drawdown por sim (negativo)

    ret_final = (final / capital - 1) * 100
    meta = {
        "fuente": fuente, "n_trades": n_trades, "n_sims": n_sims, "capital": capital,
        "riesgo_pct": riesgo_pct * 100,
        "ret_med_pct": float(np.median(ret_final)),
        "ret_p5_pct": float(np.percentile(ret_final, 5)),
        "ret_p95_pct": float(np.percentile(ret_final, 95)),
        "prob_positivo": float((final > capital).mean()),
        "dd_medio_pct": float(np.median(dd) * 100),
        "dd_peor_pct": float(np.percentile(dd, 5) * 100),
        "prob_ruina_50": float((dd <= -0.50).mean()),     # caída ≥50% en algún punto
    }
    return equity, meta


def _plot_precio(paths, meta, ticker):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    muestra = paths[np.random.choice(len(paths), min(200, len(paths)), replace=False)]
    ax.plot(muestra.T, color="tab:blue", alpha=0.05)
    for p, col in [(5, "tab:red"), (50, "black"), (95, "tab:green")]:
        ax.plot(np.percentile(paths, p, axis=0), color=col, lw=1.8, label=f"P{p}")
    ax.axhline(meta["px0"], color="gray", ls="--", lw=0.8)
    ax.set_title(f"{ticker.upper()} · Monte Carlo {meta['n_sims']} caminos · {meta['horizon']}d ({meta['metodo']})")
    ax.set_xlabel("Días"); ax.set_ylabel("Precio"); ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _plot_sistema(equity, meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    muestra = equity[np.random.choice(len(equity), min(300, len(equity)), replace=False)]
    ax.plot(muestra.T, color="tab:blue", alpha=0.05)
    for p, col in [(5, "tab:red"), (50, "black"), (95, "tab:green")]:
        ax.plot(np.percentile(equity, p, axis=0), color=col, lw=1.8, label=f"P{p}")
    ax.axhline(meta["capital"], color="gray", ls="--", lw=0.8)
    ax.set_title(f"Monte Carlo del sistema · {meta['n_sims']} simulaciones · {meta['n_trades']} operaciones")
    ax.set_xlabel("Operación"); ax.set_ylabel("Equity (€)"); ax.legend(loc="best")
    fig.tight_layout()
    return fig


def informe_precio(ticker, meta):
    p = meta["pct"]
    return ("\n".join([
        f"=== Monte Carlo PRECIO · {ticker} · {meta['horizon']}d · {meta['n_sims']} caminos ({meta['metodo']}) ===\n",
        f"  Precio actual     : {meta['px0']:.2f}",
        f"  Mediana esperada  : {p[50]:.2f}  ({meta['esperado_pct']:+.1f}%)",
        f"  Rango P5–P95      : {p[5]:.2f} – {p[95]:.2f}",
        f"  Prob. de subir    : {meta['prob_subir']*100:.0f}%",
        f"  VaR 95% del camino: {meta['var95_pct']:+.1f}%  (peor 5% de escenarios)",
        "\n> NO es predicción: es el abanico de lo posible bajo la volatilidad histórica.",
        "> Bootstrap captura colas reales mejor que GBM. No es recomendación."]))


def informe_sistema(meta):
    rob = "ROBUSTO" if (meta["prob_positivo"] > 0.7 and meta["prob_ruina_50"] < 0.05) else "FRÁGIL / dudoso"
    return ("\n".join([
        f"=== Monte Carlo SISTEMA · {meta['n_sims']} sims · {meta['n_trades']} operaciones · fuente {meta['fuente']} ===\n",
        f"  Riesgo por operación : {meta['riesgo_pct']:.1f}% del capital",
        f"  Retorno mediano      : {meta['ret_med_pct']:+.1f}%  (P5 {meta['ret_p5_pct']:+.1f}% · P95 {meta['ret_p95_pct']:+.1f}%)",
        f"  Prob. acabar positivo: {meta['prob_positivo']*100:.0f}%",
        f"  Drawdown mediano     : {meta['dd_medio_pct']:.1f}%  · peor 5%: {meta['dd_peor_pct']:.1f}%",
        f"  Prob. RUINA (−50%)   : {meta['prob_ruina_50']*100:.1f}%",
        f"\n  Veredicto de robustez: {rob}",
        "\n> Bootstrap de tus operaciones: ¿el sistema aguanta o tuvo suerte? La prob. de ruina",
        "> y el drawdown peor-5% importan más que el retorno medio. No es recomendación."]))


def main():
    ap = argparse.ArgumentParser(description="Simulación de Monte Carlo (precio y sistema).")
    sub = ap.add_subparsers(dest="modo", required=True)
    pp = sub.add_parser("precio")
    pp.add_argument("ticker"); pp.add_argument("--horizon", type=int, default=90)
    pp.add_argument("--sims", type=int, default=3000)
    pp.add_argument("--metodo", choices=["bootstrap", "gbm"], default="bootstrap")
    ps = sub.add_parser("sistema")
    ps.add_argument("--winrate", type=float, default=0.55)
    ps.add_argument("--payoff", type=float, default=1.5)
    ps.add_argument("--riesgo", type=float, default=0.01)
    ps.add_argument("--trades", type=int, default=50)
    ps.add_argument("--sims", type=int, default=5000)
    ps.add_argument("--capital", type=float, default=10000.0)
    ps.add_argument("--usar-diario", action="store_true", help="Usa las operaciones reales del diario.")
    a = ap.parse_args()

    if a.modo == "precio":
        paths, fin, meta = simular_precio(a.ticker, a.horizon, a.sims, a.metodo)
        print("\n" + informe_precio(a.ticker.upper(), meta) + "\n")
    else:
        ret = _retornos_trade_diario() if a.usar_diario else None
        equity, meta = simular_sistema(ret, a.winrate, a.payoff, a.riesgo, a.trades, a.sims, a.capital)
        print("\n" + informe_sistema(meta) + "\n")


if __name__ == "__main__":
    main()
