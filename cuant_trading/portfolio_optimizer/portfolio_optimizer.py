"""
portfolio_optimizer — optimización media-varianza (Markowitz).

Dada una cesta de tickers, calcula la cartera de máximo Sharpe y la de mínima
volatilidad, y dibuja la frontera eficiente. Rendimientos y covarianzas
anualizados desde el histórico de precios.

Uso:
    python portfolio_optimizer.py AAPL MSFT NVDA GOOGL
    python portfolio_optimizer.py SAB.MC BBVA.MC ITX.MC REP.MC --period 3y --save
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
from scipy.optimize import minimize


# paleta Nocturne (la misma de cuant_trading/dashboard/finanzia_charts.py):
# sobre fondo oscuro el negro y el azul puro no se ven
_NEU, _ACC, _ACC2 = "#9aa0b5", "#8271e0", "#a99bf0"
_UP, _DOWN, _GOLD, _DIM = "#3fa87c", "#d4615c", "#ab7f28", "#6f7486"
_CIAN, _AMB, _ROSA = "#2d9ec4", "#ab7f28", "#c25b86"

def _cmap(plt, nombre, respaldo):
    """Rampa del tema Nocturne; si la herramienta se ejecuta fuera del panel
    (sin finanzia_charts cargado), usa la rampa estándar de matplotlib."""
    try:
        return plt.get_cmap(nombre)
    except Exception:
        return respaldo


def precios(tickers, period):
    data = {}
    for t in tickers:
        h = yf.Ticker(t).history(period=period, auto_adjust=True)
        if h.empty:
            print(f"  (sin datos: {t})"); continue
        s = h["Close"].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        data[t.upper()] = s
    if len(data) < 2:
        raise SystemExit("Necesito al menos 2 tickers con datos.")
    return pd.DataFrame(data).dropna()


def main():
    ap = argparse.ArgumentParser(description="Optimizador de cartera Markowitz.")
    ap.add_argument("tickers", nargs="+")
    ap.add_argument("--period", default="3y")
    ap.add_argument("--rf", type=float, default=0.0, help="Tasa libre de riesgo anual (0.03=3%).")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()

    px = precios(a.tickers, a.period)
    rets = px.pct_change().dropna()
    mu = rets.mean() * 252
    cov = rets.cov() * 252
    n = len(mu)
    names = list(mu.index)

    def perf(w):
        r = w @ mu
        v = np.sqrt(w @ cov @ w)
        return r, v

    def neg_sharpe(w):
        r, v = perf(w)
        return -(r - a.rf) / v

    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1},)
    bnds = tuple((0, 1) for _ in range(n))
    w0 = np.repeat(1/n, n)

    res_sh = minimize(neg_sharpe, w0, method="SLSQP", bounds=bnds, constraints=cons)
    res_mv = minimize(lambda w: perf(w)[1], w0, method="SLSQP", bounds=bnds, constraints=cons)
    w_sh, w_mv = res_sh.x, res_mv.x

    def report(nombre, w):
        r, v = perf(w)
        sh = (r - a.rf) / v
        port_rets = rets @ w
        var_95 = np.percentile(port_rets, 5) * 100
        cvar_95 = port_rets[port_rets <= np.percentile(port_rets, 5)].mean() * 100
        print(f"\n=== {nombre} ===")
        print(f"  Retorno esp. anual : {r*100:.1f}%")
        print(f"  Volatilidad anual  : {v*100:.1f}%")
        print(f"  Sharpe             : {sh:.2f}")
        print(f"  VaR diario (95%)   : {var_95:+.2f}%")
        print(f"  CVaR diario (95%)  : {cvar_95:+.2f}%")
        print(f"  Pesos:")
        for nm, wi in sorted(zip(names, w), key=lambda x: -x[1]):
            if wi > 0.005:
                print(f"    {nm:<10} {wi*100:5.1f}%")

    print(f"\nCartera de {n} activos · {a.period} · rf={a.rf*100:.1f}%")
    report("Máximo Sharpe", w_sh)
    report("Mínima volatilidad", w_mv)
    print()

    if a.save:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # frontera por muestreo aleatorio
        rng = np.random.default_rng(42)
        N = 4000
        W = rng.random((N, n)); W /= W.sum(axis=1, keepdims=True)
        R = W @ mu.values
        V = np.sqrt((W @ cov.values * W).sum(axis=1))
        S = (R - a.rf) / V
        fig, ax = plt.subplots(figsize=(10, 6))
        sc = ax.scatter(V*100, R*100, c=S, cmap=_cmap(plt, "nocturne_seq", "viridis"), s=8, alpha=0.55)
        plt.colorbar(sc, label="Sharpe")
        rs, vs = perf(w_sh); rm, vm = perf(w_mv)
        ax.scatter(vs*100, rs*100, marker="*", s=300, color=_ACC, label="Máx Sharpe", zorder=5)
        ax.scatter(vm*100, rm*100, marker="*", s=300, color=_GOLD, label="Mín volatilidad", zorder=5)
        ax.set_xlabel("Volatilidad anual %"); ax.set_ylabel("Retorno esperado anual %")
        ax.set_title("Frontera eficiente"); ax.legend()
        fig.tight_layout()
        out = "frontera_eficiente.png"; fig.savefig(out, dpi=110)
        print(f"Frontera guardada: {out}\n")


if __name__ == "__main__":
    main()
