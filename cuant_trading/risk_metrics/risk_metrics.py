"""
risk_metrics — riesgo de cartera: VaR, CVaR (Expected Shortfall), drawdown, correlación.

Mide el RIESGO (lo que sí es estimable con rigor) en vez de prometer dirección.
Para una watchlist calcula, por ticker y para la cartera equiponderada:

  - Volatilidad anualizada (σ·√252).
  - VaR histórico 95 % / 99 %: pérdida diaria que solo se supera el 5 % / 1 % de los días.
  - CVaR / Expected Shortfall 95 %: pérdida MEDIA en ese peor 5 % (lo que el VaR ignora).
  - Máximo drawdown: peor caída pico-valle del histórico.
  - Matriz de correlación (diversificación real entre los activos).

VaR/CVaR históricos (no asumen normalidad: usan la distribución empírica de retornos).

Uso:
    python risk_metrics.py AAPL MSFT SAB.MC
    python risk_metrics.py SAB.MC BBVA.MC IBE.MC --period 3y --conf 0.99
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


def _precios(tickers, period):
    series = {}
    for tk in tickers:
        h = yf.Ticker(tk).history(period=period, auto_adjust=True)
        if h.empty:
            continue
        s = h["Close"].copy()
        s.index = pd.to_datetime(s.index)
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_localize(None)
        series[tk] = s.astype(float)
    if not series:
        raise ValueError("Ningún ticker con datos.")
    return pd.DataFrame(series).dropna()


def _drawdown(precios):
    """Máximo drawdown (fracción negativa) de una serie de precios."""
    curva = precios / precios.cummax() - 1.0
    return float(curva.min())


def _metricas_serie(ret, conf=0.95):
    a = 1 - conf
    var = float(np.quantile(ret, a))                 # cuantil bajo (pérdida)
    cvar = float(ret[ret <= var].mean()) if (ret <= var).any() else var
    vol_anual = float(ret.std() * np.sqrt(252))
    return var, cvar, vol_anual


def analizar(tickers, period="3y", conf=0.95):
    px = _precios(tickers, period)
    ret = px.pct_change().dropna()

    filas = []
    for tk in px.columns:
        var, cvar, vol = _metricas_serie(ret[tk], conf)
        filas.append({
            "Ticker": tk,
            "Vol anual": f"{vol*100:.1f}%",
            f"VaR {int(conf*100)}% (1d)": f"{var*100:.2f}%",
            f"CVaR {int(conf*100)}% (1d)": f"{cvar*100:.2f}%",
            "Max drawdown": f"{_drawdown(px[tk])*100:.1f}%",
        })

    # cartera equiponderada
    w = np.repeat(1.0 / len(px.columns), len(px.columns))
    ret_port = ret.values @ w
    ret_port = pd.Series(ret_port, index=ret.index)
    curva_port = (1 + ret_port).cumprod()
    var_p, cvar_p, vol_p = _metricas_serie(ret_port, conf)
    filas.append({
        "Ticker": "CARTERA (=peso)",
        "Vol anual": f"{vol_p*100:.1f}%",
        f"VaR {int(conf*100)}% (1d)": f"{var_p*100:.2f}%",
        f"CVaR {int(conf*100)}% (1d)": f"{cvar_p*100:.2f}%",
        "Max drawdown": f"{_drawdown(curva_port)*100:.1f}%",
    })
    tabla = pd.DataFrame(filas)
    corr = ret.corr().round(2)
    meta = {"corr_media": float(corr.values[np.triu_indices(len(corr), 1)].mean()) if len(corr) > 1 else float("nan"),
            "vol_cartera": vol_p, "var_cartera": var_p, "cvar_cartera": cvar_p,
            "n_dias": len(ret)}
    return tabla, corr, curva_port, meta


def _plot(corr, curva_port, tickers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                   gridspec_kw={"width_ratios": [1.2, 1]})
    # drawdown de la cartera
    dd = curva_port / curva_port.cummax() - 1.0
    ax1.fill_between(dd.index, dd.values * 100, 0, color="tab:red", alpha=0.3)
    ax1.plot(dd.index, dd.values * 100, color="tab:red", lw=1.0)
    ax1.set_title("Drawdown cartera equiponderada"); ax1.set_ylabel("%"); ax1.set_xlabel("Fecha")
    # heatmap correlación
    im = ax2.imshow(corr.values, vmin=-1, vmax=1, cmap="RdYlGn_r")
    ax2.set_xticks(range(len(corr))); ax2.set_yticks(range(len(corr)))
    ax2.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax2.set_yticklabels(corr.index, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax2.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax2.set_title("Correlación de retornos diarios")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def forecast(tickers, period="3y", conf=0.95):
    """Adaptador dashboard. Devuelve (fig, tabla, corr, meta)."""
    tabla, corr, curva_port, meta = analizar(tickers, period, conf)
    return _plot(corr, curva_port, tickers), tabla, corr.reset_index().rename(columns={"index": ""}), meta


def main():
    ap = argparse.ArgumentParser(description="Riesgo de cartera: VaR, CVaR, drawdown, correlación.")
    ap.add_argument("tickers", nargs="*", default=["AAPL", "MSFT", "SAB.MC"])
    ap.add_argument("--period", default="3y")
    ap.add_argument("--conf", type=float, default=0.95)
    a = ap.parse_args()
    tickers = a.tickers or ["AAPL", "MSFT", "SAB.MC"]

    tabla, corr, curva_port, meta = analizar(tickers, a.period, a.conf)
    print(f"\n=== Riesgo de cartera · {meta['n_dias']} días · confianza {int(a.conf*100)}% ===\n")
    print(tabla.to_string(index=False))
    print(f"\nCorrelación media entre activos: {meta['corr_media']:.2f} "
          f"({'baja → buena diversificación' if meta['corr_media'] < 0.4 else 'alta → poca diversificación'})")
    print("\nMatriz de correlación:")
    print(corr.to_string())
    print("\n> VaR/CVaR históricos (distribución empírica, sin asumir normalidad).")
    print("> CVaR = pérdida media en el peor 5%; siempre peor que el VaR. No es asesoramiento.\n")


if __name__ == "__main__":
    main()
