"""
neuralprophet_forecast — forecast con NeuralProphet.

NeuralProphet es el sucesor de Prophet sobre PyTorch: combina la
descomposición interpretable de Prophet (tendencia + estacionalidad) con
componentes autorregresivos (AR-Net) entrenados como red neuronal. Devuelve
cuantiles para la banda de incertidumbre.

A medio camino entre Prophet (puro estadístico) y la LSTM (red pura): mantiene
interpretabilidad y añade memoria autorregresiva.

Uso:
    python neuralprophet_forecast.py SAB.MC
    python neuralprophet_forecast.py AAPL --horizon 120 --epochs 60 --save
"""
import argparse
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("NP").setLevel(logging.ERROR)
logging.getLogger("neuralprophet").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import torch  # antes de neuralprophet
import yfinance as yf

HORIZONS = [30, 90, 120]


def descargar(ticker, period="3y"):
    h = yf.Ticker(ticker).history(period=period, auto_adjust=False)
    if h.empty:
        raise ValueError(f"Ticker '{ticker}' sin datos.")
    h = h.reset_index()
    ds = pd.to_datetime(h["Date"])
    if getattr(ds.dt, "tz", None) is not None:
        ds = ds.dt.tz_localize(None)
    df = pd.DataFrame({"ds": ds.dt.normalize(), "y": h["Close"].astype(float)}).dropna()
    return df.reset_index(drop=True)


def entrenar_predecir(df, horizon=120, epochs=60, n_lags=20):
    from neuralprophet import NeuralProphet, set_log_level
    set_log_level("ERROR")
    m = NeuralProphet(
        n_lags=n_lags,                 # memoria autorregresiva (AR-Net)
        n_forecasts=horizon,
        quantiles=[0.1, 0.9],          # banda 80 %
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        epochs=epochs,
        learning_rate=0.01,
    )
    m.fit(df, freq="B", progress=None, minimal=True)
    future = m.make_future_dataframe(df, periods=horizon, n_historic_predictions=False)
    fcst = m.predict(future)
    # get_latest_forecast colapsa la diagonal yhat1..yhatH en una sola serie 'origin-0'
    latest = m.get_latest_forecast(fcst, include_history_data=False, include_previous_forecasts=0)
    return _extraer(latest, horizon)


def _extraer(latest, horizon):
    """De get_latest_forecast: columnas 'origin-0', 'origin-0 10.0%', 'origin-0 90.0%'."""
    y_col = next((c for c in latest.columns if c == "origin-0"), None)
    if y_col is None:
        y_col = next(c for c in latest.columns if c.startswith("origin-0") and "%" not in c)
    lo_col = next((c for c in latest.columns if c.startswith("origin-0") and ("10.0%" in c or "10%" in c)), None)
    hi_col = next((c for c in latest.columns if c.startswith("origin-0") and ("90.0%" in c or "90%" in c)), None)
    fechas = latest["ds"].reset_index(drop=True)
    yhat = latest[y_col].astype(float).values
    lo = latest[lo_col].astype(float).values if lo_col else yhat
    hi = latest[hi_col].astype(float).values if hi_col else yhat
    n = min(len(yhat), len(fechas), len(lo), len(hi))
    return fechas[:n], yhat[:n], lo[:n], hi[:n]


def resumen(fechas, yhat, lo, hi, px, horizontes=HORIZONS):
    filas = []
    for h in horizontes:
        if h > len(yhat):
            continue
        i = h - 1
        ancho = (hi[i] - lo[i]) / yhat[i] if yhat[i] else float("nan")
        conf = "ALTA" if ancho < 0.20 else "MEDIA" if ancho < 0.45 else "BAJA"
        filas.append({
            "Horizonte": f"{h} días", "Fecha": pd.Timestamp(fechas[i]).date().isoformat(),
            "Precio esperado": round(float(yhat[i]), 3),
            "Banda 80% inf": round(float(lo[i]), 3), "Banda 80% sup": round(float(hi[i]), 3),
            "Variación %": round((yhat[i] / px - 1) * 100, 2), "Confianza": conf,
        })
    return pd.DataFrame(filas)


def forecast(ticker, period="3y", horizon=120, epochs=60):
    df = descargar(ticker, period)
    if len(df) < 200:
        raise ValueError(f"Histórico insuficiente ({len(df)} sesiones).")
    px = float(df["y"].iloc[-1])
    fechas, yhat, lo, hi = entrenar_predecir(df, horizon, epochs)
    tabla = resumen(fechas, yhat, lo, hi, px)
    fig = _plot(df, fechas, yhat, lo, hi, ticker)
    return fig, tabla, {"precio_actual": px, "n": len(df)}


def _plot(df, fechas, yhat, lo, hi, ticker):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    hist = df.iloc[-250:]
    ax.plot(hist["ds"], hist["y"], color="black", lw=1.1, label="Histórico (1 año)")
    ax.plot(fechas, yhat, color="tab:green", lw=1.5, label="NeuralProphet")
    ax.fill_between(fechas, lo, hi, color="tab:green", alpha=0.15, label="Banda 80 %")
    ax.set_title(f"{ticker.upper()} — Forecast NeuralProphet ({len(yhat)} días hábiles)")
    ax.set_xlabel("Fecha"); ax.set_ylabel("Precio"); ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description="Forecast con NeuralProphet.")
    ap.add_argument("ticker", nargs="?", default="SAB.MC")
    ap.add_argument("--horizon", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--period", default="3y")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()

    print(f"\nDescargando {a.ticker.upper()} y entrenando NeuralProphet ({a.epochs} épocas)...")
    fig, tabla, meta = forecast(a.ticker, a.period, a.horizon, a.epochs)
    print(f"Cierre actual: {meta['precio_actual']:.3f} · {meta['n']} sesiones\n")
    print(tabla.to_string(index=False))
    print("\nConfianza por ancho de banda: <20% ALTA · 20-45% MEDIA · >45% BAJA")
    print("> Forecast estadístico (AR-Net + estacionalidad). No es recomendación.\n")
    if a.save:
        from pathlib import Path
        out = Path(__file__).resolve().parent / f"{a.ticker.replace('.','_').replace('^','')}_neuralprophet.png"
        fig.savefig(out, dpi=110)
        print(f"Gráfico guardado: {out}\n")


if __name__ == "__main__":
    main()
