"""
lstm_forecast — forecast con red neuronal LSTM (PyTorch).

Red recurrente que aprende patrones de la secuencia de precios mediante una
ventana deslizante y proyecta de forma recursiva 30/90/120 días hábiles.
La banda de confianza se estima con la desviación de los residuos de un
holdout, ensanchándose con el horizonte (como un paseo aleatorio).

A diferencia de Prophet/AutoGluon (estadísticos), aquí el modelo es una red
neuronal entrenada desde cero en cada llamada (~pocos segundos en CPU).

Uso:
    python lstm_forecast.py SAB.MC
    python lstm_forecast.py AAPL --horizon 120 --window 30 --epochs 120 --save
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
import torch  # antes que libs pesadas (evita líos de DLL en Windows)
import torch.nn as nn
import yfinance as yf

HORIZONS = [30, 90, 120]
torch.manual_seed(42)
np.random.seed(42)


def descargar(ticker, period="3y"):
    h = yf.Ticker(ticker).history(period=period, auto_adjust=False)
    if h.empty:
        raise ValueError(f"Ticker '{ticker}' sin datos.")
    h = h.reset_index()
    ds = pd.to_datetime(h["Date"])
    if getattr(ds.dt, "tz", None) is not None:
        ds = ds.dt.tz_localize(None)
    return pd.DataFrame({"ds": ds.dt.normalize(), "y": h["Close"].astype(float)}).dropna().reset_index(drop=True)


class LSTM(nn.Module):
    def __init__(self, hidden=32, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _ventanas(serie, w):
    X, y = [], []
    for i in range(len(serie) - w):
        X.append(serie[i:i + w])
        y.append(serie[i + w])
    return np.array(X), np.array(y)


def entrenar_predecir(df, horizon=120, window=30, epochs=120, hidden=32):
    """Devuelve (fechas_futuras, yhat, banda_inf, banda_sup, resid_std_pct)."""
    precios = df["y"].values.astype(np.float32)
    pmin, pmax = precios.min(), precios.max()
    scal = (precios - pmin) / (pmax - pmin + 1e-9)        # normalizar [0,1]

    X, y = _ventanas(scal, window)
    Xt = torch.tensor(X).unsqueeze(-1)                     # [N, W, 1]
    yt = torch.tensor(y).unsqueeze(-1)                     # [N, 1]

    model = LSTM(hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    lossf = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(Xt), yt)
        loss.backward()
        opt.step()

    # residuos one-step sobre los últimos 90 (en escala precio) → banda
    model.eval()
    with torch.no_grad():
        pred_in = model(Xt).squeeze(-1).numpy()
    real_in = y
    n_hold = min(90, len(real_in))
    resid = (pred_in[-n_hold:] - real_in[-n_hold:]) * (pmax - pmin)
    resid_std = float(np.std(resid)) if len(resid) else 0.0

    # forecast recursivo
    ventana = list(scal[-window:])
    futuros = []
    with torch.no_grad():
        for _ in range(horizon):
            x = torch.tensor(np.array(ventana[-window:], dtype=np.float32)).reshape(1, window, 1)
            nxt = float(model(x).item())
            futuros.append(nxt)
            ventana.append(nxt)
    yhat = np.array(futuros) * (pmax - pmin) + pmin        # desnormalizar

    z = 1.2816                                             # banda 80 %
    pasos = np.arange(1, horizon + 1)
    margen = z * resid_std * np.sqrt(pasos)
    lo, hi = yhat - margen, yhat + margen

    last = df["ds"].iloc[-1]
    fechas = pd.bdate_range(last + pd.tseries.offsets.BDay(1), periods=horizon)
    return fechas, yhat, lo, hi, resid_std / df["y"].iloc[-1] * 100


def resumen(fechas, yhat, lo, hi, px, horizontes=HORIZONS):
    filas = []
    for h in horizontes:
        if h > len(yhat):
            continue
        i = h - 1
        ancho = (hi[i] - lo[i]) / yhat[i] if yhat[i] else float("nan")
        conf = "ALTA" if ancho < 0.20 else "MEDIA" if ancho < 0.45 else "BAJA"
        filas.append({
            "Horizonte": f"{h} días", "Fecha": fechas[i].date().isoformat(),
            "Precio esperado": round(float(yhat[i]), 3),
            "Banda 80% inf": round(float(lo[i]), 3), "Banda 80% sup": round(float(hi[i]), 3),
            "Variación %": round((yhat[i] / px - 1) * 100, 2),
            "Confianza": conf,
        })
    return pd.DataFrame(filas)


def forecast(ticker, period="3y", horizon=120, window=30, epochs=120):
    """Pipeline completo. Devuelve (fig, tabla, meta)."""
    df = descargar(ticker, period)
    if len(df) < window + 120:
        raise ValueError(f"Histórico insuficiente para LSTM ({len(df)} sesiones).")
    px = float(df["y"].iloc[-1])
    fechas, yhat, lo, hi, resid_pct = entrenar_predecir(df, horizon, window, epochs)
    tabla = resumen(fechas, yhat, lo, hi, px)
    fig = _plot(df, fechas, yhat, lo, hi, ticker)
    meta = {"precio_actual": px, "resid_pct": round(resid_pct, 2), "n": len(df)}
    return fig, tabla, meta


def _plot(df, fechas, yhat, lo, hi, ticker):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    hist = df.iloc[-250:]
    ax.plot(hist["ds"], hist["y"], color="black", lw=1.1, label="Histórico (1 año)")
    ax.plot(fechas, yhat, color="tab:red", lw=1.5, label="LSTM forecast")
    ax.fill_between(fechas, lo, hi, color="tab:red", alpha=0.15, label="Banda 80 %")
    ax.set_title(f"{ticker.upper()} — Forecast LSTM ({len(yhat)} días hábiles)")
    ax.set_xlabel("Fecha"); ax.set_ylabel("Precio"); ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description="Forecast con red LSTM (PyTorch).")
    ap.add_argument("ticker", nargs="?", default="SAB.MC")
    ap.add_argument("--horizon", type=int, default=120)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--period", default="3y")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()

    print(f"\nDescargando {a.ticker.upper()} y entrenando LSTM (ventana {a.window}, {a.epochs} épocas)...")
    fig, tabla, meta = forecast(a.ticker, a.period, a.horizon, a.window, a.epochs)
    print(f"Cierre actual: {meta['precio_actual']:.3f} · {meta['n']} sesiones · error residual ~{meta['resid_pct']}%\n")
    print(tabla.to_string(index=False))
    print("\nConfianza por ancho de banda: <20% ALTA · 20-45% MEDIA · >45% BAJA")
    print("> Red neuronal entrenada desde cero. Forecast estadístico, no recomendación.\n")
    if a.save:
        from pathlib import Path
        out = Path(__file__).resolve().parent / f"{a.ticker.replace('.','_').replace('^','')}_lstm.png"
        fig.savefig(out, dpi=110)
        print(f"Gráfico guardado: {out}\n")


if __name__ == "__main__":
    main()
