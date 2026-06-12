"""
autogluon_forecast — forecast AutoML con AutoGluon TimeSeries.

Implementación local del notebook "Forecasting_Sabadell_IBEX_BancaEuropea110":
  1. Descarga el ticker (y opcionalmente IBEX 35 + banca europea como covariables).
  2. Entrena varios modelos (SeasonalNaive, ETS, Theta, tabulares, Chronos2, TFT)
     y los combina en un WeightedEnsemble — todo automático.
  3. Devuelve la predicción con CUANTILES 0.1-0.9 (distribución completa de
     incertidumbre, más rica que un intervalo único).

Diferencias vs el notebook original:
  - Parametrizado: cualquier ticker, horizonte y time_limit por CLI.
  - `fillna(method='ffill')` (deprecado) → `.ffill()`.
  - Resumen con lectura de confianza por amplitud de cuantiles.
  - Leaderboard de modelos impreso (saber quién gana, no solo el ensemble).

Uso:
    python autogluon_forecast.py SAB.MC
    python autogluon_forecast.py SAB.MC --horizon 110 --covariables --save
    python autogluon_forecast.py AAPL --horizon 90 --time-limit 300 --preset medium_quality

Nota: el entrenamiento tarda lo que marque --time-limit (defecto 180 s).
"""
import argparse
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import warnings
warnings.filterwarnings("ignore")

# IMPORTANTE: torch debe importarse ANTES que autogluon en Windows.
# AutoGluon carga lightgbm/xgboost (OpenMP) y si torch llega después su
# c10.dll falla con WinError 1114. Importándolo primero no hay conflicto.
import torch  # noqa: F401

from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

COVARIABLES = {"ibex": "^IBEX", "banca_eu": "EXV1.DE"}


def descargar(ticker, period="3y", con_covariables=False):
    """DataFrame [timestamp, target(, ibex, banca_eu), item_id]."""
    if con_covariables:
        symbols = [ticker] + list(COVARIABLES.values())
        raw = yf.download(symbols, period=period, interval="1d", progress=False)["Close"]
        raw = raw.ffill().dropna()
        df = raw.reset_index()
        ren = {"Date": "timestamp", ticker: "target"}
        ren.update({v: k for k, v in COVARIABLES.items()})
        df = df.rename(columns=ren)
    else:
        raw = yf.download(ticker, period=period, interval="1d", progress=False)
        if raw.empty:
            raise SystemExit(f"Sin datos para '{ticker}'.")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        df = raw.reset_index()[["Date", "Close"]].rename(columns={"Date": "timestamp", "Close": "target"})
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df["item_id"] = ticker.upper()
    return df.dropna(subset=["target"]).reset_index(drop=True)


def entrenar_y_predecir(df, horizon=110, preset="medium_quality", time_limit=180):
    """Entrena AutoGluon y devuelve (predicciones, leaderboard)."""
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
    import tempfile

    train = TimeSeriesDataFrame.from_data_frame(df, id_column="item_id", timestamp_column="timestamp")
    predictor = TimeSeriesPredictor(
        prediction_length=horizon,
        target="target",
        eval_metric="MASE",
        freq="B",
        path=tempfile.mkdtemp(prefix="ag_forecast_"),
        verbosity=1,
    )
    predictor.fit(train, presets=preset, time_limit=time_limit)
    preds = predictor.predict(train)
    lb = predictor.leaderboard(silent=True) if hasattr(predictor, "leaderboard") else None
    return preds, lb


def resumen(preds, df, horizontes=(30, 90, 110)):
    """Tabla por horizonte con mediana, banda 10-90 % y lectura de confianza."""
    px = float(df["target"].iloc[-1])
    filas = []
    for h in horizontes:
        if h > len(preds):
            continue
        row = preds.iloc[h - 1]
        med = float(row["mean"]); lo = float(row["0.1"]); hi = float(row["0.9"])
        ancho_rel = (hi - lo) / med if med else float("nan")
        conf = "ALTA" if ancho_rel < 0.20 else "MEDIA" if ancho_rel < 0.45 else "BAJA"
        filas.append({
            "Horizonte": f"{h} días",
            "Fecha": preds.index.get_level_values("timestamp")[h - 1].date().isoformat(),
            "Mediana": round(med, 3),
            "P10": round(lo, 3),
            "P90": round(hi, 3),
            "Variación %": round((med / px - 1) * 100, 2),
            "Ancho banda %": round(ancho_rel * 100, 1),
            "Confianza": conf,
        })
    return pd.DataFrame(filas), px


def plot(df, preds, ticker, out_png=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6))
    hist = df.iloc[-250:]
    ax.plot(hist["timestamp"], hist["target"], color="black", lw=1.1, label="Histórico (1 año)")
    ts = preds.index.get_level_values("timestamp")
    ax.plot(ts, preds["mean"], color="red", lw=1.4, label="Mediana AutoGluon")
    ax.fill_between(ts, preds["0.1"], preds["0.9"], color="red", alpha=0.15, label="P10–P90")
    ax.fill_between(ts, preds["0.3"], preds["0.7"], color="red", alpha=0.25, label="P30–P70")
    ax.set_title(f"{ticker.upper()} — AutoGluon TimeSeries ({len(preds)} días hábiles, cuantiles)")
    ax.set_xlabel("Fecha"); ax.set_ylabel("Precio"); ax.legend(loc="upper left")
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=110)
        print(f"Gráfico guardado: {out_png}")
    return fig


def main():
    ap = argparse.ArgumentParser(description="Forecast AutoML con AutoGluon TimeSeries.")
    ap.add_argument("ticker", nargs="?", default="SAB.MC")
    ap.add_argument("--horizon", type=int, default=110)
    ap.add_argument("--period", default="3y")
    ap.add_argument("--preset", default="medium_quality",
                    choices=["fast_training", "medium_quality", "high_quality", "best_quality"])
    ap.add_argument("--time-limit", type=int, default=180, help="Segundos máximos de entrenamiento.")
    ap.add_argument("--covariables", action="store_true",
                    help="Añade IBEX 35 y banca europea (EXV1.DE) como past covariates.")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()

    print(f"\nDescargando {a.ticker.upper()}" + (" + IBEX + banca EU" if a.covariables else "") + "...")
    df = descargar(a.ticker, a.period, a.covariables)
    print(f"{len(df)} sesiones ({df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()})")

    print(f"\nEntrenando AutoGluon (preset {a.preset}, máx {a.time_limit}s)...")
    preds, lb = entrenar_y_predecir(df, a.horizon, a.preset, a.time_limit)

    if lb is not None and len(lb):
        print("\n=== Leaderboard (MASE, mayor=mejor con signo invertido) ===")
        cols = [c for c in ["model", "score_val", "fit_time_marginal"] if c in lb.columns]
        print(lb[cols].to_string(index=False))

    tabla, px = resumen(preds, df, horizontes=(30, 90, min(a.horizon, 120)))
    print(f"\n=== Proyección {a.ticker.upper()} (cierre actual {px:.3f}) ===")
    print(tabla.to_string(index=False))
    print("\nConfianza por ancho de banda P10-P90: <20% ALTA · 20-45% MEDIA · >45% BAJA")
    print("> Forecast estadístico — no es recomendación de inversión.\n")

    if a.save:
        out = Path(__file__).resolve().parent / f"{a.ticker.replace('.','_').replace('^','')}_autogluon.png"
        plot(df, preds, a.ticker, out)
        csv = Path(__file__).resolve().parent / f"{a.ticker.replace('.','_').replace('^','')}_autogluon_preds.csv"
        preds.reset_index().to_csv(csv, index=False)
        print(f"Cuantiles guardados: {csv}\n")


if __name__ == "__main__":
    main()
