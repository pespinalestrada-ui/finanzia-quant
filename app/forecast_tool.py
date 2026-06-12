"""
Herramienta de forecasting financiero — FinanzIA.

Introduce un ticker (cualquiera de Yahoo Finance) y obtén:
  - Proyección a 30, 90 y 120 días hábiles.
  - Banda de confianza (intervalo 80 % de Prophet).
  - Nivel de confianza por horizonte (Alta / Media / Baja) calculado con un
    backtest sobre holdout + el ancho relativo de la banda.
  - Informe ejecutivo en castellano.

Motor: Prophet (modelo de cabecera del proyecto — intervalos nativos +
interpretabilidad). Si el ticker es SAB.MC, añade los eventos OPA BBVA como
holidays automáticamente.

Lanzar:
    cd app
    python forecast_tool.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yfinance as yf
from prophet import Prophet

# Eventos OPA solo aplican a SAB.MC
try:
    from src.data_loader import OPA_BBVA_EVENTS
except Exception:
    OPA_BBVA_EVENTS = None

HORIZONS = [30, 90, 120]          # días hábiles
MAX_H = max(HORIZONS)
INTERVAL_WIDTH = 0.80             # banda 80 %


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

def descargar(ticker: str, period: str = "3y") -> pd.DataFrame:
    """Descarga histórico diario de Yahoo Finance. Devuelve df['ds','y']."""
    t = yf.Ticker(ticker)
    hist = t.history(period=period, auto_adjust=False)
    if hist.empty:
        raise ValueError(f"Ticker '{ticker}' sin datos en Yahoo Finance.")
    hist = hist.reset_index()
    fecha_col = "Date" if "Date" in hist.columns else hist.columns[0]
    ds = pd.to_datetime(hist[fecha_col])
    if getattr(ds.dt, "tz", None) is not None:
        ds = ds.dt.tz_localize(None)
    df = pd.DataFrame({"ds": ds.dt.normalize(), "y": hist["Close"].astype(float)})
    df = df.dropna().reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

def _make_prophet(ticker: str) -> Prophet:
    kwargs = dict(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        interval_width=INTERVAL_WIDTH,
        changepoint_prior_scale=0.10,
    )
    if ticker.upper() == "SAB.MC" and OPA_BBVA_EVENTS is not None:
        kwargs["holidays"] = OPA_BBVA_EVENTS
    return Prophet(**kwargs)


def _mape(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = actual != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(actual[mask] - pred[mask]) / np.abs(actual[mask])) * 100)


def backtest_por_horizonte(df: pd.DataFrame, ticker: str) -> dict[int, float]:
    """
    Entrena sobre df[:-MAX_H], predice MAX_H días hábiles y mide el MAPE
    acumulado a cada horizonte. Devuelve {h: mape_h}.
    """
    if len(df) <= MAX_H + 120:
        return {h: float("nan") for h in HORIZONS}

    train = df.iloc[:-MAX_H].copy()
    test = df.iloc[-MAX_H:].copy().reset_index(drop=True)

    m = _make_prophet(ticker)
    m.fit(train)
    fut = m.make_future_dataframe(periods=MAX_H, freq="B", include_history=False)
    fcst = m.predict(fut)
    # alinear por fecha con el test real
    merged = test.merge(fcst[["ds", "yhat"]], on="ds", how="inner")
    if merged.empty:
        # fallback: alinear por posición
        n = min(len(test), len(fcst))
        merged = pd.DataFrame({"y": test["y"].values[:n], "yhat": fcst["yhat"].values[:n]})

    out = {}
    for h in HORIZONS:
        sub = merged.iloc[:h]
        out[h] = _mape(sub["y"].values, sub["yhat"].values)
    return out


def nivel_confianza(mape_h: float, rel_band: float) -> tuple[str, int]:
    """
    Combina fiabilidad del backtest (100-MAPE) y estrechez de banda
    (1-ancho_relativo) en un score 0-100 y una etiqueta.
    """
    reliab = max(0.0, 100.0 - (mape_h if not np.isnan(mape_h) else 50.0))
    tight = max(0.0, 100.0 - min(100.0, rel_band * 100.0))
    score = int(round(0.6 * reliab + 0.4 * tight))
    if score >= 70:
        etiqueta = "ALTA"
    elif score >= 50:
        etiqueta = "MEDIA"
    else:
        etiqueta = "BAJA"
    return etiqueta, score


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------

def forecast(ticker: str, period: str = "3y"):
    """
    Devuelve (fig, tabla_df, informe_md, meta).
    meta = dict con datos auxiliares (precio_actual, etc.).
    """
    ticker = ticker.strip().upper()
    df = descargar(ticker, period=period)
    if len(df) < 250:
        raise ValueError(f"Histórico insuficiente para '{ticker}' ({len(df)} sesiones; mínimo 250).")

    precio_actual = float(df["y"].iloc[-1])
    fecha_actual = df["ds"].iloc[-1]

    # backtest de fiabilidad por horizonte
    mapes = backtest_por_horizonte(df, ticker)

    # modelo final con TODO el histórico
    m = _make_prophet(ticker)
    m.fit(df)
    fut = m.make_future_dataframe(periods=MAX_H, freq="B", include_history=False)
    fcst = m.predict(fut)

    # filas a cada horizonte (posición h-1)
    filas = []
    for h in HORIZONS:
        row = fcst.iloc[h - 1]
        yhat = float(row["yhat"])
        lo = float(row["yhat_lower"])
        hi = float(row["yhat_upper"])
        rel_band = (hi - lo) / yhat if yhat else float("nan")
        etiqueta, score = nivel_confianza(mapes.get(h, float("nan")), rel_band)
        var = (yhat / precio_actual - 1) * 100
        filas.append({
            "Horizonte": f"{h} días",
            "Fecha objetivo": row["ds"].date().isoformat(),
            "Precio esperado": round(yhat, 3),
            "Banda 80% inf": round(lo, 3),
            "Banda 80% sup": round(hi, 3),
            "Variación %": round(var, 2),
            "Confianza": f"{etiqueta} ({score})",
            "MAPE backtest %": (round(mapes[h], 2) if not np.isnan(mapes.get(h, float("nan"))) else "n/d"),
        })
    tabla = pd.DataFrame(filas)

    fig = _plot(df, fcst, ticker, filas)
    informe = _informe(ticker, precio_actual, fecha_actual, filas, mapes)
    meta = {"precio_actual": precio_actual, "fecha_actual": fecha_actual.date().isoformat(),
            "n_sesiones": len(df)}
    return fig, tabla, informe, meta


def _plot(df, fcst, ticker, filas):
    fig, ax = plt.subplots(figsize=(11, 5))
    hist = df.iloc[-260:]
    ax.plot(hist["ds"], hist["y"], color="black", linewidth=1.2, label="Histórico (1 año)")
    ax.plot(fcst["ds"], fcst["yhat"], color="tab:blue", linewidth=1.5, label="Forecast")
    ax.fill_between(fcst["ds"], fcst["yhat_lower"], fcst["yhat_upper"],
                    color="tab:blue", alpha=0.18, label="Banda 80 %")
    colores = {"30 días": "tab:green", "90 días": "tab:orange", "120 días": "tab:red"}
    for f in filas:
        fecha = pd.to_datetime(f["Fecha objetivo"])
        ax.axvline(fecha, color=colores.get(f["Horizonte"], "gray"), linestyle="--", alpha=0.5)
        ax.scatter([fecha], [f["Precio esperado"]], color=colores.get(f["Horizonte"], "gray"), zorder=5)
        ax.annotate(f"{f['Horizonte']}\n{f['Precio esperado']:.2f}",
                    (fecha, f["Precio esperado"]), textcoords="offset points",
                    xytext=(6, 8), fontsize=8, color=colores.get(f["Horizonte"], "gray"))
    ax.set_title(f"{ticker} — Forecast 30/90/120 días hábiles (Prophet, banda 80 %)")
    ax.set_xlabel("Fecha"); ax.set_ylabel("Precio")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig


def _tendencia(var120: float) -> str:
    if var120 > 8:    return "claramente alcista"
    if var120 > 2:    return "moderadamente alcista"
    if var120 > -2:   return "lateral"
    if var120 > -8:   return "moderadamente bajista"
    return "claramente bajista"


def _informe(ticker, precio_actual, fecha_actual, filas, mapes):
    var120 = filas[-1]["Variación %"]
    tend = _tendencia(var120)
    confs = [f["Confianza"].split()[0] for f in filas]
    conf_global = "ALTA" if confs.count("ALTA") >= 2 else ("BAJA" if confs.count("BAJA") >= 2 else "MEDIA")

    lineas = []
    lineas.append(f"# Informe de previsión — {ticker}")
    lineas.append("")
    lineas.append(f"*Generado por la herramienta FinanzIA el {datetime.utcnow().date()}.*")
    lineas.append("")
    lineas.append("## Situación de partida")
    lineas.append(f"- Precio de cierre más reciente: **{precio_actual:.3f}** (a {fecha_actual.date()}).")
    lineas.append(f"- Tendencia proyectada a 120 días: **{tend}** ({var120:+.1f} %).")
    lineas.append(f"- Nivel de confianza global del forecast: **{conf_global}**.")
    lineas.append("")
    lineas.append("## Proyección por horizonte")
    for f in filas:
        lineas.append(
            f"- **{f['Horizonte']}** (→ {f['Fecha objetivo']}): "
            f"precio esperado **{f['Precio esperado']:.3f}** "
            f"({f['Variación %']:+.2f} %), banda 80 % [{f['Banda 80% inf']:.3f}, {f['Banda 80% sup']:.3f}], "
            f"confianza **{f['Confianza']}**."
        )
    lineas.append("")
    lineas.append("## Cómo leer la confianza")
    lineas.append(
        "El nivel de confianza (0-100) combina dos señales: (a) la fiabilidad del modelo "
        "medida con un *backtest* sobre los últimos 120 días hábiles —error MAPE por horizonte— "
        "y (b) la estrechez de la banda de predicción al 80 %. **ALTA ≥ 70 · MEDIA 50-69 · BAJA < 50**. "
        "La confianza baja con el horizonte: a 120 días la incertidumbre es mayor que a 30."
    )
    mape_txt = " · ".join(
        f"{h}d: {(f'{mapes[h]:.1f}%' if not np.isnan(mapes.get(h, float('nan'))) else 'n/d')}"
        for h in HORIZONS
    )
    lineas.append(f"- Error de backtest (MAPE) por horizonte: {mape_txt}.")
    lineas.append("")
    lineas.append("## Advertencia")
    lineas.append(
        "> Esta proyección es un ejercicio estadístico basado en el histórico de precios. "
        "**No constituye recomendación de inversión.** Eventos no anticipados (resultados, noticias "
        "regulatorias, shocks de mercado) pueden mover el precio fuera de la banda proyectada."
    )
    return "\n".join(lineas)


# ---------------------------------------------------------------------------
# UI Gradio
# ---------------------------------------------------------------------------

def _run_ui(ticker, period):
    try:
        fig, tabla, informe, meta = forecast(ticker, period=period)
        encabezado = (f"**{ticker.strip().upper()}** · {meta['n_sesiones']} sesiones · "
                      f"cierre {meta['precio_actual']:.3f} ({meta['fecha_actual']})")
        return fig, tabla, encabezado + "\n\n" + informe
    except Exception as e:
        import matplotlib.pyplot as plt
        f, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", wrap=True)
        ax.axis("off")
        return f, pd.DataFrame(), f"**Error:** {e}"


def build_app():
    import gradio as gr
    with gr.Blocks(title="FinanzIA — Herramienta de Forecasting") as app:
        gr.Markdown("# FinanzIA — Herramienta de Forecasting")
        gr.Markdown("Introduce un **ticker** de Yahoo Finance (ej. `SAB.MC`, `AAPL`, `^IBEX`, `BTC-EUR`). "
                    "Proyección a 30 / 90 / 120 días hábiles con nivel de confianza e informe.")
        with gr.Row():
            ticker_in = gr.Textbox(label="Ticker", value="SAB.MC", scale=3)
            period_in = gr.Dropdown(["2y", "3y", "5y", "10y"], value="3y", label="Histórico", scale=1)
            btn = gr.Button("Generar forecast", variant="primary", scale=1)
        plot_out = gr.Plot(label="Proyección")
        tabla_out = gr.Dataframe(label="Resumen por horizonte", wrap=True)
        informe_out = gr.Markdown(label="Informe")
        btn.click(_run_ui, inputs=[ticker_in, period_in], outputs=[plot_out, tabla_out, informe_out])
        ticker_in.submit(_run_ui, inputs=[ticker_in, period_in], outputs=[plot_out, tabla_out, informe_out])
    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7861, inbrowser=False)
