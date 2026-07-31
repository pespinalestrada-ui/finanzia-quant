"""
App Gradio del proyecto final FinanzIA — Forecast SAB.MC con escenarios OPA.

Lanzar con:
    cd app
    python gradio_app.py

3 pestañas:
  1. Histórico SAB.MC + eventos OPA
  2. Forecast 90 días (selección de modelo)
  3. Narrativa GenAI del agente
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cuant_trading" / "dashboard"))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

# tema Nocturne: el mismo que la Mesa cuantitativa, para que las dos apps del
# proyecto se vean igual. Al importar finanzia_charts se aplican las rcParams.
from finanzia_charts import C, style, band, marker
from finanzia_theme import HEAD, THEME, CSS, TOPBAR_HTML

from src.data_loader import OPA_BBVA_EVENTS


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

DATA = ROOT / "data"


def cargar_historico() -> pd.DataFrame:
    sab = pd.read_csv(DATA / "sab_5y_clean.csv", parse_dates=["Date"])
    return sab


def cargar_forecast(modelo: str) -> pd.DataFrame | None:
    mapping = {
        "Prophet + OPA holidays":              DATA / "sab_forecast_prophet_90d.csv",
        "PyCaret univariante":                 DATA / "sab_forecast_pycaret_90d.csv",
        "PyCaret multi-horizonte (IBEX+FX)":   DATA / "sab_forecast_pycaret_horizontes_90d.csv",
        "Prophet + OPA + IBEX + tipo BCE":     DATA / "sab_forecast_regresores_90d.csv",
    }
    path = mapping.get(modelo)
    if path is None or not path.exists():
        return None
    fcst = pd.read_csv(path, parse_dates=["ds"])
    return fcst


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_historico():
    sab = cargar_historico()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(sab["Date"], sab["Close"], color=C.text, lw=1.3, label="Cierre SAB.MC")
    for ev_date in OPA_BBVA_EVENTS["ds"]:
        ax.axvline(ev_date, color=C.gold, ls="--", lw=1, alpha=0.55)
    # una sola entrada de leyenda para los hitos, no una por línea
    ax.plot([], [], color=C.gold, ls="--", lw=1, label="Hitos de la OPA BBVA")
    style(ax, titulo="Banco Sabadell — cierre diario y los hitos de la OPA",
          kicker="SAB.MC · SERIE COMPLETA · CIERRE AJUSTADO", ylabel="EUR", xlabel="Fecha")
    plt.tight_layout()
    return fig


def plot_forecast(modelo):
    sab = cargar_historico()
    fcst = cargar_forecast(modelo)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(sab["Date"].iloc[-365:], sab["Close"].iloc[-365:],
            color=C.neutral, lw=1.3, label="Histórico (último año)")
    if fcst is None:
        ax.text(0.5, 0.5, f"Falta el CSV de '{modelo}'.\nEjecuta el notebook que lo genera.",
                transform=ax.transAxes, ha="center", va="center", color=C.neutral, fontsize=11.5)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.grid(False)
        return fig
    ax.plot(fcst["ds"], fcst["yhat"], color=C.acc, lw=1.8, label="Forecast 90 días")
    if "yhat_lower" in fcst.columns:
        band(ax, fcst["ds"], fcst["yhat_lower"], fcst["yhat_upper"], label="Banda 80 %")
        marker(ax, fcst["ds"].iloc[-1], float(fcst["yhat"].iloc[-1]),
               f"{float(fcst['yhat'].iloc[-1]):.3f} €")
    style(ax, titulo=f"SAB.MC — previsión a 90 días", kicker=modelo,
          ylabel="EUR", xlabel="Fecha")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Narrativa GenAI
# ---------------------------------------------------------------------------

def narrativa(modelo):
    # Carga perezosa para no fallar si no están los CSVs
    try:
        import importlib.util
        nb_agent_path = ROOT / "notebooks" / "06_agente_explicacion.ipynb"
        # Importamos las funciones replicándolas aquí (más simple que ejecutar el .ipynb)
        from datetime import datetime, timezone
        import numpy as np

        sab = pd.read_csv(DATA / "sab_5y_clean.csv", parse_dates=["Date"])
        last = sab.iloc[-1]; first = sab.iloc[0]
        resumen = {
            "cierre_actual": round(float(last["Close"]), 3),
            "rango": f"{first['Date'].date()} -> {last['Date'].date()}",
            "sesiones": len(sab),
            "retorno_5y_pct": round((last["Close"]/first["Close"]-1)*100, 1),
            "volatilidad_anualizada_pct": round(float(sab["Close"].pct_change().std()*np.sqrt(252)*100), 2),
        }

        fcst = cargar_forecast(modelo)
        if fcst is None:
            return f"No hay forecast disponible para '{modelo}'. Ejecuta el notebook correspondiente."

        fin = fcst.iloc[-1]
        var = (float(fin["yhat"])/resumen["cierre_actual"] - 1) * 100
        if var > 5:    tendencia = "al alza significativa"
        elif var > 1:  tendencia = "ligeramente alcista"
        elif var > -1: tendencia = "lateral"
        elif var > -5: tendencia = "ligeramente bajista"
        else:          tendencia = "a la baja significativa"

        hitos = OPA_BBVA_EVENTS.sort_values("ds").tail(3)
        bullets = "\n".join(f"- {row['ds'].date()}: {row['holiday']}" for _, row in hitos.iterrows())

        return f"""## Informe FinanzIA — SAB.MC

*Generado el {datetime.now(timezone.utc).date()} a partir del modelo '{modelo}'.*

**Situación actual:** cierre **{resumen['cierre_actual']} EUR** sobre serie de {resumen['sesiones']} sesiones ({resumen['rango']}). Retorno 5 años **{resumen['retorno_5y_pct']}%**, volatilidad anualizada **{resumen['volatilidad_anualizada_pct']}%**.

**Previsión 90 días:** trayectoria **{tendencia}** hacia **{fin['ds'].date()}**. Precio esperado **{float(fin['yhat']):.3f} EUR** ({var:+.2f}% sobre cierre actual). IC 80%: [{float(fin.get('yhat_lower', float('nan'))):.3f}, {float(fin.get('yhat_upper', float('nan'))):.3f}] EUR.

**Hitos OPA recientes considerados:**
{bullets}

> Esta previsión NO constituye recomendación de inversión. Combina forecast estadístico con eventos discretos de la OPA hostil BBVA-Sabadell.
"""
    except Exception as e:
        return f"Error generando narrativa: {e}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

MODELOS = [
    "Prophet + OPA holidays",
    "PyCaret univariante",
    "PyCaret multi-horizonte (IBEX+FX)",
    "Prophet + OPA + IBEX + tipo BCE",
]


def _topbar():
    """Barra superior con el último cierre REAL del CSV cacheado. Sin chips de
    apertura de mercado: esta app trabaja sobre CSV, no en vivo, y un indicador
    de 'mercado abierto' aquí sería decorativo."""
    tk, precio, cambio = "SAB.MC", "—", ""
    try:
        sab = cargar_historico()
        c = sab["Close"].astype(float).dropna()
        precio = f"{c.iloc[-1]:.3f} €".replace(".", ",")
        if len(c) > 1:
            cambio = f"{(c.iloc[-1] / c.iloc[-2] - 1) * 100:+.2f}%".replace(".", ",")
    except Exception:
        pass                       # sin CSV cacheado: se queda en '—'
    return TOPBAR_HTML(kicker="Forecast SAB.MC · OPA BBVA", mercados=False,
                       ticker=tk, precio=precio, cambio=cambio, capital="",
                       nota="UPV/EHU · proyecto final")


with gr.Blocks(title="FinanzIA — Forecast SAB.MC bajo OPA BBVA",
               head=HEAD, theme=THEME, css=CSS) as app:
    gr.HTML(_topbar())
    gr.Markdown("Forecast a 90 días de **Banco Sabadell** con los hitos de la **OPA hostil de "
                "BBVA** como eventos discretos. Microtítulo IA Generativa aplicada a Finanzas · "
                "UPV/EHU 2025-2026.")

    with gr.Tabs():
        with gr.Tab("1 · Histórico"):
            gr.Markdown("Serie de cierre diario con los hitos de la OPA marcados.")
            with gr.Row():
                btn_h = gr.Button("Actualizar gráfico", variant="primary", scale=0)
            plot_h = gr.Plot(show_label=False)
            btn_h.click(plot_historico, outputs=plot_h)
            app.load(plot_historico, outputs=plot_h)

        with gr.Tab("2 · Forecast 90 d"):
            gr.Markdown("Selecciona el modelo cuya previsión quieres visualizar.")
            with gr.Row():
                modelo_sel = gr.Dropdown(MODELOS, value=MODELOS[0], label="Modelo", scale=3)
                btn_f = gr.Button("Generar forecast", variant="primary", scale=0)
            plot_f = gr.Plot(show_label=False)
            btn_f.click(plot_forecast, inputs=modelo_sel, outputs=plot_f)

        with gr.Tab("3 · Narrativa GenAI"):
            gr.Markdown("Informe redactado por el agente (template-based · módulo 10 → smolagents).")
            with gr.Row():
                modelo_n = gr.Dropdown(MODELOS, value=MODELOS[0], label="Modelo", scale=3)
                btn_n = gr.Button("Generar informe", variant="primary", scale=0)
            out_n = gr.Markdown()
            btn_n.click(narrativa, inputs=modelo_n, outputs=out_n)


if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
