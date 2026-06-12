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

import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr

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
    ax.plot(sab["Date"], sab["Close"], color="black", linewidth=1)
    for ev_date in OPA_BBVA_EVENTS["ds"]:
        ax.axvline(ev_date, color="red", linestyle="--", alpha=0.3)
    ax.set_title("Banco Sabadell (SAB.MC) — cierre diario · líneas rojas = hitos OPA BBVA")
    ax.set_xlabel("Fecha"); ax.set_ylabel("EUR")
    plt.tight_layout()
    return fig


def plot_forecast(modelo):
    sab = cargar_historico()
    fcst = cargar_forecast(modelo)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(sab["Date"].iloc[-365:], sab["Close"].iloc[-365:], color="black", label="Histórico (último año)")
    if fcst is None:
        ax.text(0.5, 0.5, f"Falta el CSV de '{modelo}'.\nEjecuta el notebook que lo genera.",
                transform=ax.transAxes, ha="center", va="center")
        return fig
    ax.plot(fcst["ds"], fcst["yhat"], color="tab:blue", label=f"Forecast 90d ({modelo})")
    if "yhat_lower" in fcst.columns:
        ax.fill_between(fcst["ds"], fcst["yhat_lower"], fcst["yhat_upper"],
                        color="tab:blue", alpha=0.2, label="IC 80%")
    ax.set_title(f"SAB.MC — Forecast 90 días · {modelo}")
    ax.set_xlabel("Fecha"); ax.set_ylabel("EUR")
    ax.legend(loc="upper left")
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
        from datetime import datetime
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

*Generado el {datetime.utcnow().date()} a partir del modelo '{modelo}'.*

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


with gr.Blocks(title="FinanzIA — Forecast SAB.MC bajo OPA BBVA") as app:
    gr.Markdown("# FinanzIA — Forecast Banco Sabadell (SAB.MC) bajo escenario OPA BBVA")
    gr.Markdown("Proyecto final · Microtítulo IA Generativa aplicada a Finanzas · UPV/EHU 2025-2026")

    with gr.Tabs():
        with gr.Tab("1 · Histórico"):
            gr.Markdown("Serie de cierre diario con los hitos de la OPA marcados.")
            btn_h = gr.Button("Actualizar gráfico")
            plot_h = gr.Plot()
            btn_h.click(plot_historico, outputs=plot_h)
            app.load(plot_historico, outputs=plot_h)

        with gr.Tab("2 · Forecast 90 d"):
            gr.Markdown("Selecciona el modelo cuya previsión quieres visualizar.")
            modelo_sel = gr.Dropdown(MODELOS, value=MODELOS[0], label="Modelo")
            btn_f = gr.Button("Generar forecast")
            plot_f = gr.Plot()
            btn_f.click(plot_forecast, inputs=modelo_sel, outputs=plot_f)

        with gr.Tab("3 · Narrativa GenAI"):
            gr.Markdown("Informe redactado por el agente (template-based · módulo 10 → smolagents).")
            modelo_n = gr.Dropdown(MODELOS, value=MODELOS[0], label="Modelo")
            btn_n = gr.Button("Generar informe")
            out_n = gr.Markdown()
            btn_n.click(narrativa, inputs=modelo_n, outputs=out_n)


if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
