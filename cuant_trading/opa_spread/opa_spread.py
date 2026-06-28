"""
opa_spread — spread de arbitraje de fusión (merger-arb) de la OPA BBVA → Sabadell.

Es el ÁNGULO DIFERENCIAL del proyecto convertido en señal medible y tradeable
(a diferencia del precio a 90 días, que no bate al azar): la OPA es un canje de
acciones, así que existe un precio implícito de la oferta y un spread frente al
precio de mercado de SAB.

  valor_oferta_por_SAB = (1 / canje) · precio_BBVA + efectivo_por_SAB
  spread               = precio_SAB / valor_oferta − 1

  - spread < 0  → SAB cotiza por DEBAJO de la oferta: el mercado descuenta riesgo
    de que la OPA NO se complete (o exige mejora). Hueco de arbitraje si crees que sí.
  - spread ≈ 0  → mercado da la operación casi por hecha a los términos actuales.
  - spread > 0  → SAB por ENCIMA: mercado espera mejora de oferta o contra-opa.

Probabilidad implícita de éxito (modelo de 2 estados):
  precio_SAB = p·valor_oferta + (1−p)·precio_fracaso
  → p = (precio_SAB − precio_fracaso) / (valor_oferta − precio_fracaso)

Los términos del canje son PARÁMETROS (ajústalos a la oferta vigente); por defecto
toma el canje original (1 BBVA por 4,83 SAB) sin efectivo. No es asesoramiento.

Uso:
    python opa_spread.py
    python opa_spread.py --canje 4.83 --efectivo 0.70 --periodo 3y
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

# eventos OPA del proyecto (src/data_loader.py)
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
try:
    from data_loader import OPA_BBVA_EVENTS
except Exception:
    OPA_BBVA_EVENTS = None

CANJE_DEFAULT = 4.83        # acciones SAB por 1 acción BBVA (canje original)
EFECTIVO_DEFAULT = 0.0      # € en efectivo por acción SAB (ajusta a términos vigentes)
PRE_ANUNCIO = "2024-04-29"  # día previo al 1er hito → ancla del escenario "fracaso"


def _serie(ticker, period):
    h = yf.Ticker(ticker).history(period=period, auto_adjust=False)
    if h.empty:
        raise ValueError(f"Ticker '{ticker}' sin datos.")
    s = h["Close"].copy()
    s.index = pd.to_datetime(s.index)
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    s.index = s.index.normalize()
    return s.astype(float)


def calcular(canje=CANJE_DEFAULT, efectivo=EFECTIVO_DEFAULT, period="3y",
             sab="SAB.MC", bbva="BBVA.MC", precio_fracaso=None):
    sab_s = _serie(sab, period)
    bbva_s = _serie(bbva, period)
    df = pd.DataFrame({"sab": sab_s, "bbva": bbva_s}).dropna()
    if df.empty:
        raise ValueError("Sin solape de fechas entre SAB y BBVA.")
    df["oferta"] = (1.0 / canje) * df["bbva"] + efectivo
    df["spread_pct"] = (df["sab"] / df["oferta"] - 1.0) * 100.0

    # ancla del escenario "fracaso": precio SAB pre-anuncio (o el override del usuario)
    if precio_fracaso is None:
        idx = (df.index - pd.Timestamp(PRE_ANUNCIO)).to_series().abs()
        precio_fracaso = float(df["sab"].iloc[int(idx.values.argmin())]) if len(df) else float("nan")

    ult = df.iloc[-1]
    oferta_hoy, sab_hoy = float(ult["oferta"]), float(ult["sab"])
    spread_hoy = float(ult["spread_pct"])
    denom = oferta_hoy - precio_fracaso
    prob = (sab_hoy - precio_fracaso) / denom if denom != 0 else float("nan")
    prob = float(min(1.0, max(0.0, prob))) if not np.isnan(prob) else float("nan")

    meta = {
        "sab_hoy": sab_hoy, "bbva_hoy": float(ult["bbva"]), "oferta_hoy": oferta_hoy,
        "spread_hoy": spread_hoy, "precio_fracaso": float(precio_fracaso),
        "prob_exito": prob, "canje": canje, "efectivo": efectivo,
        "spread_medio_30d": float(df["spread_pct"].iloc[-30:].mean()) if len(df) >= 30 else spread_hoy,
    }
    return df, meta


def _plot(df, meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(df.index, df["sab"], color="tab:blue", lw=1.2, label="SAB (mercado)")
    ax1.plot(df.index, df["oferta"], color="tab:red", lw=1.2, ls="--", label="Valor oferta BBVA")
    ax1.set_ylabel("€ / acción SAB"); ax1.legend(loc="upper left")
    ax1.set_title("OPA BBVA → Sabadell · precio de mercado vs valor implícito de la oferta")
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.plot(df.index, df["spread_pct"], color="tab:purple", lw=1.1)
    ax2.fill_between(df.index, df["spread_pct"], 0, color="tab:purple", alpha=0.15)
    ax2.set_ylabel("Spread %"); ax2.set_xlabel("Fecha")
    if OPA_BBVA_EVENTS is not None:
        for d in pd.to_datetime(OPA_BBVA_EVENTS["ds"]).unique():
            if df.index.min() <= d <= df.index.max():
                ax1.axvline(d, color="gray", ls=":", lw=0.7, alpha=0.6)
                ax2.axvline(d, color="gray", ls=":", lw=0.7, alpha=0.6)
    fig.tight_layout()
    return fig


def resumen_tabla(meta):
    return pd.DataFrame([
        {"Métrica": "Precio SAB (mercado)", "Valor": f"{meta['sab_hoy']:.3f} €"},
        {"Métrica": "Precio BBVA", "Valor": f"{meta['bbva_hoy']:.3f} €"},
        {"Métrica": f"Valor oferta (canje 1×{meta['canje']} + {meta['efectivo']:.2f}€)",
         "Valor": f"{meta['oferta_hoy']:.3f} €"},
        {"Métrica": "Spread hoy", "Valor": f"{meta['spread_hoy']:+.2f} %"},
        {"Métrica": "Spread medio 30d", "Valor": f"{meta['spread_medio_30d']:+.2f} %"},
        {"Métrica": "Precio escenario 'fracaso' (pre-anuncio)", "Valor": f"{meta['precio_fracaso']:.3f} €"},
        {"Métrica": "Prob. implícita de éxito de la OPA",
         "Valor": f"{meta['prob_exito']*100:.0f} %" if not np.isnan(meta['prob_exito']) else "n/d"},
    ])


def lectura(meta):
    s = meta["spread_hoy"]
    if s < -1.5:
        tesis = ("SAB cotiza por DEBAJO de la oferta: el mercado descuenta riesgo de no "
                 "completarse o exige mejora. Si crees que la OPA sale, hay hueco de arbitraje.")
    elif s > 1.5:
        tesis = ("SAB cotiza por ENCIMA de la oferta: el mercado espera mejora de los términos "
                 "o un escenario alternativo (contra-opa).")
    else:
        tesis = "Spread ~0: el mercado da la operación casi por hecha a los términos actuales."
    return tesis


def forecast(canje=CANJE_DEFAULT, efectivo=EFECTIVO_DEFAULT, period="3y", precio_fracaso=None):
    """Adaptador para el dashboard. Devuelve (fig, tabla, meta, texto)."""
    df, meta = calcular(canje, efectivo, period, precio_fracaso=precio_fracaso)
    return _plot(df, meta), resumen_tabla(meta), meta, lectura(meta)


def main():
    ap = argparse.ArgumentParser(description="Spread de arbitraje de la OPA BBVA→Sabadell.")
    ap.add_argument("--canje", type=float, default=CANJE_DEFAULT, help="Acciones SAB por 1 BBVA.")
    ap.add_argument("--efectivo", type=float, default=EFECTIVO_DEFAULT, help="€ efectivo por SAB.")
    ap.add_argument("--periodo", default="3y")
    ap.add_argument("--fracaso", type=float, default=None, help="Precio SAB si la OPA fracasa.")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()

    df, meta = calcular(a.canje, a.efectivo, a.periodo, precio_fracaso=a.fracaso)
    print("\n=== OPA BBVA → Sabadell · spread de arbitraje de fusión ===")
    print(resumen_tabla(meta).to_string(index=False))
    print(f"\nLectura: {lectura(meta)}")
    print("> Términos del canje configurables (--canje/--efectivo). NO es asesoramiento de inversión.\n")
    if a.save:
        out = Path(__file__).resolve().parent / "opa_spread.png"
        _plot(df, meta).savefig(out, dpi=110)
        print(f"Gráfico guardado: {out}\n")


if __name__ == "__main__":
    main()
