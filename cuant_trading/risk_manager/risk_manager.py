"""
risk_manager — convierte SEÑALES en un PLAN de órdenes con control de riesgo.

El puente entre "qué operar" (signal_engine) y "ejecutar" (Alpaca paper). Para cada
señal accionable calcula CUÁNTO con reglas de riesgo de verdad:

  - Volatility targeting: cada posición se dimensiona para una vol objetivo
    (peso ∝ vol_objetivo / vol_activo) → las más volátiles pesan menos.
  - Máximo de posiciones: solo las N señales de mayor convicción (|score|).
  - Tope de exposición: la suma de pesos no supera el 100 % (sin apalancar).
  - Stop por ATR: stop = precio ∓ k·ATR; de ahí el riesgo en € por posición.
  - Tope de riesgo diario: avisa si el riesgo total planificado supera tu límite.

Vol GARCH(1,1) y ATR vienen de position_sizer. No es recomendación de inversión.

Uso:
    python risk_manager.py AAPL MSFT NVDA GOOGL AMZN JPM XOM KO --capital 10000
    python risk_manager.py --file watchlist.txt --capital 20000 --target-vol 0.12 --max-pos 4
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

_SUITE = Path(__file__).resolve().parents[1]
for p in (_SUITE / "signal_engine", _SUITE / "position_sizer", _SUITE / "veredicto_backtest"):
    sys.path.insert(0, str(p))
import signal_engine as SE
import position_sizer as PS


def _vol_anual(ticker, df):
    """Vol anualizada: GARCH si se puede; si no, realizada 60d."""
    try:
        v = PS.garch_volatility(ticker)
        if v and not np.isnan(v) and v > 0:
            return float(v)
    except Exception:
        pass
    r = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    return float(r.iloc[-60:].std() * np.sqrt(252)) if len(r) >= 60 else 0.30


def generar_plan(tickers, capital=10000.0, target_vol=0.15, max_pos=5,
                 atr_mult=2.0, umbral=0.35, riesgo_dia_pct=0.06, peso_max=0.35):
    """Devuelve (plan_df, meta). plan_df = órdenes a colocar con tamaño y stop."""
    señales = SE.generar(tickers, umbral)
    if señales.empty:
        return pd.DataFrame(), {"mensaje": "Sin señales."}
    acc = señales[señales["Señal"].str.contains("COMPRAR|VENDER")].copy()
    if acc.empty:
        return pd.DataFrame(), {"mensaje": "Ninguna señal accionable (todo MANTENER)."}
    acc["abs"] = acc["Score"].abs()
    acc = acc.sort_values("abs", ascending=False).head(int(max_pos))

    # pesos por volatility targeting
    info = []
    for _, row in acc.iterrows():
        tk = row["Ticker"]
        df = SE.VB.descargar(tk, "1y")
        if df is None or df.empty:
            continue
        vol = _vol_anual(tk, df)
        try:
            atr, px = PS.atr_actual(tk)
        except Exception:
            atr, px = np.nan, float(row["Precio"])
        if not (atr and atr > 0):
            atr = px * 0.02
        peso = min(peso_max, target_vol / vol) if vol > 0 else 0.0
        info.append({"Ticker": tk, "Señal": row["Señal"], "Score": row["Score"],
                     "Precio": px, "vol": vol, "atr": atr, "peso_raw": peso})
    if not info:
        return pd.DataFrame(), {"mensaje": "Sin datos para dimensionar."}
    plan = pd.DataFrame(info)
    # normaliza para no pasar del 100 % de exposición
    suma = plan["peso_raw"].sum()
    plan["peso"] = plan["peso_raw"] / suma * min(1.0, suma) if suma > 0 else 0.0

    filas, riesgo_total, exp_total = [], 0.0, 0.0
    for _, r in plan.iterrows():
        lado = "LONG" if "COMPRAR" in r["Señal"] else "SHORT"
        coste_obj = r["peso"] * capital
        shares = int(coste_obj // r["Precio"])
        if shares < 1:
            continue
        stop = r["Precio"] - atr_mult * r["atr"] if lado == "LONG" else r["Precio"] + atr_mult * r["atr"]
        riesgo = abs(r["Precio"] - stop) * shares
        coste = shares * r["Precio"]
        riesgo_total += riesgo; exp_total += coste
        filas.append({
            "Ticker": r["Ticker"], "Lado": lado, "Señal": r["Señal"].split()[-1],
            "Precio": round(r["Precio"], 3), "Acciones": shares,
            "Coste €": round(coste, 2), "Peso %": round(r["peso"] * 100, 1),
            "Vol anual %": round(r["vol"] * 100, 1),
            "Stop": round(stop, 3), "Riesgo €": round(riesgo, 2),
        })
    plan_df = pd.DataFrame(filas)
    meta = {
        "capital": capital, "n_ordenes": len(plan_df),
        "exposicion_pct": round(exp_total / capital * 100, 1),
        "riesgo_total_eur": round(riesgo_total, 2),
        "riesgo_total_pct": round(riesgo_total / capital * 100, 2),
        "limite_riesgo_pct": riesgo_dia_pct * 100,
        "excede_riesgo": riesgo_total > riesgo_dia_pct * capital,
    }
    return plan_df, meta


def informe(plan_df, meta):
    if "mensaje" in meta:
        return meta["mensaje"]
    L = [f"=== Plan de órdenes · capital {meta['capital']:.0f} € · {meta['n_ordenes']} posiciones ===\n"]
    L.append(plan_df.to_string(index=False))
    L.append(f"\n  Exposición total : {meta['exposicion_pct']}% del capital")
    aviso = "  ⚠️ EXCEDE tu límite de riesgo diario" if meta["excede_riesgo"] else "  ✓ dentro del límite"
    L.append(f"  Riesgo total     : {meta['riesgo_total_eur']:.2f} € ({meta['riesgo_total_pct']}%) · "
             f"límite {meta['limite_riesgo_pct']:.0f}% → {aviso}")
    L.append("\n> Vol targeting + stop ATR. Este plan es la entrada del bucle de ejecución (paper).")
    L.append("> Las órdenes las disparas TÚ. No es recomendación de inversión.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Plan de órdenes con control de riesgo (sizing del sistema).")
    ap.add_argument("tickers", nargs="*",
                    default=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "XOM", "KO", "WMT"])
    ap.add_argument("--file")
    ap.add_argument("--capital", type=float, default=10000.0)
    ap.add_argument("--target-vol", type=float, default=0.15)
    ap.add_argument("--max-pos", type=int, default=5)
    ap.add_argument("--atr-mult", type=float, default=2.0)
    ap.add_argument("--umbral", type=float, default=0.35)
    a = ap.parse_args()
    tickers = a.tickers or []
    if a.file and Path(a.file).exists():
        tickers = [l.strip() for l in Path(a.file).read_text().splitlines() if l.strip()]

    print(f"\nDimensionando plan sobre {len(tickers)} tickers (capital {a.capital:.0f} €)...")
    plan_df, meta = generar_plan(tickers, a.capital, a.target_vol, a.max_pos, a.atr_mult, a.umbral)
    print("\n" + informe(plan_df, meta) + "\n")


if __name__ == "__main__":
    main()
