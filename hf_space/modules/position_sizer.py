"""
position_sizer — tamaño de posición y gestión de riesgo.

Calcula cuántas acciones comprar dado un riesgo máximo por operación.
Soporta stop manual o stop por ATR. Devuelve riesgo en €, R-múltiplos a los
objetivos y la fracción de Kelly sugerida (si das win-rate y payoff).

Uso (stop manual):
    python position_sizer.py --capital 10000 --risk 1 --entry 50 --stop 47

Uso (stop por ATR, lo descarga del ticker):
    python position_sizer.py --ticker AAPL --capital 20000 --risk 0.5 --atr-mult 2

Kelly:
    python position_sizer.py --capital 10000 --risk 1 --entry 50 --stop 47 \
        --winrate 0.55 --payoff 1.8
"""
import argparse
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import pandas as pd


def atr_actual(ticker, n=14):
    import yfinance as yf
    h = yf.Ticker(ticker).history(period="3mo", auto_adjust=False)
    if h.empty:
        raise SystemExit(f"Sin datos para '{ticker}'.")
    hl = h["High"] - h["Low"]
    hc = (h["High"] - h["Close"].shift()).abs()
    lc = (h["Low"] - h["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1/n, adjust=False).mean().iloc[-1]), float(h["Close"].iloc[-1])


def main():
    ap = argparse.ArgumentParser(description="Tamaño de posición y riesgo.")
    ap.add_argument("--capital", type=float, required=True, help="Capital total de la cuenta (€).")
    ap.add_argument("--risk", type=float, default=1.0, help="Riesgo máximo por operación (%% del capital).")
    ap.add_argument("--entry", type=float, help="Precio de entrada.")
    ap.add_argument("--stop", type=float, help="Precio de stop-loss.")
    ap.add_argument("--ticker", help="Si se da, descarga precio y ATR para stop automático.")
    ap.add_argument("--atr-mult", type=float, default=2.0, help="Múltiplo de ATR para el stop.")
    ap.add_argument("--targets", default="1,2,3", help="Objetivos en R-múltiplos (coma).")
    ap.add_argument("--winrate", type=float, help="Tasa de acierto (0-1) para Kelly.")
    ap.add_argument("--payoff", type=float, help="Ratio ganancia media / pérdida media para Kelly.")
    a = ap.parse_args()

    entry = a.entry
    stop = a.stop
    if a.ticker:
        atr, px = atr_actual(a.ticker)
        entry = entry or px
        stop = stop or (entry - a.atr_mult * atr)
        print(f"\n{a.ticker.upper()}: precio {px:.3f} · ATR(14) {atr:.3f} · stop {a.atr_mult}×ATR = {stop:.3f}")
    if entry is None or stop is None:
        raise SystemExit("Faltan --entry/--stop (o usa --ticker para stop por ATR).")
    if stop >= entry:
        raise SystemExit("El stop debe estar por debajo de la entrada (posición larga).")

    riesgo_eur = a.capital * a.risk / 100
    riesgo_accion = entry - stop
    shares = int(riesgo_eur // riesgo_accion)
    coste = shares * entry
    riesgo_real = shares * riesgo_accion
    pct_capital = coste / a.capital * 100

    print(f"\n=== Tamaño de posición ===")
    print(f"  Capital            : {a.capital:,.2f} €")
    print(f"  Riesgo objetivo    : {a.risk:.2f}% = {riesgo_eur:,.2f} €")
    print(f"  Entrada / Stop     : {entry:.3f} / {stop:.3f}  (riesgo/acción {riesgo_accion:.3f})")
    print(f"  → ACCIONES         : {shares}")
    print(f"  → Coste posición   : {coste:,.2f} €  ({pct_capital:.1f}% del capital)")
    print(f"  → Riesgo real      : {riesgo_real:,.2f} €")

    print(f"\n=== Objetivos (R-múltiplos) ===")
    for r in [float(x) for x in a.targets.split(",")]:
        precio_obj = entry + r * riesgo_accion
        ganancia = shares * r * riesgo_accion
        print(f"  {r:.0f}R → precio {precio_obj:.3f}  (ganancia {ganancia:,.2f} €)")

    if a.winrate and a.payoff:
        # Kelly: f = W - (1-W)/R
        f = a.winrate - (1 - a.winrate) / a.payoff
        print(f"\n=== Kelly ===")
        print(f"  Win-rate {a.winrate:.0%}, payoff {a.payoff:.2f}")
        print(f"  Fracción Kelly completa : {f*100:.1f}% del capital por operación")
        print(f"  Kelly fraccionada (1/2) : {f*50:.1f}%  (recomendado, menos varianza)")
        if f <= 0:
            print("  ⚠ Kelly ≤ 0: la estrategia no tiene ventaja, no operar.")
    print()


if __name__ == "__main__":
    main()
