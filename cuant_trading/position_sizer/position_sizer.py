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


def garch_volatility(ticker):
    """Calcula proxy GARCH(1,1) usando EWMA (lambda=0.94, estándar RiskMetrics)."""
    import yfinance as yf
    h = yf.Ticker(ticker).history(period="1y", auto_adjust=False)
    if h.empty:
        return float("nan")
    retornos = h["Close"].pct_change().dropna()
    # Volatilidad diaria usando EWMA lambda=0.94 (span ~ 32)
    vol_diaria = retornos.ewm(alpha=0.06, adjust=False).std().iloc[-1]
    # Volatilidad anualizada
    return float(vol_diaria * np.sqrt(252))


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
    ap.add_argument("--vol-target", type=float, default=0.0, help="Volatilidad anual objetivo (ej. 0.15 para 15%). Activa Volatility Targeting.")
    a = ap.parse_args()

    entry = a.entry
    stop = a.stop
    if a.ticker:
        atr, px = atr_actual(a.ticker)
        vol_anual = garch_volatility(a.ticker)
        entry = entry or px
        stop = stop or (entry - a.atr_mult * atr)
        print(f"\n{a.ticker.upper()}: precio {px:.3f} · ATR(14) {atr:.3f} · stop {a.atr_mult}×ATR = {stop:.3f}")
        if not np.isnan(vol_anual):
            print(f"Proxy GARCH (EWMA) volatilidad anual: {vol_anual*100:.1f}%")
    if entry is None or stop is None:
        raise SystemExit("Faltan --entry/--stop (o usa --ticker para stop por ATR).")
    if stop >= entry:
        raise SystemExit("El stop debe estar por debajo de la entrada (posición larga).")

    riesgo_accion = entry - stop

    kelly_f = 0.0
    kelly_rec = 0.0
    if a.winrate and a.payoff:
        kelly_f = a.winrate - (1 - a.winrate) / a.payoff
        kelly_rec = max(0.0, kelly_f / 2)

    if a.vol_target > 0 and a.ticker and not np.isnan(vol_anual):
        weight = a.vol_target / vol_anual
        if kelly_rec > 0:
            peso_final = min(weight, kelly_rec)
            modo_txt = f"Vol Target acotado por Kelly ({peso_final*100:.1f}% del capital)"
        else:
            peso_final = weight
            modo_txt = f"Volatility Targeting ({weight*100:.1f}% del capital)"
        coste_obj = a.capital * peso_final
        shares = max(1, int(coste_obj // entry))
        riesgo_eur = shares * riesgo_accion
    else:
        riesgo_eur = a.capital * a.risk / 100
        shares = max(1, int(riesgo_eur // riesgo_accion))
        modo_txt = f"Riesgo fijo {a.risk}% del capital"

    coste = shares * entry
    riesgo_real = shares * riesgo_accion
    pct_capital = coste / a.capital * 100

    print(f"\n=== Tamaño de posición ===")
    print(f"  Modo de sizing     : {modo_txt}")
    print(f"  Capital            : {a.capital:,.2f} €")
    print(f"  Entrada / Stop     : {entry:.3f} / {stop:.3f}  (riesgo/acción {riesgo_accion:.3f})")
    print(f"  → ACCIONES         : {shares}")
    print(f"  → Coste posición   : {coste:,.2f} €  ({pct_capital:.1f}% del capital)")
    print(f"  → Riesgo real      : {riesgo_real:,.2f} €  ({riesgo_real/a.capital*100:.2f}%)")

    print(f"\n=== Objetivos (R-múltiplos) ===")
    for r in [float(x) for x in a.targets.split(",")]:
        precio_obj = entry + r * riesgo_accion
        ganancia = shares * r * riesgo_accion
        print(f"  {r:.0f}R → precio {precio_obj:.3f}  (ganancia {ganancia:,.2f} €)")

    if a.winrate and a.payoff:
        print(f"\n=== Kelly ===")
        print(f"  Win-rate {a.winrate:.0%}, payoff {a.payoff:.2f}")
        print(f"  Fracción Kelly completa : {kelly_f*100:.1f}% del capital por operación")
        print(f"  Kelly fraccionada (1/2) : {kelly_rec*100:.1f}%  (recomendado, menos varianza)")
        if kelly_f <= 0:
            print("  ⚠ Kelly ≤ 0: la estrategia no tiene ventaja, no operar.")
    print()


if __name__ == "__main__":
    main()
