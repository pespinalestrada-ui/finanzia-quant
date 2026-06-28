"""
alerts — vigilancia de una watchlist: dispara avisos cuando algo cambia.

Autónomo y ligero (no carga modelos pesados): calcula sus propios indicadores y
revisa reglas por ticker. Pensado para ejecutarse periódicamente (manual, .bat o
tarea programada) y registrar los avisos en alerts_log.csv.

Reglas:
  - RSI(14) < 30 (sobreventa) / > 70 (sobrecompra).
  - Pico de volatilidad: vol realizada 5d con z-score > 2 frente a su media 60d.
  - Movimiento brusco: |retorno de hoy| > 2·σ diaria.
  - Cruce de medias: precio cruza hoy SMA50 (al alza/baja) respecto a ayer.
  - Proximidad a extremos: a < 2 % del máximo/mínimo de 52 semanas.

Uso:
    python alerts.py AAPL MSFT SAB.MC
    python alerts.py --file watchlist.txt --no-log
"""
import argparse
import sys
from datetime import datetime
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

LOG = Path(__file__).resolve().parent / "alerts_log.csv"


def _hist(ticker, period="1y"):
    h = yf.Ticker(ticker).history(period=period, auto_adjust=False)
    if h.empty:
        return None
    h = h.reset_index()
    return h


def _rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-9))


def revisar(ticker):
    h = _hist(ticker)
    if h is None or len(h) < 80:
        return []
    c = h["Close"].astype(float).dropna().reset_index(drop=True)
    if len(c) < 80:
        return []
    px = float(c.iloc[-1])
    avisos = []

    # RSI
    rsi = float(_rsi(c).iloc[-1])
    if rsi < 30:
        avisos.append(("RSI", f"sobreventa (RSI {rsi:.0f})"))
    elif rsi > 70:
        avisos.append(("RSI", f"sobrecompra (RSI {rsi:.0f})"))

    # volatilidad: z-score de la vol realizada 5d vs media 60d
    ret = np.log(c / c.shift(1))
    vol5 = ret.rolling(5).std()
    z = (vol5.iloc[-1] - vol5.iloc[-60:].mean()) / (vol5.iloc[-60:].std() + 1e-9)
    if z > 2:
        avisos.append(("Volatilidad", f"pico de volatilidad (z={z:.1f})"))

    # movimiento brusco hoy
    r_hoy = float(ret.iloc[-1])
    sd = float(ret.iloc[-60:].std())
    if abs(r_hoy) > 2 * sd:
        avisos.append(("Movimiento", f"variación brusca hoy {r_hoy*100:+.1f}% (>2σ)"))

    # cruce SMA50
    sma50 = c.rolling(50).mean()
    if len(sma50.dropna()) > 2:
        ayer_arriba = c.iloc[-2] > sma50.iloc[-2]
        hoy_arriba = c.iloc[-1] > sma50.iloc[-1]
        if hoy_arriba and not ayer_arriba:
            avisos.append(("Tendencia", "precio cruza SMA50 al ALZA"))
        elif ayer_arriba and not hoy_arriba:
            avisos.append(("Tendencia", "precio cruza SMA50 a la BAJA"))

    # proximidad a extremos 52s
    hi52, lo52 = float(c.iloc[-252:].max()), float(c.iloc[-252:].min())
    if px >= hi52 * 0.98:
        avisos.append(("Extremos", f"a <2% del máx. 52s ({hi52:.2f})"))
    elif px <= lo52 * 1.02:
        avisos.append(("Extremos", f"a <2% del mín. 52s ({lo52:.2f})"))

    return [(ticker, px, cat, txt) for cat, txt in avisos]


def escanear(tickers):
    filas = []
    for tk in tickers:
        try:
            filas.extend(revisar(tk.upper()))
        except Exception:
            continue
    return filas


def main():
    ap = argparse.ArgumentParser(description="Vigilancia de watchlist con avisos.")
    ap.add_argument("tickers", nargs="*", default=["AAPL", "MSFT", "SAB.MC"])
    ap.add_argument("--file", help="Fichero con un ticker por línea.")
    ap.add_argument("--no-log", action="store_true", help="No escribir alerts_log.csv.")
    a = ap.parse_args()

    tickers = a.tickers or ["AAPL", "MSFT", "SAB.MC"]
    if a.file and Path(a.file).exists():
        tickers = [l.strip() for l in Path(a.file).read_text().splitlines() if l.strip()]

    ahora = datetime.now().strftime("%Y-%m-%d %H:%M")
    filas = escanear(tickers)
    print(f"\n=== Alertas · {ahora} · {len(tickers)} tickers ===\n")
    if not filas:
        print("Sin alertas: ningún ticker cumple las reglas ahora mismo.\n")
        return
    df = pd.DataFrame([{"Ticker": t, "Precio": round(p, 3), "Tipo": cat, "Aviso": txt}
                       for t, p, cat, txt in filas])
    print(df.to_string(index=False))
    print()
    if not a.no_log:
        df.insert(0, "fecha", ahora)
        df.to_csv(LOG, mode="a", header=not LOG.exists(), index=False, encoding="utf-8")
        print(f"Registrado en {LOG.name} ({len(df)} avisos).\n")


if __name__ == "__main__":
    main()
