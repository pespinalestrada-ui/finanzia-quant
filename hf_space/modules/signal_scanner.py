"""
signal_scanner — escáner de señales accionables sobre una watchlist.

Detecta por cada ticker:
  - Golden / Death cross (SMA50 cruza SMA200 en las últimas sesiones).
  - RSI en sobreventa (<30) o sobrecompra (>70).
  - Cruce de MACD (línea sobre/bajo su señal, reciente).
  - Breakout de Bollinger (cierre fuera de la banda 20,2).
Devuelve las señales y un sesgo neto BUY / SELL / NEUTRAL por ticker.

Uso:
    python signal_scanner.py AAPL MSFT NVDA SAB.MC BBVA.MC
    python signal_scanner.py --file watchlist.txt --only-actionable
"""
import argparse
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import pandas as pd
import yfinance as yf

LOOKBACK_CROSS = 5  # sesiones para considerar un cruce "reciente"


def _rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def _macd(close, fast=12, slow=26, signal=9):
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig


def señales_ticker(ticker):
    h = yf.Ticker(ticker).history(period="1y", auto_adjust=False)
    if h.empty or len(h) < 60:
        return None
    c = h["Close"]
    px = float(c.iloc[-1])
    sigs = []          # (texto, dir) dir: +1 compra, -1 venta
    sma50 = c.rolling(50).mean()
    sma200 = c.rolling(200).mean() if len(c) >= 200 else None

    # cruce de medias
    if sma200 is not None:
        diff = (sma50 - sma200)
        recent = diff.iloc[-LOOKBACK_CROSS-1:]
        if (recent.iloc[0] <= 0) and (recent.iloc[-1] > 0):
            sigs.append(("Golden cross (SMA50>200)", +1))
        elif (recent.iloc[0] >= 0) and (recent.iloc[-1] < 0):
            sigs.append(("Death cross (SMA50<200)", -1))

    # RSI
    r = float(_rsi(c).iloc[-1])
    if r < 30:
        sigs.append((f"RSI sobreventa ({r:.0f})", +1))
    elif r > 70:
        sigs.append((f"RSI sobrecompra ({r:.0f})", -1))

    # MACD cruce reciente
    line, sigl = _macd(c)
    d = line - sigl
    rec = d.iloc[-LOOKBACK_CROSS-1:]
    if (rec.iloc[0] <= 0) and (rec.iloc[-1] > 0):
        sigs.append(("Cruce MACD alcista", +1))
    elif (rec.iloc[0] >= 0) and (rec.iloc[-1] < 0):
        sigs.append(("Cruce MACD bajista", -1))

    # Bollinger breakout
    ma = c.rolling(20).mean().iloc[-1]; sd = c.rolling(20).std().iloc[-1]
    if px > ma + 2*sd:
        sigs.append(("Breakout sobre banda sup", -1))   # caro, posible reversión
    elif px < ma - 2*sd:
        sigs.append(("Bajo banda inf (rebote?)", +1))

    score = sum(d for _, d in sigs)
    bias = "BUY" if score > 0 else "SELL" if score < 0 else "NEUTRAL"
    return dict(Ticker=ticker.upper(), Precio=round(px, 3),
                Señales="; ".join(s for s, _ in sigs) if sigs else "—",
                Sesgo=bias, Fuerza=score)


def scan(tickers):
    rows = []
    for t in tickers:
        try:
            r = señales_ticker(t)
            if r:
                rows.append(r)
        except Exception:
            pass
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Fuerza", ascending=False, key=lambda s: s.abs()).reset_index(drop=True)
        df.index += 1
    return df


def main():
    ap = argparse.ArgumentParser(description="Escáner de señales.")
    ap.add_argument("tickers", nargs="*")
    ap.add_argument("--file")
    ap.add_argument("--only-actionable", action="store_true", help="Oculta NEUTRAL.")
    a = ap.parse_args()

    tickers = list(a.tickers)
    if a.file:
        with open(a.file, encoding="utf-8") as f:
            tickers += [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if not tickers:
        tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "SAB.MC", "BBVA.MC"]

    print(f"\nEscaneando {len(tickers)} tickers...\n")
    df = scan(tickers)
    if df.empty:
        raise SystemExit("Sin datos.")
    if a.only_actionable:
        df = df[df["Sesgo"] != "NEUTRAL"]
    if df.empty:
        print("Sin señales accionables hoy.\n"); return
    print(df.to_string())
    print("\nSesgo: BUY (señales netas de compra) / SELL / NEUTRAL. Fuerza = suma de señales.\n")


if __name__ == "__main__":
    main()
