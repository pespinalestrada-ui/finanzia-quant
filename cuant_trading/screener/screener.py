"""
screener — escáner de una watchlist.

Para cada ticker calcula momentum 1m/3m, RSI(14), distancia a la SMA50,
volatilidad (ATR%) y tendencia (SMA50 vs SMA200). Ordena por un score
compuesto de momentum + fuerza de tendencia. Útil para barrido diario.

Uso:
    python screener.py AAPL MSFT NVDA SAB.MC BBVA.MC
    python screener.py --file watchlist.txt --sort momentum
    (watchlist.txt = un ticker por línea)
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


def _rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def analizar(ticker):
    h = yf.Ticker(ticker).history(period="1y", auto_adjust=False)
    if h.empty or len(h) < 60:
        return None
    c = h["Close"]
    px = float(c.iloc[-1])
    sma50 = c.rolling(50).mean().iloc[-1]
    sma200 = c.rolling(200).mean().iloc[-1] if len(c) >= 200 else np.nan
    mom1m = (px / c.iloc[-21] - 1) * 100 if len(c) > 21 else np.nan
    mom3m = (px / c.iloc[-63] - 1) * 100 if len(c) > 63 else np.nan
    rsi = float(_rsi(c).iloc[-1])
    ret = c.pct_change()
    atr_pct = float(ret.rolling(14).std().iloc[-1] * np.sqrt(252) * 100)  # vol anualizada %
    dist_sma50 = (px / sma50 - 1) * 100
    tend = 1 if (not np.isnan(sma200) and sma50 > sma200) else (-1 if not np.isnan(sma200) else 0)
    # score: momentum medio + bonus tendencia alcista, penaliza RSI extremo
    score = np.nanmean([mom1m, mom3m]) + tend * 5 - max(0, abs(rsi - 50) - 30) * 0.2
    return dict(Ticker=ticker.upper(), Precio=round(px, 3), Mom1m=round(mom1m, 1),
                Mom3m=round(mom3m, 1), RSI=round(rsi, 1), VolAnual=round(atr_pct, 1),
                vsSMA50=round(dist_sma50, 1), Tend=("ALC" if tend > 0 else "BAJ" if tend < 0 else "?"),
                Score=round(score, 1))


def main():
    ap = argparse.ArgumentParser(description="Escáner de watchlist.")
    ap.add_argument("tickers", nargs="*", help="Tickers separados por espacio.")
    ap.add_argument("--file", help="Fichero con un ticker por línea.")
    ap.add_argument("--sort", default="score", choices=["score", "momentum", "rsi"], help="Criterio de orden.")
    a = ap.parse_args()

    tickers = list(a.tickers)
    if a.file:
        with open(a.file, encoding="utf-8") as f:
            tickers += [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if not tickers:
        tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "SAB.MC", "BBVA.MC"]

    print(f"\nAnalizando {len(tickers)} tickers...\n")
    rows = []
    for t in tickers:
        try:
            r = analizar(t)
            if r:
                rows.append(r)
            else:
                print(f"  (sin datos suficientes: {t})")
        except Exception as e:
            print(f"  (error {t}: {e})")

    if not rows:
        raise SystemExit("Nada que mostrar.")
    df = pd.DataFrame(rows)
    key = {"score": "Score", "momentum": "Mom3m", "rsi": "RSI"}[a.sort]
    df = df.sort_values(key, ascending=False).reset_index(drop=True)
    df.index += 1
    print(df.to_string())
    print("\nLeyenda: Mom=momentum % | vsSMA50=distancia a media 50 | VolAnual=volatilidad anualizada % | Tend=tendencia\n")


if __name__ == "__main__":
    main()
