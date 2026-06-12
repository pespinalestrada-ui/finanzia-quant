"""
indicators — indicadores técnicos de un ticker.

Calcula RSI(14), MACD(12,26,9), Bandas de Bollinger(20,2), ATR(14) y SMA/EMA.
Imprime la señal actual de cada indicador y guarda un gráfico.

Uso:
    python indicators.py AAPL
    python indicators.py SAB.MC --period 2y --save
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


def descargar(ticker, period="1y"):
    h = yf.Ticker(ticker).history(period=period, auto_adjust=False)
    if h.empty:
        raise SystemExit(f"Sin datos para '{ticker}'.")
    h = h.reset_index()
    h["Date"] = pd.to_datetime(h["Date"]).dt.tz_localize(None)
    return h


def rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / down
    return 100 - 100 / (1 + rs)


def macd(close, fast=12, slow=26, signal=9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def bollinger(close, n=20, k=2):
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std()
    return ma, ma + k * sd, ma - k * sd


def atr(df, n=14):
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def main():
    ap = argparse.ArgumentParser(description="Indicadores técnicos de un ticker.")
    ap.add_argument("ticker")
    ap.add_argument("--period", default="1y")
    ap.add_argument("--save", action="store_true", help="Guardar gráfico PNG.")
    a = ap.parse_args()

    df = descargar(a.ticker, a.period)
    c = df["Close"]
    df["RSI"] = rsi(c)
    df["MACD"], df["MACD_sig"], df["MACD_hist"] = macd(c)
    df["BB_mid"], df["BB_up"], df["BB_lo"] = bollinger(c)
    df["ATR"] = atr(df)
    df["SMA50"] = c.rolling(50).mean()
    df["SMA200"] = c.rolling(200).mean()

    last = df.iloc[-1]
    px = last["Close"]
    print(f"\n=== {a.ticker.upper()} — {last['Date'].date()} — precio {px:.3f} ===")

    sigs = []
    r = last["RSI"]
    sigs.append(("RSI(14)", f"{r:.1f}", "SOBREVENTA (compra)" if r < 30 else "SOBRECOMPRA (venta)" if r > 70 else "neutral"))
    macd_s = "alcista" if last["MACD"] > last["MACD_sig"] else "bajista"
    sigs.append(("MACD", f"{last['MACD']:.3f} vs {last['MACD_sig']:.3f}", f"cruce {macd_s}"))
    bbpos = "sobre banda sup (caro)" if px > last["BB_up"] else "bajo banda inf (barato)" if px < last["BB_lo"] else "dentro"
    sigs.append(("Bollinger", f"[{last['BB_lo']:.2f}, {last['BB_up']:.2f}]", bbpos))
    sigs.append(("ATR(14)", f"{last['ATR']:.3f}", f"{last['ATR']/px*100:.1f}% del precio (volatilidad)"))
    trend = "ALCISTA (50>200)" if last["SMA50"] > last["SMA200"] else "BAJISTA (50<200)"
    sigs.append(("Tendencia SMA", f"50={last['SMA50']:.2f} 200={last['SMA200']:.2f}", trend))

    w = max(len(s[0]) for s in sigs)
    for nombre, val, sig in sigs:
        print(f"  {nombre:<{w}}  {val:<24}  {sig}")
    print()

    if a.save:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1, 1]})
        d = df.iloc[-260:]
        ax[0].plot(d["Date"], d["Close"], "k", lw=1, label="Close")
        ax[0].plot(d["Date"], d["BB_up"], "b--", lw=0.7, alpha=0.6)
        ax[0].plot(d["Date"], d["BB_lo"], "b--", lw=0.7, alpha=0.6)
        ax[0].plot(d["Date"], d["SMA50"], "tab:orange", lw=0.8, label="SMA50")
        ax[0].plot(d["Date"], d["SMA200"], "tab:red", lw=0.8, label="SMA200")
        ax[0].set_title(f"{a.ticker.upper()} — precio + Bollinger + SMA"); ax[0].legend(fontsize=8)
        ax[1].plot(d["Date"], d["RSI"], "tab:purple", lw=1); ax[1].axhline(70, color="r", ls="--", lw=0.6)
        ax[1].axhline(30, color="g", ls="--", lw=0.6); ax[1].set_ylabel("RSI")
        ax[2].bar(d["Date"], d["MACD_hist"], color="gray", width=1)
        ax[2].plot(d["Date"], d["MACD"], "b", lw=0.8); ax[2].plot(d["Date"], d["MACD_sig"], "r", lw=0.8)
        ax[2].set_ylabel("MACD")
        fig.tight_layout()
        out = f"{a.ticker.replace('.','_').replace('^','')}_indicadores.png"
        fig.savefig(out, dpi=110)
        print(f"Gráfico guardado: {out}\n")


if __name__ == "__main__":
    main()
