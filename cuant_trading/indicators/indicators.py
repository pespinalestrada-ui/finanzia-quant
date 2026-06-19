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


def stochastic(df, n=14, d=3):
    """Oscilador estocástico %K y %D. <20 sobreventa, >80 sobrecompra."""
    bajo = df["Low"].rolling(n).min()
    alto = df["High"].rolling(n).max()
    k = 100 * (df["Close"] - bajo) / (alto - bajo)
    return k, k.rolling(d).mean()


def williams_r(df, n=14):
    """Williams %R. -80/-100 sobreventa, 0/-20 sobrecompra (escala invertida)."""
    alto = df["High"].rolling(n).max()
    bajo = df["Low"].rolling(n).min()
    return -100 * (alto - df["Close"]) / (alto - bajo)


def cci(df, n=20):
    """Commodity Channel Index. >+100 fuerte alza, <-100 fuerte baja."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    ma = tp.rolling(n).mean()
    md = (tp - ma).abs().rolling(n).mean()
    return (tp - ma) / (0.015 * md)


def adx(df, n=14):
    """Average Directional Index — fuerza de tendencia (no dirección). >25 tendencia fuerte."""
    up = df["High"].diff()
    dn = -df["Low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atr_
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean(), plus_di, minus_di


def obv(df):
    """On-Balance Volume — flujo de volumen acumulado. Divergencias avisan giros."""
    dirn = np.sign(df["Close"].diff()).fillna(0)
    return (dirn * df["Volume"]).cumsum()


def mfi(df, n=14):
    """Money Flow Index — RSI ponderado por volumen. <20 sobreventa, >80 sobrecompra."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    mf = tp * df["Volume"]
    pos = mf.where(tp > tp.shift(), 0.0).rolling(n).sum()
    neg = mf.where(tp < tp.shift(), 0.0).rolling(n).sum()
    return 100 - 100 / (1 + pos / neg.replace(0, np.nan))


def calcular_todos(df):
    """Rellena el DataFrame con todos los indicadores. Reutilizable por el dashboard."""
    c = df["Close"]
    df["RSI"] = rsi(c)
    df["MACD"], df["MACD_sig"], df["MACD_hist"] = macd(c)
    df["BB_mid"], df["BB_up"], df["BB_lo"] = bollinger(c)
    df["ATR"] = atr(df)
    df["SMA50"] = c.rolling(50).mean()
    df["SMA200"] = c.rolling(200).mean()
    df["STOCH_K"], df["STOCH_D"] = stochastic(df)
    df["WILLR"] = williams_r(df)
    df["CCI"] = cci(df)
    df["ADX"], df["DI_POS"], df["DI_NEG"] = adx(df)
    df["OBV"] = obv(df)
    df["MFI"] = mfi(df)
    return df


def señales_dict(df):
    """Lista de (indicador, valor, lectura) para todos los indicadores. Reutilizable."""
    last = df.iloc[-1]; px = last["Close"]
    out = []
    r = last["RSI"]
    out.append(("RSI(14)", f"{r:.1f}", "sobreventa" if r < 30 else "sobrecompra" if r > 70 else "neutral"))
    out.append(("MACD", f"{last['MACD']:.3f} vs {last['MACD_sig']:.3f}", "cruce alcista" if last["MACD"] > last["MACD_sig"] else "cruce bajista"))
    bb = "sobre banda sup (caro)" if px > last["BB_up"] else "bajo banda inf (barato)" if px < last["BB_lo"] else "dentro"
    out.append(("Bollinger", f"[{last['BB_lo']:.2f}, {last['BB_up']:.2f}]", bb))
    out.append(("ATR(14)", f"{last['ATR']:.3f}", f"{last['ATR']/px*100:.1f}% del precio (volatilidad)"))
    out.append(("Tendencia SMA", f"50/200", "ALCISTA" if last["SMA50"] > last["SMA200"] else "BAJISTA"))
    k = last["STOCH_K"]
    out.append(("Estocástico %K", f"{k:.0f}", "sobreventa" if k < 20 else "sobrecompra" if k > 80 else "neutral"))
    wr = last["WILLR"]
    out.append(("Williams %R", f"{wr:.0f}", "sobreventa" if wr < -80 else "sobrecompra" if wr > -20 else "neutral"))
    cc = last["CCI"]
    out.append(("CCI(20)", f"{cc:.0f}", "fuerte alza" if cc > 100 else "fuerte baja" if cc < -100 else "neutral"))
    ax_ = last["ADX"]
    fuerza = "tendencia FUERTE" if ax_ > 25 else "tendencia débil/lateral"
    dirn = "alcista" if last["DI_POS"] > last["DI_NEG"] else "bajista"
    out.append(("ADX(14)", f"{ax_:.0f}", f"{fuerza} ({dirn})"))
    mf = last["MFI"]
    out.append(("MFI(14)", f"{mf:.0f}", "sobreventa" if mf < 20 else "sobrecompra" if mf > 80 else "neutral"))
    obv_slope = "subiendo" if df["OBV"].iloc[-1] > df["OBV"].iloc[-10] else "bajando"
    out.append(("OBV (volumen)", f"{obv_slope}", "flujo acumulado " + obv_slope))
    return out


def main():
    ap = argparse.ArgumentParser(description="Indicadores técnicos de un ticker.")
    ap.add_argument("ticker")
    ap.add_argument("--period", default="1y")
    ap.add_argument("--save", action="store_true", help="Guardar gráfico PNG.")
    a = ap.parse_args()

    df = descargar(a.ticker, a.period)
    df = calcular_todos(df)

    last = df.iloc[-1]
    px = last["Close"]
    print(f"\n=== {a.ticker.upper()} — {last['Date'].date()} — precio {px:.3f} ===")

    sigs = señales_dict(df)
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
