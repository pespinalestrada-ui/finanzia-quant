"""
backtester — backtest vectorizado de estrategias simples.

Estrategias:
  sma   : cruce de medias (rápida vs lenta).
  rsi   : compra RSI<30, vende RSI>70.
  bb    : compra bajo banda inferior Bollinger, vende sobre la superior.

Métricas: retorno total, CAGR, Sharpe anualizado, max drawdown, win rate,
nº operaciones, y comparación con buy & hold. Guarda curva de equity.

Uso:
    python backtester.py AAPL --strategy sma --fast 50 --slow 200
    python backtester.py SAB.MC --strategy rsi --period 5y --save
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


def señales(df, strat, fast, slow):
    c = df["Close"]
    pos = pd.Series(0.0, index=df.index)
    if strat == "sma":
        f, s = c.rolling(fast).mean(), c.rolling(slow).mean()
        pos[f > s] = 1.0
    elif strat == "rsi":
        r = _rsi(c)
        state = 0
        out = []
        for v in r:
            if v < 30: state = 1
            elif v > 70: state = 0
            out.append(state)
        pos = pd.Series(out, index=df.index, dtype=float)
    elif strat == "bb":
        ma = c.rolling(20).mean(); sd = c.rolling(20).std()
        up, lo = ma + 2*sd, ma - 2*sd
        state = 0; out = []
        for px, l, u in zip(c, lo, up):
            if px < l: state = 1
            elif px > u: state = 0
            out.append(state)
        pos = pd.Series(out, index=df.index, dtype=float)
    return pos.shift(1).fillna(0)  # entrar al día siguiente (sin look-ahead)


def metricas(equity, rets, pos):
    n = len(equity)
    años = n / 252
    tot = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1/años) - 1 if años > 0 else np.nan
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else np.nan
    dd = (equity / equity.cummax() - 1).min()
    trades = int((pos.diff() == 1).sum())
    # win rate por operación
    wins = tot_ops = 0
    in_pos = False; entry = 0
    for i in range(len(pos)):
        if pos.iloc[i] == 1 and not in_pos:
            in_pos = True; entry = equity.iloc[i]
        elif pos.iloc[i] == 0 and in_pos:
            in_pos = False; tot_ops += 1
            if equity.iloc[i] > entry: wins += 1
    wr = wins / tot_ops * 100 if tot_ops else np.nan
    return dict(ret_total=tot, cagr=cagr, sharpe=sharpe, max_dd=dd, trades=trades, win_rate=wr)


def main():
    ap = argparse.ArgumentParser(description="Backtest vectorizado.")
    ap.add_argument("ticker")
    ap.add_argument("--strategy", default="sma", choices=["sma", "rsi", "bb"])
    ap.add_argument("--fast", type=int, default=50)
    ap.add_argument("--slow", type=int, default=200)
    ap.add_argument("--period", default="5y")
    ap.add_argument("--cost", type=float, default=0.001, help="Coste por operación (0.001=0.1%).")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()

    h = yf.Ticker(a.ticker).history(period=a.period, auto_adjust=False)
    if h.empty:
        raise SystemExit(f"Sin datos para '{a.ticker}'.")
    h = h.reset_index()
    h["Date"] = pd.to_datetime(h["Date"]).dt.tz_localize(None)

    pos = señales(h, a.strategy, a.fast, a.slow)
    rets_mkt = h["Close"].pct_change().fillna(0)
    costes = pos.diff().abs().fillna(0) * a.cost
    rets_str = pos * rets_mkt - costes
    eq_str = (1 + rets_str).cumprod()
    eq_bh = (1 + rets_mkt).cumprod()

    m = metricas(eq_str, rets_str, pos)
    bh = metricas(eq_bh, rets_mkt, pd.Series(1.0, index=h.index))

    print(f"\n=== Backtest {a.ticker.upper()} · estrategia {a.strategy.upper()} · {a.period} ===")
    if a.strategy == "sma":
        print(f"    (SMA {a.fast}/{a.slow}, coste {a.cost*100:.2f}%/op)\n")
    print(f"{'Métrica':<16}{'Estrategia':>14}{'Buy&Hold':>14}")
    print("-"*44)
    print(f"{'Retorno total':<16}{m['ret_total']*100:>13.1f}%{bh['ret_total']*100:>13.1f}%")
    print(f"{'CAGR':<16}{m['cagr']*100:>13.1f}%{bh['cagr']*100:>13.1f}%")
    print(f"{'Sharpe':<16}{m['sharpe']:>14.2f}{bh['sharpe']:>14.2f}")
    print(f"{'Max drawdown':<16}{m['max_dd']*100:>13.1f}%{bh['max_dd']*100:>13.1f}%")
    print(f"{'Win rate':<16}{m['win_rate']:>13.1f}%{'—':>14}")
    print(f"{'Operaciones':<16}{m['trades']:>14}{'1':>14}")
    veredicto = "BATE a buy&hold" if m['ret_total'] > bh['ret_total'] else "NO bate a buy&hold"
    print(f"\n  → La estrategia {veredicto}.\n")

    if a.save:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(h["Date"], eq_str, label=f"{a.strategy.upper()} (x{eq_str.iloc[-1]:.2f})", lw=1.4)
        ax.plot(h["Date"], eq_bh, label=f"Buy&Hold (x{eq_bh.iloc[-1]:.2f})", lw=1.2, alpha=0.8)
        ax.set_title(f"{a.ticker.upper()} — equity (1€ inicial)"); ax.legend()
        ax.set_xlabel("Fecha"); ax.set_ylabel("Capital (×)")
        fig.tight_layout()
        out = f"{a.ticker.replace('.','_').replace('^','')}_{a.strategy}_equity.png"
        fig.savefig(out, dpi=110); print(f"Curva guardada: {out}\n")


if __name__ == "__main__":
    main()
