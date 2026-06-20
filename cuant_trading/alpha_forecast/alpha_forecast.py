"""
alpha_forecast — predicción DIRECCIONAL a corto plazo + volatilidad (GARCH).

En vez de predecir el PRECIO a 90 días (que no bate al azar — eficiencia de
mercado), aquí se ataca lo que SÍ tiene señal medible:

  1. DIRECCIÓN del retorno a corto plazo (5-10 días hábiles) con un clasificador
     de boosting (LightGBM) sobre features ingenierizadas leak-free, validado con
     walk-forward PURGADO + embargo (López de Prado) para evitar fuga de etiquetas.
  2. VOLATILIDAD con GARCH(1,1): la volatilidad SÍ es predecible (clustering),
     a diferencia de la dirección a 90d.

Técnicas usadas:
  - Estadística/econometría: GARCH(1,1), retornos log, z-scores, test binomial, AUC.
  - Física/series: exponente de Hurst (R/S) → régimen (tendencia vs reversión),
    volatilidad realizada, entropía de la distribución de retornos.
  - ML / ingeniería de datos: features sin look-ahead, escalado solo en train,
    walk-forward con purga y embargo, calibración de probabilidad, balance de clases.

Uso:
    python alpha_forecast.py AAPL
    python alpha_forecast.py AAPL SAB.MC MSFT --horizon 5 --period 8y
"""
import argparse
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import warnings
warnings.filterwarnings("ignore")
from math import erf, sqrt

import numpy as np
import pandas as pd
import yfinance as yf


def descargar(ticker, period="8y"):
    h = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if h.empty:
        raise ValueError(f"Ticker '{ticker}' sin datos.")
    h = h.reset_index()
    h["Date"] = pd.to_datetime(h["Date"]).dt.tz_localize(None)
    return h[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Física / matemáticas: exponente de Hurst (R/S)
# ---------------------------------------------------------------------------
def hurst_rs(serie):
    """Exponente de Hurst por rango reescalado. <0.5 reversión, ~0.5 aleatorio, >0.5 tendencia."""
    x = np.asarray(serie, dtype=float)
    n = len(x)
    if n < 20:
        return 0.5
    escalas = [n // k for k in (2, 4, 8, 16) if n // k >= 8]
    rs = []
    for s in escalas:
        trozos = n // s
        vals = []
        for i in range(trozos):
            seg = x[i*s:(i+1)*s]
            z = seg - seg.mean()
            r = np.ptp(np.cumsum(z))
            sd = seg.std()
            if sd > 0:
                vals.append(r / sd)
        if vals:
            rs.append((s, np.mean(vals)))
    if len(rs) < 2:
        return 0.5
    ls = np.log([s for s, _ in rs]); lr = np.log([v for _, v in rs])
    return float(np.polyfit(ls, lr, 1)[0])


# ---------------------------------------------------------------------------
# Ingeniería de features (LEAK-FREE: todo usa solo el pasado en cada fila t)
# ---------------------------------------------------------------------------
def features(df):
    c = df["Close"]; v = df["Volume"]
    r = np.log(c / c.shift(1))                       # retorno log
    f = pd.DataFrame(index=df.index)
    for lag in (1, 2, 3, 5, 10):
        f[f"ret_lag{lag}"] = r.shift(lag - 1)        # retorno reciente
    f["mom5"] = c.pct_change(5)
    f["mom10"] = c.pct_change(10)
    f["mom20"] = c.pct_change(20)
    # RSI(14) Wilder
    up = r.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    dn = (-r.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    f["rsi"] = 100 - 100 / (1 + up / (dn + 1e-9))
    # volatilidad realizada
    f["vol5"] = r.rolling(5).std()
    f["vol20"] = r.rolling(20).std()
    f["vol_ratio"] = f["vol5"] / (f["vol20"] + 1e-9)
    # distancia z a medias
    sma20 = c.rolling(20).mean(); sd20 = c.rolling(20).std()
    f["z_sma20"] = (c - sma20) / (sd20 + 1e-9)
    sma50 = c.rolling(50).mean()
    f["dist_sma50"] = (c - sma50) / (sma50 + 1e-9)
    # volumen z
    f["vol_z"] = (v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-9)
    # rango intradía (proxy de stress)
    f["hl_range"] = (df["High"] - df["Low"]) / c
    # Hurst rolling (régimen) + autocorrelación corta
    f["hurst"] = r.rolling(60).apply(hurst_rs, raw=False)
    f["acf1"] = r.rolling(20).apply(lambda s: s.autocorr(1) if s.std() > 0 else 0, raw=False)
    f["dow"] = pd.to_datetime(df["Date"]).dt.dayofweek
    return f, r


# ---------------------------------------------------------------------------
# Clasificador direccional con walk-forward PURGADO + embargo
# ---------------------------------------------------------------------------
def backtest_direccion(df, horizon=5, step=5, train_min=500):
    import lightgbm as lgb
    f, r = features(df)
    c = df["Close"]
    # objetivo: signo del retorno futuro a 'horizon' días
    fwd = c.shift(-horizon) / c - 1
    y = (fwd > 0).astype(int)
    data = f.copy(); data["y"] = y
    data = data.dropna()
    X = data.drop(columns="y"); Y = data["y"]
    idx = X.index.to_numpy()
    n = len(X)
    if n < train_min + horizon + 50:
        return None

    preds, reales, probs = [], [], []
    t = train_min
    while t < n - 1:
        # EMBARGO: las últimas 'horizon' filas de train tienen etiqueta que mira al test → se purgan
        tr_end = t - horizon
        if tr_end < 100:
            t += step; continue
        Xtr, Ytr = X.iloc[:tr_end], Y.iloc[:tr_end]
        Xte = X.iloc[t:t+step]; Yte = Y.iloc[t:t+step]
        if Xte.empty:
            break
        # balance de clases
        w = Ytr.map({0: (Ytr == 1).mean(), 1: (Ytr == 0).mean()}).clip(lower=0.1)
        clf = lgb.LGBMClassifier(n_estimators=120, num_leaves=15, max_depth=4,
                                 learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                                 min_child_samples=30, reg_lambda=1.0, random_state=42,
                                 verbosity=-1)
        clf.fit(Xtr, Ytr, sample_weight=w)
        p = clf.predict_proba(Xte)[:, 1]
        probs.extend(p); preds.extend((p > 0.5).astype(int)); reales.extend(Yte.values)
        t += step

    preds = np.array(preds); reales = np.array(reales); probs = np.array(probs)
    N = len(preds)
    da = float((preds == reales).mean())
    z = (da - 0.5) / sqrt(0.25 / N)
    pval = 1 - 0.5 * (1 + erf(z / sqrt(2)))
    # AUC
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(reales, probs)) if len(set(reales)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    # acierto solo en señales de alta convicción (|p-0.5|>0.1)
    conv = np.abs(probs - 0.5) > 0.1
    da_conv = float((preds[conv] == reales[conv]).mean()) if conv.sum() > 10 else float("nan")
    return dict(N=N, da=da, z=z, pval=pval, auc=auc, da_conv=da_conv, n_conv=int(conv.sum()))


# ---------------------------------------------------------------------------
# GARCH(1,1): la volatilidad SÍ es predecible
# ---------------------------------------------------------------------------
def backtest_volatilidad(df):
    try:
        from arch import arch_model
    except Exception:
        return None
    r = (np.log(df["Close"] / df["Close"].shift(1)).dropna()) * 100   # % para estabilidad numérica
    if len(r) < 600:
        return None
    # walk-forward one-step: predice vol de mañana, compara con |retorno| real
    preds, reales = [], []
    start = len(r) - 250
    for t in range(start, len(r) - 1):
        am = arch_model(r.iloc[:t], vol="Garch", p=1, q=1, mean="constant", dist="t")
        res = am.fit(disp="off", show_warning=False)
        fvar = res.forecast(horizon=1, reindex=False).variance.iloc[-1, 0]
        preds.append(np.sqrt(fvar)); reales.append(abs(r.iloc[t]))
    preds = np.array(preds); reales = np.array(reales)
    # ¿la vol predicha explica el |retorno| real? (proxy de predictibilidad)
    if preds.std() > 0 and reales.std() > 0:
        corr = float(np.corrcoef(preds, reales)[0, 1])
    else:
        corr = float("nan")
    # baseline: vol constante (std histórica). R2 vs |retorno|
    base = np.full_like(reales, reales.mean())
    ss_res = float(((reales - preds * np.sqrt(2/np.pi)) ** 2).sum())   # E|N|=sigma*sqrt(2/pi)
    ss_tot = float(((reales - base) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return dict(n=len(preds), corr=corr, r2=r2)


def main():
    ap = argparse.ArgumentParser(description="Alpha: dirección a corto + volatilidad GARCH.")
    ap.add_argument("tickers", nargs="*", default=["AAPL"])
    ap.add_argument("--horizon", type=int, default=5, help="Días hábiles para la dirección.")
    ap.add_argument("--period", default="8y")
    ap.add_argument("--no-garch", action="store_true")
    a = ap.parse_args()
    tickers = a.tickers or ["AAPL"]

    filas = []
    for tk in tickers:
        tk = tk.upper()
        try:
            df = descargar(tk, a.period)
        except Exception as e:
            print(f"\n[{tk}] sin datos: {e}"); continue
        print(f"\n=== {tk} · dirección a {a.horizon} días (walk-forward purgado) ===")
        d = backtest_direccion(df, a.horizon)
        if d:
            bate = "SÍ bate al azar ✅" if d["pval"] < 0.05 else "no bate al azar"
            print(f"  Acierto direccional : {d['da']*100:.1f}%  (N={d['N']})  z={d['z']:+.2f}  p={d['pval']:.3f}  → {bate}")
            print(f"  AUC                 : {d['auc']:.3f}  (0.5=azar)")
            if not np.isnan(d["da_conv"]):
                print(f"  Acierto alta conv.  : {d['da_conv']*100:.1f}%  ({d['n_conv']} señales con |p-0.5|>0.1)")
            filas.append((tk, d))
        else:
            print("  histórico insuficiente.")
        if not a.no_garch:
            g = backtest_volatilidad(df)
            if g:
                lect = ("vol predecible (modesto)" if g["corr"] > 0.2
                        else "vol débilmente predecible" if g["corr"] > 0.08
                        else "sin predictibilidad clara de vol")
                print(f"  GARCH(1,1) vol      : corr(vol_pred, |ret|)={g['corr']:.2f}  → {lect}")

    if filas:
        print("\n" + "=" * 64)
        print("RESUMEN dirección a corto plazo")
        print(f"{'Ticker':<9}{'DA%':>7}{'p':>7}{'AUC':>7}  Veredicto")
        for tk, d in filas:
            v = "VENTAJA real" if d["pval"] < 0.05 else "sin ventaja significativa"
            print(f"{tk:<9}{d['da']*100:>6.0f}{d['pval']:>7.2f}{d['auc']:>7.2f}  {v}")
        sig = [tk for tk, d in filas if d["pval"] < 0.05]
        print("\n> Dirección a 5d con ML está EN el azar (50-52%), no por debajo como el precio a 90d.")
        if sig:
            print(f"> Con ventaja significativa (p<0.05): {', '.join(sig)}. Opera solo esos, alta convicción.")
        else:
            print("> Ninguno bate al azar de forma significativa: en large-caps líquidas no hay edge fácil.")
        print("> Esto es eficiencia de mercado, MEDIDA — no un fallo. No fuerces un número ganador (sobreajuste).\n")


if __name__ == "__main__":
    main()
