"""
veredicto_backtest — ¿el VEREDICTO predice de verdad? Validación honesta antes de
automatizar nada (operar sobre una señal sin validar = perder dinero).

Reconstruye el score TÉCNICO del Veredicto en CADA fecha pasada usando solo datos
hasta ese día (point-in-time, sin leakage) y mide si predice el retorno futuro:

  - Information Coefficient (IC): correlación score ↔ retorno futuro (Spearman).
  - Retornos por QUINTIL del score (¿el quintil alto rinde más que el bajo? ¿monótono?).
  - Cartera long-short cross-section: largo top, corto bottom → Sharpe anualizado.
  - PSR (Probabilistic Sharpe Ratio) y DEFLATED Sharpe (Bailey & López de Prado):
    corrigen por asimetría/curtosis y por las MUCHAS configuraciones probadas
    (multiple testing). Es la prueba de que el edge es real y no data-snooping.

Nota de honestidad: se valida el NÚCLEO TÉCNICO del Veredicto (tendencia, ADX,
osciladores, MACD, momentum, OBV), que es point-in-time y reproducible. Los pilares
de Prophet y sentimiento NO se backtestean aquí (no son point-in-time sin leakage /
no hay histórico de noticias). Es decir: mide la parte que SÍ se puede validar limpio.

Uso:
    python veredicto_backtest.py AAPL MSFT NVDA GOOGL AMZN META JPM XOM
    python veredicto_backtest.py --file watchlist.txt --horizon 10 --trials 20
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
import yfinance as yf

_SUITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SUITE / "indicators"))
import indicators as IND

EULER = 0.5772156649015329

# pesos de los pilares técnicos (los mismos que el Veredicto, sin forecast/sentimiento)
PESOS = {"tend": 0.15, "adx": 0.08, "osc": 0.14, "macd": 0.08, "mom": 0.12, "obv": 0.05}
WSUM = sum(PESOS.values())


def descargar(ticker, period="6y"):
    h = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if h.empty:
        return None
    h = h[["Open", "High", "Low", "Close", "Volume"]].dropna()
    h.index = pd.to_datetime(h.index)
    if getattr(h.index, "tz", None) is not None:
        h.index = h.index.tz_localize(None)
    return h


def score_historico(df):
    """Score técnico del Veredicto en [-1,+1] para CADA fila, point-in-time (leak-free)."""
    d = IND.calcular_todos(df.copy())
    c = d["Close"]
    # tendencia
    s_tend = np.where(d["SMA50"] > d["SMA200"], 0.6, -0.6) + np.where(c > d["SMA50"], 0.4, -0.4)
    # ADX (gate de fuerza de tendencia)
    dir_adx = np.where(d["DI_POS"] > d["DI_NEG"], 1.0, -1.0)
    s_adx = np.where(d["ADX"] > 25, dir_adx * 0.7, 0.0)
    # consenso de 5 osciladores
    v_rsi = np.where(d["RSI"] < 30, 1, np.where(d["RSI"] > 70, -1, 0))
    v_k = np.where(d["STOCH_K"] < 20, 1, np.where(d["STOCH_K"] > 80, -1, 0))
    v_wr = np.where(d["WILLR"] < -80, 1, np.where(d["WILLR"] > -20, -1, 0))
    v_mfi = np.where(d["MFI"] < 20, 1, np.where(d["MFI"] > 80, -1, 0))
    v_cci = np.where(d["CCI"] < -100, 1, np.where(d["CCI"] > 100, -1, 0))
    s_osc = (v_rsi + v_k + v_wr + v_mfi + v_cci) / 5.0
    # MACD
    s_macd = np.where(d["MACD"] > d["MACD_sig"], 0.5, -0.5)
    # momentum 3 meses (~63 sesiones)
    mom = c / c.shift(63) - 1.0
    s_mom = np.clip(mom / 0.15, -1, 1).fillna(0).values
    # OBV (pendiente 10 sesiones)
    s_obv = np.where(d["OBV"] > d["OBV"].shift(10), 0.4, -0.4)

    score = (PESOS["tend"] * s_tend + PESOS["adx"] * s_adx + PESOS["osc"] * s_osc
             + PESOS["macd"] * s_macd + PESOS["mom"] * s_mom + PESOS["obv"] * s_obv) / WSUM
    s = pd.Series(score, index=d.index)
    # invalida las primeras filas sin SMA200/indicadores completos
    s[d["SMA200"].isna() | d["ADX"].isna()] = np.nan
    return s


def _psr(sr, T, skew, kurt, sr_bench=0.0):
    """Probabilistic Sharpe Ratio: P(SR_verdadero > sr_bench). sr/sr_bench por observación."""
    from scipy.stats import norm
    denom = np.sqrt(max(1e-9, 1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2))
    z = (sr - sr_bench) * np.sqrt(max(1, T - 1)) / denom
    return float(norm.cdf(z))


def _deflated_sharpe(sr_obs, T, skew, kurt, n_trials, sr_var):
    """Deflated Sharpe: PSR contra el Sharpe esperado del MEJOR de n_trials por azar."""
    from scipy.stats import norm
    if n_trials < 2 or sr_var <= 0:
        sr0 = 0.0
    else:
        z1 = norm.ppf(1 - 1.0 / n_trials)
        z2 = norm.ppf(1 - 1.0 / (n_trials * np.e))
        sr0 = np.sqrt(sr_var) * ((1 - EULER) * z1 + EULER * z2)
    return _psr(sr_obs, T, skew, kurt, sr_bench=sr0), sr0


def backtest(tickers, horizon=10, period="6y", q=0.2, trials=None):
    """Devuelve dict con IC, quintiles, Sharpe long-short, PSR y Deflated Sharpe."""
    scores, ret1, sharpes_tk = {}, {}, []
    pares = []  # (score, ret_fwd_horizonte) agregados para IC/quintiles
    for tk in tickers:
        df = descargar(tk, period)
        if df is None or len(df) < 260:
            continue
        s = score_historico(df)
        c = df["Close"]
        r1 = c.shift(-1) / c - 1.0                  # retorno del día siguiente (para la cartera)
        rh = c.shift(-horizon) / c - 1.0            # retorno a 'horizon' (para IC/quintiles)
        scores[tk] = s
        ret1[tk] = r1
        m = pd.concat([s, rh], axis=1).dropna()
        pares.append(m.values)
        # Sharpe individual de seguir el signo del score (proxy de dispersión entre "trials")
        sig = np.sign(s).reindex(r1.index).fillna(0)
        pnl = (sig * r1).dropna()
        if len(pnl) > 60 and pnl.std() > 0:
            sharpes_tk.append(pnl.mean() / pnl.std() * np.sqrt(252))
    if not pares:
        return {"n": 0, "mensaje": "Sin datos suficientes."}

    P = np.vstack(pares)
    sc, rf = P[:, 0], P[:, 1]
    # IC (Spearman)
    from scipy.stats import spearmanr
    ic, ic_p = spearmanr(sc, rf)

    # retornos por quintil del score
    dfq = pd.DataFrame({"score": sc, "ret": rf})
    dfq["Q"] = pd.qcut(dfq["score"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    quint = dfq.groupby("Q")["ret"].mean() * 100
    monotono = bool(quint.is_monotonic_increasing)
    ls_quint = float(quint.iloc[-1] - quint.iloc[0])

    # cartera long-short cross-section diaria (largo top-q, corto bottom-q por score)
    S = pd.DataFrame(scores).sort_index()
    R = pd.DataFrame(ret1).reindex(S.index)
    ls = []
    for fecha, fila in S.iterrows():
        v = fila.dropna()
        if len(v) < 5:
            continue
        nls = max(1, int(len(v) * q))
        top = v.nlargest(nls).index
        bot = v.nsmallest(nls).index
        rr = R.loc[fecha]
        rl, rs = rr[top].mean(), rr[bot].mean()
        if np.isfinite(rl) and np.isfinite(rs):
            ls.append(rl - rs)
    ls = pd.Series(ls).dropna()
    res = {"n": int(len(P)), "n_tickers": len(scores), "horizon": horizon,
           "ic": float(ic), "ic_p": float(ic_p), "quintiles_%": quint.round(3).to_dict(),
           "monotono": monotono, "ls_quintil_%": round(ls_quint, 3), "n_dias_ls": len(ls)}
    if len(ls) > 60 and ls.std() > 0:
        from scipy.stats import skew as sk, kurtosis as ku
        sr_d = ls.mean() / ls.std()                 # Sharpe por observación (diario)
        res["sharpe_ls"] = round(float(sr_d * np.sqrt(252)), 3)
        skw = float(sk(ls)); krt = float(ku(ls, fisher=False))   # kurtosis NO excedente
        res["psr_0"] = round(_psr(sr_d, len(ls), skw, krt, 0.0), 3)
        n_trials = trials or max(2, len(scores))
        sr_var = float(np.var(sharpes_tk) / 252) if len(sharpes_tk) > 1 else (1.0 / len(ls))
        dsr, sr0 = _deflated_sharpe(sr_d, len(ls), skw, krt, n_trials, sr_var)
        res["deflated_sharpe"] = round(float(dsr), 3)
        res["n_trials"] = n_trials
    return res


def informe(res):
    if res.get("n", 0) == 0:
        return res.get("mensaje", "Sin datos.")
    L = [f"=== Validación del Veredicto · {res['n_tickers']} tickers · horizonte {res['horizon']}d · {res['n']:,} observaciones ===\n"]
    sig_ic = "SÍ hay señal" if res["ic_p"] < 0.05 and res["ic"] > 0 else "sin señal significativa"
    L.append(f"  Information Coefficient : {res['ic']:+.4f}  (p={res['ic_p']:.3f}) → {sig_ic}")
    L.append(f"  Retornos por quintil (%) del score, a {res['horizon']}d:")
    for k, v in res["quintiles_%"].items():
        L.append(f"      Q{k}: {v:+.3f}%")
    L.append(f"  ¿Monótono creciente?     : {'SÍ ✅' if res['monotono'] else 'no'}  · spread Q5−Q1: {res['ls_quintil_%']:+.3f}%")
    if "sharpe_ls" in res:
        L.append(f"  Sharpe long-short (anual): {res['sharpe_ls']:+.2f}  ({res['n_dias_ls']} días)")
        L.append(f"  PSR (P[Sharpe>0])        : {res['psr_0']*100:.0f}%")
        edge = res["deflated_sharpe"] > 0.95
        L.append(f"  DEFLATED Sharpe          : {res['deflated_sharpe']*100:.0f}%  (corrige {res['n_trials']} pruebas) "
                 f"→ {'EDGE REAL ✅' if edge else 'NO supera multiple-testing'}")
    L.append("\n> IC>0 con p<0.05 y quintiles monótonos = el score ordena el futuro.")
    L.append("> El Deflated Sharpe es el juez final: si <95%, el 'edge' puede ser azar de probar mucho.")
    L.append("> Valida el núcleo TÉCNICO del Veredicto (sin Prophet/sentimiento). No es recomendación.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Valida si el Veredicto predice (IC, quintiles, Deflated Sharpe).")
    ap.add_argument("tickers", nargs="*",
                    default=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "XOM", "KO", "WMT"])
    ap.add_argument("--file")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--period", default="6y")
    ap.add_argument("--trials", type=int, default=None, help="Nº de configuraciones probadas (deflación).")
    a = ap.parse_args()
    tickers = a.tickers or []
    if a.file and Path(a.file).exists():
        tickers = [l.strip() for l in Path(a.file).read_text().splitlines() if l.strip()]

    print(f"\nValidando el Veredicto sobre {len(tickers)} tickers (horizonte {a.horizon}d)...")
    res = backtest(tickers, a.horizon, a.period, trials=a.trials)
    print("\n" + informe(res) + "\n")


if __name__ == "__main__":
    main()
