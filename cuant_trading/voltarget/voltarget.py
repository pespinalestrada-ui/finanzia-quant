"""
voltarget — algoritmo de VOLATILIDAD OBJETIVO: el índice, pero con el riesgo controlado.

La idea (y por qué es el único algoritmo que este proyecto ha justificado con datos):
medimos que la DIRECCIÓN del precio no es predecible, pero la VOLATILIDAD sí lo es
(GARCH: corr 0.24 con el movimiento real). Así que no se predice hacia dónde va el
mercado — se ajusta CUÁNTO se está expuesto según la tormenta prevista:

    exposición_t = vol_objetivo / vol_prevista_t      (acotada, sin apalancar por defecto)

En calma se está dentro; cuando la volatilidad se dispara, se reduce sola. El resto
del dinero queda en efectivo (que renta algo). Respaldo académico: Moreira & Muir,
"Volatility-Managed Portfolios" (Journal of Finance, 2017).

Detalles que separan un backtest honesto de uno que miente:
  · La exposición se aplica con UN DÍA DE RETRASO (solo se usa información ya conocida).
  · Se restan COSTES por rotación (comisión + horquilla) en cada ajuste.
  · BANDA de no-negociación: no se toca la posición por cambios pequeños → menos costes.
  · El efectivo RENTA (si no, la comparación contra comprar-y-mantener sería tramposa).
  · Por defecto NO se apalanca (máx 100%): un minorista no debería.

No es recomendación de inversión.

Uso:
    python voltarget.py SPY
    python voltarget.py SPY --objetivo 0.12 --metodo garch --coste-bps 5 --max 1.5
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
sys.path.insert(0, str(_SUITE / "cpcv"))
CRISIS = {
    "Crisis financiera 2008": ("2007-10-09", "2009-03-09"),
    "COVID-19 (2020)": ("2020-02-19", "2020-03-23"),
    "Inflación y tipos 2022": ("2022-01-03", "2022-10-12"),
}


# paleta Nocturne (la misma de cuant_trading/dashboard/finanzia_charts.py):
# sobre fondo oscuro el negro y el azul puro no se ven
_NEU, _ACC, _ACC2 = "#b2b6ca", "#9184d9", "#b5abfc"
_UP, _DOWN, _GOLD, _DIM = "#63b58e", "#d9736b", "#c9b273", "#75798c"


def descargar(ticker, period="20y"):
    h = yf.Ticker(ticker).history(period=period, auto_adjust=True)   # dividendos incluidos
    if h.empty:
        raise ValueError(f"'{ticker}' sin datos.")
    c = h["Close"].astype(float).dropna()
    c.index = pd.to_datetime(c.index).tz_localize(None)
    return c


def vol_prevista(r, metodo="ewma", ventana=20, refit=21):
    """Volatilidad anualizada prevista para MAÑANA, usando solo datos hasta hoy."""
    if metodo == "realizada":
        v = r.rolling(ventana).std() * np.sqrt(252)
    elif metodo == "ewma":
        # RiskMetrics: lambda 0.94; reacciona rápido sin sobreajustar
        v = r.ewm(alpha=1 - 0.94, adjust=False).std() * np.sqrt(252)
    elif metodo == "garch":
        try:
            from arch import arch_model
        except Exception:
            return vol_prevista(r, "ewma", ventana)
        rp = r * 100.0
        v = pd.Series(np.nan, index=r.index)
        inicio = max(500, ventana)
        for t in range(inicio, len(rp), refit):          # refit periódico (coste computacional)
            try:
                res = arch_model(rp.iloc[:t], vol="Garch", p=1, q=1,
                                 mean="constant", dist="t").fit(disp="off", show_warning=False)
                f = res.forecast(horizon=refit, reindex=False).variance.iloc[-1].values
                pred = np.sqrt(f) / 100.0 * np.sqrt(252)
                fin = min(t + refit, len(rp))
                v.iloc[t:fin] = pred[:fin - t]
            except Exception:
                continue
        v = v.ffill()
    else:
        raise ValueError("metodo debe ser realizada / ewma / garch")
    return v


def backtest(ticker="SPY", period="20y", objetivo=0.12, metodo="ewma", max_exp=1.0,
             coste_bps=5.0, banda=0.10, tasa_cash=0.02, ventana=20):
    """
    Devuelve (df, meta). df tiene: precio, retorno, vol prevista, exposición,
    curva de la estrategia y del comprar-y-mantener.
    """
    c = descargar(ticker, period)
    r = c.pct_change().dropna()
    v = vol_prevista(r, metodo, ventana).reindex(r.index)

    # exposición deseada; con RETRASO de 1 día (solo información ya conocida)
    w_obj = (objetivo / v).clip(upper=max_exp).clip(lower=0.0)
    w_obj = w_obj.shift(1)

    # banda de no-negociación: no ajustar por cambios pequeños (ahorra costes)
    w, ultimo = [], 0.0
    for x in w_obj.values:
        if not np.isfinite(x):
            w.append(ultimo); continue
        if abs(x - ultimo) > banda:
            ultimo = float(x)
        w.append(ultimo)
    w = pd.Series(w, index=r.index)

    rot = w.diff().abs().fillna(w.abs())                 # rotación diaria
    coste = rot * (coste_bps / 10000.0)
    cash_d = (1 + tasa_cash) ** (1 / 252) - 1            # el efectivo renta
    r_est = w * r + (1 - w).clip(lower=0) * cash_d - coste

    df = pd.DataFrame({"precio": c.reindex(r.index), "r": r, "vol_prev": v,
                       "exposicion": w, "r_est": r_est})
    df["curva_est"] = (1 + df["r_est"]).cumprod()
    df["curva_bh"] = (1 + df["r"]).cumprod()
    meta = {"ticker": ticker.upper(), "objetivo": objetivo, "metodo": metodo,
            "max_exp": max_exp, "coste_bps": coste_bps, "banda": banda,
            "tasa_cash": tasa_cash,
            "rotacion_anual": float(rot.sum() / (len(rot) / 252)),
            "coste_anual_pct": float(coste.sum() / (len(coste) / 252) * 100),
            "exposicion_media": float(w.mean()),
            "dias_fuera": int((w < 0.05).sum()),
            "exposicion_hoy": float(w.iloc[-1]),
            "vol_prev_hoy": float(v.iloc[-1]) if np.isfinite(v.iloc[-1]) else float("nan")}
    return df, meta


def _met(curva, r):
    años = len(r) / 252.0
    cagr = float(curva.iloc[-1] ** (1 / max(años, 1e-9)) - 1)
    vol = float(r.std() * np.sqrt(252))
    sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan
    neg = r[r < 0]
    dd_dev = float(neg.std() * np.sqrt(252)) if len(neg) > 1 else np.nan
    sortino = float(r.mean() * 252 / dd_dev) if dd_dev and dd_dev > 0 else np.nan
    dd = curva / curva.cummax() - 1
    maxdd = float(dd.min())
    ulcer = float(np.sqrt((dd ** 2).mean()) * 100)
    return {"CAGR %": round(cagr * 100, 2), "Vol %": round(vol * 100, 2),
            "Sharpe": round(sharpe, 2), "Sortino": round(sortino, 2),
            "Máx caída %": round(maxdd * 100, 1), "Ulcer": round(ulcer, 2),
            "Calmar": round(cagr / abs(maxdd), 2) if maxdd < 0 else None}


def comparar(df):
    """Tabla estrategia vs comprar-y-mantener."""
    a = _met(df["curva_est"], df["r_est"]); a["Estrategia"] = "Vol objetivo"
    b = _met(df["curva_bh"], df["r"]);      b["Estrategia"] = "Comprar y mantener"
    cols = ["Estrategia", "CAGR %", "Vol %", "Sharpe", "Sortino", "Máx caída %", "Ulcer", "Calmar"]
    return pd.DataFrame([a, b])[cols]


def crisis(ticker, objetivo, metodo, max_exp, coste_bps, banda, tasa_cash):
    """Qué habría hecho en cada crisis (con el mismo motor, periodo acotado)."""
    filas = []
    for nombre, (ini, fin) in CRISIS.items():
        try:
            d, _m = backtest(ticker, "25y", objetivo, metodo, max_exp, coste_bps, banda, tasa_cash)
            d = d.loc[ini:fin]
            if len(d) < 10:
                raise ValueError("corto")
            est = float((1 + d["r_est"]).prod() - 1) * 100
            bh = float((1 + d["r"]).prod() - 1) * 100
            filas.append({"Crisis": nombre, "Vol objetivo %": round(est, 1),
                          "Comprar y mantener %": round(bh, 1),
                          "Diferencia": round(est - bh, 1),
                          "Exposición media": round(float(d["exposicion"].mean()), 2)})
        except Exception:
            filas.append({"Crisis": nombre, "Vol objetivo %": "sin datos",
                          "Comprar y mantener %": "—", "Diferencia": "—", "Exposición media": "—"})
    return pd.DataFrame(filas)


def significancia(df):
    """
    ¿La mejora es real? Dos tests, cada uno responde a una pregunta distinta:
      · diferencia de RETORNOS (t simple): casi nunca sale significativa con datos
        diarios — el ruido es enorme. No es el test relevante aquí.
      · diferencia de SHARPE (Jobson-Korkie con corrección de Memmel, 1982/2003):
        ESTE es el correcto, porque el algoritmo busca mejor rentabilidad AJUSTADA
        AL RIESGO, no más rentabilidad.
    """
    from scipy.stats import norm
    try:
        import cpcv as CP
        s_est = CP.sharpe_con_error(df["r_est"].values)
        s_bh = CP.sharpe_con_error(df["r"].values)
    except Exception:
        s_est = s_bh = {"sharpe": np.nan, "ic95": (np.nan, np.nan)}

    j = df[["r_est", "r"]].dropna()
    a, b = j["r_est"].values, j["r"].values
    n = len(a)
    out = {"sharpe_est": s_est["sharpe"], "ic_est": s_est["ic95"],
           "sharpe_bh": s_bh["sharpe"], "ic_bh": s_bh["ic95"]}
    # test de retornos (informativo, no decisivo)
    d = a - b
    if d.std() > 0:
        t = float(d.mean() / d.std() * np.sqrt(n))
        out["p_retornos"] = round(float(2 * (1 - norm.cdf(abs(t)))), 4)
    # test de Sharpe (Jobson-Korkie / Memmel)
    if n > 30 and a.std() > 0 and b.std() > 0:
        mu_a, mu_b = a.mean(), b.mean()
        sa, sb = a.std(ddof=1), b.std(ddof=1)
        sr_a, sr_b = mu_a / sa, mu_b / sb
        rho = float(np.corrcoef(a, b)[0, 1])
        # varianza asintótica de la diferencia de Sharpes (Memmel 2003)
        var = (1.0 / n) * (2 - 2 * rho + 0.5 * (sr_a ** 2 + sr_b ** 2 - 2 * sr_a * sr_b * rho ** 2))
        if var > 0:
            z = (sr_a - sr_b) / np.sqrt(var)
            out["z_sharpe"] = round(float(z), 2)
            out["p_sharpe"] = round(float(2 * (1 - norm.cdf(abs(z)))), 4)
            out["mejora_sharpe_real"] = bool(out["p_sharpe"] < 0.05 and sr_a > sr_b)
    return out


def robustez(ticker="SPY", period="20y", coste_bps=5.0, banda=0.10, tasa_cash=0.02):
    """
    ¿Funciona solo con MI configuración o con cualquiera? Barre objetivos y métodos
    y muestra TODA la rejilla (no la mejor celda: eso sería engañarse).
    """
    filas = []
    for metodo in ("ewma", "realizada"):
        for obj in (0.08, 0.10, 0.12, 0.15, 0.20):
            try:
                df, meta = backtest(ticker, period, obj, metodo, 1.0, coste_bps, banda, tasa_cash)
                t = comparar(df)
                e, b = t.iloc[0], t.iloc[1]
                filas.append({"Método": metodo, "Objetivo vol": f"{obj*100:.0f}%",
                              "Sharpe estr.": e["Sharpe"], "Sharpe B&H": b["Sharpe"],
                              "Máx caída estr. %": e["Máx caída %"], "Máx caída B&H %": b["Máx caída %"],
                              "CAGR estr. %": e["CAGR %"], "CAGR B&H %": b["CAGR %"],
                              "¿Mejor Sharpe?": "sí" if e["Sharpe"] > b["Sharpe"] else "no",
                              "¿Menos caída?": "sí" if abs(e["Máx caída %"]) < abs(b["Máx caída %"]) else "no"})
            except Exception:
                continue
    df_r = pd.DataFrame(filas)
    if df_r.empty:
        return df_r, {}
    res = {"n": len(df_r),
           "mejor_sharpe": int((df_r["¿Mejor Sharpe?"] == "sí").sum()),
           "menos_caida": int((df_r["¿Menos caída?"] == "sí").sum())}
    return df_r, res


def explicar(tabla, meta, sig, cri):
    """Lectura en cristiano de todo el resultado."""
    est = tabla.iloc[0]; bh = tabla.iloc[1]
    L = []
    mejor_sharpe = est["Sharpe"] > bh["Sharpe"]
    menos_caida = abs(est["Máx caída %"]) < abs(bh["Máx caída %"])
    L.append(f"**Qué hace hoy:** estar al **{meta['exposicion_hoy']*100:.0f}%** invertido "
             f"(volatilidad prevista {meta['vol_prev_hoy']*100:.1f}%, objetivo {meta['objetivo']*100:.0f}%). "
             f"El resto, en efectivo.")
    L.append(f"- **Sharpe {est['Sharpe']} vs {bh['Sharpe']}** — "
             f"{'mejor relación rentabilidad/riesgo ✅' if mejor_sharpe else 'no mejora la relación rentabilidad/riesgo'}.")
    L.append(f"- **Peor caída {est['Máx caída %']}% vs {bh['Máx caída %']}%** — "
             f"{'sufres menos en lo peor ✅' if menos_caida else 'no reduce la peor caída'}.")
    L.append(f"- **Rentabilidad {est['CAGR %']}% vs {bh['CAGR %']}% anual** — "
             f"{'renta más' if est['CAGR %'] > bh['CAGR %'] else 'renta algo menos, pero con menos riesgo (mira la volatilidad)'}.")
    L.append(f"- **Coste de operar: {meta['coste_anual_pct']:.2f}%/año** "
             f"({meta['rotacion_anual']:.1f} de rotación anual). Ya está descontado de los números.")
    L.append(f"- Estuviste **fuera del mercado** (menos del 5% invertido) {meta['dias_fuera']} sesiones.")
    if sig and sig.get("p_sharpe") is not None:
        real = sig.get("mejora_sharpe_real")
        L.append(f"- **¿La mejora de Sharpe es real o suerte?** p = {sig['p_sharpe']} "
                 f"(test de Jobson-Korkie) → "
                 f"{'SÍ, estadísticamente significativa ✅' if real else 'no se puede afirmar con seguridad'}.")
        if sig.get("p_retornos") is not None:
            L.append(f"  *(La diferencia de rentabilidad pura da p={sig['p_retornos']}, pero ese no es "
                     f"el test relevante: el objetivo es rentabilidad AJUSTADA AL RIESGO.)*")
    if not cri.empty and "Diferencia" in cri.columns:
        difs = pd.to_numeric(cri["Diferencia"], errors="coerce").dropna()
        if len(difs):
            gana = int((difs > 0).sum())
            L.append(f"- **En las crisis**: lo hizo mejor que comprar-y-mantener en **{gana} de {len(difs)}**. "
                     f"Ahí es donde este algoritmo justifica su existencia.")
    L.append("\n> Lo que busca NO es ganar más, sino **ganar parecido pasando menos miedo**: "
             "menos caída y menos volatilidad. Si además renta igual o más, es un extra.")
    return "\n".join(L)


def _plot(df, meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1]})
    ax1.plot(df.index, df["curva_bh"], color=_NEU, lw=1.3, label="Comprar y mantener")
    ax1.plot(df.index, df["curva_est"], color=_ACC, lw=1.6, label="Volatilidad objetivo")
    ax1.set_yscale("log"); ax1.set_ylabel("Valor de 1 € (log)")
    ax1.set_title(f"{meta['ticker']} · volatilidad objetivo {meta['objetivo']*100:.0f}% "
                  f"({meta['metodo']}, costes {meta['coste_bps']}bps)")
    ax1.legend(loc="upper left")
    ax2.fill_between(df.index, df["exposicion"] * 100, color=_ACC, alpha=0.35)
    ax2.plot(df.index, df["exposicion"] * 100, color=_ACC, lw=0.8)
    ax2.set_ylabel("% invertido"); ax2.set_xlabel("Fecha"); ax2.set_ylim(0, max(105, meta["max_exp"] * 105))
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description="Algoritmo de volatilidad objetivo.")
    ap.add_argument("ticker", nargs="?", default="SPY")
    ap.add_argument("--period", default="20y")
    ap.add_argument("--objetivo", type=float, default=0.12, help="Volatilidad anual objetivo (0.12 = 12%%).")
    ap.add_argument("--metodo", choices=["ewma", "realizada", "garch"], default="ewma")
    ap.add_argument("--max", dest="max_exp", type=float, default=1.0, help="Exposición máxima (1.0 = sin apalancar).")
    ap.add_argument("--coste-bps", type=float, default=5.0)
    ap.add_argument("--banda", type=float, default=0.10, help="No ajustar si el cambio es menor que esto.")
    ap.add_argument("--cash", type=float, default=0.02, help="Rentabilidad anual del efectivo.")
    a = ap.parse_args()

    print(f"\nBacktesteando volatilidad objetivo en {a.ticker.upper()} ({a.metodo})...")
    df, meta = backtest(a.ticker, a.period, a.objetivo, a.metodo, a.max_exp,
                        a.coste_bps, a.banda, a.cash)
    tabla = comparar(df)
    sig = significancia(df)
    cri = crisis(a.ticker, a.objetivo, a.metodo, a.max_exp, a.coste_bps, a.banda, a.cash)
    print(f"\n=== {meta['ticker']} · {len(df):,} sesiones ===\n")
    print(tabla.to_string(index=False))
    print("\n=== En las crisis ===")
    print(cri.to_string(index=False))
    print("\n=== Lectura ===")
    print(explicar(tabla, meta, sig, cri).replace("**", ""))
    print()


if __name__ == "__main__":
    main()
