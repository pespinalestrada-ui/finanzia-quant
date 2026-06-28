"""
conformal_forecast — bandas de predicción CALIBRADAS (split conformal prediction).

Problema que ataca: el backtest demostró que la banda 80 % de Prophet solo cubre
el 14-29 % real (miente), y el "nivel de confianza" = 100−MAPE es una heurística
sin garantía. La revisión de código (consejo de Claude) lo marcó como P0.

Solución: **split conformal prediction** (Vovk; Lei et al. 2018). Distribution-free:
con los errores absolutos del modelo en validación walk-forward, el cuantil (1−α)
de esos errores da un radio de banda con **cobertura garantizada ≈ (1−α)** bajo
intercambiabilidad. No asume normalidad ni que la banda de Prophet sea correcta.

Aquí además se MIDE la cobertura fuera de muestra (split cal/test) para reportar
la cobertura real, no solo la teórica. Pasa de "banda inventada" a "banda con
cobertura demostrada".

Uso:
    python conformal_forecast.py AAPL
    python conformal_forecast.py SAB.MC --horizons 30 90 120 --alpha 0.2 --period 5y
"""
import argparse
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import yfinance as yf
from prophet import Prophet

HORIZONS = (30, 90, 120)


def descargar(ticker, period="5y"):
    h = yf.Ticker(ticker).history(period=period, auto_adjust=False)
    if h.empty:
        raise ValueError(f"Ticker '{ticker}' sin datos.")
    h = h.reset_index()
    ds = pd.to_datetime(h["Date"])
    if getattr(ds.dt, "tz", None) is not None:
        ds = ds.dt.tz_localize(None)
    df = pd.DataFrame({"ds": ds.dt.normalize(), "y": h["Close"].astype(float)}).dropna()
    return df.reset_index(drop=True)


def _prophet():
    return Prophet(daily_seasonality=False, weekly_seasonality=True,
                   yearly_seasonality=True, interval_width=0.80,
                   changepoint_prior_scale=0.05)


def residuos_walk_forward(df, horizons=HORIZONS, n_origenes=30, min_train=400):
    """
    UNA pasada walk-forward: en cada origen entrena Prophet y, de un solo forecast,
    extrae el error absoluto a cada horizonte. Devuelve dict {h: np.array(|err|)}.

    Sin leakage: cada origen entrena solo con su pasado. El conjunto de errores es
    la base de la calibración conformal.
    """
    hmax = max(horizons)
    n = len(df)
    ultimo = n - hmax - 1
    primero = max(min_train, ultimo - n_origenes * 12)   # ~12 sesiones entre cortes
    if ultimo <= primero:
        return {h: np.array([]) for h in horizons}
    cortes = np.unique(np.linspace(primero, ultimo, n_origenes, dtype=int))

    errores = {h: [] for h in horizons}
    for t in cortes:
        train = df.iloc[: t + 1]
        fecha0 = train["ds"].iloc[-1]
        m = _prophet()
        m.fit(train)
        fut = m.make_future_dataframe(periods=hmax + 5, freq="B", include_history=False)
        fc = m.predict(fut)
        for h in horizons:
            yhat_h = float(fc.iloc[h - 1]["yhat"])
            fecha_obj = fecha0 + pd.tseries.offsets.BDay(h)
            idx = (df["ds"] - fecha_obj).abs().idxmin()
            real_h = float(df["y"].iloc[idx])
            errores[h].append(abs(yhat_h - real_h))
    return {h: np.array(v) for h, v in errores.items()}


def _cobertura_oos(scores, alpha):
    """
    Split cal/test honesto: calibra el radio con el 60 % de los errores y mide qué
    fracción del 40 % restante cae dentro de ese radio. Es la cobertura REAL, no la
    teórica. Devuelve (q_calibrado_con_todo, cobertura_test, n_test).
    """
    s = np.sort(scores)
    n = len(s)
    q_full = float(np.quantile(scores, 1 - alpha, method="higher")) if n else float("nan")
    if n < 8:
        return q_full, float("nan"), 0
    k = int(n * 0.6)
    cal, test = scores[:k], scores[k:]
    q_cal = float(np.quantile(cal, 1 - alpha, method="higher"))
    cobertura = float((test <= q_cal).mean())
    return q_full, cobertura, len(test)


def forecast(ticker, period="5y", horizons=HORIZONS, alpha=0.2, n_origenes=30):
    """
    Pipeline. Devuelve (fig, tabla, meta).
    Banda conformal por horizonte + cobertura empírica medida + confianza calibrada.
    """
    df = descargar(ticker, period)
    if len(df) < 450:
        raise ValueError(f"Histórico insuficiente para conformal ({len(df)} sesiones).")
    px = float(df["y"].iloc[-1])

    scores = residuos_walk_forward(df, horizons, n_origenes)

    # forecast final: entrena con TODO y proyecta
    m = _prophet()
    m.fit(df)
    hmax = max(horizons)
    fut = m.make_future_dataframe(periods=hmax + 5, freq="B", include_history=False)
    fc = m.predict(fut)

    filas, banda = [], {}
    for h in horizons:
        if h > len(fc):
            continue
        yhat_h = float(fc.iloc[h - 1]["yhat"])
        sc = scores.get(h, np.array([]))
        if len(sc) < 8:
            continue
        q, cob, ntest = _cobertura_oos(sc, alpha)
        lo, hi = yhat_h - q, yhat_h + q
        banda[h] = (fc.iloc[h - 1]["ds"], yhat_h, lo, hi)
        ancho_rel = (hi - lo) / yhat_h if yhat_h else float("nan")
        # confianza calibrada = combina cobertura REAL y estrechez de banda
        if not np.isnan(cob):
            ok_cob = abs(cob - (1 - alpha)) <= 0.10        # cubre lo que promete (±10pp)
            if ok_cob and ancho_rel < 0.25:
                conf = "ALTA"
            elif ok_cob:
                conf = "MEDIA"
            else:
                conf = "BAJA"
        else:
            conf = "BAJA"
        filas.append({
            "Horizonte": f"{h} días",
            "Fecha": pd.Timestamp(fc.iloc[h - 1]["ds"]).date().isoformat(),
            "Precio esperado": round(yhat_h, 3),
            f"Banda {int((1-alpha)*100)}% inf": round(lo, 3),
            f"Banda {int((1-alpha)*100)}% sup": round(hi, 3),
            "Variación %": round((yhat_h / px - 1) * 100, 2),
            "Cobertura real": f"{cob*100:.0f}%" if not np.isnan(cob) else "n/d",
            "Confianza": conf,
        })
    tabla = pd.DataFrame(filas)
    fig = _plot(df, fc, banda, ticker, alpha)
    cob_media = np.nanmean([float(r["Cobertura real"].rstrip("%")) for r in filas
                            if r["Cobertura real"] != "n/d"]) if filas else float("nan")
    meta = {"precio_actual": px, "n": len(df), "objetivo": int((1 - alpha) * 100),
            "cobertura_media": round(cob_media, 0) if not np.isnan(cob_media) else None}
    return fig, tabla, meta


def _plot(df, fc, banda, ticker, alpha):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    hist = df.iloc[-250:]
    ax.plot(hist["ds"], hist["y"], color="black", lw=1.1, label="Histórico (1 año)")
    ax.plot(pd.to_datetime(fc["ds"]), fc["yhat"], color="tab:blue", lw=1.3, label="Prophet (punto)")
    for h, (fecha, yhat_h, lo, hi) in banda.items():
        ax.errorbar(pd.Timestamp(fecha), yhat_h, yerr=[[yhat_h - lo], [hi - yhat_h]],
                    fmt="o", color="tab:red", capsize=5, lw=1.8,
                    label=f"Banda conformal {int((1-alpha)*100)}%" if h == min(banda) else None)
    ax.set_title(f"{ticker.upper()} — Forecast con banda CALIBRADA (split conformal)")
    ax.set_xlabel("Fecha"); ax.set_ylabel("Precio"); ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description="Forecast con bandas calibradas (split conformal).")
    ap.add_argument("ticker", nargs="?", default="SAB.MC")
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    ap.add_argument("--alpha", type=float, default=0.2, help="Banda (1−alpha). 0.2 → 80 %.")
    ap.add_argument("--period", default="5y")
    ap.add_argument("--origins", type=int, default=30)
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()

    print(f"\nCalibrando bandas conformal de {a.ticker.upper()} ({a.origins} orígenes walk-forward)...")
    fig, tabla, meta = forecast(a.ticker, a.period, tuple(a.horizons), a.alpha, a.origins)
    print(f"Cierre {meta['precio_actual']:.3f} · {meta['n']} sesiones · objetivo {meta['objetivo']}% "
          f"· cobertura real media {meta['cobertura_media']}%\n")
    print(tabla.to_string(index=False))
    print(f"\nLa banda conformal SE MIDE: 'Cobertura real' = % de veces que el precio real cayó")
    print(f"dentro de la banda en validación. Si ≈{meta['objetivo']}%, la banda dice la verdad.")
    print("> Split conformal prediction (distribution-free). Forecast estadístico, no recomendación.\n")
    if a.save:
        from pathlib import Path
        out = Path(__file__).resolve().parent / f"{a.ticker.replace('.','_').replace('^','')}_conformal.png"
        fig.savefig(out, dpi=110)
        print(f"Gráfico guardado: {out}\n")


if __name__ == "__main__":
    main()
