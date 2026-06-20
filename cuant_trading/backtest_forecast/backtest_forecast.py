"""
backtest_forecast — mide la EFECTIVIDAD real del forecast (margen de acierto).

Implementa las métricas que un análisis riguroso exige (recomendadas por la
revisión del código), con validación WALK-FORWARD de origen móvil (no holdout
único):

  1. Tasa de acierto DIRECCIONAL (DA) + test binomial de significancia.
     ¿Acierta el SIGNO del movimiento más que una moneda al aire?
  2. Theil's U2 vs paseo aleatorio.  U2<1 = bate al "mañana ≈ hoy"; U2≥1 = no.
  3. MAPE medio ± desviación sobre todos los orígenes (no un único número).
  4. Cobertura empírica de la banda 80 % (¿la banda dice la verdad?).

Sin leakage: en cada origen se entrena SOLO con el pasado y se predice el futuro;
el baseline es el último precio conocido (paseo aleatorio). Eventos OPA NO se
pasan como holidays en el backtest (serían información futura).

Uso:
    python backtest_forecast.py AAPL
    python backtest_forecast.py AAPL SAB.MC MSFT --horizons 30 90 --origins 18
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


def walk_forward(df, horizon, n_origenes=15, min_train=400):
    """
    Origen móvil: reparte n_origenes puntos de corte en el histórico, entrena
    con el pasado de cada corte y compara la predicción a 'horizon' días hábiles
    con el valor real. Devuelve un DataFrame con una fila por origen.
    """
    n = len(df)
    ultimo_origen = n - horizon - 1
    primer_origen = max(min_train, ultimo_origen - n_origenes * 12)  # ~12 bdays entre cortes
    if ultimo_origen <= primer_origen:
        return pd.DataFrame()
    cortes = np.linspace(primer_origen, ultimo_origen, n_origenes, dtype=int)

    filas = []
    for t in cortes:
        train = df.iloc[: t + 1]
        precio0 = float(train["y"].iloc[-1])
        fecha0 = train["ds"].iloc[-1]
        m = _prophet()
        m.fit(train)
        fut = m.make_future_dataframe(periods=horizon + 5, freq="B", include_history=False)
        fc = m.predict(fut).iloc[horizon - 1]   # día 'horizon' del forecast
        yhat = float(fc["yhat"]); lo = float(fc["yhat_lower"]); hi = float(fc["yhat_upper"])
        # valor real ~horizon días hábiles después del origen
        fecha_obj = fecha0 + pd.tseries.offsets.BDay(horizon)
        idx = (df["ds"] - fecha_obj).abs().idxmin()
        real = float(df["y"].iloc[idx])
        filas.append({
            "precio0": precio0, "yhat": yhat, "real": real, "lo": lo, "hi": hi,
            "naive": precio0,                       # paseo aleatorio = sin cambio
            "dir_pred": np.sign(yhat - precio0), "dir_real": np.sign(real - precio0),
        })
    return pd.DataFrame(filas)


def metricas(bt):
    """Calcula DA + binomial, Theil U2, MAPE±sd, cobertura sobre el DataFrame walk-forward."""
    n = len(bt)
    err = bt["yhat"] - bt["real"]
    err_naive = bt["naive"] - bt["real"]
    # MAPE por origen
    ape = (err.abs() / bt["real"].abs()) * 100
    mape, mape_sd = float(ape.mean()), float(ape.std())
    # Theil U2 = RMSE_modelo / RMSE_naive
    rmse_m = float(np.sqrt((err ** 2).mean()))
    rmse_n = float(np.sqrt((err_naive ** 2).mean()))
    u2 = rmse_m / rmse_n if rmse_n > 0 else float("nan")
    # acierto direccional (ignora casos de dir_real==0)
    val = bt[bt["dir_real"] != 0]
    aciertos = int((val["dir_pred"] == val["dir_real"]).sum())
    nval = len(val)
    da = aciertos / nval if nval else float("nan")
    # test binomial normal (una cola, H0: DA=0.5)
    if nval > 0:
        z = (da - 0.5) / np.sqrt(0.25 / nval)
        from math import erf, sqrt
        pval = 1 - 0.5 * (1 + erf(z / sqrt(2)))   # P(Z>z)
    else:
        z = pval = float("nan")
    # cobertura banda 80%
    dentro = ((bt["real"] >= bt["lo"]) & (bt["real"] <= bt["hi"])).mean()
    return dict(n=n, nval=nval, da=da, aciertos=aciertos, z=z, pval=pval,
                u2=u2, mape=mape, mape_sd=mape_sd, cobertura=float(dentro))


def informe(ticker, horizon, mt):
    print(f"\n=== {ticker} · horizonte {horizon} días · {mt['n']} orígenes walk-forward ===")
    # 1. dirección
    if not np.isnan(mt["da"]):
        bate = "SÍ bate al azar" if mt["pval"] < 0.05 else "NO bate al azar"
        print(f"  Acierto direccional : {mt['da']*100:.1f}%  ({mt['aciertos']}/{mt['nval']})  "
              f"z={mt['z']:+.2f}  p={mt['pval']:.3f}  → {bate}")
    # 2. Theil U2
    if not np.isnan(mt["u2"]):
        veredicto_u = ("BATE al paseo aleatorio" if mt["u2"] < 0.98
                       else "IGUAL que paseo aleatorio" if mt["u2"] < 1.02
                       else "PEOR que paseo aleatorio")
        print(f"  Theil's U2          : {mt['u2']:.3f}  → {veredicto_u}")
    # 3. MAPE
    print(f"  MAPE                : {mt['mape']:.2f}% ± {mt['mape_sd']:.2f}  (media ± sd entre orígenes)")
    # 4. cobertura
    cal = "bien calibrada" if 0.72 <= mt["cobertura"] <= 0.88 else "MAL calibrada (miente)"
    print(f"  Cobertura banda 80% : {mt['cobertura']*100:.0f}%  → {cal}")


def main():
    ap = argparse.ArgumentParser(description="Backtest de efectividad del forecast (margen de acierto).")
    ap.add_argument("tickers", nargs="*", default=["AAPL"])
    ap.add_argument("--horizons", type=int, nargs="+", default=[30, 90])
    ap.add_argument("--origins", type=int, default=15)
    ap.add_argument("--period", default="5y")
    a = ap.parse_args()
    tickers = a.tickers or ["AAPL"]

    resumen = []
    for tk in tickers:
        tk = tk.upper()
        try:
            df = descargar(tk, a.period)
        except Exception as e:
            print(f"\n[{tk}] sin datos: {e}"); continue
        for h in a.horizons:
            bt = walk_forward(df, h, a.origins)
            if bt.empty:
                print(f"\n[{tk}] histórico insuficiente para horizonte {h}."); continue
            mt = metricas(bt)
            informe(tk, h, mt)
            resumen.append((tk, h, mt))

    if resumen:
        print("\n" + "=" * 70)
        print("RESUMEN — ¿el forecast tiene ventaja real sobre el azar?")
        print(f"{'Ticker':<9}{'Horiz':>6}{'DA%':>7}{'p':>7}{'U2':>7}{'MAPE%':>8}  Veredicto")
        for tk, h, mt in resumen:
            v = "ventaja" if (mt["pval"] < 0.05 and mt["u2"] < 1.0) else "sin ventaja clara"
            print(f"{tk:<9}{h:>6}{mt['da']*100:>6.0f}{mt['pval']:>7.2f}{mt['u2']:>7.2f}{mt['mape']:>8.1f}  {v}")
        print("\n> DA = acierto direccional. p<0.05 = bate al azar. U2<1 = bate al paseo aleatorio.")
        print("> Lectura honesta: a 90 días, casi ningún forecast de precio bate al azar — y eso")
        print("  es un resultado válido y defendible, no un fallo de la herramienta.\n")


if __name__ == "__main__":
    main()
