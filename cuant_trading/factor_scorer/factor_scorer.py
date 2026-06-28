"""
factor_scorer — modelo MULTI-FACTOR (factor investing / smart beta).

La técnica que de verdad usan los fondos cuant (AQR, ETFs smart-beta) para decidir
qué comprar: en vez de predecir el precio (que no bate al azar), rankean un universo
de acciones por una nota compuesta de FACTORES con premio de riesgo demostrado
(Fama-French y posteriores):

  - Value     : barato vs fundamentales (earnings yield 1/PER, book-to-price).
  - Momentum  : retorno 12-1 meses (sube → tiende a seguir). El más robusto.
  - Quality   : ROE alto, márgenes, poca deuda.
  - Low-vol   : menor volatilidad → mejor rentabilidad ajustada a riesgo (anomalía).

Dos modos:
  1. rankear(universo): z-score CRUZADO de cada factor dentro del universo → nota
     compuesta → ranking. Es el uso institucional (comprar el top, evitar el peor).
  2. score_absoluto(ticker): nota [-1,+1] de UNA acción con umbrales fijos, para
     usarla como pilar en el Veredicto.

Datos: yfinance (precio + Ticker.info). Fundamentales pueden faltar → se tratan
como neutros. No asume normalidad para la dirección; los factores son premios de
riesgo de LARGO plazo, no señales de timing. No es recomendación de inversión.

Uso:
    python factor_scorer.py AAPL MSFT NVDA SAB.MC ITX.MC
    python factor_scorer.py --file watchlist.txt
"""
import argparse
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

# pesos por defecto de los factores (suman 1)
PESOS = {"value": 0.30, "momentum": 0.30, "quality": 0.25, "lowvol": 0.15}


def _info_precio(ticker):
    """Devuelve (info_dict, serie_close). Robusto a campos/fechas faltantes."""
    t = yf.Ticker(ticker)
    try:
        info = t.info or {}
    except Exception:
        info = {}
    h = t.history(period="2y", auto_adjust=True)
    close = h["Close"].astype(float).dropna() if not h.empty else pd.Series(dtype=float)
    return info, close


def factores_crudos(ticker):
    """Factores en bruto de un ticker (mayor = mejor en todos). NaN si no hay dato."""
    info, c = _info_precio(ticker)
    f = {"ticker": ticker.upper()}

    # --- Value: earnings yield (1/PER) + book-to-price (1/P_B). Mayor = más barato.
    per = info.get("trailingPE") or info.get("forwardPE")
    pb = info.get("priceToBook")
    ey = (1.0 / per) if (per and per > 0) else np.nan
    bp = (1.0 / pb) if (pb and pb > 0) else np.nan
    f["value"] = np.nanmean([x for x in (ey, bp) if not np.isnan(x)]) if not (np.isnan(ey) and np.isnan(bp)) else np.nan

    # --- Momentum 12-1: retorno de hace 12 meses a hace 1 mes (salta el último mes).
    if len(c) >= 252:
        f["momentum"] = float(c.iloc[-21] / c.iloc[-252] - 1.0)
    else:
        f["momentum"] = np.nan

    # --- Quality: ROE + margen − apalancamiento (debt/equity normalizado).
    roe = info.get("returnOnEquity")
    marg = info.get("profitMargins")
    de = info.get("debtToEquity")
    q = []
    if roe is not None: q.append(float(roe))
    if marg is not None: q.append(float(marg))
    if de is not None: q.append(-float(de) / 100.0)        # de suele venir en % (120 = 1.2x)
    f["quality"] = float(np.mean(q)) if q else np.nan

    # --- Low-vol: −volatilidad anualizada (menor vol = mejor → signo negativo).
    if len(c) >= 60:
        vol = float(c.pct_change().dropna().iloc[-252:].std() * np.sqrt(252))
        f["lowvol"] = -vol
        f["_vol_anual"] = vol
    else:
        f["lowvol"] = np.nan
        f["_vol_anual"] = np.nan
    f["_per"] = per if per else np.nan
    f["_roe"] = roe if roe is not None else np.nan
    return f


def rankear(tickers):
    """Z-score cruzado de cada factor en el universo → nota compuesta → ranking."""
    crudos = [factores_crudos(tk) for tk in tickers]
    df = pd.DataFrame(crudos)
    factores = ["value", "momentum", "quality", "lowvol"]

    # z-score cruzado (NaN → 0 = neutro respecto al universo)
    for fac in factores:
        col = df[fac].astype(float)
        mu, sd = col.mean(skipna=True), col.std(skipna=True)
        z = (col - mu) / sd if (sd and not np.isnan(sd) and sd > 0) else col * 0.0
        df[f"z_{fac}"] = z.fillna(0.0).clip(-3, 3)

    df["nota"] = sum(PESOS[fac] * df[f"z_{fac}"] for fac in factores)
    df = df.sort_values("nota", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    # etiqueta por terciles de la nota (relativo al universo)
    if len(df) >= 3:
        q1, q2 = df["nota"].quantile([1/3, 2/3])
        df["señal"] = np.where(df["nota"] >= q2, "COMPRAR (top)",
                       np.where(df["nota"] <= q1, "EVITAR (fondo)", "neutral"))
    else:
        df["señal"] = np.where(df["nota"] > 0, "COMPRAR (top)", "EVITAR (fondo)")
    return df


def _mapa(x, bueno, malo):
    """Mapea un valor a [-1,+1] linealmente entre 'malo'(-1) y 'bueno'(+1)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    if bueno == malo:
        return 0.0
    return float(max(-1.0, min(1.0, (x - malo) / (bueno - malo) * 2 - 1)))


def score_absoluto(ticker):
    """Nota [-1,+1] de una acción con umbrales fijos (para el pilar de Veredicto).
    Devuelve (score, lectura, detalle_dict)."""
    f = factores_crudos(ticker)
    per, roe, vol, mom = f.get("_per"), f.get("_roe"), f.get("_vol_anual"), f.get("momentum")
    s_val = _mapa(1.0 / per if (per and per > 0) else np.nan, bueno=1/8, malo=1/40)   # PER 8 bueno, 40 malo
    s_mom = _mapa(mom, bueno=0.25, malo=-0.25)                                        # ±25% 12-1
    s_qua = _mapa(roe, bueno=0.20, malo=0.0)                                          # ROE 20% bueno
    s_low = _mapa(-vol if vol is not None and not np.isnan(vol) else np.nan,
                  bueno=-0.18, malo=-0.50)                                            # vol 18% buena, 50% mala
    partes = {"value": s_val, "momentum": s_mom, "quality": s_qua, "lowvol": s_low}
    score = sum(PESOS[k] * partes[k] for k in PESOS)
    lectura = (f"PER {per:.0f} · " if (per and per > 0) else "PER n/d · ") + \
              (f"ROE {roe*100:.0f}% · " if (roe is not None and not np.isnan(roe)) else "ROE n/d · ") + \
              (f"mom12-1 {mom*100:+.0f}% · " if (mom is not None and not np.isnan(mom)) else "") + \
              (f"vol {vol*100:.0f}%" if (vol is not None and not np.isnan(vol)) else "")
    return float(max(-1.0, min(1.0, score))), lectura, partes


def main():
    ap = argparse.ArgumentParser(description="Modelo multi-factor (value/momentum/quality/low-vol).")
    ap.add_argument("tickers", nargs="*", default=["AAPL", "MSFT", "NVDA", "JPM", "XOM"])
    ap.add_argument("--file", help="Fichero con un ticker por línea.")
    a = ap.parse_args()
    tickers = a.tickers or ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]
    if a.file:
        from pathlib import Path
        if Path(a.file).exists():
            tickers = [l.strip() for l in Path(a.file).read_text().splitlines() if l.strip()]

    print(f"\nRankeando {len(tickers)} acciones por modelo multi-factor...")
    df = rankear(tickers)
    cols = ["rank", "ticker", "z_value", "z_momentum", "z_quality", "z_lowvol", "nota", "señal"]
    out = df[cols].copy()
    for c in ["z_value", "z_momentum", "z_quality", "z_lowvol", "nota"]:
        out[c] = out[c].round(2)
    print("\n" + out.to_string(index=False))
    print(f"\nPesos: {PESOS}")
    print("> z = posición vs el universo (cruzado). Nota = combinación ponderada.")
    print("> Factores = premios de riesgo de LARGO plazo (Fama-French), no timing.")
    print("> Fundamentales faltantes se tratan como neutros. No es recomendación.\n")


if __name__ == "__main__":
    main()
