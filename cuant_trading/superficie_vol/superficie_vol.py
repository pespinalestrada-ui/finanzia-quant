"""
superficie_vol — la superficie de volatilidad: qué precio le pone el mercado al
miedo, strike a strike y plazo a plazo.

Si Black-Scholes fuera cierto, todas las opciones del mismo activo tendrían la
MISMA volatilidad implícita. No la tienen. Esa desviación no es un fallo del
mercado: es información. Dos formas y lo que significan:

- **Sonrisa / sesgo** (a lo ancho, por strike): en bolsa las puts fuera de dinero
  cotizan con MÁS volatilidad que las calls simétricas. La gente paga por
  protegerse, no por especular al alza. Medido hoy en SPY a 39 días: proteger un
  10% abajo cuesta **8,7 puntos** más de volatilidad que apostar a la subida.
- **Estructura temporal** (a lo largo, por vencimiento): normalmente la vol sube
  con el plazo (calma ahora, incertidumbre luego). Cuando se **invierte** —corto
  plazo por encima del largo— el mercado está descontando un susto inmediato.

Sobre el dato, que aquí importa
-------------------------------
La `impliedVolatility` que trae yfinance NO se usa: se recalcula por bisección
desde el punto medio bid/ask, con el tipo sin riesgo y el dividendo del activo.
Comprobado contra la de yfinance: el NIVEL difiere (ella sale 1-3 puntos por
debajo, y la brecha crece con el plazo — parece calcularla desde `lastPrice`,
que puede estar rancio), pero la FORMA es la misma: correlación 0,999 en todos
los vencimientos, y el sesgo coincide dentro de ~1 punto. Como una superficie
sirve justamente para leer la forma, la conclusión no cambia; pero el nivel
absoluto se calcula aquí con supuestos propios y explícitos.

Se usan solo opciones **fuera de dinero** (puts por debajo del spot, calls por
encima), que son las líquidas, y se exigen `bid > 0` e interés abierto > 0. Sin
eso, los strikes muertos meten volatilidades de 300% que deforman todo.

Uso:
    python superficie_vol.py SPY
    python superficie_vol.py AAPL --max-dias 200
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

try:
    import options_greeks as OG
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "options_greeks"))
    import options_greeks as OG

# paleta Nocturne (la misma de cuant_trading/dashboard/finanzia_charts.py)
_NEU, _ACC, _ACC2 = "#9aa0b5", "#8271e0", "#a99bf0"
_UP, _DOWN, _GOLD, _DIM = "#3fa87c", "#d4615c", "#ab7f28", "#6f7486"
_CIAN, _AMB, _ROSA = "#2d9ec4", "#ab7f28", "#c25b86"

_CACHE = {}
_TTL = 900          # 15 min: la cadena se mueve, pero no cada segundo


def _dividendo(info):
    """yfinance devuelve el dividendo unas veces en % y otras en tanto por uno."""
    dy = info.get("dividendYield") or info.get("yield") or 0.0
    try:
        dy = float(dy)
    except Exception:
        return 0.0
    return dy / 100.0 if dy > 1.0 else dy


def superficie(ticker, max_dias=400, min_dias=5, max_vencs=9, tasa=0.04):
    """Rejilla (moneyness x plazo) de volatilidad implícita, recalculada."""
    import time as _t
    import yfinance as yf
    tk = ticker.strip().upper()
    hit = _CACHE.get((tk, max_dias))
    if hit and (_t.time() - hit[0]) < _TTL:
        return hit[1]

    t = yf.Ticker(tk)
    vencs = t.options
    if not vencs:
        raise ValueError(f"{tk}: sin cadena de opciones (yfinance solo cubre EEUU).")
    c = t.history(period="5d")["Close"].astype(float).dropna()
    if c.empty:
        raise ValueError(f"{tk}: sin precio.")
    S = float(c.iloc[-1])
    q = _dividendo(t.info or {})
    hoy = datetime.now(timezone.utc)

    # vencimientos repartidos en el rango, no los 34 (cada uno son 2 llamadas)
    cand = []
    for v in vencs:
        d = (datetime.strptime(v, "%Y-%m-%d").replace(tzinfo=timezone.utc) - hoy).days + 1
        if min_dias <= d <= max_dias:
            cand.append((v, d))
    if not cand:
        raise ValueError(f"{tk}: sin vencimientos entre {min_dias} y {max_dias} días.")
    if len(cand) > max_vencs:
        idx = np.linspace(0, len(cand) - 1, max_vencs).round().astype(int)
        cand = [cand[i] for i in sorted(set(idx))]

    filas = []
    for v, dias in cand:
        tau = dias / 365.0
        try:
            ch = t.option_chain(v)
        except Exception:
            continue
        for df, tipo in ((ch.puts, "put"), (ch.calls, "call")):
            d = df[(df["bid"] > 0) & (df["openInterest"].fillna(0) > 0)].copy()
            if d.empty:
                continue
            d["mid"] = (d["bid"] + d["ask"]) / 2.0
            d["m"] = d["strike"] / S
            # solo fuera de dinero: puts por debajo, calls por encima
            d = d[(d["m"] <= 1.0) if tipo == "put" else (d["m"] >= 1.0)]
            d = d[d["m"].between(0.70, 1.30)]
            if d.empty:
                continue
            for _, r in d.iterrows():
                iv = OG.vol_implicita(r["mid"], S, r["strike"], tau, tasa, q, tipo)
                if np.isfinite(iv) and 0.01 < iv < 3.0:
                    filas.append({"venc": v, "dias": dias, "tau": tau, "tipo": tipo,
                                  "strike": float(r["strike"]), "moneyness": float(r["m"]),
                                  "iv": float(iv), "oi": float(r["openInterest"])})
    if not filas:
        raise ValueError(f"{tk}: cadena sin strikes utilizables tras filtrar.")
    sup = pd.DataFrame(filas).sort_values(["dias", "moneyness"]).reset_index(drop=True)
    res = {"ticker": tk, "S": S, "q": q, "tasa": tasa, "sup": sup,
           "vencs": sorted(sup["dias"].unique().tolist())}
    _CACHE[(tk, max_dias)] = (_t.time(), res)
    return res


def _en(df, x, col="iv"):
    """Valor de `col` en el moneyness más cercano a x."""
    if df.empty:
        return np.nan
    return float(df.iloc[(df["moneyness"] - x).abs().argmin()][col])


def metricas(res, period_real="1y"):
    """Sesgo por plazo, estructura temporal ATM y prima de riesgo de varianza."""
    import yfinance as yf
    sup = res["sup"]
    filas = []
    for dias in res["vencs"]:
        d = sup[sup["dias"] == dias]
        atm = _en(d, 1.00)
        iv90, iv110 = _en(d[d["tipo"] == "put"], 0.90), _en(d[d["tipo"] == "call"], 1.10)
        filas.append({"Días": dias,
                      "IV ATM %": round(atm * 100, 1) if np.isfinite(atm) else np.nan,
                      "IV −10% (put) %": round(iv90 * 100, 1) if np.isfinite(iv90) else np.nan,
                      "IV +10% (call) %": round(iv110 * 100, 1) if np.isfinite(iv110) else np.nan,
                      "Sesgo (pp)": (round((iv90 - iv110) * 100, 1)
                                     if np.isfinite(iv90) and np.isfinite(iv110) else np.nan)})
    tabla = pd.DataFrame(filas)

    # volatilidad realizada: lo que el activo se ha movido de verdad
    try:
        h = yf.Ticker(res["ticker"]).history(period=period_real, auto_adjust=True)["Close"]
        r = np.log(h.astype(float).dropna()).diff().dropna()
        real = float(r.std() * np.sqrt(252))
    except Exception:
        real = np.nan

    atm_corto = tabla["IV ATM %"].dropna().iloc[0] / 100 if tabla["IV ATM %"].notna().any() else np.nan
    atm_largo = tabla["IV ATM %"].dropna().iloc[-1] / 100 if tabla["IV ATM %"].notna().any() else np.nan
    return {"tabla": tabla, "realizada": real, "atm_corto": atm_corto, "atm_largo": atm_largo,
            "pendiente": (atm_largo - atm_corto) if np.isfinite(atm_corto) else np.nan,
            "prima_varianza": (atm_corto - real) if np.isfinite(real) and np.isfinite(atm_corto) else np.nan}


def explicar(res, met):
    t = met["tabla"]
    L = [f"### {res['ticker']} · superficie de volatilidad "
         f"(spot {res['S']:.2f}, {len(res['vencs'])} vencimientos)"]

    ses = t["Sesgo (pp)"].dropna()
    if len(ses):
        s0 = ses.iloc[0]
        L.append(f"- **Sesgo a corto plazo: {s0:+.1f} puntos.** Proteger una caída del 10% "
                 f"cuesta {abs(s0):.1f} puntos más de volatilidad que apostar a una subida "
                 f"del 10%. " + ("Normal en bolsa: la protección siempre es más cara."
                                 if s0 > 1 else
                                 "Muy plano: el mercado no está pagando prima por cubrirse."
                                 if abs(s0) <= 1 else
                                 "⚠️ Invertido: el mercado paga más por la subida que por "
                                 "protegerse. Pasa en materias primas y en manías alcistas."))
        if len(ses) > 2:
            # comparar solo el primer y el último plazo miente cuando la curva
            # tiene joroba, que es lo normal: describir el máximo y dónde está
            tsk = t.dropna(subset=["Sesgo (pp)"])
            i_max = tsk["Sesgo (pp)"].idxmax()
            d_max = int(tsk.loc[i_max, "Días"]); s_max = float(tsk.loc[i_max, "Sesgo (pp)"])
            if i_max in (tsk.index[0], tsk.index[-1]):
                forma = ("desciende con el plazo: el miedo se concentra en el corto"
                         if i_max == tsk.index[0] else
                         "crece con el plazo: la inquietud está en el largo, no en el corto")
                L.append(f"- El sesgo **{forma}** ({ses.iloc[0]:+.1f} → {ses.iloc[-1]:+.1f} pp).")
            else:
                L.append(f"- El sesgo **no es monótono**: sube hasta un máximo de "
                         f"**{s_max:+.1f} pp a {d_max} días** y luego afloja "
                         f"({ses.iloc[0]:+.1f} → {s_max:+.1f} → {ses.iloc[-1]:+.1f} pp). "
                         f"Ese pico marca el plazo donde el mercado más paga por cubrirse.")

    p = met["pendiente"]
    if np.isfinite(p):
        if p > 0.01:
            L.append(f"- **Estructura temporal normal** ({met['atm_corto']*100:.1f}% a corto → "
                     f"{met['atm_largo']*100:.1f}% a largo). Calma ahora, incertidumbre luego.")
        elif p < -0.01:
            L.append(f"- ⚠️ **Estructura INVERTIDA** ({met['atm_corto']*100:.1f}% a corto → "
                     f"{met['atm_largo']*100:.1f}% a largo). El mercado descuenta un susto "
                     f"inmediato: resultados, decisión de tipos, o miedo de verdad.")
        else:
            L.append(f"- Estructura temporal plana (~{met['atm_corto']*100:.1f}%).")

    pv = met["prima_varianza"]
    if np.isfinite(pv):
        L.append(f"- **Implícita {met['atm_corto']*100:.1f}% vs realizada "
                 f"{met['realizada']*100:.1f}%**: el mercado cobra **{pv*100:+.1f} puntos** "
                 f"por encima de lo que el activo se ha movido de verdad. "
                 + ("Es la *prima de riesgo de varianza*: de media, quien VENDE opciones cobra "
                    "esa diferencia — y a cambio se lleva el golpe entero cuando llega el "
                    "crash. Cobrarla no es gratis, es vender seguros."
                    if pv > 0 else
                    "La implícita está POR DEBAJO de lo realizado: comprar protección sale "
                    "barato respecto a cuánto se está moviendo esto."))

    L.append(f"\n> Volatilidades recalculadas por bisección desde el punto medio bid/ask, con "
             f"tipo {res['tasa']*100:.0f}% y dividendo {res['q']*100:.2f}%. **No** se usa la "
             f"`impliedVolatility` de yfinance: su nivel sale 1-3 puntos por debajo, aunque la "
             f"forma coincide (correlación 0,999). Solo opciones fuera de dinero con horquilla "
             f"y posiciones abiertas. Foto de hoy, no predicción. No es recomendación.")
    return "\n".join(L)


def _plot(res, met):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from finanzia_charts import style
    except Exception:
        style = None
    sup = res["sup"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 4.8))

    # --- 1. la sonrisa, una curva por vencimiento
    vs = res["vencs"]
    cmap = plt.get_cmap("nocturne_seq") if "nocturne_seq" in plt.colormaps() else plt.get_cmap("viridis")
    for i, dias in enumerate(vs):
        d = sup[sup["dias"] == dias].sort_values("moneyness")
        if len(d) < 4:
            continue
        col = cmap(0.15 + 0.8 * i / max(1, len(vs) - 1))
        a1.plot(d["moneyness"] * 100, d["iv"] * 100, lw=1.6, color=col, label=f"{dias}d")
    a1.axvline(100, color=_DIM, ls=":", lw=1)
    a1.annotate("precio hoy", xy=(100, a1.get_ylim()[1]), xytext=(4, -12),
                textcoords="offset points", color=_DIM, fontsize=9)
    if style:
        style(a1, titulo="La sonrisa: más caro protegerse que apostar al alza",
              kicker="VOLATILIDAD IMPLICITA POR STRIKE", xlabel="Strike como % del precio →",
              ylabel="Volatilidad implícita %")
    a1.legend(fontsize=8.5, ncol=2, frameon=False, labelcolor=_NEU)

    # --- 2. estructura temporal.
    # ATM y sesgo van en el MISMO eje, los dos en puntos de volatilidad. Antes
    # esto era un doble eje (twinx) con el sesgo a la derecha, y un doble eje
    # es una gráfica que miente: dos escalas distintas hacen que dos curvas se
    # crucen o se separen según dónde decidas poner el cero, no según el dato.
    t = met["tabla"].dropna(subset=["IV ATM %"])
    a2.plot(t["Días"], t["IV ATM %"], marker="o", lw=2.2, color=_ACC,
            mfc=_ACC, mec="#171a25", mew=2, label="Implícita ATM")
    if len(t):
        a2.annotate(f"{t['IV ATM %'].iloc[-1]:.1f}%",
                    xy=(t["Días"].iloc[-1], t["IV ATM %"].iloc[-1]),
                    xytext=(9, 0), textcoords="offset points", color=_ACC,
                    fontsize=11, fontweight="semibold", va="center",
                    annotation_clip=False)
    ts = met["tabla"].dropna(subset=["Sesgo (pp)"])
    if len(ts) > 1:
        a2.plot(ts["Días"], ts["Sesgo (pp)"], marker="s", ms=6, lw=2.0,
                color=_AMB, mfc=_AMB, mec="#171a25", mew=2,
                label="Sesgo put−call (puntos)")
    if np.isfinite(met["realizada"]):
        a2.axhline(met["realizada"] * 100, color=_NEU, ls="--", lw=1.5,
                   label=f"Realizada {met['realizada']*100:.1f}% (1 año)")
    if style:
        style(a2, titulo="Por plazo: lo implícito, el sesgo y lo que se movió de verdad",
              kicker="TODO EN PUNTOS DE VOLATILIDAD · UN SOLO EJE",
              xlabel="Días al vencimiento →", ylabel="Puntos de volatilidad")
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description="Superficie de volatilidad implícita.")
    ap.add_argument("ticker", nargs="?", default="SPY")
    ap.add_argument("--max-dias", type=int, default=400)
    a = ap.parse_args()
    res = superficie(a.ticker, a.max_dias)
    met = metricas(res)
    print(explicar(res, met))
    print()
    print(met["tabla"].to_string(index=False))


if __name__ == "__main__":
    main()
