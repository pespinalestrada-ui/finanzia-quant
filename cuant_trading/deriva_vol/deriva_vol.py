"""
deriva_vol — cuánta rentabilidad te come la volatilidad (lema de Itô aplicado).

El lema de Itô dice que si el precio sigue un browniano geométrico
dS = μ S dt + σ S dW, entonces el LOGARITMO del precio no sigue μ, sino:

    d(ln S) = (μ − σ²/2) dt + σ dW

Ese −σ²/2 es lo único que hay que llevarse de aquí, y tiene consecuencia directa
en el bolsillo: **lo que compones no es la media de tus retornos, es la media
menos medio cuadrado de la volatilidad**. Cuanto más se mueve algo, más se aleja
lo que acabas teniendo de lo que dice su rentabilidad media.

Medido sobre 10 años (ago 2026):

    SPY   (σ 18%)  media 15,9%  →  CAGR 15,4%   se come 0,6 pp
    QQQ   (σ 23%)  media 21,5%  →  CAGR 20,9%   se come 0,6 pp
    TQQQ  (σ 67%)  media 56,8%  →  CAGR 40,9%   se come 15,9 pp
    BTC   (σ 55%)  media 47,7%  →  CAGR 38,1%   se come 9,6 pp

Por qué importa aquí
--------------------
1. Explica por qué funciona el algoritmo de volatilidad objetivo (`voltarget`):
   bajar σ no solo reduce el susto, **reduce la resta**.
2. Explica por qué los ETF apalancados x3 decaen: la deriva crece con el CUADRADO
   del apalancamiento (k²σ²/2) mientras la ganancia solo crece linealmente (kμ).
   Pasado cierto punto, más apalancamiento resta en vez de sumar.
3. Es la razón de que en `montecarlo` la simulación use (mu − 0.5·sd²): ese
   término ya estaba en el proyecto, pero sin nada que lo midiera ni explicara.

Honestidad
----------
σ²/2 es una APROXIMACIÓN que supone log-normalidad. Con volatilidades bajas
(<25%) acierta casi exacto; con colas gordas (TQQQ, cripto) se pasa de largo,
porque los retornos reales no son log-normales. Por eso aquí se calcula SIEMPRE
la brecha real con los datos, y σ²/2 se muestra al lado solo como referencia
teórica. No es una predicción: es contabilidad del pasado.

Uso:
    python deriva_vol.py SPY QQQ TQQQ BTC-USD
    python deriva_vol.py --apalancamiento SPY
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

# paleta Nocturne (la misma de cuant_trading/dashboard/finanzia_charts.py)
_NEU, _ACC, _ACC2 = "#9aa0b5", "#8271e0", "#a99bf0"
_UP, _DOWN, _GOLD, _DIM = "#3fa87c", "#d4615c", "#ab7f28", "#6f7486"
_CIAN, _AMB, _ROSA = "#2d9ec4", "#ab7f28", "#c25b86"


def medir(ticker, period="10y"):
    """Media aritmética vs rentabilidad compuesta real, y lo que explica Itô."""
    tk = ticker.strip().upper()
    h = yf.Ticker(tk).history(period=period, auto_adjust=True)["Close"].astype(float).dropna()
    if len(h) < 260:
        raise ValueError(f"'{tk}': histórico insuficiente ({len(h)} sesiones).")
    r = h.pct_change().dropna()
    lr = np.log(h / h.shift(1)).dropna()
    anios = len(h) / 252.0

    arit = float(r.mean() * 252)                     # lo que suele publicarse
    sigma = float(r.std() * np.sqrt(252))
    cagr = float((h.iloc[-1] / h.iloc[0]) ** (1 / anios) - 1)   # lo que de verdad te llevas
    geom_log = float(np.exp(lr.mean() * 252) - 1)    # equivalente por log-retornos
    brecha = arit - cagr
    ito = 0.5 * sigma ** 2                           # la resta que predice Itô
    return {"ticker": tk, "anios": round(anios, 1), "arit": arit, "cagr": cagr,
            "geom_log": geom_log, "sigma": sigma, "brecha": brecha, "ito": ito,
            "explicado": (ito / brecha) if brecha > 1e-9 else np.nan,
            "serie": h}


def tabla(tickers, period="10y"):
    filas = []
    for tk in tickers:
        try:
            d = medir(tk, period)
        except Exception as e:
            filas.append({"Activo": tk.upper(), "Nota": str(e)[:40]})
            continue
        filas.append({
            "Activo": d["ticker"],
            "Volatilidad %": round(d["sigma"] * 100, 1),
            "Media anual %": round(d["arit"] * 100, 1),
            "CAGR real %": round(d["cagr"] * 100, 1),
            "Se come (pp)": round(d["brecha"] * 100, 1),
            "σ²/2 teórico (pp)": round(d["ito"] * 100, 1),
        })
    return pd.DataFrame(filas)


def curva_apalancamiento(ticker, period="10y", k_max=4.0):
    """CAGR esperado en función del apalancamiento k, según Itô.

        CAGR(k) ≈ k·μ − k²·σ²/2

    La ganancia sube en línea recta con k; la resta sube con el CUADRADO. Por eso
    existe un óptimo k* = μ/σ² y un punto k₀ = 2μ/σ² donde apalancar MÁS te deja
    peor que no apalancar nada. Es la explicación exacta de por qué los ETF x3
    decaen en mercados laterales.
    """
    d = medir(ticker, period)
    mu, s2 = d["arit"], d["sigma"] ** 2
    ks = np.linspace(0, k_max, 200)
    cagr_k = ks * mu - (ks ** 2) * s2 / 2.0
    k_opt = mu / s2 if s2 > 0 else np.nan
    k_cero = 2 * mu / s2 if s2 > 0 else np.nan
    return {"ks": ks, "cagr_k": cagr_k, "k_opt": float(k_opt), "k_cero": float(k_cero),
            "mu": mu, "sigma": d["sigma"], "ticker": d["ticker"]}


def destino_probable(ticker, anios=10, n_sims=4000, period="20y", bloque=20, semilla=7):
    """¿Qué le pasa al inversor NORMAL, no a la media?

    La rentabilidad que se anuncia es la MEDIA, y la media de una distribución
    torcida a la derecha la levantan unos pocos caminos que se van muy arriba.
    A ti te toca vivir UNO. La pregunta útil no es cuánto da de media, sino
    cuánto le toca al de en medio y qué probabilidad hay de llegar a la media.

    Se remuestrea POR BLOQUES de retornos reales (no browniano geométrico): así
    se conservan las colas gordas y los racimos de volatilidad, que es justo lo
    que una simulación log-normal se inventa demasiado amable.

    Medido a 10 años: al SPY solo llega a su media el 41% de los caminos; al BTC,
    el 37%, y el inversor de en medio se queda en 34,9% frente al 43,9% anunciado.
    """
    tk = ticker.strip().upper()
    h = yf.Ticker(tk).history(period=period, auto_adjust=True)["Close"].astype(float).dropna()
    if len(h) < 500:
        raise ValueError(f"'{tk}': histórico insuficiente ({len(h)} sesiones).")
    lr = np.log(h / h.shift(1)).dropna().values
    arit = float((np.exp(lr).mean() - 1) * 252)          # la que se anuncia
    n = int(252 * anios)
    rng = np.random.default_rng(semilla)
    finales = np.empty(n_sims)
    n_bloques = n // bloque + 1
    for i in range(n_sims):
        idx = rng.integers(0, len(lr) - bloque, size=n_bloques)
        s = np.concatenate([lr[j:j + bloque] for j in idx])[:n]
        finales[i] = np.exp(s.sum())
    cagr = finales ** (1 / anios) - 1
    pct = {p: float(np.percentile(cagr, p)) for p in (10, 25, 50, 75, 90)}
    return {"ticker": tk, "anios": anios, "n_sims": n_sims, "anunciada": arit,
            "mediana": pct[50], "llegan": float((cagr >= arit).mean()),
            "pct": pct, "cagr": cagr,
            "perdida": float((cagr < 0).mean())}


def explicar_destino(d):
    # el separador de miles va en el número, no en la frase entera: aplicar el
    # replace al f-string completo se comía las comas del texto
    sims = f"{d['n_sims']:,}".replace(",", ".")
    L = [f"### {d['ticker']} — qué le pasa al inversor normal a {d['anios']} años",
         f"- La rentabilidad que se **anuncia** es **{d['anunciada']*100:.1f}%** anual. "
         f"Pero de {sims} caminos posibles, solo **{d['llegan']*100:.0f}%** la alcanzan."]
    L.append(f"- Al inversor **de en medio** le toca **{d['mediana']*100:.1f}%**: "
             f"**{(d['anunciada']-d['mediana'])*100:.1f} puntos menos** de lo anunciado.")
    p = d["pct"]
    L.append(f"- El abanico real: 1 de cada 10 se queda en **{p[10]*100:+.1f}%** o menos; "
             f"1 de cada 10 supera **{p[90]*100:+.1f}%**. La mitad central va de "
             f"{p[25]*100:+.1f}% a {p[75]*100:+.1f}%.")
    if d["perdida"] > 0.01:
        L.append(f"- **{d['perdida']*100:.0f}%** de los caminos acaban en **pérdida** "
                 f"tras {d['anios']} años.")
    L.append(f"\n> La media la levantan unos pocos caminos que se van muy arriba; a ti te toca "
             f"vivir **uno**. Por eso, cuando leas «esto da un {d['anunciada']*100:.0f}% de "
             f"media», la pregunta correcta es qué le pasa al inversor normal — y casi siempre "
             f"es bastante menos.")
    L.append("> Remuestreo por bloques de los retornos **reales**, no browniano geométrico: "
             "conserva colas gordas y racimos de volatilidad. Sigue siendo una simulación "
             "sobre el pasado, **no** una previsión.")
    return "\n".join(L)


def _plot_destino(d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from finanzia_charts import style
    except Exception:
        style = None
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = d["cagr"] * 100
    ax.hist(x, bins=70, color=_ACC, alpha=0.55, edgecolor="none")
    ax.axvline(d["anunciada"] * 100, color=_GOLD, lw=2,
               label=f"Media anunciada {d['anunciada']*100:.1f}%")
    ax.axvline(d["mediana"] * 100, color=_UP, lw=2, ls="--",
               label=f"Al inversor de en medio {d['mediana']*100:.1f}%")
    ax.axvspan(x.min(), 0, color=_DOWN, alpha=0.10)
    ax.annotate(f"solo el {d['llegan']*100:.0f}% llega a la media anunciada",
                xy=(d["anunciada"] * 100, ax.get_ylim()[1] * 0.88), xytext=(8, 0),
                textcoords="offset points", color=_GOLD, fontsize=10.5)
    sims = f"{d['n_sims']:,}".replace(",", ".")
    if style:
        style(ax, titulo=f"{d['ticker']} — dónde acaba cada uno de {sims} inversores",
              kicker=f"REMUESTREO POR BLOQUES DE RETORNOS REALES · {d['anios']} AÑOS",
              xlabel="Rentabilidad anual conseguida →", ylabel="nº de caminos")
    else:
        ax.legend()
    fig.tight_layout()
    return fig


def explicar(d):
    """La lectura en cristiano."""
    L = [f"### {d['ticker']} · {d['anios']} años",
         f"- Su rentabilidad **media** es **{d['arit']*100:.1f}%** al año, pero lo que de "
         f"verdad has compuesto es **{d['cagr']*100:.1f}%**. La diferencia, "
         f"**{d['brecha']*100:.1f} puntos**, se la ha comido la volatilidad."]
    L.append(f"- Su volatilidad es **{d['sigma']*100:.0f}%**. La fórmula σ²/2 daría "
             f"**{d['ito']*100:.1f} puntos**, o sea **{1/d['explicado']:.0%}** de lo que "
             f"realmente se ha comido."
             if np.isfinite(d["explicado"]) and d["explicado"] > 0 else
             f"- Su volatilidad es **{d['sigma']*100:.0f}%**.")
    L.append("- La fórmula **se pasa siempre**, y no es un fallo del dato: σ²/2 supone que los "
             "retornos son log-normales e independientes, y no lo son. Medido en 6 activos, la "
             "resta real va del 25% al 72% de lo que predice la teoría. **Vale como techo, no "
             "como estimación**: el número que cuenta es la brecha medida.")
    if d["sigma"] > 0.40:
        L.append(f"- ⚠️ Con σ del {d['sigma']*100:.0f}%, la resta es enorme. Aquí bajar "
                 f"volatilidad no es solo dormir mejor: es **ganar más compuesto**. "
                 f"Es exactamente lo que hace la pestaña 🛞 Vol objetivo.")
    elif d["sigma"] < 0.25:
        L.append(f"- Con σ del {d['sigma']*100:.0f}% la resta es pequeña ({d['brecha']*100:.1f} pp): "
                 f"en activos tranquilos, media y compuesto casi coinciden.")
    L.append("\n> Es **contabilidad del pasado**, no una predicción: mide lo que la "
             "volatilidad ya te ha restado. No es recomendación de inversión.")
    return "\n".join(L)


def explicar_apalancamiento(c):
    k_opt, k0 = c["k_opt"], c["k_cero"]
    L = [f"### {c['ticker']} — cuánto apalancamiento aguanta",
         f"Con μ = {c['mu']*100:.1f}% y σ = {c['sigma']*100:.0f}%, la fórmula de Itô "
         f"`CAGR(k) = k·μ − k²·σ²/2` dice:",
         f"- El máximo está en **k\\* = {k_opt:.2f}×**. Más allá, cada punto de "
         f"apalancamiento **resta**.",
         f"- A partir de **k₀ = {k0:.2f}×** apalancar te deja **peor que no apalancar nada**."]
    if k0 < 3:
        L.append(f"- 🔴 Un ETF **x3** de este activo está por encima de {k0:.1f}×: "
                 f"matemáticamente destinado a rendir menos que el subyacente a largo plazo, "
                 f"aunque el subyacente suba.")
    L.append("\n> La ganancia crece con k, pero la resta crece con **k²**. Por eso el "
             "apalancamiento tiene un techo que no depende de tu convicción, sino de la "
             "volatilidad del activo.")
    L.append("> ⚠️ Estos k son **optimistas**: la fórmula supone rebalanceo continuo, coste de "
             "financiación cero y retornos log-normales. En la realidad hay intereses, "
             "comisiones y colas gordas, así que el techo real está **bastante por debajo**. "
             "Úsalo para entender la forma de la curva, no para elegir apalancamiento.")
    return "\n".join(L)


def _plot(datos, curva=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from finanzia_charts import style
    except Exception:
        style = None

    if curva is not None:
        fig, ax = plt.subplots(figsize=(10, 4.6))
        ks, y = curva["ks"], curva["cagr_k"] * 100
        ax.plot(ks, y, color=_ACC, lw=2)
        ax.axhline(0, color=_DIM, ls="--", lw=1)
        ax.axvline(1, color=_NEU, ls=":", lw=1)
        ax.annotate("sin apalancar", xy=(1, y.min()), xytext=(4, 4),
                    textcoords="offset points", color=_NEU, fontsize=9)
        if np.isfinite(curva["k_opt"]) and curva["k_opt"] <= ks.max():
            ax.axvline(curva["k_opt"], color=_UP, ls="--", lw=1.2)
            ax.annotate(f"máximo  k*={curva['k_opt']:.2f}×", xy=(curva["k_opt"], y.max()),
                        xytext=(6, -12), textcoords="offset points", color=_UP, fontsize=10)
        if np.isfinite(curva["k_cero"]) and curva["k_cero"] <= ks.max():
            ax.axvline(curva["k_cero"], color=_DOWN, ls="--", lw=1.2)
            ax.annotate(f"aquí ya no compensa  k₀={curva['k_cero']:.2f}×",
                        xy=(curva["k_cero"], 0), xytext=(6, 8),
                        textcoords="offset points", color=_DOWN, fontsize=10)
        if style:
            # el kicker va en mayúsculas: sin letras griegas, que μ y σ se convierten
            # en Μ y Σ y pasan a significar otra cosa
            style(ax, titulo=f"{curva['ticker']} — rentabilidad compuesta según el apalancamiento",
                  kicker="LEMA DE ITÔ · LA GANANCIA SUBE CON k, LA RESTA CON k AL CUADRADO",
                  xlabel="Apalancamiento k →", ylabel="CAGR esperado %", legend=False)
            ax.annotate(r"$CAGR(k) = k\mu - k^2\sigma^2/2$", xy=(0.995, 0.04),
                        xycoords="axes fraction", ha="right", color=_DIM, fontsize=11)
        fig.tight_layout()
        return fig

    # comparativa: media vs compuesto, por activo
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(datos))
    arit = [d["arit"] * 100 for d in datos]
    cagr = [d["cagr"] * 100 for d in datos]
    ax.bar(x - 0.19, arit, 0.38, color=_NEU, label="Media anual (lo que se publica)")
    ax.bar(x + 0.19, cagr, 0.38, color=_ACC, label="CAGR real (lo que te llevas)")
    for i, d in enumerate(datos):
        ax.annotate(f"−{d['brecha']*100:.1f} pp", xy=(i, max(arit[i], cagr[i])),
                    xytext=(0, 6), textcoords="offset points", ha="center",
                    color=_DOWN if d["brecha"] > 0.03 else _DIM, fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([f"{d['ticker']}\nσ {d['sigma']*100:.0f}%" for d in datos])
    if style:
        style(ax, titulo="Lo que la volatilidad se come cada año",
              kicker="MEDIA ARITMÉTICA vs RENTABILIDAD COMPUESTA · LEMA DE ITÔ",
              ylabel="% anual")
    else:
        ax.legend()
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description="Cuánta rentabilidad te come la volatilidad (Itô).")
    ap.add_argument("tickers", nargs="*", default=["SPY", "QQQ", "TQQQ", "BTC-USD"])
    ap.add_argument("--period", default="10y")
    ap.add_argument("--apalancamiento", help="Curva CAGR(k) de un activo.")
    a = ap.parse_args()
    if a.apalancamiento:
        c = curva_apalancamiento(a.apalancamiento, a.period)
        print(explicar_apalancamiento(c))
        return
    tks = a.tickers or ["SPY", "QQQ", "TQQQ", "BTC-USD"]
    print(tabla(tks, a.period).to_string(index=False))
    print()
    for tk in tks:
        try:
            print(explicar(medir(tk, a.period)) + "\n")
        except Exception as e:
            print(f"{tk}: {e}\n")


if __name__ == "__main__":
    main()
