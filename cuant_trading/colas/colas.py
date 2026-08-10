"""
colas — cuánto miente la campana de Gauss con tus activos (curtosis y colas gordas).

La curtosis ya estaba usada DENTRO de la estadística del proyecto (error estándar
de Lo en `cpcv`, PSR y Deflated Sharpe en `veredicto_backtest`), pero en ningún
sitio se medía ni se enseñaba. Este módulo la saca a la luz y, sobre todo, la
traduce a algo que se entiende: **cuántas veces ha fallado el modelo normal**.

El dato que lo resume todo (20 años, medido ago 2026)
-----------------------------------------------------
    activo   curtosis exc.   días |z|>4: reales vs normal   peor día
    SPY           14,8              38  vs  0,32             −9,0 σ
    QQQ            7,5              22  vs  0,32             −8,7 σ
    BTC            8,0              27  vs  0,28            −10,8 σ
    KO            11,6              36  vs  0,32             −8,3 σ

El SPY tuvo 38 días de más de 4 sigmas donde la campana predecía 0,32: **119
veces más de lo que dice el modelo**. Y su peor día, −9 sigmas, bajo una normal
ocurriría una vez cada 2,9·10¹⁹ años. El universo tiene 1,4·10¹⁰.

Por qué importa, y con qué se conecta
-------------------------------------
1. **`evt_risk`** existe por esto: el VaR calculado con la normal se queda corto
   justo el día que importa. Aquí se cuenta cuántas veces se lo saltó.
2. **`deriva_vol`**: es la razón de que σ²/2 sobrestimara la resta real. Esa
   fórmula supone log-normalidad, y esto no lo es.
3. **`cpcv` / `veredicto_backtest`**: por eso el Sharpe lleva la corrección de
   Lo (2002) por asimetría y curtosis. Un Sharpe sin corregir, con curtosis 15,
   promete una precisión que no tiene.

Uso:
    python colas.py SPY QQQ BTC-USD
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
_NEU, _ACC, _ACC2 = "#b2b6ca", "#9184d9", "#b5abfc"
_UP, _DOWN, _GOLD, _DIM = "#63b58e", "#d9736b", "#c9b273", "#75798c"


def medir(ticker, period="20y", conf=0.99):
    """Curtosis, asimetría, normalidad y cuántas veces falló el modelo normal."""
    from scipy.stats import kurtosis, skew, norm, jarque_bera
    tk = ticker.strip().upper()
    h = yf.Ticker(tk).history(period=period, auto_adjust=True)["Close"].astype(float).dropna()
    if len(h) < 400:
        raise ValueError(f"'{tk}': histórico insuficiente ({len(h)} sesiones).")
    r = h.pct_change().dropna()
    n = len(r)
    z = ((r - r.mean()) / r.std()).values

    # cuántos días extremos hubo, frente a los que predice la campana
    sig = {}
    for u in (3, 4, 5):
        sig[u] = {"reales": int((np.abs(z) > u).sum()),
                  "normal": float(2 * norm.sf(u) * n)}

    peor = float(z.min())
    p_peor = float(norm.cdf(peor))
    # cada cuántos años ocurriría ese día si el mundo fuera normal
    periodo_retorno = (1.0 / (p_peor * 252)) if p_peor > 0 else float("inf")

    # VaR: lo que dice la campana vs lo que pasó de verdad
    var_norm = float(r.mean() + norm.ppf(1 - conf) * r.std())
    var_emp = float(np.percentile(r, (1 - conf) * 100))
    excesos = int((r < var_norm).sum())
    excesos_esp = (1 - conf) * n

    return {"ticker": tk, "n": n, "anios": round(n / 252, 1),
            "curtosis_exc": float(kurtosis(z)),          # 0 = normal
            "asimetria": float(skew(z)),
            "jb_p": float(jarque_bera(z).pvalue),
            "sigmas": sig, "peor_z": peor, "periodo_retorno": periodo_retorno,
            "conf": conf, "var_norm": var_norm, "var_emp": var_emp,
            "excesos": excesos, "excesos_esp": excesos_esp,
            "vol_diaria": float(r.std()), "z": z}


def tabla(tickers, period="20y"):
    filas = []
    for tk in tickers:
        try:
            d = medir(tk, period)
        except Exception as e:
            filas.append({"Activo": tk.upper(), "Nota": str(e)[:40]})
            continue
        s4 = d["sigmas"][4]
        filas.append({
            "Activo": d["ticker"],
            "Años": d["anios"],
            "Curtosis exc.": round(d["curtosis_exc"], 1),
            "Asimetría": round(d["asimetria"], 2),
            "Días >4σ (reales)": s4["reales"],
            "Días >4σ (normal)": round(s4["normal"], 2),
            "Veces más": (round(s4["reales"] / s4["normal"], 0) if s4["normal"] > 0 else np.nan),
            "Peor día (σ)": round(d["peor_z"], 1),
        })
    return pd.DataFrame(filas)


def _anios_legible(a):
    if not np.isfinite(a):
        return "nunca"
    if a < 1e3:
        return f"{a:,.0f} años".replace(",", ".")
    if a < 1e6:
        return f"{a/1e3:,.0f} mil años".replace(",", ".")
    if a < 1e9:
        return f"{a/1e6:,.0f} millones de años".replace(",", ".")
    return f"{a:.0e} años"


def explicar(d):
    L = [f"### {d['ticker']} · {d['anios']} años ({d['n']:,} sesiones)".replace(",", ".")]
    k = d["curtosis_exc"]
    L.append(f"- **Curtosis en exceso {k:.1f}** (una campana normal vale 0). "
             + ("Colas **muy** gordas." if k > 6 else
                "Colas gordas." if k > 2 else "Cerca de lo normal."))
    if d["asimetria"] < -0.15:
        L.append(f"- Asimetría **{d['asimetria']:+.2f}**: los días malos son más "
                 f"violentos que los buenos.")
    elif d["asimetria"] > 0.15:
        L.append(f"- Asimetría **{d['asimetria']:+.2f}**: los días buenos son más "
                 f"violentos que los malos.")
    s4 = d["sigmas"][4]
    if s4["normal"] > 0:
        L.append(f"- Hubo **{s4['reales']} días** de más de 4σ. La campana predecía "
                 f"**{s4['normal']:.2f}**: **{s4['reales']/s4['normal']:.0f} veces más** "
                 f"de lo que dice el modelo.")
    L.append(f"- El peor día fue de **{d['peor_z']:.1f}σ** ({d['peor_z']*d['vol_diaria']*100:.1f}%). "
             f"Si el mundo fuera normal, un día así tocaría **una vez cada "
             f"{_anios_legible(d['periodo_retorno'])}**.")
    L.append(f"- VaR al {d['conf']*100:.0f}%: la campana dice **{d['var_norm']*100:.2f}%**, "
             f"pero el dato real es **{d['var_emp']*100:.2f}%**. Se lo saltó "
             f"**{d['excesos']} veces** cuando esperaba {d['excesos_esp']:.0f}.")
    L.append("\n> Por eso este proyecto **no se fía de la normal**: el riesgo de cola se mide "
             "con EVT en 📉 EVT, el Sharpe lleva la corrección de Lo por asimetría y curtosis, "
             "y la resta de σ²/2 de 📉 Deriva vol. sale sobrestimada justo por esto. "
             "Medición del pasado, no predicción.")
    return "\n".join(L)


def _plot(datos):
    """Histograma en escala logarítmica: es la única forma de VER las colas."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    try:
        from finanzia_charts import style
    except Exception:
        style = None

    n = len(datos)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.4), squeeze=False)
    for ax, d in zip(axes[0], datos):
        z = d["z"]
        lim = max(6, min(12, abs(d["peor_z"]) + 1))
        bins = np.linspace(-lim, lim, 90)
        ax.hist(z, bins=bins, density=True, color=_ACC, alpha=0.55, edgecolor="none",
                label="lo que pasó")
        gx = np.linspace(-lim, lim, 400)
        ax.plot(gx, norm.pdf(gx), color=_GOLD, lw=1.8, label="campana normal")
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1)
        for u, c in ((4, _DOWN),):
            ax.axvline(-u, color=c, ls="--", lw=1)
            ax.axvline(u, color=c, ls="--", lw=1)
        s4 = d["sigmas"][4]
        ax.annotate(f"{s4['reales']} días fuera de ±4σ\nla normal decía {s4['normal']:.2f}",
                    xy=(0.02, 0.04), xycoords="axes fraction", color=_DOWN, fontsize=9.5)
        if style:
            style(ax, titulo=f"{d['ticker']}  ·  curtosis {d['curtosis_exc']:.1f}",
                  kicker="RETORNOS DIARIOS EN SIGMAS · ESCALA LOGARITMICA",
                  xlabel="desviaciones típicas", ylabel="densidad (log)")
        else:
            ax.set_title(d["ticker"]); ax.legend()
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description="Curtosis y colas gordas: cuánto miente la normal.")
    ap.add_argument("tickers", nargs="*", default=["SPY", "QQQ", "BTC-USD"])
    ap.add_argument("--period", default="20y")
    a = ap.parse_args()
    tks = a.tickers or ["SPY", "QQQ", "BTC-USD"]
    print(tabla(tks, a.period).to_string(index=False))
    print()
    for tk in tks:
        try:
            print(explicar(medir(tk, a.period)) + "\n")
        except Exception as e:
            print(f"{tk}: {e}\n")


if __name__ == "__main__":
    main()
