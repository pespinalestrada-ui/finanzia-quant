"""
portfolio_lab — laboratorio de carteras: métricas de riesgo avanzadas, carteras
de leyenda como referencia, y comportamiento en las crisis reales.

Complementa la suite por el lado del LARGO PLAZO. Todo es medición histórica
(mirar atrás con rigor), nada de predicción.

1) MÉTRICAS que van más allá del Sharpe:
   · Sortino  — como el Sharpe, pero solo castiga la volatilidad MALA (a la baja).
   · Omega    — cuántos euros de ganancia hay por cada euro de pérdida.
   · Ulcer    — mide el "dolor": la profundidad Y duración de las caídas.
   · Calmar   — rentabilidad anual dividida por la peor caída sufrida.
   · Peor racha + tiempo de recuperación (¿cuánto tardó en volver a máximos?).
   · VaR / CVaR históricos.

2) CARTERAS DE LEYENDA como referencia (mejor que compararse solo con el índice):
   60/40 · Permanente de Browne · All Weather de Dalio · Buffett 90/10 ·
   Bogleheads 3 fondos · Golden Butterfly.

3) CRISIS REALES: qué habría pasado con tu cartera en 2008, COVID-19 y la
   inflación/tipos de 2022.

4) CONTRIBUCIÓN AL RIESGO: qué activo aporta el riesgo de verdad (no es lo mismo
   que su peso: un 10% en algo muy volátil puede ser el 40% del riesgo).

5) CAPTURA alcista/bajista frente a un índice: cuánto sigues cuando sube y
   cuánto sufres cuando baja.

No es recomendación de inversión.

Uso:
    python portfolio_lab.py --cartera "SPY:60, AGG:40"
    python portfolio_lab.py --legendarias
    python portfolio_lab.py --cartera "VTI:50, GLD:20, TLT:30" --bench SPY
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

# --- carteras de leyenda (réplica con ETFs líquidos) -------------------------
LEGENDARIAS = {
    "60/40 clásica":        {"SPY": 60, "AGG": 40},
    "Permanente (Browne)":  {"SPY": 25, "TLT": 25, "GLD": 25, "SHY": 25},
    "All Weather (Dalio)":  {"SPY": 30, "TLT": 40, "IEF": 15, "GLD": 7.5, "DBC": 7.5},
    "Buffett 90/10":        {"SPY": 90, "SHY": 10},
    "Bogleheads 3 fondos":  {"VTI": 60, "VXUS": 20, "BND": 20},
    "Golden Butterfly":     {"VTI": 20, "IWN": 20, "TLT": 20, "SHY": 20, "GLD": 20},
}

CRISIS = {
    "Crisis financiera 2008": ("2007-10-09", "2009-03-09"),
    "COVID-19 (2020)":        ("2020-02-19", "2020-03-23"),
    "Inflación y tipos 2022": ("2022-01-03", "2022-10-12"),
}


def _precios(tickers, period="15y", start=None, end=None):
    """Precios total-return (dividendos incluidos) alineados."""
    series = {}
    for tk in tickers:
        t = yf.Ticker(tk)
        h = t.history(start=start, end=end, auto_adjust=True) if start else \
            t.history(period=period, auto_adjust=True)
        if h.empty:
            continue
        s = h["Close"].astype(float).dropna()
        if s.empty:
            continue
        s.index = pd.to_datetime(s.index).tz_localize(None)
        series[tk] = s
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).dropna()


def parsear_cartera(txt):
    """'SPY:60, AGG:40' → {'SPY': 0.6, 'AGG': 0.4} (normaliza a 1)."""
    pesos = {}
    for parte in str(txt).replace(";", ",").split(","):
        if not parte.strip():
            continue
        if ":" in parte:
            tk, w = parte.split(":", 1)
            try:
                pesos[tk.strip().upper()] = float(str(w).strip().replace("%", ""))
            except ValueError:
                continue
        else:
            pesos[parte.strip().upper()] = 1.0
    tot = sum(pesos.values())
    if tot <= 0:
        raise ValueError("Cartera vacía o pesos inválidos.")
    return {k: v / tot for k, v in pesos.items()}


def serie_cartera(pesos, period="15y", start=None, end=None, rebalanceo="mensual", min_sesiones=30):
    """Curva de valor de la cartera (base 1) con rebalanceo. Devuelve (curva, ret, cobertura).
    min_sesiones baja a ~10 para crisis cortas (el crash del COVID duró 24 sesiones)."""
    px = _precios(list(pesos), period, start, end)
    if px.empty or len(px) < min_sesiones:
        return None, None, []
    usados = [t for t in pesos if t in px.columns]
    w = np.array([pesos[t] for t in usados], dtype=float)
    w = w / w.sum()                                   # renormaliza si falta algún activo
    r = px[usados].pct_change().dropna()
    if rebalanceo == "ninguno":
        # comprar y no tocar: los pesos derivan libremente
        val = ((1 + r).cumprod() * w).sum(axis=1)
        rp = val.pct_change().dropna()
    else:
        freq = {"mensual": "M", "trimestral": "Q", "anual": "A"}.get(rebalanceo, "M")
        marcas = set(r.groupby(r.index.to_period(freq)).head(1).index)   # 1ª sesión de cada periodo
        val, pesos_act, vals = 1.0, w.copy(), []
        for fecha, fila in r.iterrows():
            if fecha in marcas:
                pesos_act = w.copy()                       # vuelve a los pesos objetivo
            crec = float(np.dot(pesos_act, 1.0 + fila.values))
            val *= crec
            pesos_act = pesos_act * (1.0 + fila.values) / crec   # deriva hasta el próximo rebalanceo
            vals.append(val)
        rp = pd.Series(vals, index=r.index).pct_change().dropna()
    curva = (1 + rp).cumprod()
    return curva, rp, usados


# --- métricas ----------------------------------------------------------------
def metricas(rp, curva=None, conf=0.95):
    """Batería completa de métricas de riesgo/rentabilidad sobre retornos diarios."""
    r = pd.Series(rp).dropna()
    if len(r) < 30:
        return {}
    if curva is None:
        curva = (1 + r).cumprod()
    años = len(r) / 252.0
    cagr = float(curva.iloc[-1] ** (1 / max(años, 1e-9)) - 1)
    vol = float(r.std() * np.sqrt(252))
    sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan
    # Sortino: solo castiga la volatilidad a la baja
    neg = r[r < 0]
    dd_dev = float(neg.std() * np.sqrt(252)) if len(neg) > 1 else np.nan
    sortino = float(r.mean() * 252 / dd_dev) if dd_dev and dd_dev > 0 else np.nan
    # Omega: ganancias totales / pérdidas totales (umbral 0)
    gan, per = float(r[r > 0].sum()), float(-r[r < 0].sum())
    omega = gan / per if per > 0 else np.inf
    # drawdown, Ulcer, Calmar, recuperación
    pico = curva.cummax()
    dd = curva / pico - 1.0
    maxdd = float(dd.min())
    ulcer = float(np.sqrt((dd ** 2).mean()) * 100)
    calmar = float(cagr / abs(maxdd)) if maxdd < 0 else np.nan
    bajo_agua = int((dd < -1e-9).sum())
    # tiempo hasta recuperar el máximo tras el peor drawdown
    i_min = dd.idxmin()
    post = curva.loc[i_min:]
    nivel = float(pico.loc[i_min])
    rec = post[post >= nivel]
    dias_rec = int(len(post.loc[:rec.index[0]])) if len(rec) else -1
    var = float(np.quantile(r, 1 - conf))
    cvar = float(r[r <= var].mean()) if (r <= var).any() else var
    return {"CAGR %": round(cagr * 100, 2), "Vol anual %": round(vol * 100, 2),
            "Sharpe": round(sharpe, 2), "Sortino": round(sortino, 2) if sortino == sortino else None,
            "Omega": round(omega, 2) if np.isfinite(omega) else None,
            "Ulcer": round(ulcer, 2), "Calmar": round(calmar, 2) if calmar == calmar else None,
            "Máx caída %": round(maxdd * 100, 1),
            "Días bajo agua": bajo_agua,
            "Días en recuperar": dias_rec if dias_rec > 0 else "aún no",
            f"VaR {int(conf*100)}% día %": round(var * 100, 2),
            f"CVaR {int(conf*100)}% día %": round(cvar * 100, 2)}


def contribucion_riesgo(pesos, period="10y"):
    """Qué parte del riesgo aporta cada activo (no es su peso: es w·(Σw)/σ²)."""
    px = _precios(list(pesos), period)
    if px.empty:
        return pd.DataFrame()
    usados = [t for t in pesos if t in px.columns]
    w = np.array([pesos[t] for t in usados]); w = w / w.sum()
    r = px[usados].pct_change().dropna()
    S = r.cov().values * 252
    var_p = float(w @ S @ w)
    mcr = S @ w                                   # contribución marginal
    ctr = w * mcr / var_p if var_p > 0 else np.zeros_like(w)
    return pd.DataFrame({"Activo": usados,
                         "Peso %": (w * 100).round(1),
                         "Vol activo %": (np.sqrt(np.diag(S)) * 100).round(1),
                         "Contrib. riesgo %": (ctr * 100).round(1)}).sort_values(
        "Contrib. riesgo %", ascending=False).reset_index(drop=True)


def captura(rp, bench="SPY", period="10y"):
    """Captura alcista/bajista: cuánto sigues al índice cuando sube y cuando baja."""
    b = _precios([bench], period)
    if b.empty:
        return {}
    rb = b[bench].pct_change().dropna()
    j = pd.concat([pd.Series(rp).rename("p"), rb.rename("b")], axis=1).dropna()
    if len(j) < 30:
        return {}
    up, dn = j[j["b"] > 0], j[j["b"] < 0]
    cu = float(up["p"].mean() / up["b"].mean()) if len(up) and up["b"].mean() != 0 else np.nan
    cd = float(dn["p"].mean() / dn["b"].mean()) if len(dn) and dn["b"].mean() != 0 else np.nan
    beta = float(np.polyfit(j["b"], j["p"], 1)[0])
    return {"captura_alcista": round(cu * 100, 0), "captura_bajista": round(cd * 100, 0),
            "beta": round(beta, 2), "corr": round(float(j["p"].corr(j["b"])), 2)}


def prueba_crisis(pesos, rebalanceo="mensual"):
    """Cómo se comportó la cartera en cada crisis histórica."""
    filas = []
    for nombre, (ini, fin) in CRISIS.items():
        # el crash del COVID duró 24 sesiones: no se puede exigir 30
        curva, rp, usados = serie_cartera(pesos, start=ini, end=fin, rebalanceo=rebalanceo,
                                          min_sesiones=10)
        if curva is None:
            filas.append({"Crisis": nombre, "Rentabilidad %": "sin datos",
                          "Peor caída %": "—", "Activos con datos": 0})
            continue
        dd = float((curva / curva.cummax() - 1).min())
        filas.append({"Crisis": nombre,
                      "Rentabilidad %": round(float(curva.iloc[-1] - 1) * 100, 1),
                      "Peor caída %": round(dd * 100, 1),
                      "Activos con datos": f"{len(usados)}/{len(pesos)}"})
    return pd.DataFrame(filas)


def comparar_legendarias(period="15y", rebalanceo="mensual", extra=None):
    """Tabla de métricas de las carteras de leyenda (+ la tuya si la pasas)."""
    filas, curvas = [], {}
    fuentes = dict(LEGENDARIAS)
    if extra:
        fuentes = {"⭐ TU CARTERA": extra, **fuentes}
    for nombre, pesos in fuentes.items():
        pesos_n = {k: v / sum(pesos.values()) for k, v in pesos.items()}
        curva, rp, usados = serie_cartera(pesos_n, period=period, rebalanceo=rebalanceo)
        if curva is None:
            continue
        m = metricas(rp, curva)
        m = {"Cartera": nombre, **m}
        m["Activos"] = f"{len(usados)}/{len(pesos)}"
        filas.append(m)
        curvas[nombre] = curva
    return pd.DataFrame(filas), curvas


# --- explicaciones en cristiano ---------------------------------------------
LEYENDA_METRICAS = """**Cómo leer la tabla** (cada columna, en cristiano):

| Columna | Qué significa | Qué es "bueno" |
|---|---|---|
| **CAGR %** | Lo que habrías ganado de media cada año | Más alto |
| **Vol anual %** | Cuánto se mueve la cartera (los sustos) | Más bajo |
| **Sharpe** | Rentabilidad por cada unidad de riesgo | >1 bien · >1.5 muy bien |
| **Sortino** | Igual que el Sharpe, pero solo castiga las CAÍDAS (subir mucho no es "riesgo") | Suele ser mayor que el Sharpe |
| **Omega** | Cuántos euros ganas por cada euro que pierdes | >1 gana más de lo que pierde |
| **Ulcer** | El "dolor": mide profundidad **y** duración de las caídas | Más bajo = se sufre menos |
| **Calmar** | Rentabilidad anual dividida por la peor caída | >0.5 decente |
| **Máx caída %** | Lo peor que llegaste a perder desde un máximo | Más cerca de 0 |
| **Días en recuperar** | Cuánto tardó en volver a máximos tras esa caída | Menos = más llevadero |
"""


def explicar_metricas(fila):
    """Interpreta la fila de métricas de UNA cartera, en lenguaje llano."""
    L = []
    sh, so = fila.get("Sharpe"), fila.get("Sortino")
    if sh is not None:
        cal = ("excelente" if sh >= 1.5 else "buena" if sh >= 1.0 else
               "aceptable" if sh >= 0.5 else "floja")
        L.append(f"- **Sharpe {sh}** — relación rentabilidad/riesgo **{cal}**.")
    if so is not None and sh:
        if so > sh * 1.15:
            L.append(f"- **Sortino {so} > Sharpe {sh}** — buena señal: sus movimientos fuertes son "
                     f"sobre todo **al alza**; la volatilidad que sufre es menos dolorosa de lo que parece.")
        else:
            L.append(f"- **Sortino {so}** — parecido al Sharpe: la volatilidad se reparte entre subidas y bajadas.")
    om = fila.get("Omega")
    if om:
        L.append(f"- **Omega {om}** — por cada euro perdido, gana **{om:.2f} €**.")
    dd, rec = fila.get("Máx caída %"), fila.get("Días en recuperar")
    if dd is not None:
        aguante = ("suave" if abs(dd) < 15 else "notable" if abs(dd) < 30 else "muy dura")
        extra = (f" y tardó **{rec} sesiones (~{int(rec/21)} meses)** en recuperarse"
                 if isinstance(rec, (int, float)) and rec > 0 else "")
        L.append(f"- **Peor caída {dd}%** — una racha {aguante}{extra}. "
                 f"Pregúntate: ¿habrías aguantado sin vender?")
    ul = fila.get("Ulcer")
    if ul is not None:
        L.append(f"- **Ulcer {ul}** — {'poco' if ul < 5 else 'bastante'} 'dolor' acumulado "
                 f"(mide caídas profundas Y largas, no solo la peor).")
    return "\n".join(L)


def explicar_contribucion(tabla):
    """Explica la descomposición del riesgo."""
    if tabla is None or tabla.empty:
        return ""
    top = tabla.iloc[0]
    peso, riesgo = float(top["Peso %"]), float(top["Contrib. riesgo %"])
    aviso = ""
    if riesgo > peso * 1.4:
        aviso = (f" **Ojo**: pesa el {peso:.0f}% pero aporta el {riesgo:.0f}% del riesgo — "
                 f"tu cartera depende de este activo mucho más de lo que parece por el peso.")
    return (f"**Quién pone el riesgo de verdad:** *{top['Activo']}* aporta el **{riesgo:.0f}%** "
            f"del riesgo total.{aviso}\n\n"
            "> El peso NO es el riesgo: un 10% en algo muy volátil puede ser el 40% de tus sustos. "
            "Por eso los fondos reparten por *riesgo*, no por dinero.")


def explicar_captura(cap, bench="SPY"):
    """Explica captura alcista/bajista y beta."""
    if not cap:
        return ""
    cu, cd, b = cap["captura_alcista"], cap["captura_bajista"], cap["beta"]
    if cu > cd + 5:
        lectura = "✅ **Buen perfil**: captura más subida que bajada."
    elif cd > cu + 5:
        lectura = "⚠️ **Perfil malo**: sufre más las bajadas de lo que aprovecha las subidas."
    else:
        lectura = "➖ Perfil simétrico: sube y baja en la misma proporción que el índice."
    return (f"**Frente a {bench}:** cuando el índice sube, tú captas el **{cu:.0f}%**; "
            f"cuando baja, sufres el **{cd:.0f}%** (beta {b}). {lectura}\n\n"
            "> Ideal: captura alcista alta y bajista baja. Una cartera defensiva capta menos de "
            "ambas: gana menos en bonanza pero duerme mejor en las crisis.")


def explicar_crisis(tabla):
    """Explica el comportamiento en crisis."""
    if tabla is None or tabla.empty:
        return ""
    vals = pd.to_numeric(tabla["Rentabilidad %"], errors="coerce").dropna()
    if vals.empty:
        return "> Sin datos suficientes en los periodos de crisis."
    peor = tabla.loc[pd.to_numeric(tabla["Rentabilidad %"], errors="coerce").idxmin()]
    return (f"**La peor fue {peor['Crisis']}**: {peor['Rentabilidad %']}% "
            f"(llegó a caer {peor['Peor caída %']}%).\n\n"
            "> Esto es lo que de verdad importa: no cuánto ganas en los años buenos, sino si "
            "aguantas los malos sin vender en el peor momento. Si estos números te quitarían el "
            "sueño, la cartera es demasiado agresiva para ti.")


def _plot(curvas, titulo="Carteras de leyenda"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for nombre, c in curvas.items():
        ancho = 2.4 if nombre.startswith("⭐") else 1.2
        ax.plot(c.index, c.values, lw=ancho, label=nombre)
    ax.set_yscale("log")
    ax.set_title(titulo + " · crecimiento de 1 € (escala log, dividendos incluidos)")
    ax.set_xlabel("Fecha"); ax.set_ylabel("Valor de 1 €")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description="Laboratorio de carteras: métricas, leyendas y crisis.")
    ap.add_argument("--cartera", help='Pesos: "SPY:60, AGG:40"')
    ap.add_argument("--legendarias", action="store_true", help="Compara solo las carteras clásicas.")
    ap.add_argument("--period", default="15y")
    ap.add_argument("--rebalanceo", choices=["mensual", "trimestral", "anual", "ninguno"], default="mensual")
    ap.add_argument("--bench", default="SPY")
    a = ap.parse_args()

    mia = parsear_cartera(a.cartera) if a.cartera else None
    print(f"\nAnalizando ({a.period}, rebalanceo {a.rebalanceo})...")
    tabla, curvas = comparar_legendarias(a.period, a.rebalanceo, mia)
    cols = ["Cartera", "CAGR %", "Vol anual %", "Sharpe", "Sortino", "Omega", "Ulcer",
            "Calmar", "Máx caída %", "Días en recuperar"]
    print("\n=== Métricas comparadas ===")
    print(tabla[cols].to_string(index=False))
    if mia:
        print("\n=== Tu cartera en las crisis ===")
        print(prueba_crisis(mia, a.rebalanceo).to_string(index=False))
        print("\n=== Contribución al riesgo ===")
        print(contribucion_riesgo(mia).to_string(index=False))
        _c, rp, _u = serie_cartera(mia, period=a.period, rebalanceo=a.rebalanceo)
        cap = captura(rp, a.bench, a.period)
        if cap:
            print(f"\n=== Frente a {a.bench} ===")
            print(f"  Captura alcista {cap['captura_alcista']:.0f}% · bajista {cap['captura_bajista']:.0f}% "
                  f"· beta {cap['beta']} · correlación {cap['corr']}")
    print("\n> Sortino castiga solo las caídas · Omega = ganancias/pérdidas · Ulcer mide el dolor")
    print("> (profundidad y duración) · Calmar = rentabilidad por unidad de peor caída.")
    print("> Todo es medición histórica, no predicción. No es recomendación de inversión.\n")


if __name__ == "__main__":
    main()
