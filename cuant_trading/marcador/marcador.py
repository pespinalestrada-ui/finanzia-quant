"""
marcador — ¿lo que dijo el sistema se cumplió?

Todo lo demás que mide este proyecto es RETROSPECTIVO: se calcula sobre el pasado
y arrastra el riesgo de sobreajuste. Este módulo hace la única medición que no se
puede sobreajustar: **apunta la llamada antes de conocer el resultado** y la
comprueba después.

Cómo se puntúa (tres marcadores, definición operativa, sin interpretación)
--------------------------------------------------------------------------
1. **Dirección**: COMPRAR y subió / VENDER y bajó. Se compara contra la TASA BASE
   del propio activo: qué % de todas las ventanas de ese plazo fueron al alza.
   Acertar el 54% en algo que sube el 54% de las veces es habilidad CERO.

2. **Exceso sobre la deriva**: media de `signo(señal)·(r_llamada − r_medio)`.
   Mide si el MOMENTO elegido aporta algo por encima de estar siempre dentro.
   No es "vs quedarse quieto" literal: para un COMPRAR, comprar y mantener ese
   mismo activo es exactamente lo mismo que no hacer nada, y la comparación se
   anularía sola sin medir nada.

3. **Exceso sobre el mercado**: lo mismo contra el SPY.

Los MANTENER se apuntan pero NO puntúan en dirección: no afirman nada. Sí cuentan
para saber cuántas veces el sistema se mojó de verdad.

Honestidad
----------
- Con n < 30 no se da porcentaje: se dice "muestra insuficiente". Un 70% sobre 7
  llamadas es ruido.
- Los plazos de 30 y 90 días solapan entre llamadas cercanas: el estadístico usa
  **n efectivo = n / (plazo/21)**, la misma corrección que `cpcv` y
  `veredicto_tune`.
- Lo esperable, por todo lo ya medido en este proyecto, es que NO haya ventaja.
  Ese resultado es información, no una avería.

Uso:
    python marcador.py --reconstruir "AAPL, MSFT, KO"
    python marcador.py --resolver
    python marcador.py
"""
import argparse
import sys
from datetime import datetime, timedelta
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

# paleta Nocturne (la misma de cuant_trading/dashboard/finanzia_charts.py)
_NEU, _ACC, _ACC2 = "#9aa0b5", "#8271e0", "#a99bf0"
_UP, _DOWN, _GOLD, _DIM = "#3fa87c", "#d4615c", "#ab7f28", "#6f7486"

_SUITE = Path(__file__).resolve().parents[1]
CSV = _SUITE.parent / "data" / "llamadas.csv"
PLAZOS = (7, 30, 90)
MIN_N = 30                 # por debajo, no se da porcentaje
PASO_INDEP = 21            # sesiones que se consideran una observación nueva

COLS = ["id", "fecha", "fuente", "origen", "ticker", "senal", "score", "confianza",
        "precio_0", "var_esperada_pct",
        "precio_7", "precio_30", "precio_90",
        "spy_0", "spy_7", "spy_30", "spy_90",
        "resuelto_7", "resuelto_30", "resuelto_90"]

_PX = {}                   # caché de series de precio dentro del proceso


# --- almacén ----------------------------------------------------------------
def _cargar():
    if not CSV.exists():
        return pd.DataFrame(columns=COLS)
    try:
        df = pd.read_csv(CSV)
    except Exception:
        # un CSV corrupto no puede tumbar el panel: se aparta y se empieza limpio
        try:
            CSV.rename(CSV.with_suffix(".csv.bak"))
        except Exception:
            pass
        return pd.DataFrame(columns=COLS)
    for c in COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[COLS]


def _guardar(df):
    CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV, index=False)


def _serie(ticker, period="6y"):
    """Cierre ajustado, cacheado en proceso. Devuelve None si no hay datos."""
    tk = ticker.strip().upper()
    if tk in _PX:
        return _PX[tk]
    try:
        h = yf.Ticker(tk).history(period=period, auto_adjust=True)["Close"]
        c = h.astype(float).dropna()
        c.index = pd.to_datetime(c.index).tz_localize(None).normalize()
        _PX[tk] = c if len(c) else None
    except Exception:
        _PX[tk] = None
    return _PX[tk]


# --- registro ---------------------------------------------------------------
def registrar(ticker, senal, score=None, confianza=None, precio_0=None,
              fuente="veredicto", origen="vivo", var_esperada=None, fecha=None):
    """Apunta una llamada. Devuelve True si se escribió, False si ya existía.

    Deduplica por (ticker, fuente, día): abrir AAPL cinco veces mientras trasteas
    anota UNA. Sin esto, el ticker que más miras domina el marcador."""
    tk = str(ticker).strip().upper()
    f = pd.Timestamp(fecha or datetime.now()).normalize()
    df = _cargar()
    if len(df):
        dup = ((df["ticker"] == tk) & (df["fuente"] == fuente) &
               (pd.to_datetime(df["fecha"], errors="coerce").dt.normalize() == f))
        if dup.any():
            return False
    if precio_0 is None:
        s = _serie(tk)
        precio_0 = float(s.iloc[-1]) if s is not None and len(s) else np.nan
    spy = _serie("SPY")
    spy0 = _precio_en(spy, f)
    fila = {c: np.nan for c in COLS}
    fila.update({"id": f"{tk}-{fuente}-{f.date()}", "fecha": f.date().isoformat(),
                 "fuente": fuente, "origen": origen, "ticker": tk,
                 "senal": str(senal).upper(), "score": score, "confianza": confianza,
                 "precio_0": precio_0, "var_esperada_pct": var_esperada,
                 "spy_0": spy0})
    _guardar(pd.concat([df, pd.DataFrame([fila])], ignore_index=True))
    return True


def _precio_en(serie, fecha, tolerancia=6):
    """Precio del primer día HÁBIL en o después de `fecha`. NaN si aún no existe."""
    if serie is None or not len(serie):
        return np.nan
    f = pd.Timestamp(fecha).normalize()
    if f > serie.index[-1]:
        return np.nan
    idx = serie.index.searchsorted(f)
    if idx >= len(serie):
        return np.nan
    if (serie.index[idx] - f).days > tolerancia:
        return np.nan
    return float(serie.iloc[idx])


# --- resolución -------------------------------------------------------------
def resolver(verbose=False):
    """Rellena los plazos ya vencidos. Idempotente: lo ya resuelto no se toca."""
    df = _cargar()
    if df.empty:
        return 0
    hoy = pd.Timestamp(datetime.now()).normalize()
    n = 0
    for tk in df["ticker"].dropna().unique():
        s = _serie(tk)
        if s is None:
            continue
        m = df["ticker"] == tk
        for h in PLAZOS:
            col, rcol = f"precio_{h}", f"resuelto_{h}"
            pend = m & df[col].isna()
            if not pend.any():
                continue
            for i in df.index[pend]:
                objetivo = pd.Timestamp(df.at[i, "fecha"]).normalize() + timedelta(days=h)
                if objetivo > hoy:
                    continue                       # aún no ha vencido
                p = _precio_en(s, objetivo)
                if np.isfinite(p):
                    df.at[i, col] = p
                    df.at[i, rcol] = hoy.date().isoformat()
                    n += 1
    # el SPY, una vez para todas las filas
    spy = _serie("SPY")
    if spy is not None:
        for h in PLAZOS:
            col = f"spy_{h}"
            pend = df[col].isna() & df[f"precio_{h}"].notna()
            for i in df.index[pend]:
                objetivo = pd.Timestamp(df.at[i, "fecha"]).normalize() + timedelta(days=h)
                df.at[i, col] = _precio_en(spy, objetivo)
        pend0 = df["spy_0"].isna()
        for i in df.index[pend0]:
            df.at[i, "spy_0"] = _precio_en(spy, pd.Timestamp(df.at[i, "fecha"]))
    if n:
        _guardar(df)
    if verbose:
        print(f"resueltos {n} plazos")
    return n


# --- referencia: qué hace ese activo por sí solo -----------------------------
def _base(ticker, h):
    """(prob. de subir, retorno medio) de TODAS las ventanas de `h` días.

    Es el listón. Si el sistema acierta lo mismo que esto, no aporta nada."""
    s = _serie(ticker)
    if s is None or len(s) < h + 40:
        return np.nan, np.nan
    r = (s.shift(-h) / s - 1.0).dropna()
    if not len(r):
        return np.nan, np.nan
    return float((r > 0).mean()), float(r.mean())


def _n_efectivo(n, h):
    """Las ventanas de 30 y 90 días de llamadas cercanas comparten datos: 500
    observaciones solapadas no son 500 independientes."""
    return max(1.0, n / max(1.0, h / PASO_INDEP))


def _veredicto(suficiente, n, ventaja_pp, exceso, p_dir, p_der):
    """Etiqueta con SIGNO. Un resultado significativo puede ser malo: si el
    sistema acierta menos que el propio activo, hay que decirlo con esas
    palabras, no llamarlo 'ventaja'."""
    if not suficiente:
        return f"muestra insuficiente ({n}/{MIN_N})"
    sig_dir = np.isfinite(p_dir) and p_dir < 0.05
    sig_der = np.isfinite(p_der) and p_der < 0.05
    if not (sig_dir or sig_der):
        return "sin diferencia"
    # manda el signo: los dos criterios suelen ir juntos, y si no, decide la
    # direccion porque es la que se lee en la tabla
    peor = (ventaja_pp < 0) if sig_dir else (exceso < 0)
    return "PEOR que el activo solo" if peor else "ventaja significativa"


def marcador(origen="vivo", fuente=None, df=None):
    """Los tres marcadores por plazo. Devuelve dict con tabla y resumen."""
    from scipy.stats import binomtest, t as t_dist
    d = _cargar() if df is None else df.copy()
    if len(d):
        d = d[d["origen"] == origen]
        if fuente:
            d = d[d["fuente"] == fuente]
    if d.empty:
        return {"n_total": 0, "filas": [], "mensaje": "Todavía no hay llamadas apuntadas."}

    filas = []
    for h in PLAZOS:
        sub = d[d[f"precio_{h}"].notna() & d["precio_0"].notna()].copy()
        n_venc = len(sub)
        fila = {"Plazo": f"{h} días", "Vencidas": n_venc}
        direccional = sub[sub["senal"].isin(["COMPRAR", "VENDER"])].copy()
        if n_venc == 0 or direccional.empty:
            fila.update({"Acierto %": None, "Tasa base %": None, "Ventaja pp": None,
                         "Exceso deriva %": None, "Exceso SPY %": None,
                         "Veredicto": "sin vencer"})
            filas.append(fila)
            continue

        direccional["r"] = direccional[f"precio_{h}"] / direccional["precio_0"] - 1.0
        direccional["sg"] = np.where(direccional["senal"] == "COMPRAR", 1.0, -1.0)
        base = [_base(tk, h) for tk in direccional["ticker"]]
        direccional["p_base"] = [b[0] for b in base]
        direccional["r_base"] = [b[1] for b in base]

        acierto = float((direccional["sg"] * direccional["r"] > 0).mean())
        p0 = float(np.nanmean(direccional["p_base"]))
        # para un VENDER, el listón es la probabilidad de BAJAR
        p0_ajust = float(np.nanmean(np.where(direccional["sg"] > 0,
                                             direccional["p_base"],
                                             1 - direccional["p_base"])))
        n = len(direccional)
        n_ef = _n_efectivo(n, h)

        exc_der = direccional["sg"] * (direccional["r"] - direccional["r_base"])
        con_spy = direccional[direccional[f"spy_{h}"].notna() & direccional["spy_0"].notna()]
        if len(con_spy):
            r_spy = con_spy[f"spy_{h}"] / con_spy["spy_0"] - 1.0
            sg_spy = np.where(con_spy["senal"] == "COMPRAR", 1.0, -1.0)
            r_c = con_spy[f"precio_{h}"] / con_spy["precio_0"] - 1.0
            exc_spy = pd.Series(sg_spy * (r_c - r_spy))
        else:
            exc_spy = pd.Series(dtype=float)

        def _t(serie):
            s2 = serie.dropna()
            if len(s2) < 3 or s2.std(ddof=1) == 0:
                return np.nan, np.nan
            ne = _n_efectivo(len(s2), h)
            tt = float(s2.mean() / (s2.std(ddof=1) / np.sqrt(ne)))
            p = float(2 * (1 - t_dist.cdf(abs(tt), df=max(1, ne - 1))))
            return tt, p

        t_der, p_der = _t(exc_der)
        t_spy, p_spy = _t(exc_spy)
        # Test de DOS COLAS: el marcador tiene que poder decir tambien que el
        # sistema es PEOR que el activo solo. Con una cola ("greater"), un
        # resultado significativamente malo salia con p~1 y se anunciaba como
        # bueno. Paso de verdad en la primera reconstruccion: -16 pp de ventaja
        # etiquetados como "ventaja significativa".
        if n_ef >= 5 and np.isfinite(p0_ajust):
            exitos = int(round(acierto * n_ef))
            p_dir = float(binomtest(min(exitos, int(n_ef)), int(n_ef),
                                    min(max(p0_ajust, 1e-6), 1 - 1e-6),
                                    alternative="two-sided").pvalue)
        else:
            p_dir = np.nan

        suficiente = n >= MIN_N
        fila.update({
            "Acierto %": round(acierto * 100, 1) if suficiente else None,
            "Tasa base %": round(p0_ajust * 100, 1) if suficiente else None,
            "Ventaja pp": round((acierto - p0_ajust) * 100, 1) if suficiente else None,
            "Exceso deriva %": round(float(exc_der.mean()) * 100, 2) if suficiente else None,
            "Exceso SPY %": round(float(exc_spy.mean()) * 100, 2) if (suficiente and len(exc_spy)) else None,
            "Veredicto": _veredicto(suficiente, n, acierto - p0_ajust,
                                    float(exc_der.mean()), p_dir, p_der),
            "_n": n, "_n_ef": round(n_ef, 1), "_p_dir": p_dir, "_p_der": p_der,
            "_p_spy": p_spy, "_exc_der": exc_der, "_p0": p0,
        })
        filas.append(fila)

    total = len(d)
    mojadas = int((d["senal"].isin(["COMPRAR", "VENDER"])).sum())
    return {"n_total": total, "n_mojadas": mojadas, "filas": filas,
            "pendientes": int(d[f"precio_{PLAZOS[-1]}"].isna().sum()),
            "desde": str(pd.to_datetime(d["fecha"], errors="coerce").min().date())
                     if total else None}


# --- reconstrucción histórica ------------------------------------------------
def reconstruir(tickers, period="3y", paso=PASO_INDEP, borrar=True):
    """Genera llamadas del NÚCLEO TÉCNICO en fechas pasadas, sin mirar al futuro.

    OJO: no reconstruye el Veredicto completo. `score_historico` solo cubre los
    pilares que se calculan con precio (tendencia, ADX, osciladores, MACD,
    momentum, OBV). Quedan fuera el forecast de Prophet (habría que reentrenarlo
    en cada fecha) y los factores/ROIC (yfinance solo da los fundamentales de HOY:
    usarlos para reconstruir 2023 sería mirar al futuro). Por eso se etiqueta
    'reconstruido' y NUNCA se suma al marcador en vivo.

    `paso`: una llamada cada N sesiones. A diario, dos llamadas consecutivas a 90
    días compartirían 89 días de datos."""
    sys.path.insert(0, str(_SUITE / "veredicto_backtest"))
    sys.path.insert(0, str(_SUITE / "indicators"))
    import veredicto_backtest as VB

    df = _cargar()
    if borrar and len(df):
        df = df[df["origen"] != "reconstruido"]
        _guardar(df)

    n = 0
    for tk in tickers:
        tk = tk.strip().upper()
        try:
            h = VB.descargar(tk, period)
            if h is None or len(h) < 300:
                continue
            s = VB.score_historico(h)
        except Exception:
            continue
        c = h["Close"].astype(float)
        idx = s.dropna().index
        for fecha in idx[::paso]:
            sc = float(s.loc[fecha])
            senal = "COMPRAR" if sc >= 0.35 else "VENDER" if sc <= -0.35 else "MANTENER"
            registrar(tk, senal, score=round(sc, 4), precio_0=float(c.loc[fecha]),
                      fuente="veredicto", origen="reconstruido", fecha=fecha)
            n += 1
    resolver()
    return n


# --- salida -----------------------------------------------------------------
def tabla(res):
    if not res.get("filas"):
        return pd.DataFrame()
    cols = ["Plazo", "Vencidas", "Acierto %", "Tasa base %", "Ventaja pp",
            "Exceso deriva %", "Exceso SPY %", "Veredicto"]
    return pd.DataFrame([{k: f.get(k) for k in cols} for f in res["filas"]])


def explicar(res, etiqueta="En vivo"):
    if not res.get("n_total"):
        return (f"### {etiqueta}\nTodavía no hay llamadas apuntadas. Sal un "
                f"★ Veredicto o un Forecast y quedará registrado aquí.")
    L = [f"### {etiqueta} · {res['n_total']} llamadas desde {res['desde']}",
         f"- Se mojó en **{res['n_mojadas']}** de {res['n_total']} "
         f"({res['n_mojadas']/res['n_total']*100:.0f}%); el resto fueron MANTENER."]
    hay = False
    for f in res["filas"]:
        if f.get("Acierto %") is None:
            L.append(f"- **{f['Plazo']}**: {f['Veredicto']} "
                     f"({f['Vencidas']} vencidas).")
            continue
        hay = True
        vent = f["Ventaja pp"]
        L.append(
            f"- **{f['Plazo']}**: acierta el **{f['Acierto %']}%** frente a una "
            f"tasa base del **{f['Tasa base %']}%** → ventaja **{vent:+.1f} pp**. "
            f"Exceso sobre la deriva **{f['Exceso deriva %']:+.2f}%**. "
            f"n={f['_n']} (efectivo {f['_n_ef']}). **{f['Veredicto']}**.")
    if hay:
        L.append("\n> La **tasa base** es lo que hace ese activo por sí solo. Acertar el "
                 "54% en algo que sube el 54% de las veces es habilidad **cero**: lo que "
                 "cuenta es la ventaja, no el acierto.")
        # Aviso de régimen: sin esto, el marcador se lee como "el sistema es malo"
        # cuando parte de la explicación es "el periodo fue excepcionalmente alcista".
        bases = [x["Tasa base %"] for x in res["filas"] if x.get("Tasa base %") is not None]
        if bases and max(bases) >= 62:
            L.append(f"> ⚠️ **Ojo al periodo**: la tasa base llega al {max(bases):.0f}% a plazo largo, o sea que el mercado subió casi siempre. En un tramo así, CUALQUIER sistema que se quede fuera o se ponga corto pierde contra estar dentro. Parte de la desventaja es del RÉGIMEN, no del sistema: hay que volver a mirarlo tras un mercado bajista antes de dar el veredicto por definitivo.")
    L.append("> Los plazos largos **solapan** entre llamadas cercanas, así que el "
             "estadístico usa muestra efectiva, no el número bruto.")
    return "\n".join(L)


def _plot(res, etiqueta="En vivo"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from finanzia_charts import style, msg_fig
    except Exception:
        style = msg_fig = None
    filas = [f for f in res.get("filas", []) if f.get("Acierto %") is not None]
    if not filas:
        if msg_fig:
            return msg_fig("Aún no hay plazos vencidos con muestra suficiente.\n"
                           "El marcador se llena solo según pasen los días.")
        f, a = plt.subplots(figsize=(8, 2)); a.axis("off")
        a.text(.5, .5, "sin datos", ha="center"); return f

    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(filas))
    acc = [f["Acierto %"] for f in filas]
    base = [f["Tasa base %"] for f in filas]
    ax.bar(x - 0.19, base, 0.38, color=_NEU, label="Tasa base (el activo solo)")
    ax.bar(x + 0.19, acc, 0.38, color=_ACC, label="Acierto del sistema")
    for i, f in enumerate(filas):
        v = f["Ventaja pp"]
        col = _UP if (f["Veredicto"] == "ventaja significativa") else _DIM
        ax.annotate(f"{v:+.1f} pp", xy=(i, max(acc[i], base[i])), xytext=(0, 7),
                    textcoords="offset points", ha="center", color=col, fontsize=10.5,
                    fontweight="semibold")
    ax.axhline(50, color=_DIM, ls=":", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([f["Plazo"] for f in filas])
    if style:
        style(ax, titulo=f"{etiqueta} — acierto frente a lo que hace el activo solo",
              kicker="SI LAS DOS BARRAS SON IGUALES, EL SISTEMA NO APORTA",
              ylabel="% de acierto")
    else:
        ax.legend()
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description="Marcador: ¿se cumplió lo que dijo?")
    ap.add_argument("--reconstruir", help="Tickers separados por comas.")
    ap.add_argument("--period", default="3y")
    ap.add_argument("--resolver", action="store_true")
    ap.add_argument("--origen", default="vivo", choices=["vivo", "reconstruido"])
    a = ap.parse_args()
    if a.reconstruir:
        tks = [t.strip() for t in a.reconstruir.replace(",", " ").split() if t.strip()]
        print(f"reconstruidas {reconstruir(tks, a.period)} llamadas")
    if a.resolver:
        resolver(verbose=True)
    res = marcador(origen=a.origen)
    print(explicar(res, "Reconstruido (núcleo técnico)" if a.origen == "reconstruido" else "En vivo"))
    t = tabla(res)
    if not t.empty:
        print()
        print(t.to_string(index=False))


if __name__ == "__main__":
    main()
