"""
kpis — los cuatro KPI de rentabilidad con los que se juzga un negocio.

  ROE  = Beneficio neto / Fondos propios            ¿cuánto renta el dinero del dueño?
  ROA  = Beneficio neto / Activo total              ¿cuánto renta TODO lo que maneja?
  BPA  = Beneficio neto / Nº de acciones            ¿cuánto beneficio toca por acción?
  ROIC = NOPAT / Capital invertido                  ¿cuánto renta el capital que trabaja?

Por qué los cuatro y no solo el ROE
-----------------------------------
El ROE se puede inflar con deuda sin mejorar el negocio ni un céntimo: si pides
prestado y recompras acciones, los fondos propios bajan y el ROE sube. Ejemplo
real (jul 2026): Apple tiene ROE 149% y ROA 27%. Esa brecha de 122 puntos no es
excelencia operativa, es apalancamiento. El ROA y el ROIC no se dejan inflar
así, y por eso hay que mirarlos juntos.

El ROIC es el único que yfinance no da hecho. Se calcula:

    NOPAT = EBIT x (1 - tipo impositivo)
    ROIC  = NOPAT / Capital invertido        (deuda + fondos propios - caja)

yfinance publica `EBIT`, `Tax Rate For Calcs` e `Invested Capital` en los estados
financieros anuales, así que sale de la fuente, no de una estimación.

Limitaciones que hay que decir en voz alta
------------------------------------------
1. **Bancos y aseguradoras**: no tienen EBIT ni capital invertido en el sentido
   industrial. El ROIC NO aplica y un ROA del 0,8% es NORMAL en banca (Santander),
   no un desastre. Comparar un banco con una tecnológica por ROA es un error.
2. Son ratios **contables y retrospectivos**: dicen cómo fue el negocio, no cómo
   irá la acción. Ninguno de los cuatro predice el retorno futuro.
3. Un año suelto no dice nada. Lo que informa es el **nivel sostenido**: por eso
   este módulo saca la serie de 3-4 años, no solo el último dato.

Uso:
    python kpis.py AAPL
    python kpis.py --comparar "AAPL, MSFT, KO, SAN.MC"
"""
import argparse
import sys

import numpy as np
import pandas as pd
import yfinance as yf

# paleta Nocturne (la misma de cuant_trading/dashboard/finanzia_charts.py):
# sobre fondo oscuro el negro y el azul puro no se ven
_NEU, _ACC, _ACC2 = "#b2b6ca", "#9184d9", "#b5abfc"
_UP, _DOWN, _GOLD, _DIM = "#63b58e", "#d9736b", "#c9b273", "#75798c"

# sectores donde el ROIC industrial no significa nada
_FINANCIERO = ("bank", "banco", "insurance", "capital markets", "financial",
               "asset management", "credit", "mortgage")


def _fila(bs_o_inc, *nombres):
    """Primera fila del estado financiero que case con alguno de los nombres."""
    for n in nombres:
        if n in bs_o_inc.index:
            return bs_o_inc.loc[n]
    return None


def _es_financiera(info):
    txt = f"{info.get('sector', '')} {info.get('industry', '')}".lower()
    return any(p in txt for p in _FINANCIERO)


def kpis(ticker):
    """Los 4 KPI del último ejercicio + el contexto para interpretarlos."""
    tk = ticker.strip().upper()
    t = yf.Ticker(tk)
    info = t.info or {}
    if not info.get("shortName") and not info.get("longName"):
        raise ValueError(f"Ticker '{tk}' sin datos.")

    fin = _es_financiera(info)
    d = {
        "ticker": tk,
        "nombre": info.get("shortName") or info.get("longName") or tk,
        "sector": info.get("sector") or "n/d",
        "financiera": fin,
        # los tres directos
        "roe": info.get("returnOnEquity"),
        "roa": info.get("returnOnAssets"),
        "bpa": info.get("trailingEps"),
        "bpa_fwd": info.get("forwardEps"),
        # contexto imprescindible para no malinterpretar el ROE
        "deuda_equity": info.get("debtToEquity"),
        "margen_neto": info.get("profitMargins"),
        "per": info.get("trailingPE"),
        "roic": None,
        "roic_nota": "",
    }

    # --- ROIC: el único que hay que calcular --------------------------------
    if fin:
        d["roic_nota"] = ("No aplica: en un banco no existe el 'capital invertido' "
                          "industrial. Mírale el ROE y el ROA de banca.")
    else:
        try:
            inc, bs = t.income_stmt, t.balance_sheet
            ebit = _fila(inc, "EBIT", "Operating Income")
            tasa = _fila(inc, "Tax Rate For Calcs")
            cap = _fila(bs, "Invested Capital")
            if ebit is None or cap is None:
                d["roic_nota"] = "Sin EBIT o capital invertido publicados."
            else:
                e = float(ebit.iloc[0])
                tt = float(tasa.iloc[0]) if tasa is not None and not pd.isna(tasa.iloc[0]) else 0.21
                c = float(cap.iloc[0])
                if c > 0:
                    d["roic"] = (e * (1 - tt)) / c
                    d["roic_nota"] = f"NOPAT = EBIT x (1 − {tt*100:.0f}% impuestos)."
        except Exception as e:
            d["roic_nota"] = f"No calculable ({str(e)[:60]})."

    # --- la brecha ROE−ROA: cuánto del ROE es apalancamiento ----------------
    if d["roe"] is not None and d["roa"] is not None:
        d["brecha"] = d["roe"] - d["roa"]
        d["multiplicador"] = (d["roe"] / d["roa"]) if d["roa"] not in (None, 0) else None
    else:
        d["brecha"] = d["multiplicador"] = None
    return d


def historico(ticker, anios=4):
    """ROE / ROA / ROIC año a año. Un dato suelto no dice nada; el nivel
    sostenido sí: eso es lo que separa un buen negocio de un buen año."""
    tk = ticker.strip().upper()
    t = yf.Ticker(tk)
    inc, bs = t.income_stmt, t.balance_sheet
    if inc is None or inc.empty or bs is None or bs.empty:
        return pd.DataFrame()

    neto = _fila(inc, "Net Income Common Stockholders", "Net Income",
                 "Net Income From Continuing Operation Net Minority Interest")
    ebit = _fila(inc, "EBIT", "Operating Income")
    tasa = _fila(inc, "Tax Rate For Calcs")
    eq = _fila(bs, "Stockholders Equity", "Total Equity Gross Minority Interest")
    act = _fila(bs, "Total Assets")
    cap = _fila(bs, "Invested Capital")

    filas = []
    for col in list(inc.columns)[:anios]:
        f = {"Ejercicio": str(pd.Timestamp(col).date())}

        def _v(serie, columna=col):
            if serie is None or columna not in serie.index:
                return np.nan
            v = serie.get(columna)
            return float(v) if v is not None and not pd.isna(v) else np.nan

        n, e_, a_, c_ = _v(neto), _v(eq), _v(act), _v(cap)
        f["ROE %"] = round(n / e_ * 100, 1) if e_ and not np.isnan(n) and e_ != 0 else np.nan
        f["ROA %"] = round(n / a_ * 100, 1) if a_ and not np.isnan(n) and a_ != 0 else np.nan
        eb, tt = _v(ebit), _v(tasa)
        tt = tt if not np.isnan(tt) else 0.21
        f["ROIC %"] = round(eb * (1 - tt) / c_ * 100, 1) if (c_ and not np.isnan(eb) and c_ != 0) else np.nan
        filas.append(f)
    return pd.DataFrame(filas).iloc[::-1].reset_index(drop=True)   # del más viejo al más nuevo


def comparar(tickers):
    """Tabla con los 4 KPI de varias empresas. Comparar solo tiene sentido
    DENTRO del mismo sector: un banco y una tecnológica no juegan al mismo juego."""
    filas = []
    for tk in tickers:
        try:
            d = kpis(tk)
        except Exception as e:
            filas.append({"Ticker": tk.upper(), "Nombre": f"error: {str(e)[:40]}"})
            continue
        pct = lambda v: round(v * 100, 1) if isinstance(v, (int, float)) and v is not None else np.nan
        filas.append({
            "Ticker": d["ticker"],
            "Nombre": d["nombre"][:26],
            "Sector": (d["sector"] or "n/d")[:20],
            "ROE %": pct(d["roe"]),
            "ROA %": pct(d["roa"]),
            "ROIC %": pct(d["roic"]),
            "BPA": round(d["bpa"], 2) if d["bpa"] is not None else np.nan,
            "Margen %": pct(d["margen_neto"]),
            "Deuda/Equity": round(d["deuda_equity"], 0) if d["deuda_equity"] is not None else np.nan,
            "ROE/ROA": round(d["multiplicador"], 1) if d["multiplicador"] else np.nan,
        })
    return pd.DataFrame(filas)


# --- lectura en cristiano ---------------------------------------------------

LEYENDA = """**Qué mide cada uno** (todos son *rentabilidad*: cuánto beneficio saca de lo que maneja)

- **ROE** — beneficio ÷ dinero de los accionistas. *"Por cada 100 € que han puesto los
  dueños, ¿cuánto ganan al año?"* Se infla con deuda: ojo.
- **ROA** — beneficio ÷ TODO el activo (lo propio y lo prestado). *"Por cada 100 € que
  la empresa mueve, ¿cuánto gana?"* La deuda no lo maquilla.
- **ROIC** — beneficio operativo después de impuestos ÷ capital que de verdad trabaja.
  El más honesto de los cuatro y el más difícil de manipular.
- **BPA** — beneficio ÷ nº de acciones. *"¿Cuánto beneficio toca a cada acción tuya?"*
  Es el único en euros, no en %; sube solo con recomprar acciones aunque no se gane más.

**La regla que hace útil mirarlos juntos:** `ROE ÷ ROA` es, aproximadamente, cuántas
veces está apalancada la empresa. Si sale 1-2, el ROE es negocio. Si sale 5 o más,
buena parte de ese ROE es deuda, no talento."""


def explicar(d, hist=None):
    """Lectura de los KPI de una empresa, sin jerga."""
    L = [f"### {d['nombre']} ({d['ticker']}) · {d['sector']}"]
    pc = lambda v: f"{v*100:.1f}%" if isinstance(v, (int, float)) and v is not None else "n/d"

    L.append(f"- **ROE {pc(d['roe'])}** · **ROA {pc(d['roa'])}** · "
             f"**ROIC {pc(d['roic']) if d['roic'] is not None else 'n/d'}** · "
             f"**BPA {d['bpa'] if d['bpa'] is not None else 'n/d'}**")

    # el punto clave: cuánto del ROE es apalancamiento.
    # En banca un multiplicador de 12-20x ES el modelo de negocio (prestan dinero
    # ajeno), así que marcarlo como alarma sería un falso positivo.
    m = d.get("multiplicador")
    if m and d["financiera"]:
        L.append(f"- El ROE es **{m:.1f} veces** el ROA, pero eso en un banco es **lo normal**: "
                 f"su negocio consiste precisamente en mover dinero ajeno. Aquí el "
                 f"multiplicador no es una señal de alarma, es el sector.")
    elif m:
        if m < 2:
            L.append(f"- El ROE es **{m:.1f} veces** el ROA: la empresa apenas usa deuda, "
                     f"así que ese ROE **es negocio de verdad**.")
        elif m < 4:
            L.append(f"- El ROE es **{m:.1f} veces** el ROA: usa deuda de forma normal. "
                     f"Parte del ROE viene de ahí, no todo es mérito operativo.")
        else:
            L.append(f"- ⚠️ El ROE es **{m:.1f} veces** el ROA. Esa brecha es **apalancamiento**: "
                     f"buena parte del ROE viene de deuda o de recomprar acciones, no de "
                     f"ganar más dinero. El ROE solo, aquí, te engañaría.")

    if d["roic"] is not None:
        r = d["roic"] * 100
        if r > 15:
            L.append(f"- ROIC {r:.1f}%: **alto**. El capital que trabaja renta bastante más "
                     f"de lo que suele costar financiarlo (~8-10%). Señal de negocio con ventaja.")
        elif r > 8:
            L.append(f"- ROIC {r:.1f}%: **correcto**, en la zona de lo que cuesta el capital. "
                     f"Ni crea ni destruye valor de forma clara.")
        else:
            L.append(f"- ROIC {r:.1f}%: **bajo**. Renta menos de lo que cuesta financiarse: "
                     f"cada euro que reinvierte vale menos de un euro.")
    elif d["roic_nota"]:
        L.append(f"- ROIC: {d['roic_nota']}")

    if d["financiera"]:
        L.append("- 🏦 Es una **financiera**: un ROA del 0,5-1% es NORMAL en banca (mueven "
                 "activo prestado enorme). No la compares por ROA con una industrial.")

    if hist is not None and not hist.empty and len(hist) >= 3:
        col = "ROIC %" if hist["ROIC %"].notna().sum() >= 3 else "ROE %"
        s = hist[col].dropna()
        if len(s) >= 3:
            tendencia = ("**mejorando**" if s.iloc[-1] > s.iloc[0] * 1.1 else
                         "**empeorando**" if s.iloc[-1] < s.iloc[0] * 0.9 else "**estable**")
            L.append(f"- En {len(s)} ejercicios el {col.replace(' %','')} va de "
                     f"{s.iloc[0]:.1f}% a {s.iloc[-1]:.1f}%: {tendencia}. "
                     f"Un año bueno es suerte; **cuatro seguidos es un negocio**.")

    L.append("\n> Son ratios **contables y del pasado**: dicen cómo ha ido el negocio, "
             "**no** hacia dónde va la acción. Ninguno de los cuatro predice el retorno. "
             "Sirven para descartar, no para acertar. No es recomendación de inversión.")
    return "\n".join(L)


_CACHE_ROIC = {}
_TTL_ROIC = 6 * 3600          # las cuentas anuales no cambian en todo el día


def roic_ticker(ticker):
    """ROIC (último ejercicio, mediana) sin pedir `.info`. Devuelve (ult, mediana, n).

    Versión ligera de `historico()` para usarla en bucle sobre una watchlist: son
    2 llamadas de red por ticker en vez de 3, y quedan cacheadas en el proceso.
    Sobre 10 valores la diferencia es de segundos, no de milisegundos.

    No hace falta mirar el sector: un banco no publica EBIT, así que el ROIC sale
    NaN solo, sin necesidad de detectarlo. Devuelve NaN en vez de lanzar: en un
    barrido, un valor raro no puede tumbar la tabla entera."""
    import time as _t
    tk = ticker.strip().upper()
    hit = _CACHE_ROIC.get(tk)
    if hit and (_t.time() - hit[0]) < _TTL_ROIC:
        return hit[1]
    fuera = (np.nan, np.nan, 0)
    try:
        t = yf.Ticker(tk)
        inc, bs = t.income_stmt, t.balance_sheet
        if inc is None or inc.empty or bs is None or bs.empty:
            _CACHE_ROIC[tk] = (_t.time(), fuera)
            return fuera
        ebit = _fila(inc, "EBIT", "Operating Income")
        tasa = _fila(inc, "Tax Rate For Calcs")
        cap = _fila(bs, "Invested Capital")
        if ebit is None or cap is None:
            _CACHE_ROIC[tk] = (_t.time(), fuera)
            return fuera
        vals = []
        for col in inc.columns:
            if col not in cap.index:
                continue
            e, c = ebit.get(col), cap.get(col)
            tt = tasa.get(col) if tasa is not None else None
            if e is None or c is None or pd.isna(e) or pd.isna(c) or float(c) == 0:
                continue
            tt = float(tt) if tt is not None and not pd.isna(tt) else 0.21
            vals.append(float(e) * (1 - tt) / float(c))
        if not vals:
            _CACHE_ROIC[tk] = (_t.time(), fuera)
            return fuera
        res = (float(vals[0]), float(np.median(vals)), len(vals))
    except Exception:
        res = fuera
    _CACHE_ROIC[tk] = (_t.time(), res)
    return res


def pilar_calidad(ticker):
    """Pilar de CALIDAD para el Veredicto, basado en ROIC. Devuelve (score, lectura, w).

    Por qué ROIC y no ROE: el ROE se infla con deuda, y un pilar de calidad que
    premie el apalancamiento premia justo lo contrario de lo que quiere medir.

    Cómo puntúa
    -----------
    Nivel: se usa la MEDIANA de los ejercicios disponibles, no el último dato.
    Un año bueno es suerte; la mediana resiste a un ejercicio raro.

        score = (ROIC_mediano − 10%) / 15%     acotado a [−1, +1]

    El 10% es el coste aproximado del capital: por debajo, cada euro reinvertido
    vale menos de un euro (destruye valor); por encima, lo crea.

    Tendencia: si el ROIC se ha desplomado (último < 60% del primero) se recorta
    el score a la mitad. Caso real, XOM 2022-2025: ROIC de 22,7% a 9,7%. Puntuar
    eso como negocio sano por su media seria enganarse.

    Peso: PROPORCIONAL a la conviccion, igual que el pilar de forecast. Un ROIC
    del 10% (ni fu ni fa) casi no participa. Si pesara fijo, meter un pilar
    neutro dividiendo la suma de pesos empujaria TODOS los veredictos hacia
    MANTENER, que es justo el fallo que ya se ha corregido otras veces aqui.

    Devuelve (None, motivo, 0.0) cuando no aplica: bancos y aseguradoras no
    tienen capital invertido industrial, y forzar un numero ahi seria inventarlo.
    """
    d = kpis(ticker)
    if d["financiera"]:
        return None, "no aplica (financiera: sin capital invertido industrial)", 0.0
    if d["roic"] is None:
        return None, (d["roic_nota"] or "sin datos de ROIC"), 0.0

    hist = historico(ticker, 4)
    serie = hist["ROIC %"].dropna() / 100.0 if not hist.empty and "ROIC %" in hist else pd.Series(dtype=float)
    nivel = float(serie.median()) if len(serie) >= 2 else float(d["roic"])

    score = max(-1.0, min(1.0, (nivel - 0.10) / 0.15))
    nota = ""
    if len(serie) >= 3 and serie.iloc[0] > 0 and serie.iloc[-1] < 0.6 * serie.iloc[0]:
        # El castigo tiene que empeorar el score SIEMPRE. Un `score *= 0.5` a secas
        # sube los negativos hacia cero: Intel, con el ROIC cayendo del 5% al 1%,
        # salía premiada por desplomarse. Se ramifica por signo.
        score = score * 0.5 if score > 0 else max(-1.0, score * 1.25)
        nota = f" · ⚠️ en caída ({serie.iloc[0]*100:.0f}%→{serie.iloc[-1]*100:.0f}%)"

    w = max(0.04, 0.12 * min(1.0, abs(score)))
    calif = ("crea valor" if score > 0.33 else
             "destruye valor" if score < -0.33 else "en el coste del capital")
    lectura = (f"ROIC {nivel*100:.1f}% ({len(serie) if len(serie) else 1} ejerc., mediana) "
               f"· {calif}{nota}")
    return score, lectura, w


def pilar_red(ticker):
    """Pilar de CALIDAD para el Veredicto Cripto. Devuelve (score, lectura, peso).

    NO es un ROIC ni pretende serlo: una cripto no tiene EBIT ni capital invertido
    porque no es una empresa (comprobado: `income_stmt` y `balance_sheet` vienen
    vacíos, y ROE/ROA/BPA vienen a None en BTC, ETH, SOL y ADA). Este pilar
    responde a la MISMA pregunta que el ROIC —¿esto trata bien al que pone el
    dinero?— pero con lo único que hay: datos de red.

    Dos componentes
    ---------------
    1. **Liquidez** = volumen negociado en 24 h, en dólares ABSOLUTOS.

       Se usa el volumen absoluto y no el turnover (volumen ÷ capitalización)
       porque el turnover INVIERTE el orden: medido hoy, BTC sale el peor
       (1,46%) y ADA el mejor (10,3%), cuando lo de ADA es rotación
       especulativa. Lo que decide si puedes salir sin mover el precio son los
       dólares que se cruzan, no su proporción. Escala logarítmica: 1 M$ es malo,
       ~32 M$ neutro, 1.000 M$ bueno.

       No se añade la capitalización como tercer componente: log(volumen) y
       log(capitalización) correlacionan 0,98 en las 13 monedas medidas, así que
       sería contar lo mismo dos veces.

    2. **Dilución** = circulante ÷ suministro máximo. Lo que queda por emitir te
       diluye, igual que una empresa que no para de sacar acciones nuevas. 95%
       emitido es bueno; 65% es malo.

       Si NO hay tope (ETH, SOL, DOGE) el componente queda NEUTRO, no negativo:
       sin tope no significa inflación garantizada — Ethereum quema comisiones y
       puede ser deflacionaria. Penalizarlo sería opinar, no medir.

    Devuelve (None, motivo, 0.0) si el dato viene incompleto, que pasa (UNI-USD
    daba volumen 0,0 M$ y capitalización 0,00 B$): un fallo de datos no puede
    puntuar como si fuera una moneda ilíquida de verdad.
    """
    tk = ticker.strip().upper()
    try:
        info = yf.Ticker(tk).info or {}
    except Exception as e:
        return None, f"sin datos ({str(e)[:40]})", 0.0

    vol = info.get("volume24Hr") or info.get("volume")
    mc = info.get("marketCap")
    if not vol or not mc or vol < 1e5 or mc < 1e6:
        return None, "datos de red incompletos en la fuente", 0.0

    s_liq = max(-1.0, min(1.0, (np.log10(float(vol)) - 7.5) / 1.5))
    partes = [(s_liq, 0.6)]
    txt = [f"volumen 24h {vol/1e6:,.0f} M$"]

    circ, mx = info.get("circulatingSupply"), info.get("maxSupply")
    if circ and mx and mx > 0:
        rel = float(circ) / float(mx)
        s_dil = max(-1.0, min(1.0, (rel - 0.80) / 0.15))
        partes.append((s_dil, 0.4))
        txt.append(f"{rel*100:.0f}% del máximo ya emitido")
    else:
        txt.append("sin tope de emisión (neutro: sin tope ≠ inflación)")

    score = sum(s * w for s, w in partes) / sum(w for _, w in partes)
    w = max(0.04, 0.12 * min(1.0, abs(score)))
    calif = ("red sólida" if score > 0.33 else
             "red frágil" if score < -0.33 else "en la media")
    return float(score), " · ".join(txt) + f" → {calif}", float(w)


def _plot(hist, nombre, financiera=False):
    """Evolución de los tres ratios.

    Cuando el ROE se dispara por apalancamiento (Apple: 150-200% frente a un ROA
    del 27%) meterlos en un solo eje deja ROA y ROIC pegados al suelo, ilegibles.
    En ese caso se parte en dos paneles: el de arriba para el ROE con su escala,
    el de abajo para ROA y ROIC con la suya."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _vacio(msg):
        f, a = plt.subplots(figsize=(10, 2.2))
        a.text(0.5, 0.5, msg, ha="center", va="center", color=_NEU, fontsize=11.5)
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
        a.grid(False)
        f.tight_layout()
        return f

    if hist is None or hist.empty:
        return _vacio("Sin estados financieros publicados para este valor.")

    x = hist["Ejercicio"]
    hay = {c: (c in hist.columns and hist[c].notna().any()) for c in ("ROE %", "ROA %", "ROIC %")}
    if not any(hay.values()):
        return _vacio("Sin datos suficientes para calcular la rentabilidad.")

    # ¿el ROE se sale de escala frente a los otros dos?
    otros = [hist[c].max() for c in ("ROA %", "ROIC %") if hay[c] and hist[c].notna().any()]
    roe_max = hist["ROE %"].max() if hay["ROE %"] else 0
    partir = bool(otros) and hay["ROE %"] and roe_max > 3 * max(otros)

    try:
        from finanzia_charts import style
    except Exception:
        style = None

    def _remate(ax, titulo=None, kicker=None, ylabel="%"):
        # sin leyenda: las series se etiquetan al final de su propia línea, que
        # no puede pisar los datos (la caja de leyenda sí lo hacía)
        if style:
            style(ax, titulo=titulo, kicker=kicker, ylabel=ylabel, legend=False)
        else:
            if titulo:
                ax.set_title(titulo)
            ax.set_ylabel(ylabel)
        ax.margins(x=0.13)

    # ROIC en el acento principal: es el protagonista y el menos manipulable
    COLOR = {"ROE %": _GOLD, "ROA %": _NEU, "ROIC %": _ACC}

    def _serie(ax, col):
        ax.plot(x, hist[col], marker="o", ms=5, lw=1.8, color=COLOR[col])
        s = hist[col].dropna()
        if len(s):
            ax.annotate(f" {col.replace(' %', '')} {s.iloc[-1]:.1f}%",
                        xy=(len(hist) - 1, hist[col].iloc[-1]), xytext=(7, 0),
                        textcoords="offset points", color=COLOR[col], fontsize=10.5,
                        va="center", ha="left", annotation_clip=False)

    def _coste_capital(ax):
        # en banca el ROA nunca se compara con el coste del capital: pintar la
        # referencia ahí sugeriría que un ROA del 0,8% es un fracaso, y no lo es
        if financiera:
            return
        ax.axhline(10, color=_DIM, ls="--", lw=1)
        ax.annotate("~10% = lo que suele costar el capital", xy=(0, 10),
                    xycoords=("axes fraction", "data"), xytext=(4, 5),
                    textcoords="offset points", ha="left", color=_DIM, fontsize=9)

    if partir:
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True,
                                     gridspec_kw={"height_ratios": [1, 1.3]})
        _serie(a1, "ROE %")
        _remate(a1, titulo=f"{nombre} — rentabilidad por ejercicio",
                kicker="ROE · ROA · ROIC · CUENTAS ANUALES", ylabel="ROE %")
        for c in ("ROA %", "ROIC %"):
            if hay[c]:
                _serie(a2, c)
        _coste_capital(a2)
        _remate(a2, ylabel="ROA y ROIC %")
        nota = ("escala propia: en banca el ROE va siempre muy por encima del ROA"
                if financiera else
                "escala propia: el ROE va muy por encima del ROA (apalancamiento)")
        a1.annotate(nota, xy=(0, 1), xycoords="axes fraction", xytext=(0, -13),
                    textcoords="offset points", color=_DIM, fontsize=9, va="top")
    else:
        fig, ax = plt.subplots(figsize=(10, 4.6))
        for c in ("ROE %", "ROA %", "ROIC %"):
            if hay[c]:
                _serie(ax, c)
        _coste_capital(ax)
        _remate(ax, titulo=f"{nombre} — rentabilidad por ejercicio",
                kicker="ROE · ROA · ROIC · CUENTAS ANUALES")
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description="Los 4 KPI de rentabilidad: ROE, ROA, BPA, ROIC.")
    ap.add_argument("ticker", nargs="?", default="AAPL")
    ap.add_argument("--comparar", help="Lista separada por comas.")
    ap.add_argument("--anios", type=int, default=4)
    a = ap.parse_args()
    if a.comparar:
        tks = [t.strip() for t in a.comparar.replace(",", " ").split() if t.strip()]
        print(comparar(tks).to_string(index=False))
        return
    d = kpis(a.ticker)
    h = historico(a.ticker, a.anios)
    print(explicar(d, h))
    if not h.empty:
        print("\n" + h.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
