"""
veredicto_tune — búsqueda honesta de la configuración del Veredicto.

Barre configuraciones (pesos de los pilares + umbral) y se queda con la mejor.
Eso, hecho a la ligera, es **p-hacking**: si pruebas 200 combinaciones sobre la
misma serie, la mejor lo será por azar aunque no haya ninguna ventaja real. Así
que aquí la búsqueda va con tres cortafuegos:

1. **CPCV** (Combinatorial Purged Cross-Validation, López de Prado). La mejor
   configuración se elige DENTRO de muestra y se juzga FUERA, sobre bloques que
   no ha visto, con purgado del horizonte y embargo.

2. **PBO** (Probability of Backtest Overfitting): en cada partición se mira si la
   ganadora dentro de muestra cae por debajo de la mediana fuera. Un PBO alto
   significa "lo que ganó en el pasado no gana después" — es decir, la búsqueda
   está encontrando ruido.

3. **Deflated Sharpe con el recuento ACUMULADO de pruebas.** Este es el punto
   clave y el que hace que el proceso siga siendo honesto tanda tras tanda: el
   número de configuraciones probadas se guarda en disco y NUNCA se reinicia. Si
   hoy pruebas 10 y mañana otras 10, el listón que hay que superar sube, porque
   has tenido 20 oportunidades de acertar por suerte, no 10. Sin esta memoria,
   repetir la búsqueda cada pocas horas "encontraría" un ganador tarde o
   temprano garantizado, y sería mentira.

Resultado esperado, dicho de antemano para no engañarse luego: lo más probable
es que NINGUNA configuración sobreviva al deflactado. Eso no es un fracaso del
experimento, es el resultado.

Uso:
    python veredicto_tune.py --n 10
    python veredicto_tune.py --estado
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from math import comb
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SUITE = HERE.parent
for p in (SUITE / "veredicto_backtest", SUITE / "indicators", SUITE / "cpcv"):
    sys.path.insert(0, str(p))
import veredicto_backtest as VB

EULER = 0.5772156649015329
ESTADO = SUITE.parent / "data" / "tune_veredicto.json"

PILARES = ["tend", "adx", "osc", "macd", "mom", "obv"]
UNIVERSO = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "XOM", "KO", "JNJ"]


# --- estado persistente ------------------------------------------------------
def cargar_estado():
    """Ledger de TODAS las configuraciones probadas alguna vez. No se reinicia."""
    if ESTADO.exists():
        try:
            return json.loads(ESTADO.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"pruebas": [], "tandas": 0}


def guardar_estado(est):
    ESTADO.parent.mkdir(parents=True, exist_ok=True)
    ESTADO.write_text(json.dumps(est, indent=1, ensure_ascii=False), encoding="utf-8")


# --- generación de configuraciones -------------------------------------------
def generar(n, semilla, ya_probadas):
    """n configuraciones nuevas: pesos Dirichlet + umbral. Evita repetir las ya vistas."""
    rng = np.random.default_rng(semilla)
    vistos = {c["firma"] for c in ya_probadas}
    out = []
    intentos = 0
    while len(out) < n and intentos < n * 200:
        intentos += 1
        w = rng.dirichlet(np.ones(len(PILARES)))
        pesos = {k: round(float(v), 3) for k, v in zip(PILARES, w)}
        umbral = float(rng.choice([0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]))
        firma = "|".join(f"{pesos[k]:.3f}" for k in PILARES) + f"|{umbral:.2f}"
        if firma in vistos:
            continue
        vistos.add(firma)
        out.append({"pesos": pesos, "umbral": umbral, "firma": firma})
    return out


def config_actual():
    """La configuración que el Veredicto usa hoy, como referencia a batir."""
    p = dict(VB.PESOS)
    firma = "|".join(f"{p[k]:.3f}" for k in PILARES) + "|0.35"
    return {"pesos": p, "umbral": 0.35, "firma": firma, "nombre": "ACTUAL"}


# --- datos -------------------------------------------------------------------
def preparar(tickers, period="8y", horizonte=10):
    """Componentes por ticker (una vez) + retorno futuro alineado.

    Los indicadores se calculan UNA vez; después, evaluar una configuración es un
    producto escalar. Sin esto, barrer 10 configuraciones costaría 10 descargas."""
    comps, fwd = {}, {}
    for tk in tickers:
        try:
            df = VB.descargar(tk, period)
        except Exception:
            continue
        if df is None or len(df) < 400:
            continue
        comps[tk] = VB.componentes(df)
        c = df["Close"]
        fwd[tk] = c.shift(-horizonte) / c - 1.0
    if len(comps) < 2:
        return None, None
    idx = sorted(set().union(*[c.index for c in comps.values()]))
    C = {k: v.reindex(idx) for k, v in comps.items()}
    R = pd.DataFrame({k: v.reindex(idx) for k, v in fwd.items()})
    return C, R


def scores_de(C, pesos):
    """Matriz de scores (filas=fechas, cols=tickers) para un juego de pesos."""
    return pd.DataFrame({tk: VB.score_de_componentes(comp, pesos) for tk, comp in C.items()})


def _retornos(S, R, idx, umbral):
    s = S.iloc[idx].values
    r = R.iloc[idx].values
    m = np.isfinite(s) & np.isfinite(r) & (np.abs(s) > umbral)
    if m.sum() < 10:
        return np.array([])
    return np.sign(s[m]) * r[m]


def tasa_exito(r):
    """Fracción de operaciones que acaban en verde. OJO: por sí sola no dice si
    ganas dinero — se puede acertar el 90% y perderlo todo en el 10% restante.
    Siempre hay que mirarla junto al Sharpe y a la ganancia media."""
    if len(r) == 0:
        return np.nan
    return float((r > 0).mean())


def payoff(r):
    """Ganancia media de los aciertos ÷ pérdida media de los fallos. Es el
    contrapeso de la tasa de acierto: si es < 1, aciertas mucho y cobras poco."""
    g, p = r[r > 0], r[r < 0]
    if len(g) == 0 or len(p) == 0 or p.mean() == 0:
        return np.nan
    return float(g.mean() / abs(p.mean()))


def p_binomial(tasa, n_ops, horizonte, n_pruebas):
    """¿Esa tasa de acierto bate a la moneda al aire, habiendo probado n_pruebas?

    Dos correcciones imprescindibles:
    - **Solape**: los retornos a `horizonte` días comparten barras, así que la
      muestra efectiva es n_ops/horizonte, no n_ops. Sin esto cualquier tasa sale
      significativa por tener "miles" de operaciones que en realidad son cientos.
    - **Multiplicidad (Šidák)**: con n_pruebas configuraciones, la probabilidad de
      que ALGUNA parezca buena por azar es 1-(1-p)^n. Se corrige el p-valor.
    """
    from scipy.stats import binomtest
    n_ef = max(5, int(n_ops / max(1, horizonte)))
    exitos = int(round(tasa * n_ef))
    if not np.isfinite(tasa) or n_ef < 5:
        return np.nan, np.nan
    p = binomtest(exitos, n_ef, 0.5, alternative="greater").pvalue
    p_corr = 1.0 - (1.0 - p) ** max(1, n_pruebas)     # Šidák
    return float(p), float(min(1.0, p_corr))


def _sharpe(r, horizonte):
    """Sharpe anualizado. Los retornos a 'horizonte' días SOLAPAN, así que se
    anualiza con sqrt(252/horizonte), no con sqrt(252)."""
    if len(r) < 20 or not np.isfinite(r).all() or r.std(ddof=1) == 0:
        return np.nan
    return float(r.mean() / r.std(ddof=1) * np.sqrt(252.0 / horizonte))


# --- evaluación CPCV ---------------------------------------------------------
def evaluar(configs, C, R, horizonte=10, n_bloques=6, k_prueba=2, embargo_pct=0.01):
    """Sharpe dentro y fuera de muestra de cada configuración, por partición CPCV."""
    n = len(R)
    bloques = np.array_split(np.arange(n), n_bloques)
    emb = max(1, int(n * embargo_pct))
    S = {c["firma"]: scores_de(C, c["pesos"]) for c in configs}

    particiones = []
    for test_ids in combinations(range(n_bloques), k_prueba):
        idx_test = np.concatenate([bloques[i] for i in test_ids])
        prohibido = set()
        for i in test_ids:                      # purgado + embargo
            b = bloques[i]
            ini, fin = b[0] - horizonte, b[-1] + horizonte + emb
            prohibido.update(range(max(0, ini), min(n, fin + 1)))
        idx_train = np.array([i for i in range(n) if i not in prohibido])
        if len(idx_train) < 150 or len(idx_test) < 30:
            continue
        fila = {"test": test_ids, "is": {}, "oos": {}, "r_oos": {}}
        for c in configs:
            f = c["firma"]
            fila["is"][f] = _sharpe(_retornos(S[f], R, idx_train, c["umbral"]), horizonte)
            r_o = _retornos(S[f], R, idx_test, c["umbral"])
            fila["oos"][f] = _sharpe(r_o, horizonte)
            fila["r_oos"][f] = r_o
        particiones.append(fila)
    return particiones, comb(n_bloques, k_prueba)


def pbo_de(particiones, firmas):
    """Probability of Backtest Overfitting: ¿la ganadora IS cae bajo la mediana OOS?"""
    peores = 0
    validas = 0
    for p in particiones:
        cand = [(f, p["is"][f]) for f in firmas if np.isfinite(p["is"].get(f, np.nan))]
        if not cand:
            continue
        mejor = max(cand, key=lambda x: x[1])[0]
        vals = [p["oos"][f] for f in firmas if np.isfinite(p["oos"].get(f, np.nan))]
        if not vals or not np.isfinite(p["oos"].get(mejor, np.nan)):
            continue
        validas += 1
        rank = np.mean([p["oos"][mejor] <= v for v in vals])
        peores += 1 if rank > 0.5 else 0
    return (peores / validas) if validas else np.nan, validas


def dsr(sr_obs, r, n_pruebas, var_sr, horizonte):
    """Deflated Sharpe contra el mejor esperado por azar entre n_pruebas.

    T efectivo = nº de retornos / horizonte, porque los retornos solapan. Sin esa
    corrección el DSR sale inflado y contradice al PBO."""
    from scipy.stats import norm
    if len(r) < 30 or not np.isfinite(sr_obs):
        return np.nan, np.nan
    T = max(20, int(len(r) / max(1, horizonte)))
    sr_p = sr_obs / np.sqrt(252.0 / horizonte)            # Sharpe por periodo
    sk = float(pd.Series(r).skew())
    ku = float(pd.Series(r).kurt()) + 3.0
    if n_pruebas < 2 or var_sr <= 0:
        sr0 = 0.0
    else:
        z1 = norm.ppf(1 - 1.0 / n_pruebas)
        z2 = norm.ppf(1 - 1.0 / (n_pruebas * np.e))
        sr0 = np.sqrt(var_sr) * ((1 - EULER) * z1 + EULER * z2)
    den = np.sqrt(max(1e-9, 1 - sk * sr_p + (ku - 1) / 4.0 * sr_p ** 2))
    z = (sr_p - sr0) * np.sqrt(max(1, T - 1)) / den
    return float(norm.cdf(z)), float(sr0 * np.sqrt(252.0 / horizonte))


# --- tanda -------------------------------------------------------------------
def tanda(n=10, tickers=None, period="8y", horizonte=10, semilla=None):
    """Una tanda de n configuraciones nuevas + la actual como referencia."""
    t0 = time.time()
    est = cargar_estado()
    tickers = tickers or UNIVERSO
    semilla = semilla if semilla is not None else (est["tandas"] * 1000 + 7)

    C, R = preparar(tickers, period, horizonte)
    if C is None:
        return {"error": "Datos insuficientes."}

    nuevas = generar(n, semilla, est["pruebas"])
    actual = config_actual()
    configs = nuevas + [actual]
    firmas_nuevas = [c["firma"] for c in nuevas]

    particiones, n_comb = evaluar(configs, C, R, horizonte)
    if not particiones:
        return {"error": "Sin particiones válidas."}

    # media OOS por configuración
    resumen = []
    for c in configs:
        f = c["firma"]
        oos = [p["oos"][f] for p in particiones if np.isfinite(p["oos"].get(f, np.nan))]
        rr = [p["r_oos"][f] for p in particiones if len(p["r_oos"].get(f, []))]
        rcat = np.concatenate(rr) if rr else np.array([])
        resumen.append({"firma": f, "pesos": c["pesos"], "umbral": c["umbral"],
                        "nombre": c.get("nombre", ""),
                        "sr_oos": float(np.mean(oos)) if oos else np.nan,
                        "n_oos": len(oos), "r": rcat,
                        "tasa": tasa_exito(rcat), "payoff": payoff(rcat),
                        "n_ops": int(len(rcat)),
                        "ret_medio": float(rcat.mean()) if len(rcat) else np.nan})

    # el ledger acumula: N nunca se reinicia entre tandas
    for c in nuevas:
        r = next(x for x in resumen if x["firma"] == c["firma"])
        est["pruebas"].append({"firma": c["firma"], "pesos": c["pesos"],
                               "umbral": c["umbral"], "sr_oos": r["sr_oos"],
                               "tasa": r["tasa"], "payoff": r["payoff"],
                               "n_ops": r["n_ops"], "ret_medio": r["ret_medio"],
                               "tanda": est["tandas"] + 1,
                               "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    est["tandas"] += 1
    guardar_estado(est)

    n_total = len(est["pruebas"])
    srs_hist = [p["sr_oos"] for p in est["pruebas"] if p.get("sr_oos") is not None
                and np.isfinite(p.get("sr_oos", np.nan))]
    var_sr = float(np.var(srs_hist, ddof=1)) / (252.0 / horizonte) if len(srs_hist) > 2 else 0.0

    solo_nuevas = [r for r in resumen if r["firma"] in firmas_nuevas and np.isfinite(r["sr_oos"])]
    ref = next(r for r in resumen if r["nombre"] == "ACTUAL")
    mejor = max(solo_nuevas, key=lambda x: x["sr_oos"]) if solo_nuevas else None

    pbo, n_val = pbo_de(particiones, [c["firma"] for c in configs])
    p_dsr, sr0 = (dsr(mejor["sr_oos"], mejor["r"], n_total, var_sr, horizonte)
                  if mejor is not None else (np.nan, np.nan))

    return {"tanda": est["tandas"], "n_nuevas": len(nuevas), "n_total": n_total,
            "tickers": len(C), "particiones": n_val, "combinaciones": n_comb,
            "mejor": mejor, "actual": ref, "resumen": resumen,
            "pbo": pbo, "dsr": p_dsr, "sr0": sr0, "var_sr": var_sr,
            "segundos": round(time.time() - t0, 1)}


def informe(res):
    if "error" in res:
        return f"**Error:** {res['error']}"
    m, a = res["mejor"], res["actual"]
    L = [f"### Tanda {res['tanda']} · {res['n_nuevas']} configuraciones nuevas "
         f"({res['n_total']} acumuladas)",
         f"{res['tickers']} valores · {res['particiones']} particiones CPCV de "
         f"{res['combinaciones']} · {res['segundos']}s"]
    if m is None:
        L.append("\nNinguna configuración nueva dio Sharpe OOS calculable.")
        return "\n".join(L)

    L.append(f"\n| | Sharpe OOS medio | umbral |\n|---|---|---|")
    L.append(f"| **Mejor de la tanda** | {m['sr_oos']:+.3f} | {m['umbral']:.2f} |")
    L.append(f"| Configuración ACTUAL | {a['sr_oos']:+.3f} | {a['umbral']:.2f} |")

    L.append(f"\n**Pesos de la mejor:** " +
             " · ".join(f"{k} {m['pesos'][k]:.2f}" for k in PILARES))

    L.append(f"\n**Los tres números que deciden si esto vale algo:**")
    L.append(f"- **PBO = {res['pbo']*100:.0f}%** — probabilidad de que la ganadora dentro de "
             f"muestra quede por debajo de la mediana fuera. Por encima del 50% la búsqueda "
             f"está encontrando ruido.")
    L.append(f"- **Listón por azar = {res['sr0']:+.3f}** de Sharpe — lo que sacaría la mejor de "
             f"{res['n_total']} configuraciones aunque ninguna tuviera ventaja.")
    L.append(f"- **Deflated Sharpe = {res['dsr']*100:.0f}%** — probabilidad de que la ventaja "
             f"sea real una vez descontadas esas {res['n_total']} oportunidades de acertar "
             f"por suerte. Hace falta ≥95%.")

    bate = m["sr_oos"] > a["sr_oos"]
    if np.isfinite(res["dsr"]) and res["dsr"] >= 0.95 and res["pbo"] < 0.5 and bate:
        L.append(f"\n✅ **Candidata que sobrevive.** Supera a la actual, el PBO está por debajo "
                 f"del 50% y el DSR pasa del 95%. Aun así, antes de cambiar nada: repítelo con "
                 f"otro universo de valores.")
    else:
        motivos = []
        if not bate:
            motivos.append("no bate a la configuración actual")
        if np.isfinite(res["pbo"]) and res["pbo"] >= 0.5:
            motivos.append(f"PBO {res['pbo']*100:.0f}% ≥ 50% (lo que gana dentro no gana fuera)")
        if not np.isfinite(res["dsr"]) or res["dsr"] < 0.95:
            motivos.append(f"DSR {res['dsr']*100:.0f}% < 95%")
        L.append(f"\n❌ **Nada que cambiar todavía**: " + "; ".join(motivos) + ".")
        L.append(f"> Esto es un resultado, no un fallo: significa que la configuración actual "
                 f"no es peor que {res['n_total']} alternativas probadas a conciencia.")

    L.append(f"\n> Cada tanda **suma** al recuento de pruebas y sube el listón, porque cada una "
             f"es otra oportunidad de acertar por suerte. El contador vive en "
             f"`data/tune_veredicto.json` y no se reinicia: si se reiniciara, repetir la "
             f"búsqueda acabaría 'encontrando' un ganador garantizado, y sería mentira.")
    return "\n".join(L)


def recalcular(tickers=None, period="8y", horizonte=10, n_bloques=6, k_prueba=2):
    """Rellena tasa/payoff en las configuraciones ya guardadas que no la tengan.

    NO incrementa el contador de pruebas: esas configuraciones ya se contaron
    cuando se probaron. Volver a medirlas no da nuevas oportunidades de acertar
    por suerte, así que el listón no debe moverse."""
    est = cargar_estado()
    faltan = [p for p in est["pruebas"] if p.get("tasa") is None]
    if not faltan:
        return 0, len(est["pruebas"])
    C, R = preparar(tickers or UNIVERSO, period, horizonte)
    if C is None:
        return 0, len(est["pruebas"])
    configs = [{"pesos": p["pesos"], "umbral": p["umbral"], "firma": p["firma"]} for p in faltan]
    particiones, _ = evaluar(configs, C, R, horizonte, n_bloques, k_prueba)
    for p in faltan:
        rr = [x["r_oos"][p["firma"]] for x in particiones if len(x["r_oos"].get(p["firma"], []))]
        r = np.concatenate(rr) if rr else np.array([])
        p["tasa"] = tasa_exito(r)
        p["payoff"] = payoff(r)
        p["n_ops"] = int(len(r))
        p["ret_medio"] = float(r.mean()) if len(r) else None
    guardar_estado(est)
    return len(faltan), len(est["pruebas"])


def clasificacion(criterio="tasa", top=8, horizonte=10):
    """Tabla de TODAS las configuraciones probadas alguna vez, ordenada por
    `tasa` (acierto) o `sharpe`, con el p-valor ya corregido por multiplicidad.

    Esto es lo que de verdad contesta a "¿cuál es la mejor?": comparar la mejor
    HISTÓRICA contra el listón que impone el total acumulado de pruebas. Mirar
    solo la ganadora de cada tanda contra su propia tanda es el error clásico."""
    est = cargar_estado()
    pr = [p for p in est["pruebas"] if p.get("tasa") is not None
          and np.isfinite(p.get("tasa", np.nan))]
    if not pr:
        return None, est
    n_total = len(est["pruebas"])
    for p in pr:
        p["p_bruto"], p["p_corr"] = p_binomial(p["tasa"], p.get("n_ops", 0),
                                               horizonte, n_total)
    clave = "tasa" if criterio == "tasa" else "sr_oos"
    pr.sort(key=lambda x: (x.get(clave) if x.get(clave) is not None else -9), reverse=True)
    return pr[:top], est


def informe_clasificacion(criterio="tasa", top=8, horizonte=10):
    pr, est = clasificacion(criterio, top, horizonte)
    if not pr:
        return "Aún no hay pruebas con tasa de acierto guardada. Lanza `--n 10`."
    n_total = len(est["pruebas"])
    L = [f"### Clasificación por {'tasa de acierto' if criterio == 'tasa' else 'Sharpe'} "
         f"· {n_total} configuraciones probadas en {est['tandas']} tandas\n",
         "| # | acierto | payoff | Sharpe OOS | ret. medio | ops | umbral | p corregido |",
         "|---|---|---|---|---|---|---|---|"]
    for i, p in enumerate(pr, 1):
        pay = f"{p['payoff']:.2f}" if p.get("payoff") and np.isfinite(p["payoff"]) else "—"
        L.append(f"| {i} | **{p['tasa']*100:.1f}%** | {pay} | {p['sr_oos']:+.2f} | "
                 f"{p['ret_medio']*100:+.2f}% | {p['n_ops']} | {p['umbral']:.2f} | "
                 f"{p['p_corr']:.3f} |")
    m = pr[0]
    L.append(f"\n**La de mayor tasa de acierto** ({m['tasa']*100:.1f}%) usa estos pesos:")
    L.append("· ".join(f"**{k}** {m['pesos'][k]:.2f}" for k in PILARES) +
             f" · umbral **{m['umbral']:.2f}**")
    L.append(f"\n- Acierta el **{m['tasa']*100:.1f}%** de {m['n_ops']} operaciones. Pero esas "
             f"operaciones **solapan** (retornos a {horizonte} días), así que la muestra "
             f"independiente es de ~{int(m['n_ops']/horizonte)}, no de {m['n_ops']}.")
    if m.get("payoff") and np.isfinite(m["payoff"]):
        if m["payoff"] < 1:
            L.append(f"- ⚠️ Su **payoff es {m['payoff']:.2f}**: gana menos en cada acierto de lo "
                     f"que pierde en cada fallo. Acertar mucho aquí no basta — por eso la tasa "
                     f"de acierto sola engaña.")
        else:
            L.append(f"- Payoff {m['payoff']:.2f}: cuando acierta gana más de lo que pierde "
                     f"cuando falla. La tasa alta sí se traduce en dinero.")
    L.append(f"- **p corregido = {m['p_corr']:.3f}** (Šidák sobre {n_total} pruebas). "
             f"Hace falta < 0,05 para poder decir que bate a la moneda al aire.")
    if m["p_corr"] < 0.05:
        L.append(f"\n✅ Sobrevive a la corrección por multiplicidad. Antes de tocar el "
                 f"Veredicto, valídala en un universo de valores distinto.")
    else:
        L.append(f"\n❌ **No sobrevive.** Con {n_total} configuraciones probadas, una tasa del "
                 f"{m['tasa']*100:.1f}% sobre ~{int(m['n_ops']/horizonte)} operaciones "
                 f"independientes entra dentro de lo que sale por azar. No cambies nada.")
    return "\n".join(L)


def validar_fuera(pesos, umbral, tickers, period="8y", horizonte=10):
    """La prueba decisiva: la misma configuración en un universo que NO se usó
    para elegirla. Si el efecto era ruido, aquí se cae."""
    C, R = preparar(tickers, period, horizonte)
    if C is None:
        return None
    S = scores_de(C, pesos)
    r = _retornos(S, R, np.arange(len(R)), umbral)
    if len(r) == 0:
        return None
    return {"tickers": len(C), "n_ops": len(r), "tasa": tasa_exito(r),
            "payoff": payoff(r), "sharpe": _sharpe(r, horizonte),
            "ret_medio": float(r.mean())}


# --- perfiles guardados y elegibles -----------------------------------------
PERFILES = SUITE.parent / "data" / "perfiles_veredicto.json"
UNIVERSO_VAL = ["WMT", "PG", "CVX", "MRK", "HD", "CAT", "BA", "PFE", "T", "VZ"]


def _veredicto_perfil(busq, val):
    """Etiqueta honesta de un perfil según SU validación fuera del universo.

    Lo que decide no es lo bien que salió en la búsqueda —ahí siempre sale
    bien, para eso se eligió— sino si aguanta en valores que no participaron
    en elegirla."""
    if val is None:
        return "⚠️ sin validar", ("Nunca se ha probado fuera del universo donde se "
                                  "eligió. No sabes si es real.")
    if val["tasa"] >= 0.52 and val["sharpe"] > 0:
        return "✅ validado", "Aguanta en valores que no se usaron para elegirla."
    if val["tasa"] >= 0.49:
        return "➖ se queda en nada", (f"Fuera de su universo acierta {val['tasa']*100:.1f}%: "
                                       f"como una moneda al aire. No aporta.")
    return "❌ se cae fuera", (f"Fuera de su universo acierta {val['tasa']*100:.1f}% con Sharpe "
                              f"{val['sharpe']:+.2f}: PEOR que el azar. Estaba sobreajustada.")


def listar_perfiles():
    if PERFILES.exists():
        try:
            return json.loads(PERFILES.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def perfil_por_nombre(nombre):
    """Devuelve (pesos, umbral, perfil) del nombre dado; la ACTUAL si no existe."""
    for p in listar_perfiles():
        if p["nombre"] == nombre:
            return p["pesos"], p["umbral"], p
    a = config_actual()
    return a["pesos"], a["umbral"], None


def construir_perfiles(top=5, period="8y", horizonte=10, universo_val=None):
    """Guarda las mejores de la clasificación CON su validación independiente.

    Se guardan también las que se caen: un perfil marcado ❌ enseña más que un
    perfil escondido. Lo que no se hace es ofrecerlas como si fueran buenas."""
    pr, est = clasificacion("tasa", top, horizonte)
    if not pr:
        return []
    val_u = universo_val or UNIVERSO_VAL
    out = []
    a = config_actual()
    v_act = validar_fuera(a["pesos"], a["umbral"], val_u, period, horizonte)
    out.append({"nombre": "ACTUAL (por defecto)", "pesos": a["pesos"], "umbral": a["umbral"],
                "origen": "la del proyecto, no sale de ninguna búsqueda",
                "busqueda": None, "validacion": v_act,
                "etiqueta": "✅ referencia",
                "aviso": "No se eligió optimizando sobre estos datos, así que no está "
                         "sobreajustada a ellos."})
    for i, p in enumerate(pr, 1):
        v = validar_fuera(p["pesos"], p["umbral"], val_u, period, horizonte)
        et, aviso = _veredicto_perfil(p, v)
        out.append({"nombre": f"#{i} acierto {p['tasa']*100:.1f}% (umbral {p['umbral']:.2f})",
                    "pesos": p["pesos"], "umbral": p["umbral"],
                    "origen": f"tanda {p.get('tanda', '?')} de la búsqueda",
                    "busqueda": {"tasa": p["tasa"], "sharpe": p["sr_oos"],
                                 "payoff": p.get("payoff"), "n_ops": p.get("n_ops"),
                                 "p_corr": p.get("p_corr")},
                    "validacion": v, "etiqueta": et, "aviso": aviso})
    PERFILES.parent.mkdir(parents=True, exist_ok=True)
    PERFILES.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


def informe_perfiles():
    ps = listar_perfiles()
    if not ps:
        return "No hay perfiles guardados. Lanza `--perfiles`."
    L = ["### Configuraciones guardadas\n",
         "| Perfil | veredicto | acierto en su búsqueda | acierto FUERA | Sharpe fuera |",
         "|---|---|---|---|---|"]
    for p in ps:
        b, v = p.get("busqueda"), p.get("validacion")
        tb = f"{b['tasa']*100:.1f}%" if b else "—"
        tv = f"{v['tasa']*100:.1f}%" if v else "—"
        sv = f"{v['sharpe']:+.2f}" if v else "—"
        L.append(f"| {p['nombre']} | {p['etiqueta']} | {tb} | **{tv}** | {sv} |")
    L.append("\n**La columna que importa es 'acierto FUERA'**: en su propio universo de "
             "búsqueda toda configuración parece buena, porque se eligió precisamente por "
             "eso. Lo que dice si sirve es cómo se porta en valores que no participaron.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Búsqueda honesta de configuración del Veredicto.")
    ap.add_argument("--n", type=int, default=10, help="Configuraciones nuevas por tanda.")
    ap.add_argument("--period", default="8y")
    ap.add_argument("--horizonte", type=int, default=10)
    ap.add_argument("--tickers", help="Lista separada por comas.")
    ap.add_argument("--estado", action="store_true", help="Solo muestra el acumulado.")
    ap.add_argument("--clasificacion", choices=["tasa", "sharpe"],
                    help="Ranking de TODAS las probadas, por acierto o por Sharpe.")
    ap.add_argument("--validar", action="store_true",
                    help="Valida la mejor por acierto en un universo distinto.")
    ap.add_argument("--perfiles", action="store_true",
                    help="Guarda las mejores CON su validación, para poder elegirlas.")
    ap.add_argument("--ver-perfiles", action="store_true", help="Lista los perfiles guardados.")
    ap.add_argument("--universo2", default="WMT,PG,CVX,MRK,HD,CAT,BA,PFE,T,VZ",
                    help="Universo de validación (no usado para elegir).")
    a = ap.parse_args()

    if a.clasificacion:
        print(informe_clasificacion(a.clasificacion, horizonte=a.horizonte))
        return

    if a.perfiles:
        ps = construir_perfiles(top=5, period=a.period, horizonte=a.horizonte)
        print(f"Guardados {len(ps)} perfiles en {PERFILES}\n")
        print(informe_perfiles())
        return

    if a.ver_perfiles:
        print(informe_perfiles())
        return

    if a.validar:
        pr, est = clasificacion("tasa", 1, a.horizonte)
        if not pr:
            print("Sin pruebas guardadas."); return
        m = pr[0]
        tks = [t.strip().upper() for t in a.universo2.replace(",", " ").split()]
        v = validar_fuera(m["pesos"], m["umbral"], tks, a.period, a.horizonte)
        print(f"Mejor por acierto: {m['tasa']*100:.1f}% en el universo de BÚSQUEDA "
              f"(Sharpe {m['sr_oos']:+.2f})")
        if not v:
            print("Sin datos en el universo de validación."); return
        print(f"La MISMA configuración en {v['tickers']} valores DISTINTOS:")
        print(f"  acierto {v['tasa']*100:.1f}% · payoff {v['payoff']:.2f} · "
              f"Sharpe {v['sharpe']:+.2f} · ret. medio {v['ret_medio']*100:+.2f}% "
              f"· {v['n_ops']} ops")
        caida = (m["tasa"] - v["tasa"]) * 100
        print(f"  caída de acierto: {caida:+.1f} puntos porcentuales")
        return

    if a.estado:
        est = cargar_estado()
        srs = [p["sr_oos"] for p in est["pruebas"] if p.get("sr_oos") is not None]
        print(f"Tandas: {est['tandas']} · configuraciones probadas: {len(est['pruebas'])}")
        if srs:
            mejor = max(est["pruebas"], key=lambda p: p.get("sr_oos") or -9)
            print(f"Mejor Sharpe OOS histórico: {mejor['sr_oos']:+.3f} (tanda {mejor['tanda']})")
        return

    tks = [t.strip().upper() for t in a.tickers.replace(",", " ").split()] if a.tickers else None
    print(informe(tanda(a.n, tks, a.period, a.horizonte)))


if __name__ == "__main__":
    main()
