"""
cpcv — Validación Cruzada Combinatoria con Purgado y Embargo (guía §10.4)
       + Sharpe con error estándar y DSR con N EFECTIVO (guía §10.2 y §10.3).

Es el escalón que le faltaba a la suite: `veredicto_backtest` ya hacía PSR/DSR y
`alpha_forecast` ya purgaba, pero la validación era de UN camino (walk-forward).
La CPCV genera MUCHOS caminos fuera de muestra y mide la dispersión — que es
donde se ve si una estrategia es robusta o tuvo suerte en un tramo.

  · Divide la historia en N bloques y usa k como prueba → C(N,k) particiones y
        φ(N,k) = (k/N)·C(N,k)   trayectorias completas fuera de muestra.
  · PURGADO: quita del entrenamiento las muestras cuyo horizonte solapa el test.
  · EMBARGO: descarta además un trozo justo después del test (ecos de correlación).
  · PBO (Probability of Backtest Overfitting): con qué frecuencia la configuración
    que gana dentro de muestra queda por debajo de la mediana fuera de muestra.

DSR con N efectivo (§10.3): si pruebas M configuraciones muy parecidas no son M
pruebas independientes. Se corrige con  N_eff = ρ̂ + (1−ρ̂)·M.

No es recomendación de inversión.

Uso:
    python cpcv.py AAPL MSFT NVDA GOOGL JPM XOM
    python cpcv.py --file watchlist.txt --bloques 6 --prueba 2
"""
import argparse
import sys
from itertools import combinations
from math import comb, erf, sqrt, log
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import norm

_SUITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SUITE / "veredicto_backtest"))
EULER = 0.5772156649015329


# --- métricas honestas -------------------------------------------------------
def sharpe_con_error(r, periodos=252):
    """Sharpe anualizado + su error estándar (§10.2). r = retornos por periodo."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 20 or r.std(ddof=1) == 0:
        return {"sharpe": float("nan"), "se": float("nan"), "ic95": (float("nan"), float("nan")), "n": n}
    sr = r.mean() / r.std(ddof=1)                       # por periodo
    from scipy.stats import skew, kurtosis
    g3, g4 = float(skew(r)), float(kurtosis(r, fisher=False))
    # error estándar de Lo (2002) con corrección por asimetría y curtosis
    var_sr = (1 - g3 * sr + (g4 - 1) / 4.0 * sr ** 2) / (n - 1)
    se = sqrt(max(var_sr, 1e-12))
    esc = sqrt(periodos)
    return {"sharpe": float(sr * esc), "se": float(se * esc), "n": n,
            "ic95": (float((sr - 1.96 * se) * esc), float((sr + 1.96 * se) * esc))}


def n_efectivo(M, rho):
    """N efectivo de pruebas (§10.3): N = ρ̂ + (1−ρ̂)·M."""
    rho = min(max(float(rho), 0.0), 1.0)
    return max(1.0, rho + (1.0 - rho) * float(M))


def dsr(sr_obs, r, M=1, rho=0.0, var_sr_pruebas=None, solape=1):
    """
    Deflated Sharpe Ratio con N EFECTIVO. sr_obs y r por periodo (no anualizados).
    'solape': si cada retorno abarca h barras y las señales son diarias, las
    observaciones NO son independientes → T efectivo = T/h (si no se corrige, el
    DSR sale inflado y contradice al PBO).
    Devuelve dict(dsr, sr0, n_eff).
    """
    r = np.asarray(r, dtype=float); r = r[np.isfinite(r)]
    T = max(20, int(len(r) / max(1, int(solape))))     # observaciones efectivas
    if len(r) < 20:
        return {"dsr": float("nan"), "sr0": float("nan"), "n_eff": float("nan")}
    from scipy.stats import skew, kurtosis
    g3, g4 = float(skew(r)), float(kurtosis(r, fisher=False))
    N = n_efectivo(M, rho)
    V = var_sr_pruebas if (var_sr_pruebas and var_sr_pruebas > 0) else 1.0 / T
    if N >= 2:
        z1 = norm.ppf(1 - 1.0 / N)
        z2 = norm.ppf(1 - 1.0 / (N * np.e))
        sr0 = sqrt(V) * ((1 - EULER) * z1 + EULER * z2)
    else:
        sr0 = 0.0
    denom = sqrt(max(1e-12, 1 - g3 * sr_obs + (g4 - 1) / 4.0 * sr_obs ** 2))
    z = (sr_obs - sr0) * sqrt(max(1, T - 1)) / denom
    return {"dsr": float(norm.cdf(z)), "sr0": float(sr0), "n_eff": float(N)}


# --- CPCV --------------------------------------------------------------------
def _score_y_retornos(tickers, period="6y", horizonte=10):
    """Matriz de scores del Veredicto (point-in-time) y retornos futuros alineados."""
    import veredicto_backtest as VB
    scores, fwd = {}, {}
    for tk in tickers:
        df = VB.descargar(tk, period)
        if df is None or len(df) < 300:
            continue
        s = VB.score_historico(df)
        c = df["Close"]
        scores[tk] = s
        fwd[tk] = c.shift(-horizonte) / c - 1.0
    if len(scores) < 2:
        return None, None
    S = pd.DataFrame(scores).sort_index()
    R = pd.DataFrame(fwd).reindex(S.index)
    return S, R


def _retorno_estrategia(S, R, idx, umbral):
    """Retorno medio de operar señales |score|>umbral en las filas 'idx'."""
    s = S.iloc[idx].values
    r = R.iloc[idx].values
    m = np.isfinite(s) & np.isfinite(r) & (np.abs(s) > umbral)
    if m.sum() < 10:
        return np.array([])
    return np.sign(s[m]) * r[m]


def cpcv(tickers, period="6y", horizonte=10, n_bloques=6, k_prueba=2, embargo_pct=0.01,
         umbrales=(0.20, 0.30, 0.40, 0.50)):
    """
    CPCV sobre la familia de umbrales del Veredicto. Devuelve dict con:
    combinaciones, trayectorias φ(N,k), Sharpe OOS por camino, PBO y DSR deflactado.
    """
    S, R = _score_y_retornos(tickers, period, horizonte)
    if S is None:
        return {"mensaje": "Datos insuficientes (≥2 tickers con histórico)."}
    n = len(S)
    bloques = np.array_split(np.arange(n), n_bloques)
    n_comb = comb(n_bloques, k_prueba)
    phi = int(k_prueba / n_bloques * n_comb)
    emb = max(1, int(n * embargo_pct))

    filas = []
    for test_ids in combinations(range(n_bloques), k_prueba):
        idx_test = np.concatenate([bloques[i] for i in test_ids])
        # PURGADO: fuera del train todo lo que solapa el horizonte del test
        # EMBARGO: además, 'emb' barras justo después de cada bloque de test
        prohibido = set()
        for i in test_ids:
            b = bloques[i]
            ini, fin = b[0] - horizonte, b[-1] + horizonte + emb
            prohibido.update(range(max(0, ini), min(n, fin + 1)))
        idx_train = np.array([i for i in range(n) if i not in prohibido])
        if len(idx_train) < 100 or len(idx_test) < 20:
            continue
        # dentro de muestra: elegimos el mejor umbral; fuera: lo evaluamos
        mejor_u, mejor_sr = None, -np.inf
        srs_is = {}
        for u in umbrales:
            r_is = _retorno_estrategia(S, R, idx_train, u)
            sr = (r_is.mean() / r_is.std(ddof=1)) if len(r_is) > 20 and r_is.std(ddof=1) > 0 else -np.inf
            srs_is[u] = sr
            if sr > mejor_sr:
                mejor_sr, mejor_u = sr, u
        r_oos_mejor = _retorno_estrategia(S, R, idx_test, mejor_u)
        sr_oos = {}
        for u in umbrales:
            r_o = _retorno_estrategia(S, R, idx_test, u)
            sr_oos[u] = (r_o.mean() / r_o.std(ddof=1)) if len(r_o) > 10 and r_o.std(ddof=1) > 0 else np.nan
        filas.append({"test": test_ids, "mejor_umbral": mejor_u, "sr_is": mejor_sr,
                      "sr_oos_mejor": sr_oos.get(mejor_u, np.nan),
                      "sr_oos_todos": sr_oos,
                      "r_oos": r_oos_mejor})
    if not filas:
        return {"mensaje": "Sin particiones válidas (prueba menos bloques)."}

    # PBO: ¿el ganador dentro de muestra cae por debajo de la mediana fuera?
    peores = 0
    for f in filas:
        vals = [v for v in f["sr_oos_todos"].values() if np.isfinite(v)]
        if not vals or not np.isfinite(f["sr_oos_mejor"]):
            continue
        rank = np.mean([f["sr_oos_mejor"] <= v for v in vals])
        peores += 1 if rank > 0.5 else 0
    pbo = peores / max(1, len(filas))

    sr_oos_list = [f["sr_oos_mejor"] for f in filas if np.isfinite(f["sr_oos_mejor"])]
    todos_r = np.concatenate([f["r_oos"] for f in filas if len(f["r_oos"])]) if filas else np.array([])
    sr_medio = float(np.mean(sr_oos_list)) if sr_oos_list else float("nan")

    # DSR con N efectivo: M = nº de umbrales probados, ρ̂ = correlación media entre ellos
    corr_media = 0.0
    try:
        mat = pd.DataFrame([{u: f["sr_oos_todos"].get(u, np.nan) for u in umbrales} for f in filas]).corr()
        vals = mat.values[np.triu_indices(len(umbrales), 1)]
        corr_media = float(np.nanmean(vals)) if len(vals) else 0.0
    except Exception:
        pass
    var_pruebas = float(np.nanvar(sr_oos_list)) if len(sr_oos_list) > 1 else None
    d = dsr(sr_medio if np.isfinite(sr_medio) else 0.0, todos_r,
            M=len(umbrales), rho=corr_media, var_sr_pruebas=var_pruebas, solape=horizonte)
    # cada retorno abarca 'horizonte' barras → ~252/horizonte operaciones independientes al año
    sh = sharpe_con_error(todos_r, periodos=252.0 / max(1, horizonte))

    return {"n_obs": n, "n_bloques": n_bloques, "k_prueba": k_prueba,
            "combinaciones": n_comb, "trayectorias": phi, "particiones_validas": len(filas),
            "embargo_barras": emb, "horizonte": horizonte,
            "sr_oos_medio": sr_medio,
            "sr_oos_p5": float(np.percentile(sr_oos_list, 5)) if sr_oos_list else float("nan"),
            "sr_oos_p95": float(np.percentile(sr_oos_list, 95)) if sr_oos_list else float("nan"),
            "pbo": float(pbo), "corr_pruebas": corr_media,
            "sharpe_anual": sh["sharpe"], "sharpe_se": sh["se"], "ic95": sh["ic95"],
            "dsr": d["dsr"], "sr0": d["sr0"], "n_eff": d["n_eff"],
            "umbral_mas_elegido": pd.Series([f["mejor_umbral"] for f in filas]).mode().iloc[0]}


def informe(res):
    if "mensaje" in res:
        return res["mensaje"]
    ic = res["ic95"]
    L = [f"=== CPCV · {res['n_obs']:,} observaciones · {res['n_bloques']} bloques, {res['k_prueba']} de prueba ===\n",
         f"  Particiones C(N,k)   : {res['combinaciones']}  →  φ(N,k) = {res['trayectorias']} trayectorias completas",
         f"  Purgado + embargo    : horizonte {res['horizonte']} barras + {res['embargo_barras']} de embargo",
         f"  Umbral más elegido   : {res['umbral_mas_elegido']}\n",
         f"  Sharpe OOS medio     : {res['sr_oos_medio']:+.3f}  (P5 {res['sr_oos_p5']:+.3f} · P95 {res['sr_oos_p95']:+.3f})",
         f"  Sharpe anualizado    : {res['sharpe_anual']:+.2f} ± {res['sharpe_se']:.2f} "
         f"(IC95 {ic[0]:+.2f} a {ic[1]:+.2f})"
         f"{'  ⚠️ el IC cruza 0' if ic[0] < 0 < ic[1] else ''}",
         f"  PBO (sobreajuste)    : {res['pbo']*100:.0f}%  "
         f"→ {'ALTO: el ganador in-sample suele fallar fuera' if res['pbo'] > 0.5 else 'aceptable'}",
         f"  Pruebas efectivas    : {res['n_eff']:.1f} (M=4, correlación media {res['corr_pruebas']:.2f})",
         f"  Listón por azar SR₀  : {res['sr0']:.4f} por periodo",
         f"  DEFLATED SHARPE      : {res['dsr']*100:.0f}%  "
         f"→ {'EDGE REAL ✅' if res['dsr'] > 0.95 else 'NO supera el descuento por multiple-testing'}"]
    L.append("\n> El IC95 del Sharpe muestra el margen de error: si cruza 0, no puedes afirmar que ganas.")
    L.append("> PBO alto = elegir 'la mejor configuración' del pasado no funciona fuera. No es recomendación.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="CPCV (purgado+embargo) + Sharpe con error + DSR con N efectivo.")
    ap.add_argument("tickers", nargs="*",
                    default=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "JPM", "XOM", "KO"])
    ap.add_argument("--file")
    ap.add_argument("--period", default="6y")
    ap.add_argument("--horizonte", type=int, default=10)
    ap.add_argument("--bloques", type=int, default=6)
    ap.add_argument("--prueba", type=int, default=2)
    a = ap.parse_args()
    tickers = a.tickers or []
    if a.file and Path(a.file).exists():
        tickers = [t.strip() for t in Path(a.file).read_text().replace("\n", ",").split(",") if t.strip()]
    print(f"\nValidando con CPCV sobre {len(tickers)} tickers (puede tardar ~1 min)...")
    res = cpcv(tickers, a.period, a.horizonte, a.bloques, a.prueba)
    print("\n" + informe(res) + "\n")


if __name__ == "__main__":
    main()
