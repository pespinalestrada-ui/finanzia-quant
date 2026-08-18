"""Pruebas del marcador, con respuesta conocida de antemano.

No comprueban que "no pete": comprueban que el estadístico NO da por habilidad
lo que es solo mercado alcista, y que no encuentra ventaja donde no la hay.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import marcador as M


def _sintetico(serie, senal="COMPRAR", n=60, paso=21):
    """Construye llamadas sobre una serie dada y devuelve la tabla resuelta."""
    idx = serie.index
    filas = []
    for i in range(0, len(idx) - 95, paso):
        f = idx[i]
        fila = {c: np.nan for c in M.COLS}
        fila.update({"id": f"T{i}", "fecha": f.date().isoformat(), "fuente": "test",
                     "origen": "vivo", "ticker": "TEST", "senal": senal,
                     "score": 0.5, "precio_0": float(serie.iloc[i])})
        for h in M.PLAZOS:
            j = min(i + h, len(serie) - 1)
            fila[f"precio_{h}"] = float(serie.iloc[j])
        filas.append(fila)
        if len(filas) >= n:
            break
    return pd.DataFrame(filas)[M.COLS]


def test_serie_siempre_sube():
    """Sube SIEMPRE lo mismo: acierto 100%, tasa base 100%, VENTAJA 0 y exceso 0.

    Si el exceso sobre la deriva no sale ~0, el marcador estaría dando por
    habilidad lo que es puramente mercado alcista. Es la prueba que importa."""
    idx = pd.bdate_range("2018-01-01", periods=1100)
    s = pd.Series(100 * (1.0005 ** np.arange(1100)), index=idx)
    M._PX["TEST"] = s
    M._PX["SPY"] = s.copy()
    d = _sintetico(s, "COMPRAR")
    res = M.marcador(df=d)
    f30 = [x for x in res["filas"] if x["Plazo"] == "30 días"][0]
    assert f30["Acierto %"] == 100.0, f30
    assert f30["Tasa base %"] == 100.0, f30
    assert abs(f30["Ventaja pp"]) < 1e-6, f"ventaja deberia ser 0, es {f30['Ventaja pp']}"
    assert abs(f30["Exceso deriva %"]) < 0.05, \
        f"exceso deberia ser ~0 en una serie de deriva constante, es {f30['Exceso deriva %']}"
    print("OK 1 · serie siempre alcista -> ventaja 0 y exceso 0")


def test_aleatorio_sin_deriva():
    """Paseo aleatorio: ~50% de acierto y el test NO debe encontrar ventaja."""
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2018-01-01", periods=900)
    s = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 900))), index=idx)
    M._PX["TEST"] = s
    M._PX["SPY"] = s.copy()
    d = _sintetico(s, "COMPRAR", n=40)
    res = M.marcador(df=d)
    f30 = [x for x in res["filas"] if x["Plazo"] == "30 días"][0]
    assert f30["Veredicto"] == "sin diferencia", \
        f"en ruido puro no puede declarar ventaja: {f30}"
    assert 25 <= f30["Acierto %"] <= 75, f30
    print(f"OK 2 · ruido puro -> '{f30['Veredicto']}' (acierto {f30['Acierto %']}%)")


def test_muestra_insuficiente():
    """Con pocas llamadas no se da porcentaje: un 70% sobre 7 es ruido."""
    idx = pd.bdate_range("2022-01-01", periods=300)
    s = pd.Series(100 * (1.001 ** np.arange(300)), index=idx)
    M._PX["TEST"] = s
    d = _sintetico(s, "COMPRAR", n=6)
    res = M.marcador(df=d)
    f30 = [x for x in res["filas"] if x["Plazo"] == "30 días"][0]
    assert f30["Acierto %"] is None, "con n<30 no puede dar porcentaje"
    assert "insuficiente" in f30["Veredicto"]
    print("OK 3 · n<30 -> 'muestra insuficiente', sin porcentaje")


def test_vender_se_puntua_al_reves():
    """Un VENDER en una serie que sube debe acertar 0%, no 100%."""
    idx = pd.bdate_range("2018-01-01", periods=1100)
    s = pd.Series(100 * (1.0008 ** np.arange(1100)), index=idx)
    M._PX["TEST"] = s
    M._PX["SPY"] = s.copy()
    d = _sintetico(s, "VENDER")
    res = M.marcador(df=d)
    f30 = [x for x in res["filas"] if x["Plazo"] == "30 días"][0]
    assert f30["Acierto %"] == 0.0, f"un VENDER en serie alcista acierta 0%: {f30}"
    assert f30["Tasa base %"] == 0.0, "el liston de un VENDER es la prob. de BAJAR"
    print("OK 4 · VENDER se puntua al reves y su tasa base tambien")


def test_mantener_no_puntua():
    """Los MANTENER se apuntan pero no entran en el acierto direccional."""
    idx = pd.bdate_range("2018-01-01", periods=1100)
    s = pd.Series(100 * (1.0008 ** np.arange(1100)), index=idx)
    M._PX["TEST"] = s; M._PX["SPY"] = s.copy()
    d1 = _sintetico(s, "COMPRAR")
    d2 = _sintetico(s, "MANTENER")
    d2["id"] = d2["id"] + "b"
    juntas = pd.concat([d1, d2], ignore_index=True)
    a = M.marcador(df=d1)["filas"][1]
    b = M.marcador(df=juntas)["filas"][1]
    assert a["Acierto %"] == b["Acierto %"], "los MANTENER no deben mover el acierto"
    assert b["_n"] == a["_n"], "solo cuentan COMPRAR/VENDER en el direccional"
    print("OK 5 · los MANTENER se guardan pero no puntuan")


def test_n_efectivo():
    """El solape tiene que reducir la muestra, nunca aumentarla."""
    assert M._n_efectivo(90, 90) == 21.0, M._n_efectivo(90, 90)
    assert M._n_efectivo(90, 30) == 63.0, M._n_efectivo(90, 30)
    assert M._n_efectivo(90, 7) == 90.0, "a 7 dias no hay que inflar la muestra"
    print("OK 6 · n efectivo corrige el solape y nunca lo infla")


def test_sistema_malo_se_llama_malo():
    """Un sistema deliberadamente equivocado tiene que salir como PEOR.

    No vale con un VENDER sobre una serie que solo sube: ahi el acierto es 0%
    pero la tasa base de bajar tambien es 0%, asi que la ventaja es cero y
    'sin diferencia' es la respuesta correcta. Hay que construir un sistema que
    lo haga peor QUE SU PROPIO LISTON: se emite la senal contraria a lo que va a
    pasar de verdad.

    Con un test de una sola cola esto salia con p~1 y se anunciaba como bueno.
    Paso de verdad en la primera reconstruccion: -16 pp de ventaja etiquetados
    como 'ventaja significativa'."""
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2018-01-01", periods=1400)
    s = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.011, 1400))), index=idx)
    M._PX["TEST"] = s; M._PX["SPY"] = s.copy()
    filas = []
    for i in range(0, len(s) - 95, 21):
        r30 = s.iloc[i + 30] / s.iloc[i] - 1.0
        fila = {c: np.nan for c in M.COLS}
        fila.update({"id": f"M{i}", "fecha": idx[i].date().isoformat(), "fuente": "test",
                     "origen": "vivo", "ticker": "TEST",
                     "senal": "VENDER" if r30 > 0 else "COMPRAR",   # siempre al reves
                     "score": -0.5, "precio_0": float(s.iloc[i])})
        for h in M.PLAZOS:
            fila[f"precio_{h}"] = float(s.iloc[min(i + h, len(s) - 1)])
        filas.append(fila)
    d = pd.DataFrame(filas)[M.COLS]
    f30 = [x for x in M.marcador(df=d)["filas"] if x["Plazo"] == "30 dias".replace("dias", "días")][0]
    assert f30["Acierto %"] == 0.0, f30
    assert f30["Veredicto"] == "PEOR que el activo solo", f30
    print(f"OK 7 · sistema al reves -> '{f30['Veredicto']}' "
          f"(acierto {f30['Acierto %']}% vs base {f30['Tasa base %']}%)")


if __name__ == "__main__":
    test_serie_siempre_sube()
    test_aleatorio_sin_deriva()
    test_muestra_insuficiente()
    test_vender_se_puntua_al_reves()
    test_mantener_no_puntua()
    test_n_efectivo()
    test_sistema_malo_se_llama_malo()
    print("\nTODAS OK")
