"""
rebalance — cartera de LARGO PLAZO guiada: pesos HRP + órdenes de rebalanceo.

Convierte el HRP (asignación robusta) en un plan de acción real:
  1. crear(tickers, capital): calcula pesos HRP y te dice cuántas acciones comprar
     de cada valor HOY. Guarda la cartera en cartera_lp.csv.
  2. revisar(): lee tu cartera guardada, mira los precios de hoy, recalcula los
     pesos objetivo y te da las órdenes de ajuste ("compra 2 de X, vende 1 de Y")
     solo donde la desviación supera el umbral (por defecto 2.5%).

Precios auto-ajustados (dividendos y splits incluidos → total return).
No es recomendación de inversión.

Uso:
    python rebalance.py crear AAPL MSFT JPM KO GLD TLT --capital 10000
    python rebalance.py revisar
"""
import argparse
import sys
from datetime import date
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

_SUITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SUITE / "hrp_portfolio"))
CSV = Path(__file__).resolve().parent / "cartera_lp.csv"
UMBRAL_DRIFT = 0.025          # solo ajustar si el peso se desvía >2.5 puntos


def _precios_hoy(tickers):
    out = {}
    for tk in tickers:
        h = yf.Ticker(tk).history(period="5d", auto_adjust=True)
        if not h.empty:
            out[tk] = float(h["Close"].iloc[-1])
    return out


def _pesos_hrp(tickers, period="4y"):
    import hrp_portfolio as HP
    ret = HP._retornos(tickers, period)
    if ret.shape[1] < 3:
        raise ValueError("Necesito al menos 3 activos con histórico.")
    return HP.hrp(ret)


def crear(tickers, capital=10000.0, period="4y"):
    """Cartera nueva: pesos HRP → nº de acciones a comprar hoy. Guarda cartera_lp.csv."""
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    w = _pesos_hrp(tickers, period)
    px = _precios_hoy(list(w.index))
    filas = []
    for tk, peso in w.items():
        p = px.get(tk)
        if not p:
            continue
        acciones = int((capital * float(peso)) // p)
        filas.append({"Ticker": tk, "Peso HRP %": round(float(peso) * 100, 1),
                      "Precio": round(p, 2), "Acciones": acciones,
                      "Coste €": round(acciones * p, 2)})
    plan = pd.DataFrame(filas).sort_values("Peso HRP %", ascending=False).reset_index(drop=True)
    guardado = plan[plan["Acciones"] > 0][["Ticker", "Acciones"]].copy()
    guardado["fecha"] = date.today().isoformat()
    guardado.to_csv(CSV, index=False)
    invertido = float(plan["Coste €"].sum())
    meta = {"capital": capital, "invertido": round(invertido, 2),
            "cash": round(capital - invertido, 2), "n": len(guardado)}
    return plan, meta


def revisar(period="4y", umbral=UMBRAL_DRIFT):
    """Rebalanceo: compara tu cartera guardada con los pesos HRP de hoy → órdenes."""
    if not CSV.exists():
        raise ValueError("No hay cartera guardada. Crea una primero (botón 'Crear cartera').")
    cart = pd.read_csv(CSV)
    tickers = cart["Ticker"].tolist()
    px = _precios_hoy(tickers)
    cart["Precio"] = cart["Ticker"].map(px)
    cart = cart.dropna(subset=["Precio"])
    cart["Valor"] = cart["Acciones"] * cart["Precio"]
    total = float(cart["Valor"].sum())
    if total <= 0:
        raise ValueError("Cartera sin valor (¿precios no disponibles?).")
    cart["Peso actual"] = cart["Valor"] / total
    w_obj = _pesos_hrp(tickers, period)
    filas = []
    for _, r in cart.iterrows():
        tk = r["Ticker"]
        obj = float(w_obj.get(tk, 0.0))
        drift = float(r["Peso actual"]) - obj
        delta_eur = -drift * total
        delta_acc = int(round(delta_eur / r["Precio"]))
        if abs(drift) > umbral and delta_acc != 0:
            orden = f"{'COMPRAR' if delta_acc > 0 else 'VENDER'} {abs(delta_acc)}"
        else:
            orden = "— (dentro del margen)"
        filas.append({"Ticker": tk, "Acciones": int(r["Acciones"]),
                      "Peso actual %": round(float(r["Peso actual"]) * 100, 1),
                      "Peso objetivo %": round(obj * 100, 1),
                      "Desvío pp": round(drift * 100, 1), "Orden": orden})
    tabla = pd.DataFrame(filas).sort_values("Desvío pp", key=abs, ascending=False).reset_index(drop=True)
    n_ord = int((tabla["Orden"] != "— (dentro del margen)").sum())
    meta = {"valor_total": round(total, 2), "n_ordenes": n_ord,
            "fecha_cartera": str(cart["fecha"].iloc[0]) if "fecha" in cart else "?"}
    return tabla, meta


def aplicar_ordenes(tabla):
    """Actualiza cartera_lp.csv aplicando las órdenes de la tabla de revisar()."""
    cart = pd.read_csv(CSV).set_index("Ticker")
    for _, r in tabla.iterrows():
        o = str(r["Orden"])
        if o.startswith("COMPRAR"):
            cart.loc[r["Ticker"], "Acciones"] += int(o.split()[1])
        elif o.startswith("VENDER"):
            cart.loc[r["Ticker"], "Acciones"] = max(0, cart.loc[r["Ticker"], "Acciones"] - int(o.split()[1]))
    cart["fecha"] = date.today().isoformat()
    cart.reset_index().to_csv(CSV, index=False)
    return True


def comparar_dca(tickers, total=12000.0, anios=5):
    """DCA (aportar cada mes) vs entrada única, mismo dinero total, cesta
    equiponderada, precios total-return. Devuelve (tabla, meta, curvas)."""
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    px = {}
    for tk in tickers:
        h = yf.Ticker(tk).history(period=f"{anios}y", auto_adjust=True)
        if not h.empty:
            s = h["Close"].astype(float)
            s.index = pd.to_datetime(s.index).tz_localize(None)
            px[tk] = s
    if not px:
        raise ValueError("Sin datos.")
    df = pd.DataFrame(px).dropna()
    # cesta equiponderada normalizada
    cesta = (df / df.iloc[0]).mean(axis=1)
    meses = df.resample("MS").first().index
    meses = [m for m in meses if m >= df.index[0]]
    aporte = total / len(meses)
    # DCA: compra 'aporte' de cesta cada mes
    unidades = 0.0
    curva_dca = pd.Series(0.0, index=df.index)
    for m in meses:
        idx = df.index[df.index >= m]
        if len(idx) == 0:
            continue
        d0 = idx[0]
        unidades += aporte / float(cesta.loc[d0])
        curva_dca.loc[d0:] = unidades * cesta.loc[d0:]
    # aportado acumulado (para comparar contra lo invertido)
    aportado = pd.Series(0.0, index=df.index)
    acum = 0.0
    for m in meses:
        idx = df.index[df.index >= m]
        if len(idx):
            acum += aporte
            aportado.loc[idx[0]:] = acum
    # entrada única al principio
    unidades_ls = total / float(cesta.iloc[0])
    curva_ls = unidades_ls * cesta
    fin_dca, fin_ls = float(curva_dca.iloc[-1]), float(curva_ls.iloc[-1])
    tabla = pd.DataFrame([
        {"Estrategia": "DCA (aportar cada mes)", "Invertido €": round(acum, 0),
         "Valor final €": round(fin_dca, 0), "Ganancia %": round((fin_dca / acum - 1) * 100, 1)},
        {"Estrategia": "Entrada única (todo al inicio)", "Invertido €": total,
         "Valor final €": round(fin_ls, 0), "Ganancia %": round((fin_ls / total - 1) * 100, 1)},
    ])
    meta = {"gana": "DCA" if fin_dca / acum > fin_ls / total else "Entrada única",
            "n_meses": len(meses), "aporte_mes": round(aporte, 0)}
    return tabla, meta, (curva_dca, curva_ls, aportado)


def _plot_dca(curvas, tickers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dca, ls, ap = curvas
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(ls.index, ls.values, color="tab:blue", lw=1.4, label="Entrada única")
    ax.plot(dca.index, dca.values, color="tab:green", lw=1.4, label="DCA (mensual)")
    ax.plot(ap.index, ap.values, color="gray", ls="--", lw=1.0, label="Dinero aportado (DCA)")
    ax.set_title("DCA vs entrada única · " + ", ".join(tickers[:6]))
    ax.set_xlabel("Fecha"); ax.set_ylabel("Valor (€)"); ax.legend(loc="best")
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description="Cartera de largo plazo guiada (HRP + rebalanceo).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crear")
    c.add_argument("tickers", nargs="+")
    c.add_argument("--capital", type=float, default=10000.0)
    sub.add_parser("revisar")
    a = ap.parse_args()
    if a.cmd == "crear":
        plan, meta = crear(a.tickers, a.capital)
        print("\n" + plan.to_string(index=False))
        print(f"\nInvertido {meta['invertido']:.0f} € de {meta['capital']:.0f} € · queda cash {meta['cash']:.0f} €")
        print("Cartera guardada en cartera_lp.csv. Revísala cada mes con 'revisar'.\n")
    else:
        tabla, meta = revisar()
        print("\n" + tabla.to_string(index=False))
        print(f"\nValor actual {meta['valor_total']:.0f} € · {meta['n_ordenes']} órdenes de ajuste.")
        print("> Rebalancear ~1 vez al mes basta. No es recomendación de inversión.\n")


if __name__ == "__main__":
    main()
