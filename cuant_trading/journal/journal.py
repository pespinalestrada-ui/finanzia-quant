"""
journal — diario de operaciones (paper trading).

Registra operaciones SIMULADAS, las cierra con su precio de salida y calcula
estadísticas honestas: win rate, expectancy, R-múltiplo medio y payoff.
Tras 20-30 operaciones sabrás si tu método tiene ventaja real ANTES de
arriesgar dinero. El win rate y el payoff resultantes son exactamente los
inputs que necesita la fracción de Kelly del position_sizer.

Persistencia: CSV en esta misma carpeta (operaciones.csv).

Uso CLI:
    python journal.py abrir AAPL --entrada 295.6 --stop 285 --acciones 10 --nota "veredicto COMPRAR +0.57"
    python journal.py cerrar 1 --salida 310.2
    python journal.py lista
    python journal.py stats
"""
import argparse
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

CSV = Path(__file__).resolve().parent / "operaciones.csv"
COLS = ["id", "fecha_apertura", "ticker", "direccion", "entrada", "stop",
        "acciones", "nota", "fecha_cierre", "salida", "pnl", "r_multiplo", "estado"]


def _load() -> pd.DataFrame:
    if CSV.exists():
        df = pd.read_csv(CSV)
        for c in COLS:
            if c not in df.columns:
                df[c] = np.nan
        return df[COLS]
    return pd.DataFrame(columns=COLS)


def _save(df: pd.DataFrame):
    df.to_csv(CSV, index=False)


def abrir(ticker, entrada, stop, acciones, nota="", direccion="LONG"):
    """Registra una operación simulada abierta. Devuelve el id asignado."""
    entrada, stop, acciones = float(entrada), float(stop), int(acciones)
    if direccion == "LONG" and stop >= entrada:
        raise ValueError("En un LONG el stop debe estar por debajo de la entrada.")
    if direccion == "SHORT" and stop <= entrada:
        raise ValueError("En un SHORT el stop debe estar por encima de la entrada.")
    df = _load()
    nid = int(df["id"].max()) + 1 if len(df) else 1
    fila = {
        "id": nid, "fecha_apertura": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "ticker": ticker.upper().strip(), "direccion": direccion,
        "entrada": entrada, "stop": stop, "acciones": acciones, "nota": nota,
        "fecha_cierre": "", "salida": np.nan, "pnl": np.nan,
        "r_multiplo": np.nan, "estado": "ABIERTA",
    }
    df = pd.concat([df, pd.DataFrame([fila])], ignore_index=True)
    _save(df)
    return nid


def cerrar(op_id, salida):
    """Cierra una operación con su precio de salida. Devuelve (pnl, r_multiplo)."""
    df = _load()
    mask = (df["id"] == int(op_id)) & (df["estado"] == "ABIERTA")
    if not mask.any():
        raise ValueError(f"No hay operación ABIERTA con id {op_id}.")
    i = df.index[mask][0]
    salida = float(salida)
    entrada = float(df.at[i, "entrada"]); stop = float(df.at[i, "stop"])
    acciones = int(df.at[i, "acciones"]); direc = df.at[i, "direccion"]
    signo = 1 if direc == "LONG" else -1
    pnl = (salida - entrada) * acciones * signo
    riesgo = abs(entrada - stop) * acciones
    r = pnl / riesgo if riesgo > 0 else np.nan
    df.at[i, "fecha_cierre"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    df.at[i, "salida"] = salida
    df.at[i, "pnl"] = round(pnl, 2)
    df.at[i, "r_multiplo"] = round(r, 2)
    df.at[i, "estado"] = "CERRADA"
    _save(df)
    return pnl, r


def lista() -> pd.DataFrame:
    return _load()


def estadisticas() -> dict:
    """Estadísticas sobre operaciones cerradas. Incluye inputs para Kelly."""
    df = _load()
    cerradas = df[df["estado"] == "CERRADA"].copy()
    n = len(cerradas)
    if n == 0:
        return {"n_cerradas": 0, "mensaje": "Aún no hay operaciones cerradas. Mínimo razonable: 20-30 para juzgar."}
    pnl = cerradas["pnl"].astype(float)
    r = cerradas["r_multiplo"].astype(float)
    wins = cerradas[pnl > 0]
    losses = cerradas[pnl <= 0]
    win_rate = len(wins) / n
    g_media = float(wins["pnl"].mean()) if len(wins) else 0.0
    p_media = float(abs(losses["pnl"].mean())) if len(losses) else 0.0
    payoff = g_media / p_media if p_media > 0 else np.nan
    expectancy_r = float(r.mean())
    kelly = win_rate - (1 - win_rate) / payoff if payoff and not np.isnan(payoff) and payoff > 0 else np.nan
    return {
        "n_cerradas": n,
        "win_rate": round(win_rate * 100, 1),
        "pnl_total": round(float(pnl.sum()), 2),
        "ganancia_media": round(g_media, 2),
        "perdida_media": round(p_media, 2),
        "payoff": round(payoff, 2) if not np.isnan(payoff) else None,
        "expectancy_R": round(expectancy_r, 2),
        "mejor_R": round(float(r.max()), 2),
        "peor_R": round(float(r.min()), 2),
        "kelly_pct": round(kelly * 100, 1) if not np.isnan(kelly) else None,
        "kelly_medio_pct": round(kelly * 50, 1) if not np.isnan(kelly) else None,
    }


def stats_texto() -> str:
    s = estadisticas()
    if s["n_cerradas"] == 0:
        return s["mensaje"]
    veredicto = ("✅ Expectancy positiva: el método gana en promedio."
                 if s["expectancy_R"] > 0 else
                 "⛔ Expectancy negativa: con estas reglas PIERDES en promedio. No pases a real.")
    fiable = "" if s["n_cerradas"] >= 20 else f"\n⚠️ Solo {s['n_cerradas']} operaciones — estadística poco fiable aún (mínimo 20-30)."
    out = [
        f"Operaciones cerradas : {s['n_cerradas']}",
        f"Win rate             : {s['win_rate']}%",
        f"P&L total (simulado) : {s['pnl_total']:+.2f} €",
        f"Ganancia media       : {s['ganancia_media']:.2f} € · Pérdida media: {s['perdida_media']:.2f} €",
        f"Payoff (G/P)         : {s['payoff']}",
        f"Expectancy           : {s['expectancy_R']:+.2f} R por operación",
        f"Mejor / peor         : {s['mejor_R']:+.2f}R / {s['peor_R']:+.2f}R",
    ]
    if s["kelly_pct"] is not None:
        out.append(f"Kelly (completa/½)   : {s['kelly_pct']}% / {s['kelly_medio_pct']}%  → úsalo en position_sizer")
    out.append("")
    out.append(veredicto + fiable)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="Diario de operaciones simuladas.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("abrir")
    a.add_argument("ticker"); a.add_argument("--entrada", type=float, required=True)
    a.add_argument("--stop", type=float, required=True)
    a.add_argument("--acciones", type=int, required=True)
    a.add_argument("--nota", default=""); a.add_argument("--short", action="store_true")

    c = sub.add_parser("cerrar")
    c.add_argument("id", type=int); c.add_argument("--salida", type=float, required=True)

    sub.add_parser("lista")
    sub.add_parser("stats")

    args = ap.parse_args()
    if args.cmd == "abrir":
        nid = abrir(args.ticker, args.entrada, args.stop, args.acciones,
                    args.nota, "SHORT" if args.short else "LONG")
        print(f"Operación #{nid} abierta ({args.ticker.upper()}).")
    elif args.cmd == "cerrar":
        pnl, r = cerrar(args.id, args.salida)
        print(f"Operación #{args.id} cerrada: P&L {pnl:+.2f} € ({r:+.2f}R).")
    elif args.cmd == "lista":
        df = lista()
        print(df.to_string(index=False) if len(df) else "Diario vacío.")
    elif args.cmd == "stats":
        print(stats_texto())


if __name__ == "__main__":
    main()
