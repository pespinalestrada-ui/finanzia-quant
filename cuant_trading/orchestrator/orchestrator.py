"""
orchestrator — el SISTEMA completo en un sitio: señales → plan/riesgo → ejecución
paper → diario.

Une las piezas del flujo algorítmico:
  1. signal_engine  → qué operar (ranking del Veredicto).
  2. risk_manager   → cuánto (vol targeting + stop ATR + tope de riesgo).
  3. alpaca_paper   → ejecuta el plan en PAPER (dinero ficticio, real-time).
  4. journal        → registra cada orden con su nota de factores → expectancy real.

SEGURIDAD: `ejecutar()` manda órdenes (de papel) y por eso la dispara EL USUARIO
(botón del dashboard o CLI con --ejecutar). El asistente nunca opera por su cuenta.
`plan_de_hoy()` es solo lectura.

Uso:
    python orchestrator.py AAPL MSFT NVDA GOOGL AMZN JPM XOM KO --capital 10000
    python orchestrator.py --file watchlist.txt --capital 10000 --ejecutar   # manda paper
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

_SUITE = Path(__file__).resolve().parents[1]
for p in (_SUITE / "risk_manager", _SUITE / "alpaca_paper", _SUITE / "journal",
          _SUITE / "factor_scorer", _SUITE / "signal_engine"):
    sys.path.insert(0, str(p))
import risk_manager as RKM


def plan_de_hoy(tickers, capital=10000.0, target_vol=0.15, max_pos=5, umbral=0.35, atr_mult=2.0):
    """Solo lectura: genera el plan de órdenes del día. Devuelve (plan_df, meta)."""
    return RKM.generar_plan(tickers, capital, target_vol, max_pos, atr_mult, umbral)


def ejecutar(plan_df, registrar=True):
    """
    Manda el plan a Alpaca PAPER y (opcional) lo registra en el diario.
    La dispara el USUARIO. Devuelve (resultados_df, resumen_str).
    """
    import alpaca_paper as AP
    import journal as JR
    try:
        import factor_scorer as FS
    except Exception:
        FS = None
    if plan_df is None or plan_df.empty:
        return pd.DataFrame(), "Plan vacío: nada que ejecutar."
    if not AP.configurada():
        return pd.DataFrame(), "Faltan claves de Alpaca (ALPACA_KEY/ALPACA_SECRET) en el .env."
    # freno de pérdida diaria: si hoy la cuenta pierde más del límite, no se opera
    try:
        fr = AP.freno_diario()
        if fr["bloqueado"]:
            return pd.DataFrame(), (f"🛑 FRENO DIARIO ACTIVO: la cuenta paper pierde {fr['pct']}% hoy "
                                    f"(límite −{fr['limite_pct']}%). No se envían órdenes hasta mañana.")
    except Exception:
        pass

    res = []
    for _, r in plan_df.iterrows():
        tk = r["Ticker"]; lado = r["Lado"]; shares = int(r["Acciones"])
        side = "buy" if lado == "LONG" else "sell"
        estado, nid = "—", "—"
        try:
            o = AP.enviar_orden(tk, shares, side)
            estado = o.get("estado", "?")
            if registrar:
                nf = None
                if FS is not None:
                    try:
                        nf, _, _ = FS.score_absoluto(tk)
                    except Exception:
                        nf = None
                try:
                    nid = JR.abrir(tk, float(r["Precio"]), float(r["Stop"]), shares,
                                   f"Sistema (orden {o.get('id','?')})", lado, nf)
                except Exception as ej:
                    nid = f"err: {ej}"
        except Exception as e:
            estado = f"rechazada: {e}"
        res.append({"Ticker": tk, "Lado": lado, "Acciones": shares,
                    "Estado Alpaca": estado, "Diario #": nid})
    rdf = pd.DataFrame(res)
    ok = rdf["Estado Alpaca"].astype(str).str.contains("accepted|new|filled|pending", case=False).sum()
    resumen = f"Enviadas {len(rdf)} órdenes PAPER · {ok} aceptadas. Registradas en el diario para medir expectancy."
    return rdf, resumen


def main():
    ap = argparse.ArgumentParser(description="Orquestador del sistema (señales→plan→paper→diario).")
    ap.add_argument("tickers", nargs="*",
                    default=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "XOM", "KO", "WMT"])
    ap.add_argument("--file")
    ap.add_argument("--capital", type=float, default=10000.0)
    ap.add_argument("--target-vol", type=float, default=0.15)
    ap.add_argument("--max-pos", type=int, default=5)
    ap.add_argument("--umbral", type=float, default=0.35)
    ap.add_argument("--ejecutar", action="store_true", help="Manda las órdenes a Alpaca PAPER.")
    a = ap.parse_args()
    tickers = a.tickers or []
    if a.file and Path(a.file).exists():
        tickers = [l.strip() for l in Path(a.file).read_text().splitlines() if l.strip()]

    print(f"\nSISTEMA · plan de hoy sobre {len(tickers)} tickers (capital {a.capital:.0f} €)...")
    plan, meta = plan_de_hoy(tickers, a.capital, a.target_vol, a.max_pos, a.umbral)
    print("\n" + RKM.informe(plan, meta))
    if not a.ejecutar:
        print("\n[SIMULACRO] No se envió nada. Añade --ejecutar para mandar el plan a Alpaca PAPER.\n")
        return
    if plan.empty:
        print("\nNada que ejecutar.\n"); return
    rdf, resumen = ejecutar(plan, registrar=True)
    print("\n=== Ejecución PAPER ===")
    print(rdf.to_string(index=False))
    print(f"\n{resumen}")
    print("> Dinero ficticio. Las órdenes las disparaste tú. No es recomendación.\n")


if __name__ == "__main__":
    main()
