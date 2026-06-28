"""
signal_engine — generador de SEÑALES del sistema (qué operar hoy).

Corre el score TÉCNICO del Veredicto (el mismo que valida `veredicto_backtest`,
point-in-time) sobre una watchlist y devuelve un ranking COMPRAR / MANTENER /
VENDER. Es la entrada del bucle de ejecución (sizing → Alpaca paper → diario).

Opcionalmente añade la nota de FACTORES (value/momentum/quality/low-vol) como
segunda capa de convicción.

Honestidad: el backtest mostró que el núcleo técnico NO supera el multiple-testing
en large-caps líquidas → trata estas señales como un marco de PAPER para medir tu
expectancy, no como alfa garantizado. No es recomendación de inversión.

Uso:
    python signal_engine.py AAPL MSFT NVDA GOOGL AMZN JPM XOM KO
    python signal_engine.py --file watchlist.txt --umbral 0.35 --factor
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
for p in (_SUITE / "veredicto_backtest", _SUITE / "factor_scorer"):
    sys.path.insert(0, str(p))
import veredicto_backtest as VB


def _senal(score, umbral):
    if score >= umbral:
        return "COMPRAR", "🟢"
    if score <= -umbral:
        return "VENDER", "🔴"
    return "MANTENER", "🟡"


def generar(tickers, umbral=0.35, con_factor=False, period="2y"):
    """Devuelve DataFrame rankeado por score técnico del Veredicto (último valor)."""
    FS = None
    if con_factor:
        try:
            import factor_scorer as FS
        except Exception:
            FS = None
    filas = []
    for tk in tickers:
        tk = tk.strip().upper()
        if not tk:
            continue
        try:
            df = VB.descargar(tk, period)
            if df is None or len(df) < 220:
                continue
            s = VB.score_historico(df).dropna()
            if s.empty:
                continue
            score = float(s.iloc[-1])
            px = float(df["Close"].iloc[-1])
            senal, emoji = _senal(score, umbral)
            fila = {"Ticker": tk, "Precio": round(px, 3), "Score": round(score, 3),
                    "Señal": f"{emoji} {senal}"}
            if FS is not None:
                try:
                    fac, _, _ = FS.score_absoluto(tk)
                    fila["Factor"] = round(fac, 3)
                except Exception:
                    fila["Factor"] = np.nan
            filas.append(fila)
        except Exception:
            continue
    df = pd.DataFrame(filas)
    if not df.empty:
        df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser(description="Generador de señales del sistema (ranking del Veredicto).")
    ap.add_argument("tickers", nargs="*",
                    default=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "XOM", "KO", "WMT"])
    ap.add_argument("--file")
    ap.add_argument("--umbral", type=float, default=0.35)
    ap.add_argument("--factor", action="store_true", help="Añade la nota de factores.")
    a = ap.parse_args()
    tickers = a.tickers or []
    if a.file and Path(a.file).exists():
        tickers = [l.strip() for l in Path(a.file).read_text().splitlines() if l.strip()]

    print(f"\nGenerando señales sobre {len(tickers)} tickers (umbral ±{a.umbral})...")
    df = generar(tickers, a.umbral, a.factor)
    if df.empty:
        print("Sin señales (datos insuficientes)."); return
    print("\n" + df.to_string(index=False))
    comprar = df[df["Señal"].str.contains("COMPRAR")]["Ticker"].tolist()
    vender = df[df["Señal"].str.contains("VENDER")]["Ticker"].tolist()
    print(f"\n  COMPRAR: {', '.join(comprar) if comprar else '—'}")
    print(f"  VENDER : {', '.join(vender) if vender else '—'}")
    print("\n> Score técnico del Veredicto (validado en veredicto_backtest). En líquidas no")
    print("> supera multiple-testing → úsalo como marco de PAPER, no como alfa garantizado.\n")


if __name__ == "__main__":
    main()
