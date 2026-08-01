"""
screener — escáner de una watchlist.

Para cada ticker calcula momentum 1m/3m, RSI(14), distancia a la SMA50,
volatilidad (ATR%) y tendencia (SMA50 vs SMA200). Ordena por un score
compuesto de momentum + fuerza de tendencia. Útil para barrido diario.

Uso:
    python screener.py AAPL MSFT NVDA SAB.MC BBVA.MC
    python screener.py --file watchlist.txt --sort momentum
    (watchlist.txt = un ticker por línea)
"""
import argparse
import sys
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import pandas as pd
import yfinance as yf

try:                                      # dentro del panel, kpis ya está en el path
    import kpis as _KP
except ImportError:                       # ejecutado suelto
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kpis"))
    try:
        import kpis as _KP
    except Exception:
        _KP = None                        # sin ROIC, pero el escáner sigue funcionando


def _rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def analizar(ticker, con_roic=True):
    """Fila del escáner. `con_roic` añade la calidad del negocio (2 llamadas de red
    más por ticker, ~0,3 s; quedan cacheadas). Ponlo a False si quieres el barrido
    puramente técnico y rápido."""
    h = yf.Ticker(ticker).history(period="1y", auto_adjust=False)
    if h.empty or len(h) < 60:
        return None
    # yfinance devuelve a veces la última fila vacía (pasa con tickers del BME):
    # sin dropna, el precio y todos los momentums salían nulos en la tabla
    c = h["Close"].astype(float).dropna()
    if len(c) < 60:
        return None
    px = float(c.iloc[-1])
    sma50 = c.rolling(50).mean().iloc[-1]
    sma200 = c.rolling(200).mean().iloc[-1] if len(c) >= 200 else np.nan
    mom1m = (px / c.iloc[-21] - 1) * 100 if len(c) > 21 else np.nan
    mom3m = (px / c.iloc[-63] - 1) * 100 if len(c) > 63 else np.nan
    rsi = float(_rsi(c).iloc[-1])
    ret = c.pct_change()
    atr_pct = float(ret.rolling(14).std().iloc[-1] * np.sqrt(252) * 100)  # vol anualizada %
    dist_sma50 = (px / sma50 - 1) * 100
    tend = 1 if (not np.isnan(sma200) and sma50 > sma200) else (-1 if not np.isnan(sma200) else 0)
    # score: momentum medio + bonus tendencia alcista, penaliza RSI extremo.
    # El ROIC NO entra aquí a propósito: este Score es de TIMING (momentum y
    # tendencia) y mezclarle un dato contable anual cambiaría su significado sin
    # avisar. Va como columna aparte: sirve para descartar "sube mucho pero el
    # negocio no gana dinero", que es justo lo que no ve un escáner técnico.
    score = np.nanmean([mom1m, mom3m]) + tend * 5 - max(0, abs(rsi - 50) - 30) * 0.2
    fila = dict(Ticker=ticker.upper(), Precio=round(px, 3), Mom1m=round(mom1m, 1),
                Mom3m=round(mom3m, 1), RSI=round(rsi, 1), VolAnual=round(atr_pct, 1),
                vsSMA50=round(dist_sma50, 1), Tend=("ALC" if tend > 0 else "BAJ" if tend < 0 else "?"),
                Score=round(score, 1))
    if con_roic and _KP is not None:
        _u, med, n = _KP.roic_ticker(ticker)
        fila["ROIC"] = round(med * 100, 1) if med == med else np.nan   # mediana de n ejercicios
        fila["Calidad"] = ("—" if med != med else
                           "crea valor" if med > 0.15 else
                           "ok" if med > 0.08 else "destruye")
    return fila


def main():
    ap = argparse.ArgumentParser(description="Escáner de watchlist.")
    ap.add_argument("tickers", nargs="*", help="Tickers separados por espacio.")
    ap.add_argument("--file", help="Fichero con un ticker por línea.")
    ap.add_argument("--sort", default="score", choices=["score", "momentum", "rsi", "roic"], help="Criterio de orden.")
    ap.add_argument("--sin-roic", action="store_true", help="Barrido solo técnico (más rápido).")
    a = ap.parse_args()

    tickers = list(a.tickers)
    if a.file:
        with open(a.file, encoding="utf-8") as f:
            tickers += [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if not tickers:
        tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "SAB.MC", "BBVA.MC"]

    print(f"\nAnalizando {len(tickers)} tickers...\n")
    rows = []
    for t in tickers:
        try:
            r = analizar(t, con_roic=not a.sin_roic)
            if r:
                rows.append(r)
            else:
                print(f"  (sin datos suficientes: {t})")
        except Exception as e:
            print(f"  (error {t}: {e})")

    if not rows:
        raise SystemExit("Nada que mostrar.")
    df = pd.DataFrame(rows)
    key = {"score": "Score", "momentum": "Mom3m", "rsi": "RSI", "roic": "ROIC"}[a.sort]
    if key not in df.columns:
        key = "Score"
    df = df.sort_values(key, ascending=False).reset_index(drop=True)
    df.index += 1
    print(df.to_string())
    print("\nLeyenda: Mom=momentum % | vsSMA50=distancia a media 50 | VolAnual=volatilidad anualizada % | "
          "Tend=tendencia | ROIC=rentabilidad del capital que trabaja (mediana de los últimos ejercicios; "
          "por encima de ~10% crea valor, por debajo lo destruye; vacío en bancos, donde no aplica)\n"
          "El Score NO incluye el ROIC: es un score de timing. Mira las dos cosas juntas.\n")


if __name__ == "__main__":
    main()
