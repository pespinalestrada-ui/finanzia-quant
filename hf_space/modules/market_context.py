"""
market_context — termómetro del mercado: Fear & Greed + VIX + fundamentales.

Responde "¿cómo está el mercado HOY?" antes de mirar ningún ticker:
  - Fear & Greed Index de bolsa (réplica del CNN, vía feargreedchart.com,
    gratis sin clave): score 0-100 + sus 5 componentes + sectores.
  - Fear & Greed cripto (alternative.me, gratis sin clave).
  - VIX con lectura de régimen de volatilidad.
  - Fundamentales de un ticker opcional (PER, beta, dividendo, rango 52 sem).

Uso:
    python market_context.py            # solo mercado
    python market_context.py AAPL       # mercado + fundamentales de AAPL
"""
import argparse
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import warnings
warnings.filterwarnings("ignore")

import requests

UA = {"User-Agent": "Mozilla/5.0"}


def fear_greed_bolsa():
    """Score 0-100 + componentes. None si el endpoint falla."""
    try:
        r = requests.get("https://feargreedchart.com/api/?action=current",
                         headers=UA, timeout=10).json()
        sc = r.get("score", {})
        return {"score": sc.get("score"), "componentes": sc.get("components", []),
                "sectores": r.get("sectors", {})}
    except Exception:
        return None


def fear_greed_cripto():
    try:
        d = requests.get("https://api.alternative.me/fng/", headers=UA, timeout=10).json()["data"][0]
        return {"score": int(d["value"]), "etiqueta": d["value_classification"]}
    except Exception:
        return None


def etiqueta_fg(score):
    if score is None: return "n/d"
    if score < 25: return "MIEDO EXTREMO"
    if score < 45: return "Miedo"
    if score < 55: return "Neutral"
    if score < 75: return "Codicia"
    return "CODICIA EXTREMA"


def vix_regimen():
    try:
        import yfinance as yf
        v = float(yf.Ticker("^VIX").history(period="5d")["Close"].iloc[-1])
        if v < 15:   reg = "CALMA (complacencia — ojo, los sustos llegan desde aquí)"
        elif v < 20: reg = "NORMAL"
        elif v < 30: reg = "NERVIOSO (volatilidad elevada, reduce tamaño)"
        else:        reg = "PÁNICO (>30 — históricamente zona de oportunidad a largo)"
        return {"vix": round(v, 2), "regimen": reg}
    except Exception:
        return None


def fundamentales(ticker):
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        def g(k, fmt="{}"):
            v = info.get(k)
            return fmt.format(v) if v is not None else "n/d"
        return {
            "Nombre": info.get("shortName", ticker.upper()),
            "PER (trailing)": g("trailingPE", "{:.1f}"),
            "PER (forward)": g("forwardPE", "{:.1f}"),
            "P/Book": g("priceToBook", "{:.2f}"),
            "Beta": g("beta", "{:.2f}"),
            "Dividendo %": g("dividendYield", "{:.2f}"),
            "Market cap": g("marketCap", "{:,}"),
            "Mín 52sem": g("fiftyTwoWeekLow", "{:.2f}"),
            "Máx 52sem": g("fiftyTwoWeekHigh", "{:.2f}"),
            "Recomendación analistas": info.get("recommendationKey", "n/d"),
        }
    except Exception as e:
        return {"error": str(e)[:120]}


def lectura_conjunta(fg, vix):
    """Frase de contexto operativo combinando F&G y VIX."""
    if not fg or fg.get("score") is None or not vix:
        return "Datos incompletos para lectura conjunta."
    s, v = fg["score"], vix["vix"]
    if s < 30 and v > 25:
        return ("Miedo alto + VIX alto: mercado castigado. Históricamente zona de "
                "acumulación para largo plazo, pero con volatilidad dolorosa a corto.")
    if s > 70 and v < 15:
        return ("Codicia + VIX dormido: complacencia. Las correcciones nacen aquí — "
                "no es momento de apalancarse.")
    if 45 <= s <= 55:
        return "Mercado neutral: ni pánico ni euforia. El análisis por ticker pesa más que el contexto."
    return "Contexto mixto: usa el sesgo del mercado como viento a favor/en contra, no como señal."


def main():
    ap = argparse.ArgumentParser(description="Termómetro de mercado.")
    ap.add_argument("ticker", nargs="?", help="Fundamentales de este ticker (opcional).")
    a = ap.parse_args()

    print("\n=== 🌡️ Termómetro del mercado ===\n")
    fg = fear_greed_bolsa()
    if fg and fg.get("score") is not None:
        print(f"Fear & Greed BOLSA : {fg['score']}/100 → {etiqueta_fg(fg['score'])}")
        for c in fg.get("componentes", []):
            print(f"    {c['name']:<11} {c['val']:>3}  ({c['desc']})")
    else:
        print("Fear & Greed bolsa : no disponible ahora")
    cr = fear_greed_cripto()
    if cr:
        print(f"\nFear & Greed CRIPTO: {cr['score']}/100 → {cr['etiqueta']}")
    vx = vix_regimen()
    if vx:
        print(f"VIX                : {vx['vix']} → {vx['regimen']}")
    print(f"\nLectura: {lectura_conjunta(fg, vx)}")

    if a.ticker:
        print(f"\n=== Fundamentales {a.ticker.upper()} ===")
        for k, v in fundamentales(a.ticker).items():
            print(f"  {k:<24}: {v}")
    print()


if __name__ == "__main__":
    main()
