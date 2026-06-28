"""
sentiment_news — análisis de sentimiento financiero de noticias con FinBERT.

Reconstrucción limpia del notebook "ANALISIS DE SENTIMIENTO" (Colab):
  1. Descarga las noticias actuales del ticker desde yfinance.
  2. Clasifica cada titular con FinBERT (ProsusAI/finbert): positive/negative/neutral.
  3. Extrae entidades nombradas (empresas, personas, lugares) con un modelo NER.
  4. Agrega un score de sentimiento global y lo cruza con el precio.

Cambios respecto al notebook original:
  - El código del .docx estaba corrompido por el traductor automático
    ("importar yfinance como yf", "modelo=", "si ... de lo contrario") — reescrito.
  - yfinance moderno (>=0.2.50) devuelve las noticias anidadas en `item['content']`
    con `pubDate` ISO en lugar de `providerPublishTime` epoch. Se soportan ambos
    formatos.
  - Añadido score agregado y modo --demo con titulares de ejemplo (como la
    primera parte del notebook).

Uso:
    python sentiment_news.py AAPL
    python sentiment_news.py SAB.MC --max-news 15
    python sentiment_news.py --demo          (titulares de ejemplo, sin red)

Primera ejecución descarga los modelos de Hugging Face (~1.6 GB, solo una vez).
"""
import argparse
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timezone
import pandas as pd

SENT_MODEL = "ProsusAI/finbert"
NER_MODEL = "dbmdz/bert-large-cased-finetuned-conll03-english"

_sent = None
_ner = None


def cargar_modelos():
    """Carga perezosa de los pipelines (la primera vez descarga los pesos)."""
    global _sent, _ner
    if _sent is None:
        from transformers import pipeline
        print("Cargando FinBERT (sentimiento financiero)...")
        _sent = pipeline("sentiment-analysis", model=SENT_MODEL)
        print("Cargando modelo NER (entidades)...")
        _ner = pipeline("ner", model=NER_MODEL, aggregation_strategy="simple")
    return _sent, _ner


def extraer_noticias(ticker_symbol, max_news=10):
    """Noticias normalizadas a [{'fecha','titular','fuente'}].

    Usa el agregador multi-fuente (Yahoo RSS + Google News RSS + yfinance), gratis y
    sin clave. Si falla, recurre a yfinance solo (comportamiento antiguo).
    """
    try:
        from news_feeds import obtener_noticias
        agg = obtener_noticias(ticker_symbol, max_news)
        if agg:
            return agg
    except Exception:
        pass
    # fallback: solo yfinance (formato viejo y nuevo)
    import yfinance as yf
    raw = (yf.Ticker(ticker_symbol).news or [])
    out = []
    for item in raw[:max_news]:
        content = item.get("content", item)
        titular = content.get("title")
        if not titular:
            continue
        fecha = None
        pub = content.get("pubDate") or content.get("displayTime")
        if pub:
            try:
                fecha = datetime.fromisoformat(pub.replace("Z", "+00:00")).date()
            except ValueError:
                pass
        if fecha is None:
            ts = item.get("providerPublishTime")
            if ts:
                fecha = datetime.fromtimestamp(ts, tz=timezone.utc).date()
        out.append({"fecha": fecha, "titular": titular, "fuente": "yfinance"})
    return out


def analizar(noticias):
    """FinBERT + NER sobre cada titular. Devuelve DataFrame."""
    sent, ner = cargar_modelos()
    rows = []
    for n in noticias:
        titular = n["titular"]
        s = sent(titular[:512])[0]
        ents = ner(titular[:512])
        entidades = ", ".join(dict.fromkeys(e["word"] for e in ents)) or "—"
        rows.append({
            "Fecha": n["fecha"].isoformat() if n["fecha"] else "—",
            "Fuente": n.get("fuente", "—").split("/")[0],
            "Titular": titular[:90] + ("…" if len(titular) > 90 else ""),
            "Sentimiento": s["label"],
            "Conf": round(float(s["score"]), 2),
            "Entidades": entidades,
        })
    return pd.DataFrame(rows)


import math

def score_global(df):
    """Score agregado en [-1, +1]: ponderado por confianza y decaimiento exponencial (noticias viejas pesan menos)."""
    if df.empty:
        return 0.0, "SIN DATOS"
    w = {"positive": 1, "negative": -1, "neutral": 0}
    
    suma_scores = 0.0
    suma_pesos = 0.0
    
    from datetime import date
    hoy = date.today()
    
    for _, r in df.iterrows():
        dias_antiguedad = 0
        if r["Fecha"] != "—":
            try:
                fecha_obj = date.fromisoformat(r["Fecha"])
                dias_antiguedad = max(0, (hoy - fecha_obj).days)
            except Exception:
                pass
        
        # lambda = 0.15 -> la vida media de una noticia es ~4.6 días
        peso_tiempo = math.exp(-0.15 * dias_antiguedad)
        peso_total = r["Conf"] * peso_tiempo
        
        suma_scores += w.get(r["Sentimiento"], 0) * peso_total
        suma_pesos += peso_total
        
    s = suma_scores / suma_pesos if suma_pesos > 0 else 0.0
    veredicto = ("POSITIVO" if s > 0.15 else "NEGATIVO" if s < -0.15 else "NEUTRAL")
    return round(s, 3), veredicto


DEMO = [
    {"fecha": datetime(2026, 5, 20).date(), "titular": "Apple launches new AI features boosting stock prices."},
    {"fecha": datetime(2026, 5, 22).date(), "titular": "EU fines Apple over App Store antitrust violations."},
    {"fecha": datetime(2026, 5, 23).date(), "titular": "Tim Cook announces spectacular revenue growth this quarter."},
]


def main():
    ap = argparse.ArgumentParser(description="Sentimiento de noticias con FinBERT.")
    ap.add_argument("ticker", nargs="?", default="AAPL")
    ap.add_argument("--max-news", type=int, default=10)
    ap.add_argument("--demo", action="store_true", help="Usa titulares de ejemplo (sin red).")
    a = ap.parse_args()

    if a.demo:
        print("\nModo demo — 3 titulares de ejemplo (los del notebook original).\n")
        noticias = DEMO
    else:
        print(f"\nExtrayendo noticias de yfinance para {a.ticker.upper()}...")
        noticias = extraer_noticias(a.ticker, a.max_news)
        if not noticias:
            print("Sin noticias disponibles ahora mismo. Prueba --demo o otro ticker.")
            return
        print(f"{len(noticias)} noticias encontradas.\n")

    df = analizar(noticias)
    print(df.to_string(index=False))

    s, veredicto = score_global(df)
    pos = (df["Sentimiento"] == "positive").sum()
    neg = (df["Sentimiento"] == "negative").sum()
    neu = (df["Sentimiento"] == "neutral").sum()
    print(f"\n=== Sentimiento agregado: {veredicto} (score {s:+.3f}) ===")
    print(f"    {pos} positivas · {neg} negativas · {neu} neutrales")

    if not a.demo:
        try:
            import yfinance as yf
            px = yf.Ticker(a.ticker).history(period="5d")["Close"]
            if len(px) >= 2:
                var = (px.iloc[-1] / px.iloc[-2] - 1) * 100
                print(f"    Última sesión {a.ticker.upper()}: {px.iloc[-1]:.2f} ({var:+.2f}%)")
        except Exception:
            pass
    print("\n> Sentimiento de titulares ≠ recomendación. Úsalo como una señal más (ver signal_scanner).\n")


if __name__ == "__main__":
    main()
