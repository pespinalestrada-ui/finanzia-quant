"""
news_feeds — agregador de noticias multi-fuente, GRATIS y SIN CLAVE.

Amplía la cobertura del análisis de sentimiento (antes solo yfinance) usando varias
fuentes públicas de RSS, la alternativa legal a depender de una Terminal Bloomberg
reenviada (propietaria, prohíbe redistribución). Solo `requests` + `xml.etree`
(sin dependencias nuevas).

Fuentes:
  - Yahoo Finance RSS  (por ticker exacto; va bien con .MC, .DE, etc.).
  - Google News RSS    (búsqueda por nombre/símbolo; cobertura amplia).
  - yfinance .news     (la que ya se usaba).

Devuelve [{'fecha': date|None, 'titular': str, 'fuente': str}] deduplicado y
ordenado por fecha descendente.

Uso directo:
    python news_feeds.py AAPL
    python news_feeds.py SAB.MC --max 20
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import warnings
warnings.filterwarnings("ignore")

import re
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

import requests

UA = {"User-Agent": "Mozilla/5.0 (compatible; FinanzIA/1.0)"}
_NOMBRE_CACHE = {}


def _parse_fecha(pub):
    """RFC822 ('Wed, 25 Jun 2026 14:30:00 GMT') o ISO → date. None si falla."""
    if not pub:
        return None
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(pub).date()
    except Exception:
        pass
    try:
        return datetime.fromisoformat(pub.replace("Z", "+00:00")).date()
    except Exception:
        return None


def _rss(url):
    """Descarga un RSS y devuelve [(titular, fecha, fuente_item)]."""
    out = []
    try:
        r = requests.get(url, headers=UA, timeout=10)
        if r.status_code != 200 or not r.content:
            return out
        root = ET.fromstring(r.content)
        for item in root.iter("item"):
            titulo = (item.findtext("title") or "").strip()
            if not titulo:
                continue
            fecha = _parse_fecha(item.findtext("pubDate"))
            src = item.findtext("source") or ""
            out.append((titulo, fecha, src.strip()))
    except Exception:
        pass
    return out


def _nombre(ticker):
    """Nombre de la empresa para la búsqueda en Google News (best-effort, cacheado)."""
    t = ticker.upper()
    if t in _NOMBRE_CACHE:
        return _NOMBRE_CACHE[t]
    nombre = None
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        nombre = info.get("shortName") or info.get("longName")
    except Exception:
        pass
    if not nombre:                                  # fallback: símbolo sin sufijo de mercado
        nombre = re.split(r"[.\-]", t)[0]
    _NOMBRE_CACHE[t] = nombre
    return nombre


def _yahoo(ticker, max_news):
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    return [{"fecha": f, "titular": ti, "fuente": "Yahoo"} for ti, f, _ in _rss(url)[:max_news]]


def _google(ticker, max_news):
    q = requests.utils.quote(f'"{_nombre(ticker)}" stock')
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    res = []
    for ti, f, src in _rss(url)[:max_news]:
        # Google News añade " - Fuente" al final del titular; lo dejamos, FinBERT lo ignora
        res.append({"fecha": f, "titular": ti, "fuente": f"Google/{src}" if src else "Google"})
    return res


def _yfinance(ticker, max_news):
    try:
        import yfinance as yf
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []
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


def _norm(t):
    """Normaliza un titular para deduplicar (minúsculas, sin puntuación ni fuente final)."""
    t = re.sub(r"\s+-\s+[^-]+$", "", t)             # quita " - Reuters" final de Google News
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def obtener_noticias(ticker, max_news=15):
    """Agrega Yahoo + Google News + yfinance, deduplica y ordena por fecha desc."""
    por_fuente = max(6, max_news)
    todas = _yahoo(ticker, por_fuente) + _google(ticker, por_fuente) + _yfinance(ticker, por_fuente)
    vistos, unicas = set(), []
    for n in todas:
        clave = _norm(n["titular"])
        if not clave or clave in vistos:
            continue
        vistos.add(clave)
        unicas.append(n)
    # orden por fecha desc (None al final)
    unicas.sort(key=lambda n: (n["fecha"] is not None, n["fecha"] or datetime.min.date()), reverse=True)
    return unicas[:max_news]


def resumen_fuentes(noticias):
    from collections import Counter
    c = Counter(n["fuente"].split("/")[0] for n in noticias)
    return ", ".join(f"{k}: {v}" for k, v in c.most_common())


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Agregador de noticias multi-fuente sin clave.")
    ap.add_argument("ticker", nargs="?", default="AAPL")
    ap.add_argument("--max", type=int, default=15)
    a = ap.parse_args()
    print(f"\nAgregando noticias de {a.ticker.upper()} (Yahoo + Google News + yfinance)...")
    noticias = obtener_noticias(a.ticker, a.max)
    if not noticias:
        print("Sin noticias ahora mismo. Prueba otro ticker.")
        return
    print(f"{len(noticias)} titulares únicos · fuentes → {resumen_fuentes(noticias)}\n")
    for n in noticias:
        f = n["fecha"].isoformat() if n["fecha"] else "  —  "
        print(f"  [{f}] ({n['fuente']}) {n['titular'][:95]}")
    print()


if __name__ == "__main__":
    main()
