# sentiment

Análisis de sentimiento financiero de noticias con **FinBERT** + extracción de
entidades (NER). Reconstrucción limpia y funcional del notebook
`ANALISIS DE SENTIMIENTO.docx` del curso.

## Qué hace
1. Descarga las noticias actuales del ticker (yfinance).
2. Clasifica cada titular con `ProsusAI/finbert`: **positive / negative / neutral** + confianza.
3. Extrae entidades (empresas, personas) con `dbmdz/bert-large-cased-finetuned-conll03-english`.
4. Score agregado en [-1, +1] ponderado por confianza → veredicto POSITIVO/NEUTRAL/NEGATIVO.

## Uso
```bash
python sentiment_news.py AAPL
python sentiment_news.py SAB.MC --max-news 15
python sentiment_news.py --demo        # titulares de ejemplo, sin red
```

**Primera ejecución descarga los modelos de Hugging Face (~1.6 GB, una sola vez,
quedan cacheados en `~/.cache/huggingface`).** Después va rápido (CPU).

## Arreglos respecto al notebook original
- El código del `.docx` estaba **corrompido por el traductor** ("importar yfinance
  como yf", "modelo=", "si ... de lo contrario") — inejacutable tal cual. Reescrito.
- **API de noticias de yfinance cambió**: el notebook usaba `providerPublishTime`
  (epoch) y `title` planos; yfinance moderno los anida en `item['content']` con
  `pubDate` ISO. Esta versión soporta ambos formatos.
- Añadido score agregado + cruce con el precio de la última sesión.

## Conexión con el proyecto final
FinBERT es de la familia de LLMs financieros que cita la rúbrica (FinGPT). Esta
herramienta cubre el "trabajo futuro" del informe (sección 8.2): sentimiento
sobre noticias como señal complementaria al forecast.
