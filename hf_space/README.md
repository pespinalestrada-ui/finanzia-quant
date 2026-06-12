---
title: FinanzIA — Mesa Cuantitativa
emoji: 📈
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: "5.49.1"
app_file: app.py
pinned: false
license: mit
---

# FinanzIA — Mesa cuantitativa

Dashboard personal de análisis cuantitativo: forecast Prophet con nivel de
confianza, indicadores técnicos, screener, señales, backtest, correlación,
cartera Markowitz, sentimiento de noticias con FinBERT, tamaño de posición,
diario de paper-trading, veredicto agregado y termómetro de mercado
(Fear & Greed + VIX).

**Uso personal y educativo. No es recomendación de inversión.**

Notas de este Space:
- El **📒 Diario** es efímero (se borra si el Space se reinicia). Para diario
  persistente, usa la versión local.
- La pestaña **Sentimiento** descarga FinBERT (~1.6 GB) en el primer uso tras
  cada reinicio — la primera consulta tarda varios minutos.
- Motor AutoGluon no incluido (excede el tier gratuito); disponible en local.
