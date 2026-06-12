# FinanzIA — Forecast bursátil + Mesa cuantitativa

Sistema de análisis cuantitativo y forecasting de acciones, 100 % local y gratuito
(sin claves de API, sin servicios de pago). Incluye un caso de estudio completo
—forecast de Banco Sabadell (SAB.MC) durante la OPA hostil de BBVA— y una suite
de herramientas de trading algorítmico con dashboard web.

> ⚠️ **Proyecto educativo.** Nada de lo que produce este software constituye
> recomendación de inversión.

## ✨ Qué incluye

### 📓 Caso de estudio: forecast SAB.MC bajo la OPA de BBVA (`notebooks/`)

| Notebook | Contenido |
|----------|-----------|
| `01_descarga_eda` | Descarga con caché (yfinance) + EDA con eventos de la OPA |
| `02_forecast_prophet` | Prophet baseline vs Prophet + hitos OPA como *holidays* |
| `03_forecast_pycaret` | AutoML con PyCaret `time_series` vs Prophet |
| `03b_pycaret_horizontes` | Multi-horizonte: IBEX mensual + EUR/USD trimestral vía `merge_asof` |
| `04_regresores_externos` | Prophet + IBEX 35 + tipos BCE — el mejor modelo comparable |
| `05_escenarios_opa` | Tres escenarios: base / OPA exitosa / OPA fallida |
| `06_agente_explicacion` | Agente 3-roles que redacta el informe en lenguaje natural |
| `07_app_gradio` | App web del caso de estudio |

Resultado clave (holdout 90 días naturales, comparación homogénea):
modelizar los hitos de la OPA como *holidays* y añadir regresores macro
**reduce el MAPE de 20.6 % a 11.2 %** — casi la mitad del error del baseline.

### 🛠️ Suite cuantitativa (`cuant_trading/`)

| Herramienta | Qué hace |
|-------------|----------|
| `dashboard/` | **UI web única con todo** (12 pestañas, Gradio) |
| `indicators/` | RSI, MACD, Bollinger, ATR, SMA + señal actual |
| `screener/` | Ranking de watchlist por momentum/tendencia |
| `signal_scanner/` | Señales accionables: cruces, RSI extremo, breakouts |
| `backtester/` | Backtest SMA/RSI/Bollinger vs buy & hold |
| `position_sizer/` | Tamaño de posición por riesgo + fracción de Kelly |
| `portfolio_optimizer/` | Markowitz: máx Sharpe + frontera eficiente |
| `correlation/` | Matriz de correlación + diversificación |
| `sentiment/` | Sentimiento de noticias con FinBERT + NER (local) |
| `market_context/` | Fear & Greed (bolsa y cripto) + VIX + fundamentales |
| `journal/` | Diario de paper-trading con expectancy y Kelly |
| `autogluon_forecast/` | Forecast AutoML con cuantiles (incluye Chronos2) |

El dashboard incluye una pestaña **★ Veredicto** que ejecuta todos los motores
sobre un ticker y agrega un sesgo COMPRAR / MANTENER / VENDER con desglose
por pilar y nivel de confianza calibrado por backtest.

### ☁️ Despliegue en Hugging Face Spaces (`hf_space/`)

Paquete autocontenido listo para subir como Space privado de Gradio
(tier gratuito): el mismo dashboard accesible desde el móvil.

## 🚀 Instalación

```bash
pip install -r requirements.txt
# para sentiment y autogluon (opcionales, pesados):
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "transformers>=4.45,<5" autogluon.timeseries
```

## ▶️ Uso

```bash
# Dashboard completo
cd cuant_trading/dashboard
python dashboard.py          # → http://127.0.0.1:7862

# Herramientas CLI (cada carpeta tiene su README)
cd cuant_trading/screener && python screener.py AAPL MSFT SAB.MC
cd cuant_trading/backtester && python backtester.py AAPL --strategy sma --save
```

## 🧠 Decisiones de diseño

- **Todo local, todo gratis**: yfinance para datos (retardo ~15 min), modelos
  open-source (Prophet, FinBERT) ejecutados en CPU. Cero claves de API.
- **Confianza medida, no inventada**: el forecast reporta un nivel de confianza
  calculado con backtest por horizonte + ancho de banda; el diario de paper
  trading mide la expectancy real antes de arriesgar dinero.
- **Honestidad estadística**: los modelos AutoML que ganan en su holdout se
  reportan con sus cautelas (frecuencias distintas, ventaja de los lags); el
  ensemble de AutoGluon delata cuánto pesa el modelo "ingenuo".

## ⚠️ Limitaciones conocidas

- yfinance no es una API oficial de Yahoo: para uso personal/educativo.
- En Windows, `torch` debe importarse antes que `autogluon` (documentado en el código).
- Los datos de fin de día no sirven para intradía.

## 📄 Licencia

MIT — úsalo, modifícalo, aprende con él.
