# backtest_forecast

Mide la **efectividad real** del forecast: ¿acierta más que el azar? Validación
**walk-forward** (origen móvil), no un holdout único. Es la prueba honesta de si
la herramienta de forecast tiene ventaja predictiva.

## Métricas (rigurosas)
- **Acierto direccional (DA)** + **test binomial**: ¿acierta el signo del
  movimiento más que una moneda al aire? `z=(DA−0.5)/√(0.25/N)`, p<0.05 = bate al azar.
- **Theil's U2** = RMSE_modelo / RMSE_naive. **<1** bate al paseo aleatorio ("mañana≈hoy"); **≥1** no.
- **MAPE medio ± sd** sobre todos los orígenes (no un número suelto).
- **Cobertura empírica de la banda 80%**: ¿la banda dice la verdad?

## Uso
```bash
python backtest_forecast.py AAPL
python backtest_forecast.py AAPL SAB.MC MSFT --horizons 30 90 --origins 14
```

## Resultado (5 años, 14 orígenes) — la verdad sin maquillaje
| Ticker | Horiz | DA | p | U2 | Cobertura | Veredicto |
|--------|-------|----|---|----|-----------|-----------|
| AAPL | 30 | 36% | 0.86 | 1.43 | 14% | sin ventaja |
| AAPL | 90 | 50% | 0.50 | 1.14 | — | sin ventaja |
| SAB.MC | 90 | 50% | 0.50 | 3.33 | 14% | sin ventaja |
| MSFT | 90 | 14% | 1.00 | 1.62 | 21% | sin ventaja |

**Conclusión honesta:** el forecast de precios a 30/90 días **no bate al azar ni al
paseo aleatorio**, y la banda de confianza está mal calibrada (cubre mucho menos
del 80% nominal). Esto **no es un fallo de la herramienta** — es la realidad de
los mercados: a esos horizontes el precio es casi un paseo aleatorio. Reportarlo
es lo que distingue un análisis serio de una demo que finge acertar.

## Implicación para el dashboard
- El "nivel de confianza" del forecast (basado en `100−MAPE` + ancho de banda) es
  una heurística, no una probabilidad. Tómalo como orientación, no como verdad.
- El Veredicto COMPRAR/MANTENER/VENDER es un resumen técnico, **no una predicción
  validada**. Úsalo con el diario de paper-trading para medir TU expectancy real.

> No es recomendación de inversión.
