# autogluon_forecast

Forecast AutoML con **AutoGluon TimeSeries** — implementación local del notebook
`Forecasting_Sabadell_IBEX_BancaEuropea110`. Entrena automáticamente 7 familias
de modelos (SeasonalNaive, ETS, Theta, tabulares, **Chronos2** fundacional,
TemporalFusionTransformer) y los combina en un ensemble ponderado.

## Diferencial vs el forecast Prophet de la suite
- **Cuantiles 0.1–0.9**: distribución completa de incertidumbre, no solo un intervalo.
- **Chronos2 incluido**: el modelo fundacional de forecast que el informe del
  proyecto cita como benchmark futuro — aquí ya corre.
- **Covariables opcionales**: IBEX 35 + banca europea (`EXV1.DE`) como *past covariates*.
- Contra: tarda lo que el `--time-limit` (minutos), Prophet va en segundos.

## Uso
```bash
python autogluon_forecast.py SAB.MC
python autogluon_forecast.py SAB.MC --horizon 110 --covariables --save
python autogluon_forecast.py AAPL --horizon 90 --preset fast_training --time-limit 60
```

## Salida
- **Leaderboard** de modelos (quién gana por MASE).
- Tabla por horizonte (30/90/110): mediana, P10, P90, variación %, **confianza
  por ancho de banda** (<20% ALTA · 20-45% MEDIA · >45% BAJA).
- Con `--save`: gráfico con doble banda (P10-P90 y P30-P70) + CSV de cuantiles.

## Hallazgo honesto del notebook original
En ambas corridas del notebook (con y sin covariables) el ensemble dio **~54-58 %
del peso a SeasonalNaive** — el modelo "ingenuo". Y las covariables IBEX/banca
**no mejoraron** el MASE (-1.6181 vs -1.6197). Lección: en series casi
paseo-aleatorio, lo simple compite muy duro. Coherente con lo visto en el
proyecto (NB 03b).
