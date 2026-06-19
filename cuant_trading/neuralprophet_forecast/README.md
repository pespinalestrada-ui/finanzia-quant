# neuralprophet_forecast

Forecast con **NeuralProphet** — el sucesor de Prophet sobre PyTorch. Combina la
descomposición interpretable de Prophet (tendencia + estacionalidad) con
componentes autorregresivos (**AR-Net**) entrenados como red neuronal.

## Uso
```bash
python neuralprophet_forecast.py SAB.MC
python neuralprophet_forecast.py AAPL --horizon 120 --epochs 60 --save
```

## Posición entre los motores
| Motor | Tipo | Memoria autorregresiva | Interpretable |
|-------|------|------------------------|---------------|
| Prophet | estadístico | no | sí |
| **NeuralProphet** | híbrido (red) | **sí (AR-Net)** | **sí** |
| LSTM | red pura | sí | no |
| AutoGluon | ensemble AutoML | según modelo | parcial |

## Detalles
- `n_lags=20`: ventana autorregresiva. `quantiles=[0.1,0.9]` → banda 80%.
- Estacionalidad semanal + anual activadas.
- `get_latest_forecast` colapsa la predicción multi-paso (diagonal) en una serie limpia.

## Instalación
```bash
pip install neuralprophet
```
Verificado compatible con torch 2.9 + transformers + autogluon del proyecto.

> No es recomendación de inversión.
