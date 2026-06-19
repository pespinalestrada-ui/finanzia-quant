# lstm_forecast

Forecast con **red neuronal LSTM** (PyTorch). Aprende patrones de la secuencia
de precios con una ventana deslizante y proyecta de forma recursiva.

## Uso
```bash
python lstm_forecast.py SAB.MC
python lstm_forecast.py AAPL --horizon 120 --window 30 --epochs 120 --save
```

## Cómo funciona
- Normaliza el precio a [0,1], ventanas de `--window` días → predice el siguiente.
- LSTM (1→32, 1 capa) + capa lineal, entrenada desde cero en cada llamada (~segundos CPU).
- Forecast recursivo: realimenta su propia predicción 120 días.
- Banda 80% por desviación de residuos del holdout, ensanchada como paseo aleatorio (√paso).

## Honestidad
Las redes recurrentes sobre precio tienden a **"seguir" el último valor** (lag):
el forecast suele salir plano y las bandas anchas. Eso es la incertidumbre real,
no un defecto — un activo a 120 días es bruma. Compáralo con Prophet/NeuralProphet
en la pestaña Forecast del dashboard.

> No requiere instalar nada extra (torch ya viene con el proyecto).
