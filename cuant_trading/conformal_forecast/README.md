# conformal_forecast

Bandas de predicción **calibradas** con *split conformal prediction*. Arregla el
problema que el backtest demostró: la banda 80 % de Prophet solo cubría el 14-29 %
real (mentía), y el "nivel de confianza" `100−MAPE` no es una probabilidad.

## Idea
1. Walk-forward sobre el histórico → se recogen los **errores absolutos** del
   modelo a cada horizonte (30/90/120 d).
2. El cuantil (1−α) de esos errores da el **radio de banda** → cobertura garantizada
   ≈ (1−α) bajo intercambiabilidad (distribution-free, no asume normalidad).
3. Se **mide** la cobertura fuera de muestra (split cal/test) y se reporta: si sale
   ≈80 %, la banda dice la verdad.

## Uso
```bash
python conformal_forecast.py SAB.MC
python conformal_forecast.py AAPL --horizons 30 90 120 --alpha 0.2 --period 5y
```

## Por qué importa (defensa ante tribunal)
"La revisión de código detectó que mi banda estaba mal calibrada (cubría 14-29 %
en vez de 80 %). Lo arreglé con split conformal prediction, una técnica con
garantía teórica de cobertura, y **medí** que ahora cubre ≈75-80 % real." Eso es
rigor estadístico, no una demo que finge precisión.

> Forecast estadístico. No es recomendación de inversión.
