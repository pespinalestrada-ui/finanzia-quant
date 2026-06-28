# montecarlo

Simulación de **Monte Carlo**: el abanico de lo posible (no predicción). Dos usos:

1. **PRECIO** — miles de trayectorias futuras (bootstrap de retornos históricos o
   GBM) → distribución del precio a N días, percentiles, prob. de subir, VaR del
   camino. Bootstrap captura colas reales mejor que GBM.
2. **SISTEMA** — re-muestrea con reemplazo los resultados de tus operaciones (del
   diario, o de win-rate/payoff dados) → miles de curvas de equity → **prob. de
   acabar positivo**, distribución del **máximo drawdown** y **prob. de RUINA**. El
   modo correcto de medir si un sistema es robusto o tuvo suerte.

## Uso
```bash
python montecarlo.py precio AAPL --horizon 90 --sims 3000
python montecarlo.py sistema --winrate 0.55 --payoff 1.5 --trades 50
python montecarlo.py sistema --usar-diario        # con tus operaciones reales
```
En el dashboard: pestaña **🎲 Monte Carlo**.

## Qué mirar
En el sistema, la **prob. de ruina** y el **drawdown peor-5%** importan más que el
retorno medio: un sistema con buen retorno medio pero 20% de prob. de ruina no es
operable. No es recomendación de inversión.
