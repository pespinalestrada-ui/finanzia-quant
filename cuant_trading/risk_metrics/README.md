# risk_metrics

Riesgo de cartera medido con rigor — lo que **sí** es estimable, frente a la
dirección del precio (que no bate al azar). Para una watchlist, por ticker y para
la cartera equiponderada:

- **Volatilidad anualizada** (σ·√252).
- **VaR 95 % / 99 %** histórico: pérdida diaria que solo se supera el 5 % / 1 % de los días.
- **CVaR / Expected Shortfall 95 %**: pérdida **media** en ese peor 5 % (lo que el VaR ignora).
- **Máximo drawdown**: peor caída pico-valle.
- **Matriz de correlación**: diversificación real entre activos.

VaR/CVaR **históricos** (distribución empírica de retornos, sin asumir normalidad).

## Uso
```bash
python risk_metrics.py AAPL MSFT SAB.MC
python risk_metrics.py SAB.MC BBVA.MC IBE.MC --period 3y --conf 0.99
```

## Lectura
- CVaR siempre es peor que el VaR: es lo que pierdes **cuando** rebasas el VaR.
- Correlación media baja (<0.4) entre los activos = la cartera diversifica de verdad
  (su volatilidad será menor que la media de las individuales).

> No es asesoramiento de inversión.
