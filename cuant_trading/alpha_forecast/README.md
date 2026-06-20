# alpha_forecast

Predicción **direccional a corto plazo** + **volatilidad (GARCH)** con técnicas
avanzadas, atacando lo que SÍ tiene señal medible — en vez del precio a 90 días,
que no bate al azar (ver `backtest_forecast`).

## Técnicas
- **ML / ingeniería de datos:** LightGBM clasificador sobre features leak-free
  (retornos lag, momentum, RSI, volatilidad realizada, z-scores, volumen),
  **walk-forward purgado con embargo** (López de Prado) — sin fuga de etiquetas,
  escalado solo en train, balance de clases, calibración de probabilidad.
- **Física / series:** exponente de **Hurst** (R/S) como feature de régimen
  (tendencia vs reversión), volatilidad realizada, rango intradía.
- **Econometría:** **GARCH(1,1)** walk-forward one-step para la volatilidad;
  test binomial de significancia; AUC.

## Uso
```bash
python alpha_forecast.py AAPL
python alpha_forecast.py AAPL SAB.MC MSFT --horizon 5 --period 8y
```

## Resultado (8 años, walk-forward purgado) — la verdad
| Ticker | Dirección 5d (DA) | p | AUC | GARCH corr |
|--------|-------------------|---|-----|-----------|
| AAPL | 50.3% | 0.42 | 0.50 | 0.11 |
| SAB.MC | 50.7% | 0.29 | 0.50 | 0.03 |
| MSFT | **51.9%** | 0.078 | 0.50 | **0.24** |

**Lo que demuestra:**
1. **Mejora real vs el precio a 90d:** ese estaba *por debajo* del azar (DA 14-50%,
   U2>1, peor que paseo aleatorio). La dirección a 5d con ML está *en* el azar
   (50-52%) — pasó de "peor que moneda" a "moneda". Eso ya es un avance medible.
2. **No hay edge significativo** en large-caps líquidas (p>0.05 en todos; MSFT
   roza con p=0.078). Esto es **eficiencia de mercado, medida con rigor** — no un
   fallo de la herramienta.
3. **La volatilidad sí tiene algo de predictibilidad** (MSFT corr 0.24) — útil para
   dimensionar posición y calibrar bandas, más que para predecir dirección.

## Por qué NO se "fuerza" un número ganador
Probar muchas configuraciones hasta que una salga significativa por azar es
**p-hacking / sobreajuste** — el error que la revisión de código advirtió. Aquí se
fija una configuración razonable, se mide honestamente, y se reporta el resultado
sea cual sea. Un acierto del 52% sin significancia NO es una estrategia.

> Conclusión defendible para un tribunal: "apliqué ML con boosting, features
> ingenierizadas, validación purgada y GARCH; medí que en acciones líquidas no
> existe edge direccional fácil — eficiencia de mercado — y que la volatilidad es
> lo único modestamente predecible." Eso es ciencia, no una demo que finge acertar.

> No es recomendación de inversión.
