# factor_scorer

Modelo **multi-factor (factor investing / smart beta)** — la técnica que de verdad
usan los fondos cuant (AQR, ETFs smart-beta) para decidir qué comprar. En vez de
predecir el precio (que no bate al azar), rankea un universo por una nota compuesta
de **factores con premio de riesgo demostrado** (Fama-French y posteriores):

- **Value** — barato vs fundamentales (earnings yield 1/PER, book-to-price).
- **Momentum** — retorno 12-1 meses (lo que sube tiende a seguir). El más robusto.
- **Quality** — ROE alto, márgenes, poca deuda.
- **Low-vol** — menor volatilidad → mejor rentabilidad ajustada a riesgo (anomalía).

## Dos modos
1. **`rankear(universo)`** — z-score **cruzado** de cada factor dentro del universo →
   nota compuesta → ranking. Uso institucional: comprar el top, evitar el fondo.
   (En el dashboard: pestaña **📊 Factores**.)
2. **`score_absoluto(ticker)`** — nota [-1,+1] de UNA acción con umbrales fijos, que
   entra como **pilar en el Veredicto** (solo acciones; la cripto no tiene PER/ROE).

## Uso
```bash
python factor_scorer.py AAPL MSFT NVDA JPM XOM KO
python factor_scorer.py --file watchlist.txt
```

## Por qué es lo correcto
El factor investing tiene **evidencia empírica de décadas** (premios Nobel), a
diferencia de adivinar el precio. Es el núcleo replicable del proceso de un fondo
cuant. El límite honesto: con datos gratis no se replica el edge de *datos
alternativos* (satélite, tarjetas) que usan Two Sigma o Renaissance — pero el motor
de factores sí, y es académicamente sólido.

> Factores = premios de riesgo de LARGO plazo, no señales de timing.
> Fundamentales faltantes se tratan como neutros. No es recomendación de inversión.
