# correlation

Matriz de correlación de los retornos de una cesta. Para no llevar varias
posiciones que en el fondo son la misma apuesta.

## Uso
```bash
python correlation.py AAPL MSFT NVDA TLT GLD
python correlation.py SAB.MC BBVA.MC ITX.MC REP.MC --period 2y --save
```

## Qué devuelve
- **Matriz de correlación** de retornos diarios.
- **Correlación media por activo** — menor = mejor diversificador.
- Par **más** y **menos** correlacionado de la cesta.
- Con `--save`: heatmap (verde = baja correlación, rojo = alta).

## Cómo usarlo
- Correlación **> 0.7** entre dos posiciones = riesgo duplicado.
- Mezcla activos de baja correlación (ej. acciones + bonos `TLT` + oro `GLD`)
  para suavizar la curva de capital.
