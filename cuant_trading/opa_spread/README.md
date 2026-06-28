# opa_spread

Spread de **arbitraje de fusión** (merger-arb) de la OPA hostil de BBVA sobre
Sabadell — el ángulo diferencial del proyecto convertido en señal medible y
tradeable (a diferencia del precio a 90 d, que no bate al azar).

## Qué calcula
La OPA es un **canje de acciones**, así que existe un precio implícito de la oferta:

```
valor_oferta_por_SAB = (1 / canje) · precio_BBVA + efectivo_por_SAB
spread = precio_SAB / valor_oferta − 1
```

- **spread < 0** → SAB por debajo de la oferta: el mercado descuenta riesgo de no
  completarse. Hueco de arbitraje si crees que la OPA sale.
- **spread ≈ 0** → operación casi descontada a los términos actuales.
- **spread > 0** → mercado espera mejora de oferta / contra-opa.

**Probabilidad implícita de éxito** (modelo de 2 estados):
`precio_SAB = p·valor_oferta + (1−p)·precio_fracaso` → se despeja `p`.

## Uso
```bash
python opa_spread.py
python opa_spread.py --canje 4.83 --efectivo 0.70 --periodo 3y --fracaso 1.74
```

## Nota importante
Los términos del canje son **parámetros** (`--canje`, `--efectivo`): ajústalos a la
oferta vigente. El periodo de aceptación de la OPA terminó el **10/10/2025** (ver
`OPA_BBVA_EVENTS` en `src/data_loader.py`), así que con datos posteriores la
herramienta es sobre todo **retrospectiva/ilustrativa** de la metodología.

> No es asesoramiento de inversión.
