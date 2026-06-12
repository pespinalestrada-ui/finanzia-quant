# position_sizer

Cuántas acciones comprar para arriesgar solo un % fijo del capital. Gestión de
riesgo, no predicción.

## Uso
```bash
# Stop manual
python position_sizer.py --capital 10000 --risk 1 --entry 50 --stop 47

# Stop automático por ATR (descarga el ticker)
python position_sizer.py --ticker AAPL --capital 20000 --risk 0.5 --atr-mult 2

# Con Kelly (necesita win-rate y payoff de tu estrategia)
python position_sizer.py --capital 10000 --risk 1 --entry 50 --stop 47 --winrate 0.55 --payoff 1.8
```

## Qué devuelve
- **Acciones** a comprar y **coste** de la posición.
- **Riesgo real** en € (= capital × risk%).
- **Objetivos en R-múltiplos** (1R, 2R, 3R…) con precio y ganancia.
- **Fracción de Kelly** (completa y a la mitad) si das win-rate + payoff.

## Regla de oro
Arriesga **0.5–2 %** por operación. La R es la unidad: 1R = lo que pierdes si salta el stop.
