# indicators

Indicadores técnicos de un ticker con la señal actual de cada uno.

**Calcula:** RSI(14, Wilder), MACD(12,26,9), Bandas de Bollinger(20,2), ATR(14), SMA50/200.

## Uso
```bash
python indicators.py AAPL
python indicators.py SAB.MC --period 2y --save
```

## Salida
- Tabla en consola: valor + interpretación (sobrecompra/sobreventa, cruce MACD, posición en banda, volatilidad, tendencia SMA50 vs SMA200).
- Con `--save`: PNG de 3 paneles (precio+Bollinger+SMA, RSI, MACD).

## Cómo leerlo
- **RSI < 30** sobreventa (posible rebote) · **> 70** sobrecompra.
- **MACD** cruce alcista/bajista de la línea sobre su señal.
- **ATR%** = volatilidad relativa; útil para fijar stops (ver `position_sizer`).
- **SMA50 > SMA200** = tendencia de fondo alcista (golden cross).
