# performance

Monitor de rendimiento del sistema **vs benchmark (SPY)**. Cierra el bucle: operas
en paper → mides si tu sistema **bate a comprar-y-mantener**.

Lee el diario (operaciones cerradas) y calcula:
- **Curva de equity** (capital simulado acumulado) + gráfico vs SPY.
- **Retorno total, win rate, profit factor, expectancy R, máx drawdown**.
- **Benchmark SPY** (buy & hold) en el mismo periodo → ¿lo bates?
- Comparación **factor ALTO vs BAJO** (vía el diario).

## Uso
```bash
python performance.py
python performance.py --capital 10000 --benchmark SPY
```
En el dashboard: pestaña **📈 Rendimiento**.

## El listón real
Batir a **SPY comprar-y-mantener** (neto, tras costes, con muchas operaciones) es la
vara de medir honesta. Si no lo bates, **indexarte es mejor** — y saberlo con tus
propios datos vale más que cualquier backtest optimista.

> No es recomendación de inversión.
