# veredicto_backtest

¿El **Veredicto** predice de verdad? Validación honesta **antes** de automatizar
nada — operar sobre una señal sin validar es la forma más rápida de perder dinero.

Reconstruye el **score técnico del Veredicto** en cada fecha pasada (point-in-time,
sin leakage) y mide si predice el retorno futuro:

- **Information Coefficient (IC)** — correlación score ↔ retorno futuro (Spearman).
- **Retornos por quintil** del score — ¿el quintil alto rinde más? ¿es monótono?
- **Sharpe long-short** cross-section (largo top / corto bottom).
- **PSR** y **Deflated Sharpe** (Bailey & López de Prado) — corrigen por asimetría/
  curtosis y por las **muchas configuraciones probadas** (multiple testing). El
  Deflated Sharpe es el juez final: si <95%, el "edge" puede ser azar de probar mucho.

## Uso
```bash
python veredicto_backtest.py AAPL MSFT NVDA GOOGL AMZN META JPM XOM
python veredicto_backtest.py --file watchlist.txt --horizon 10 --trials 20
```

## Resultado medido (10 large-caps US, 10d) — la verdad
IC ≈ −0.02, quintiles no monótonos, Sharpe long-short ≈ 0.38, **Deflated Sharpe ≈ 43%**
→ el núcleo técnico **no supera el multiple-testing**. Es decir: en líquidas no hay
edge fácil (mercado eficiente). Reportarlo es lo correcto: el Veredicto vale como
**resumen discrecional**, no como alfa automático.

## Alcance honesto
Valida la parte **point-in-time y reproducible** del Veredicto (tendencia, ADX,
osciladores, MACD, momentum, OBV). NO incluye Prophet ni sentimiento (no son
backtesteables sin leakage / sin histórico de noticias).

> No es recomendación de inversión.
