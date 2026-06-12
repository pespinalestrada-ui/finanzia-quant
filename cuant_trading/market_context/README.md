# market_context

Termómetro del mercado: contexto ANTES de mirar ningún ticker.

## Fuentes (todas gratis, sin clave)
- **Fear & Greed bolsa** — réplica del índice de CNN vía feargreedchart.com:
  score 0-100 + 5 componentes (volatilidad, momentum, put/call, refugio, bonos basura).
- **Fear & Greed cripto** — alternative.me.
- **VIX** — yfinance, con lectura de régimen (<15 calma · 15-20 normal · 20-30 nervioso · >30 pánico).
- **Fundamentales del ticker** — yfinance `.info`: PER, P/Book, beta, dividendo,
  market cap, rango 52 semanas, recomendación de analistas.

## Uso
```bash
python market_context.py            # solo mercado
python market_context.py SAB.MC     # mercado + fundamentales
```
También como pestaña **🌡️ Mercado** en el dashboard.

## Cómo usarlo
- Miedo extremo (<25) históricamente coincide con suelos; codicia extrema (>75) con techos. Indicador **contrarian**.
- VIX >30 = reduce tamaño de posición aunque la señal sea buena.
- La lectura conjunta del script combina ambos en una frase operativa.
