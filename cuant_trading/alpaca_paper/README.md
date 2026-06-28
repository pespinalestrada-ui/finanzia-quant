# alpaca_paper

Conector a **Alpaca PAPER trading** — el salto a "vivo" del flujo intradía **sin
riesgo**: cuenta de papel (dinero ficticio, ~100.000 $) con **datos en tiempo real**
(feed IEX gratis) y **órdenes simuladas**.

Usa la API REST de Alpaca **directamente con `requests`** → ninguna dependencia nueva.

## Funciones
- `cuenta()` — equity, cash, buying power, P&L del día.
- `posiciones()` — posiciones abiertas (paper).
- `cotizacion(sym)` — último precio real-time (IEX).
- `ordenes()` — últimas órdenes.
- `enviar_orden(sym, qty, side)` — manda una orden **paper**. La dispara **el usuario**
  (botón del dashboard o CLI con `--confirmar`); el asistente nunca opera por su cuenta.

## Claves (seguridad)
Se leen del `.env` de la raíz con `os.getenv`, **nunca** se hardcodean ni se suben a
GitHub (`.env` está en `.gitignore`):
```
ALPACA_KEY=PK...        (Key ID del panel Paper)
ALPACA_SECRET=...       (Secret; se muestra 1 sola vez)
```
Panel: https://app.alpaca.markets/paper/dashboard/overview → caja **API Keys**.

## Uso
```bash
python alpaca_paper.py cuenta
python alpaca_paper.py precio AAPL
python alpaca_paper.py posiciones
python alpaca_paper.py comprar AAPL 1 --confirmar     # orden paper (requiere --confirmar)
```
En el dashboard: pestaña **🦙 Alpaca Paper** (refrescar cuenta, cotización, enviar orden).

## El flujo correcto
intradía con costes (validar) → **Alpaca paper** (probar en vivo, real-time, sin
arriesgar) → real (solo si paper confirma expectancy positiva, y la decisión es tuya).

> Dinero ficticio. No es recomendación de inversión. El asistente no ejecuta órdenes
> ni mueve dinero: las órdenes las disparas tú.
