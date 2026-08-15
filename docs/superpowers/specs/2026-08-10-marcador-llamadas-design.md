# Marcador de llamadas — ¿lo que dijo el sistema se cumplió?

**Fecha:** 2026-08-10
**Estado:** diseño aprobado, pendiente de plan de implementación

## El problema

El panel emite veredictos (COMPRAR / MANTENER / VENDER) y previsiones de precio,
pero **nadie lleva la cuenta de si se cumplieron**. Todo lo que se ha medido hasta
ahora es retrospectivo (`veredicto_backtest`, `cpcv`, 🔬 Validar): se calcula sobre
el pasado, con el riesgo de sobreajuste que eso arrastra y que este proyecto ya ha
documentado.

Falta la única medición que **no se puede sobreajustar**: apuntar la llamada antes
de conocer el resultado y comprobarla después.

## Lo que ya sabemos, y que este marcador va a confirmar o desmentir

Medido en sesiones anteriores: el forecast de precio a 30-90 días **no bate al
azar** (DA ~14-50%, Theil U2>1) y la dirección a 5 días con ML se queda en 50-52%
(p>0,05). La búsqueda de configuraciones del Veredicto (80 probadas) no encontró
ninguna que aguantara fuera de su universo.

Por tanto **lo esperable es que el marcador se acerque al 50%**. Eso no invalida
la herramienta: la hace útil. Un marcador honesto que dice "no hay ventaja" evita
operar de más, que es la vía más directa por la que un minorista pierde dinero.

## Alcance

**Entra:**
- Registro de llamadas del ★ Veredicto y del 1 · Forecast.
- Resolución automática a 7, 30 y 90 días.
- Tres definiciones de acierto en paralelo (dirección, contra no hacer nada,
  contra el mercado), cada una contra su tasa base y con intervalo de confianza.
- Reconstrucción histórica del núcleo técnico para tener marcador desde el día 1.
- Pestaña 🎯 Marcador en el grupo ⚙️ Sistema algo.

**No entra (otra iteración):**
- Calibración por nivel de confianza declarada. No hay muestra para partir en
  grupos; con 40 llamadas repartidas en tres niveles no se puede decir nada.
- Alertas, notificaciones o comparación entre configuraciones.
- Registro de las demás pantallas (Señales, Factores, Vol objetivo).

## Arquitectura

Módulo nuevo `cuant_trading/marcador/marcador.py` y un enganche corto en
`tab_veredicto` y `tab_forecast`. Cuatro funciones con una responsabilidad cada
una, que se comunican **por el fichero** y no por memoria:

| función | qué hace | depende de |
|---|---|---|
| `registrar(...)` | apunta una llamada nueva | nada, solo escribe |
| `resolver()` | busca precios vencidos y los rellena | yfinance |
| `marcador(...)` | cuenta aciertos y calcula significancia | solo el CSV |
| `reconstruir(...)` | genera llamadas históricas del núcleo técnico | `veredicto_backtest` |

Almacén: `data/llamadas.csv` (queda fuera del repo por el `.gitignore` de `data/`).
Un CSV, no una base de datos: se puede abrir en un notebook sin nada de por medio.

## Modelo de datos

Una fila por llamada, escrita **antes** de conocer ningún resultado:

```
id, fecha, fuente, origen, ticker, senal, score, confianza,
precio_0, var_esperada_pct,
precio_7, precio_30, precio_90,
spy_0, spy_7, spy_30, spy_90,
resuelto_7, resuelto_30, resuelto_90
```

- `fuente`: `veredicto` | `forecast`
- `origen`: `vivo` | `reconstruido`
- `senal`: `COMPRAR` | `MANTENER` | `VENDER`
- `score`: el `total` del Veredicto en [−1, +1]; para el forecast, la variación
  esperada a 90 días
- `resuelto_*`: fecha en que se rellenó ese plazo, o vacío

### Tres decisiones sobre qué se guarda

1. **Los MANTENER también se apuntan.** Un sistema que solo registra sus apuestas
   se autoengaña. Saber cuántas veces dijo "no sé" es saber cuántas veces estaba
   opinando de verdad.
2. **Una llamada por (ticker, fuente, día).** Abrir AAPL cinco veces mientras se
   trastea anota una sola. Sin deduplicar, un ticker que se mira mucho domina el
   marcador.
3. **Se guardan los PRECIOS, no los aciertos.** Al resolver se guarda el precio a
   7/30/90 días, no un "acertó sí/no". Así se puede cambiar la definición de
   acierto sin perder el histórico — necesario, porque hay tres definiciones.

## Cómo se puntúa

Tres marcadores sobre el mismo dato. Los tres tienen definición **operativa**,
sin margen de interpretación:

1. **Dirección** (tasa de acierto) — COMPRAR y `precio_h > precio_0`; VENDER y
   `precio_h < precio_0`. Se compara contra la **tasa base**: qué porcentaje de
   todas las ventanas de ese plazo, en ese activo, fueron al alza.

2. **Contra no hacer nada** (exceso sobre la deriva) — media de
   `signo(señal) · (r_llamada − r_medio_del_activo)`, donde `r_medio_del_activo`
   es el retorno medio de TODAS las ventanas de ese plazo en ese activo.

   > **Por qué así y no "vs quedarse quieto" literal.** Para una llamada de
   > COMPRAR, "no hacer nada" comparado con comprar y mantener ese mismo activo
   > da exactamente lo mismo: la comparación se anula sola y no mide nada. Lo que
   > de verdad se quiere saber es si la llamada aporta **por encima de la deriva
   > del activo**, es decir, si el momento elegido añade algo a estar siempre
   > dentro. Ese es el confusor que tumbó el análisis de rotura de simetría en
   > una sesión anterior: el momentum parecía funcionar en calma y solo estaba
   > capturando la subida del mercado.

3. **Contra el mercado** (exceso sobre el SPY) — media de
   `signo(señal) · (r_llamada − r_spy)` en el mismo plazo.

En 2 y 3 la métrica es un **exceso de retorno medio en %**, no una tasa de
acierto, y se acompaña de su t-estadístico con n efectivo. En 1 la métrica es una
tasa de acierto contra la tasa base. No se mezclan.

Los MANTENER **no puntúan** en dirección (no afirman nada), pero sí cuentan para
la métrica "cuántas veces se mojó".

### La comparación que decide si el marcador vale algo

Nunca se enseña el acierto solo. Siempre contra la **tasa base**:

```
acierto del sistema    58%   (n=41)
tasa base del activo   54%   ← ese activo sube el 54% de las veces igualmente
ventaja                +4 pp     IC 95%: [−11, +19]  →  sin diferencia
```

La tasa base se calcula sobre el mismo activo y el mismo plazo en el histórico
disponible. Un sistema que acierta el 54% de los COMPRAR en un activo que sube el
54% de las veces tiene habilidad **cero**, y el marcador lo dirá así.

### Suelo de muestra

Con **n < 30** el marcador no da porcentaje: dice "muestra insuficiente". Un 70%
sobre 7 llamadas es ruido, y enseñarlo sería el autoengaño que el resto del
proyecto evita.

### Corrección por solape

Los plazos de 30 y 90 días de llamadas cercanas comparten datos. El intervalo de
confianza usa **n efectivo = n / (plazo/21)**, la misma corrección de muestra
efectiva que ya aplica `cpcv` y `veredicto_tune`.

## Reconstrucción histórica

`veredicto_backtest.score_historico()` recalcula el score día a día sin mirar al
futuro, así que reconstruir es honesto. **Pero no reconstruye el mismo sistema que
corre en vivo**, y esto hay que decirlo en la pantalla, no solo aquí:

| pilar | ¿reconstruible? | por qué |
|---|---|---|
| Tendencia, ADX, osciladores, MACD, momentum, OBV | sí | solo necesitan precio |
| Forecast (Prophet) | no en la práctica | habría que reentrenar Prophet en cada fecha |
| Factores, Calidad (ROIC) | **no** | yfinance solo da los fundamentales de HOY; usarlos para reconstruir 2023 sería mirar al futuro |

Por eso:
- El marcador reconstruido se etiqueta **"núcleo técnico"**.
- El de en vivo se etiqueta **"Veredicto completo"**.
- **No se suman ni se comparan como iguales.** Son dos sistemas distintos.

**Muestreo: una llamada cada 21 sesiones por ticker.** Reconstruir a diario haría
que dos llamadas consecutivas a 90 días compartieran 89 días de datos: parecerían
500 observaciones cuando hay ~6 independientes. Con paso mensual el solape baja
mucho, y aun así se aplica la corrección de n efectivo.

Sobre una watchlist de 10 valores y 3 años salen ~340 llamadas.

## Cuándo se resuelve

Dos disparadores, ninguno manual:

- **Al abrir la pestaña**, usando la caché de precios que ya existe (`_dl`, TTL 30
  min).
- **En la tanda de las 5 horas**, añadiendo una línea a `tanda_programada.py`.

`resolver()` es **idempotente**: si una llamada ya tiene precio a 30 días no lo
vuelve a pedir. Se puede invocar mil veces sin duplicar ni recalcular.

## Interfaz

Pestaña **🎯 Marcador** en el grupo ⚙️ Sistema algo, junto a 🔬 Validar y 🧪 CPCV,
que es donde vive lo de "¿esto funciona de verdad?".

Estructura:

1. **Una frase de estado arriba**, que es lo único que hay que leer:
   > En vivo: 12 llamadas, ninguna vencida todavía. Vuelve en una semana.
   > Núcleo técnico reconstruido (340 llamadas, 3 años): acierta el 53% a 30 días
   > frente a una tasa base del 54%. Ventaja −1 pp, IC [−8, +6]: ninguna.
2. **Tabla 3×3**: filas 7/30/90 días, columnas dirección / exceso sobre la deriva
   / exceso sobre el SPY. La primera columna en tasa de acierto contra tasa base;
   las otras dos en exceso de retorno medio con su t. Cada celda dice si la
   diferencia es significativa o no.
3. **Gráfica**: ventaja acumulada con su banda de confianza, que se estrecha
   según entran llamadas. Enseña visualmente que al principio no se sabe nada.
4. Separación visual clara entre el bloque **en vivo** y el **reconstruido**.

## Errores

- Ticker sin datos, deslistado o yfinance caído: esa llamada **queda sin
  resolver** y se reintenta en la siguiente pasada. No rompe la pestaña.
- La pestaña devuelve `msg_fig` en caso de error, como el resto del panel.
- Si `data/llamadas.csv` se corrompe, se renombra a `.bak` y se empieza limpio,
  avisando en pantalla. Nunca se pierde el fichero anterior en silencio.
- El registro es **best-effort**: si `registrar()` falla, el Veredicto responde
  igual. Medir no puede romper lo que se mide.

## Pruebas

Tres con respuesta conocida de antemano, más una de no-regresión:

1. **Serie sintética que siempre sube lo mismo** → los COMPRAR dan 100% en
   dirección, tasa base 100% (ventaja 0 pp) y **exceso sobre la deriva ≈ 0**. Si
   el exceso no sale ~0, el descuento de la deriva está mal implementado y el
   marcador estaría dando por habilidad lo que es solo mercado alcista.
2. **Serie aleatoria sin deriva** → acierto ≈ 50% y el test dice "no
   significativo". Si dice que hay ventaja, el estadístico está roto.
3. **Idempotencia** → llamar `resolver()` tres veces seguidas deja el CSV
   idéntico.
4. **No-regresión** → el mismo ticker analizado con y sin marcador activo devuelve
   exactamente el mismo veredicto. Medir no altera lo medido.

## Riesgos asumidos

- **El marcador probablemente dirá que no hay ventaja.** Es el resultado esperado
  y es información, no un fallo. La pantalla lo enmarca así desde el principio
  para que no se lea como una avería.
- **La reconstrucción no es prospectiva.** Está etiquetada, pero sigue siendo una
  medición sobre el pasado con las limitaciones de siempre.
- **Tarda en madurar.** El marcador en vivo a 90 días necesita tres meses para su
  primer dato. Por eso existe la reconstrucción.
