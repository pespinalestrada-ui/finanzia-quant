# 📖 Guía de usuario — FinanzIA

_Herramienta de análisis y educación. No es recomendación de inversión. Practica siempre primero sin dinero real._

## 🚀 Empieza aquí (2 minutos)
Cada pantalla lleva una etiqueta: **[Básico]** empieza por estas · **[Intermedio]** cuando te sientas cómodo · **[Avanzado]** matemática fina, sáltatelas al principio.

**La regla de oro:** nunca pongas dinero real hasta haberlo practicado de mentira (Diario y Alpaca Paper). Predecir el precio exacto no se puede; el valor está en controlar el riesgo y no engañarse.

**Tus primeros 15 minutos:**
1. Abre 🌡️ Mercado y mira si hay miedo o codicia.
2. Escribe una empresa en ★ Veredicto y pulsa 'Analizar TODO'.
3. Si dice COMPRAR, mira el plan sugerido (acciones y stop).
4. Apúntalo en 📒 Diario. Tras 20-30 operaciones sabrás si ganarías.

**Según lo que quieras hacer:**

| Quiero… | Ve a |
|---|---|
| Saber si comprar/vender UNA empresa | ★ Veredicto |
| Elegir entre VARIAS empresas | 📊 Factores  (y luego ★ Veredicto) |
| Ver el ambiente general del mercado | 🌡️ Mercado |
| Saber cuántas acciones comprar | 9 · Tamaño posición |
| Ver el riesgo de mi lista | 🛡️ Riesgo |
| Practicar sin dinero real | 📒 Diario  +  🦙 Alpaca Paper |
| Que me avisen de movimientos | 🔔 Alertas |
| Ver noticias de una empresa | 8 · Sentimiento |
| Saber si mi método de verdad funciona | 📈 Rendimiento  +  🔬 Validar Veredicto |

## Bloque 1 · Para el día a día (empieza por aquí)

### 📊 Factores  `[Básico]`
**Para qué sirve:** Te ordena una lista de empresas de mejor a peor, según si están baratas, si vienen subiendo, si la empresa es sólida y si son tranquilas. Como un ranking para elegir candidatas.

**Cómo se usa:** Escribe varias empresas separadas por comas (ej. AAPL, MSFT, NVDA) y pulsa Rankear.

**Qué significa lo que ves:**
- La de arriba (rank 1) es la 'mejor nota' del grupo.
- La columna 'señal' te dice: COMPRAR (las mejores), EVITAR (las peores) o neutral.
- Los números z son solo la posición dentro del grupo: + es bueno, − es malo.
- **z_quality** = lo sólida que es la empresa. Dentro de esa nota mandan el **ROIC** (40%), el ROE (25%), el margen (20%) y la poca deuda (15%).
- La columna **ROIC %** es lo que renta el capital que de verdad trabaja, mediana de varios años. Por encima de ~10% crea valor; por debajo lo destruye. Sale vacía en bancos, donde no aplica.
- Puede pasar que una empresa con ROIC malo salga arriba: el ranking pesa también lo barata que está y su empuje. Mira la nota Y el ROIC, no solo el puesto.

**👉 Qué hago con esto:** Úsala para PRESELECCIONAR. Quédate con las 2-3 de arriba y luego míralas a fondo en la pantalla ★ Veredicto.

### ★ Veredicto  `[Básico]`
**Para qué sirve:** La pantalla estrella. Analiza UNA empresa por todos lados y te da un resumen: COMPRAR, MANTENER o VENDER, con el porqué.

**Cómo se usa:** Escribe una empresa (ej. AAPL) y pulsa 'Analizar TODO'. Los dos checkbox son opcionales (déjalos sin marcar al principio).

**Qué significa lo que ves:**
- Arriba, el veredicto en grande con un color: 🟢 comprar, 🟡 esperar, 🔴 vender.
- La tabla son los 'motivos' (tendencia, fuerza, etc.). No hace falta entender cada fila.
- MANTENER = ni claro sí ni claro no, mejor esperar.
- Entre los motivos verás **Calidad del negocio (ROIC)**: mide si la empresa gana más de lo que le cuesta el dinero (~10%). Por encima crea valor, por debajo lo destruye, y avisa si va cayendo. Usa la mediana de 4 años, no el último dato.
- Ese motivo **pesa según lo claro que esté**: una empresa del montón casi no cuenta, una que destruye valor sí. En bancos no aparece: el ROIC no aplica ahí.

**👉 Qué hago con esto:** Es tu pantalla principal. Si dice MANTENER, no operes. NO es un consejo garantizado: es un resumen para ayudarte a decidir.

### 🌡️ Mercado  `[Básico]`
**Para qué sirve:** El 'tiempo' de la bolsa antes de mirar empresas: ¿hay miedo o codicia?, ¿está nervioso el mercado?

**Cómo se usa:** Pulsa 'Medir mercado' (opcional: escribe una empresa para ver sus datos básicos).

**Qué significa lo que ves:**
- Miedo extremo suele ser buen momento (la gente vende por pánico). Codicia extrema, cuidado.
- El VIX alto = mercado nervioso: conviene ir con menos dinero.

**👉 Qué hago con esto:** Míralo al empezar el día. Si hay pánico o euforia extrema, sé más prudente.

### 💰 Rentabilidad  `[Básico]`
**Para qué sirve:** Los cuatro números con los que se juzga si una empresa es un buen NEGOCIO (no si la acción va a subir): ROE, ROA, ROIC y BPA.

**Cómo se usa:** Escribe una empresa y pulsa 'Medir rentabilidad'. En el segundo hueco puedes poner otras para compararlas, pero solo tiene sentido comparar empresas del MISMO sector.

**Qué significa lo que ves:**
- **ROE**: por cada 100 € que han puesto los dueños, cuánto gana al año. Es el más famoso y el más tramposo: sube solo con endeudarse.
- **ROA**: por cada 100 € que la empresa mueve (suyos y prestados), cuánto gana. La deuda no lo maquilla.
- **ROIC**: el más honesto. Lo que renta el dinero que de verdad trabaja en el negocio. Por encima de ~10% (lo que cuesta el capital) crea valor; por debajo, lo destruye.
- **BPA**: cuánto beneficio toca a cada acción. El único en euros, no en %.
- El aviso de **'el ROE es X veces el ROA'**: si sale 1-2 el ROE es negocio de verdad; si sale 5 o más, buena parte es deuda.
- La gráfica saca 4 ejercicios: un año bueno es suerte, cuatro seguidos es un negocio.

**👉 Qué hago con esto:** Úsalo para DESCARTAR, no para acertar. Son datos contables del pasado: dicen cómo ha ido la empresa, no hacia dónde va la acción. En bancos el ROIC no aplica y un ROA del 0,8% es normal, no un desastre.

### 8 · Sentimiento  `[Básico]`
**Para qué sirve:** Lee las noticias recientes de una empresa y te dice si suenan buenas, malas o neutras.

**Cómo se usa:** Escribe una empresa y pulsa 'Analizar noticias'. La primera vez tarda ~1 min (carga el modelo).

**Qué significa lo que ves:**
- Un resumen: POSITIVO / NEGATIVO / NEUTRAL.
- Las noticias viejas pesan menos que las de hoy.

**👉 Qué hago con esto:** Úsalo como contexto: ¿hay alguna noticia gorda detrás del movimiento? Es una señal más, no la decisión.

### 2 · Indicadores  `[Básico]`
**Para qué sirve:** Los 'termómetros' técnicos clásicos de una empresa (si está cara/barata a corto, si cruza medias, etc.).

**Cómo se usa:** Escribe una empresa y pulsa Calcular.

**Qué significa lo que ves:**
- Cada indicador da una lectura (sobreventa/sobrecompra, cruce al alza, etc.).
- Ninguno decide solo; el ★ Veredicto ya los junta por ti.

**👉 Qué hago con esto:** Para curiosear el detalle técnico. Si no quieres complicarte, usa directamente el Veredicto.

### 3 · Screener  `[Básico]`
**Para qué sirve:** Pasa un filtro rápido a una lista de empresas y te dice cuáles tienen más empuje ahora mismo.

**Cómo se usa:** Escribe varias empresas y pulsa Escanear.

**Qué significa lo que ves:**
- Ordenadas por fuerza/empuje.
- La columna **ROIC** y **Calidad** te dicen si además el NEGOCIO gana dinero: 'crea valor', 'ok' o 'destruye'. Vacío en bancos.
- El Score **no** incluye el ROIC a propósito: mide empuje, no calidad. Sirve para cazar el caso 'sube mucho pero la empresa no gana dinero'.

**👉 Qué hago con esto:** Para encontrar candidatas rápido. Luego confírmalas en el ★ Veredicto.

### 4 · Señales  `[Básico]`
**Para qué sirve:** Busca avisos claros de compra/venta (cruces, rebotes) en una lista de empresas.

**Cómo se usa:** Escribe varias empresas y pulsa 'Buscar señales'.

**Qué significa lo que ves:**
- Solo muestra lo accionable ahora.

**👉 Qué hago con esto:** Un radar de oportunidades. Confirma siempre con el Veredicto antes de nada.

### 9 · Tamaño posición  `[Básico]`
**Para qué sirve:** Te dice CUÁNTAS acciones comprar para arriesgar solo lo que tú decidas (p. ej. el 1% de tu dinero).

**Cómo se usa:** Pon tu capital y el % de riesgo. Pon entrada y stop, o solo el ticker para que los calcule solos.

**Qué significa lo que ves:**
- Te da el número de acciones.
- El 'stop' es el precio al que saldrías si va mal, para no perder de más.

**👉 Qué hago con esto:** Úsala SIEMPRE antes de comprar: define cuánto puedes perder ANTES de entrar. Regla de oro.

### 🛡️ Riesgo  `[Básico]`
**Para qué sirve:** Mide cuánto podrías perder con tu lista de empresas: en un día malo, en el peor momento, y si están muy 'pegadas' entre sí.

**Cómo se usa:** Escribe tu lista de empresas y pulsa 'Medir riesgo'.

**Qué significa lo que ves:**
- VaR = pérdida fea de un día malo. Drawdown = la peor caída que habrías sufrido.
- Si las empresas están poco relacionadas, diversificas bien (menos sustos).

**👉 Qué hago con esto:** Antes de juntar varias posiciones, comprueba que no es una bomba de riesgo.

### 1 · Forecast  `[Básico]`
**Para qué sirve:** Dibuja una predicción del precio a futuro con una banda de '¿hasta dónde podría ir?'.

**Cómo se usa:** Escribe una empresa, elige el motor (deja Prophet) y pulsa Forecast.

**Qué significa lo que ves:**
- La línea es la predicción; la banda, la incertidumbre (cuanto más lejos, más ancha).
- AVISO IMPORTANTE: a 90 días esto NO acierta mejor que el azar. Tómalo flojo.

**👉 Qué hago con esto:** Míralo como orientación, NUNCA como una bola de cristal. El valor real está en el riesgo, no en adivinar el precio.

## Bloque 2 · Practicar y operar (sin dinero real)

### 📒 Diario  `[Básico]`
**Para qué sirve:** Un cuaderno de operaciones DE MENTIRA. Apuntas tus compras/ventas imaginarias y te dice si tu método gana o pierde.

**Cómo se usa:** Abrir = apuntar una compra (empresa, precio, stop). Cerrar = poner el precio al que 'saliste'. Refrescar para ver tus números.

**Qué significa lo que ves:**
- Con 20-30 operaciones ya sabes si tu método tiene ventaja.
- 'Expectancy' = cuánto ganas de media por operación. Positivo = bien.

**👉 Qué hago con esto:** ÚSALO antes de jugarte un euro real. Si en mentira pierdes, en real también.

### 🦙 Alpaca Paper  `[Intermedio]`
**Para qué sirve:** Como el Diario pero conectado a una cuenta de prácticas REAL (Alpaca): dinero ficticio, precios en vivo, órdenes de mentira.

**Cómo se usa:** Refrescar = ver tu cuenta. Precio = cotización en vivo. Enviar orden = TÚ pulsas el botón para 'comprar' de mentira.

**Qué significa lo que ves:**
- Es dinero FICTICIO (empiezas con ~100.000 $ de prácticas).
- Cada orden puede apuntarse sola en tu Diario.

**👉 Qué hago con esto:** Practica aquí semanas antes de pensar en real. Las órdenes las disparas tú; el programa nunca opera solo. Necesita una clave gratis de Alpaca.

### ⏱️ Intradía  `[Intermedio]`
**Para qué sirve:** Para operar dentro del mismo día. Lo primero: el 🚦 SEMÁFORO te dice si hoy conviene operar al alza, a la baja o no operar. Y el backtest prueba estrategias DESCONTANDO los costes.

**Cómo se usa:** '🚦 Semáforo de HOY' = ¿opero o no? (usa datos EN VIVO de Alpaca si tienes clave). '📡 Snapshot EN VIVO' = foto en tiempo real. 'Backtest' = prueba una estrategia (orb/vwap/ema9) con costes. 'Escanear varios' = compara empresas.

**Qué significa lo que ves:**
- El semáforo: 🟢 sesgo largo · 🔴 sesgo corto · 🟡 no operes hoy (la respuesta más frecuente y a menudo la correcta).
- En el backtest, lo importante es la ganancia NETA (tras costes), no la bruta.
- Hay 3 estrategias para comparar: rotura de apertura (orb), retorno al VWAP (vwap) y pullback a la media (ema9).

**👉 Qué hago con esto:** Empieza el día por el semáforo. Si está 🟡, no operes: no operar también es ganar. Si operas, respeta el stop que te sugiere.

### 🪙 Veredicto Cripto  `[Intermedio]`
**Para qué sirve:** El mismo ★ Veredicto pero para criptomonedas (Bitcoin, Ethereum…).

**Cómo se usa:** Escribe el cripto en formato BTC-USD, ETH-EUR… y pulsa Analizar.

**Qué significa lo que ves:**
- Igual que el Veredicto normal: COMPRAR / MANTENER / VENDER.
- Incluye el 'miedo y codicia' del mundo cripto.
- Aquí **no** verás el motivo de calidad por ROIC: una cripto no tiene cuentas anuales, así que no hay ROIC que calcular. No es un olvido.
- En su lugar está **Calidad de la red**, que mide dos cosas que sí existen: cuánto **volumen** mueve al día (si es poco, no podrás salir sin hundir el precio) y cuánto queda **por emitir** (lo que falta te diluye, como una empresa que no para de sacar acciones).
- Ojo: eso mide la *fontanería*, no si la moneda vale algo. Una memecoin muy negociada puntúa bien aquí y sigue siendo una memecoin.

**👉 Qué hago con esto:** Recuerda que la cripto se mueve muchísimo más: ve con más cuidado y menos dinero.

## Bloque 3 · El sistema automático (avanzado)

### 📡 Señales  `[Intermedio]`
**Para qué sirve:** Te dice 'qué tocaría operar hoy' de tu lista, según el método del Veredicto.

**Cómo se usa:** Escribe tu lista y pulsa 'Generar señales'.

**Qué significa lo que ves:**
- Un ranking con COMPRAR / MANTENER / VENDER.

**👉 Qué hago con esto:** Es el primer paso del 'piloto automático'. Trátalo como prácticas, no como dinero seguro.

### ⚖️ Plan / Riesgo  `[Intermedio]`
**Para qué sirve:** Convierte esas señales en un plan concreto: cuántas acciones de cada una, con su stop y su riesgo.

**Cómo se usa:** Pon tu capital y pulsa 'Generar plan'.

**Qué significa lo que ves:**
- Te dice acciones, coste y riesgo de cada posición.
- Avisa si te pasas de riesgo.

**👉 Qué hago con esto:** Revisa que el riesgo total te parece asumible antes de nada.

### 🤖 Sistema  `[Avanzado]`
**Para qué sirve:** Todo junto y de un clic: genera el plan y, si TÚ confirmas, lo manda a la cuenta de prácticas (Alpaca) y lo apunta en el Diario.

**Cómo se usa:** 1) 'Generar plan'. 2) Marca 'Confirmo enviar a PAPER' y pulsa 'Ejecutar'.

**Qué significa lo que ves:**
- Sin marcar la casilla, no envía nada.
- Es dinero ficticio.

**👉 Qué hago con esto:** El 'robot' de prácticas. Empléalo solo cuando entiendas los pasos anteriores.

### 📈 Rendimiento  `[Intermedio]`
**Para qué sirve:** ¿Lo estás haciendo mejor que comprar el mercado y no tocar nada? Te lo compara con el índice (SPY).

**Cómo se usa:** Pon tu capital y pulsa 'Medir rendimiento'.

**Qué significa lo que ves:**
- Tu ganancia vs SPY (comprar y mantener).
- Si no le ganas, indexarte (comprar el índice) sería mejor.

**👉 Qué hago con esto:** La prueba de la verdad. Si tu sistema no bate a SPY, mejor algo simple.

### 🎲 Monte Carlo  `[Avanzado]`
**Para qué sirve:** Simula MILES de futuros posibles para ver el abanico de lo que podría pasar (no una predicción, sino el rango).

**Cómo se usa:** Precio: una empresa. Sistema: pon tu win-rate/payoff (o usa tu diario).

**Qué significa lo que ves:**
- En 'sistema', lo clave es la PROBABILIDAD DE RUINA (acabar reventado) y la peor caída.
- Un sistema con buena media pero alta ruina NO sirve.

**👉 Qué hago con esto:** Úsalo para ver si tu método aguanta la mala suerte, no solo la buena.

### 🔬 Validar Veredicto  `[Avanzado]`
**Para qué sirve:** Comprueba con datos del pasado si el Veredicto realmente acierta o solo lo parece.

**Cómo se usa:** Escribe varias empresas y pulsa Validar. Tarda ~1 min.

**Qué significa lo que ves:**
- El número clave ('Deflated Sharpe') dice si la ventaja es real o suerte.
- Resultado honesto habitual: en grandes empresas, no hay ventaja fácil.

**👉 Qué hago con esto:** La prueba de honestidad. Te recuerda que esto mide, no promete milagros.

### 🛞 Vol objetivo  `[Intermedio]`
**Para qué sirve:** EL ALGORITMO del proyecto. No intenta adivinar si el mercado sube o baja (eso ya medimos que no se puede): ajusta CUÁNTO estás invertido según la tormenta prevista. En calma, dentro; cuando la volatilidad se dispara, se retira solo y el resto queda en efectivo.

**Cómo se usa:** Escribe un activo (SPY, QQQ, tu ETF...), elige la volatilidad que quieres soportar (12% es un buen punto de partida) y pulsa Probar algoritmo. El botón 🔬 Robustez comprueba si funciona con cualquier configuración o solo con una afortunada.

**Qué significa lo que ves:**
- **Qué hace hoy**: el % que deberías tener invertido ahora mismo.
- La tabla compara el algoritmo contra comprar y mantener: fíjate sobre todo en la **peor caída** y el **Sharpe**.
- **En las crisis**: cuánto habrías perdido en 2008, COVID y 2022 frente a estar siempre dentro.
- El **coste de operar ya está descontado** de todos los números.

**👉 Qué hago con esto:** Lo que busca no es ganar más, sino **ganar parecido pasando mucho menos miedo**. Si la peor caída baja a la mitad y el Sharpe sube, está haciendo su trabajo. Practícalo en paper antes de aplicarlo con dinero.

### 🧪 CPCV  `[Avanzado]`
**Para qué sirve:** La validación más dura de todas: parte la historia en bloques y prueba la estrategia en MUCHAS combinaciones distintas fuera de muestra, con cortafuegos para que el futuro no se cuele.

**Cómo se usa:** Escribe tu universo y pulsa Validar. Tarda 1-2 minutos.

**Qué significa lo que ves:**
- **PBO**: probabilidad de sobreajuste. Alto (>50%) = elegir 'la mejor configuración' del pasado NO funciona después.
- **Sharpe ± margen de error**: si el intervalo cruza 0, no puedes afirmar que ganas.
- **Deflated Sharpe**: descuenta el azar de haber probado muchas veces. Necesita >95% para creerse.

**👉 Qué hago con esto:** Es el juez más severo. Si una idea sobrevive aquí, es seria; si no, mejor saberlo antes de arriesgar dinero.

### 📊 Backtest sist.  `[Avanzado]`
**Para qué sirve:** Prueba la estrategia completa sobre años pasados (con costes) y la compara con el índice.

**Cómo se usa:** Escribe un universo de empresas y pulsa Backtestear.

**Qué significa lo que ves:**
- Ganancia, riesgo y peor caída vs SPY.
- Que ganara en el pasado NO garantiza el futuro.

**👉 Qué hago con esto:** Mídelo, pero no te lo creas a pies juntillas: un buen pasado puede ser casualidad.

## Bloque 4 · Matemática de fondo (solo si te pica la curiosidad)

### 🔗 Pairs (cointegración)  `[Avanzado]`
**Para qué sirve:** Busca PAREJAS de activos que se mueven juntos. Cuando se separan mucho, apuestas a que vuelven a juntarse.

**Cómo se usa:** Escribe varias empresas y pulsa 'Buscar pares'.

**Qué significa lo que ves:**
- Te da parejas y cuándo están 'estiradas' (z alto).
- Es una técnica neutral (no dependes de que la bolsa suba).

**👉 Qué hago con esto:** Avanzado. Es de lo poco con ventaja real, pero requiere cuidado. Para más adelante.

### 🧮 HRP Cartera  `[Avanzado]`
**Para qué sirve:** Reparte tu dinero entre varios activos de forma más robusta que el método clásico (que es inestable).

**Cómo se usa:** Escribe varios activos y pulsa 'Comparar asignación'.

**Qué significa lo que ves:**
- Compara métodos: el robusto suele dar menos sustos (menos volatilidad).

**👉 Qué hago con esto:** Si algún día gestionas una cartera de varios activos, usa este reparto en vez del clásico.

### 📉 EVT Colas  `[Avanzado]`
**Para qué sirve:** Mide bien el riesgo de CRASH (las caídas gordas), que los métodos normales subestiman.

**Cómo se usa:** Escribe una empresa/índice y pulsa 'Medir cola'.

**Qué significa lo que ves:**
- Compara el riesgo extremo 'normal' vs el real: el normal casi siempre se queda corto.

**👉 Qué hago con esto:** Para entender cuánto puede doler de verdad un día negro.

### 🌀 Régimen (HMM)  `[Avanzado]`
**Para qué sirve:** Detecta en qué 'modo' está el mercado: calma alcista, nervios, o lateral.

**Cómo se usa:** Escribe un índice (ej. SPY) y pulsa 'Detectar régimen'.

**Qué significa lo que ves:**
- 🟢 calma = bien para seguir tendencia. 🔴 nervios = baja el tamaño. 🟡 lateral = cautela.

**👉 Qué hago con esto:** Úsalo como semáforo general antes de operar tendencia.

### 🎯 Meta-labeling  `[Avanzado]`
**Para qué sirve:** Un segundo filtro de inteligencia artificial que decide SI hacer caso a una señal (menos operaciones, pero mejores).

**Cómo se usa:** Escribe una empresa y pulsa 'Medir'. Tarda ~1-2 min.

**Qué significa lo que ves:**
- Compara: con filtro aciertas más, pero operas menos.
- A veces ayuda, a veces no: te lo dice con números.

**👉 Qué hago con esto:** Curiosidad técnica. Muestra cómo afinar señales sin engañarte.

### 🧲 RMT (correlación)  `[Avanzado]`
**Para qué sirve:** Limpia el 'ruido' de cómo se relacionan los activos entre sí, usando una idea de la física.

**Cómo se usa:** Escribe varios activos (4 o más) y pulsa 'Limpiar correlación'.

**Qué significa lo que ves:**
- Te dice cuántas relaciones son señal de verdad y cuántas son ruido.

**👉 Qué hago con esto:** Mejora el reparto de cartera cuando manejas muchos activos.

### 🛰️ Kalman (pairs)  `[Avanzado]`
**Para qué sirve:** Versión 'lista' de las parejas: la relación entre dos activos cambia con el tiempo, y esto la sigue al día.

**Cómo se usa:** Escribe dos activos (A y B) y pulsa 'Kalman dinámico'. Ej: KO y PEP.

**Qué significa lo que ves:**
- Compara la relación 'fija' vs la 'que se adapta'. La que se adapta es más fiable.

**👉 Qué hago con esto:** Si haces pares, usa esta versión: no se queda anclada al pasado.

### 🎲 Opciones  `[Avanzado]`
**Para qué sirve:** Valora una OPCIÓN (el derecho a comprar o vender algo a un precio fijado) con la fórmula Black-Scholes, y te da sus 'Griegas': a qué es sensible tu posición.

**Cómo se usa:** Escribe un ticker de EEUU para ver la cadena real de opciones, o déjalo vacío y juega con strike/días/volatilidad para una opción teórica.

**Qué significa lo que ves:**
- **Delta**: cuánto gana la opción si el activo sube 1 € (y, aproximadamente, la probabilidad de acabar valiendo algo).
- **Gamma**: cómo se acelera el Delta. Alto = la posición se mueve cada vez más rápido.
- **Vega**: cuánto gana si sube la volatilidad. Comprar opciones es apostar a que habrá movimiento.
- **Theta**: lo que te quita el reloj cada día. El comprador pierde solo con que pase el tiempo.

**👉 Qué hago con esto:** Antes de tocar una opción, mira Theta: si es alta, el tiempo juega en tu contra. Es el único apartado que trata derivados; el resto de la app son acciones.

### 📉 Deriva vol.  `[Intermedio]`
**Para qué sirve:** Cuánta rentabilidad se te come la volatilidad cada año. La rentabilidad que se publica es la MEDIA; la que de verdad acumulas es más baja, y la diferencia crece muy rápido con lo que se mueve el activo.

**Cómo se usa:** Escribe varios activos separados por comas y pulsa Medir. Marca la casilla para ver, además, hasta cuánto apalancamiento aguanta el primero.

**Qué significa lo que ves:**
- Dos barras por activo: la **media anual** (lo que se publica) y el **CAGR real** (lo que te llevas). Encima, los puntos que se pierden.
- SPY se deja 0,6 puntos al año; TQQQ se deja **15,9**. La diferencia entre ambos no es la rentabilidad: es la volatilidad.
- La curva de apalancamiento tiene forma de campana: hay un punto óptimo y, pasado otro, apalancar te deja PEOR que no apalancar.

**👉 Qué hago con esto:** Es la razón matemática de que 🛞 Vol objetivo funcione: bajar la volatilidad no solo quita sustos, sube lo que acumulas. Y explica por qué un ETF x3 puede perder a largo plazo aunque el índice suba. Ojo: los apalancamientos que salen son OPTIMISTAS (no cuentan intereses ni comisiones).

### ⏳ OU óptimo  `[Avanzado]`
**Para qué sirve:** Para pares (dos activos que van de la mano): calcula el 'muelle' que los une y te dice el momento EXACTO de cerrar, con matemáticas en vez de a ojo.

**Cómo se usa:** Escribe dos activos (ej. KO y PEP, o EWA y EWC), tu capital y el coste de operar. Pulsa Calibrar.

**Qué significa lo que ves:**
- **θ (fuerza del muelle)** y **half-life**: cómo de rápido vuelve el spread a su sitio. Corto = tradeable.
- **σ_eq**: cuánto se mueve el spread por puro azar. El stop se pone MÁS ALLÁ de eso.
- **SALIDA ÓPTIMA**: el nivel donde compensa cerrar, descontando lo que cuesta operar. No es 'cerrar en la media': suele compensar esperar algo más.

**👉 Qué hago con esto:** Si operas pares, usa este umbral en vez de decidir por intuición. Cuanto más caro sea operar, más lejos se pone la salida (la herramienta ya lo calcula).

### 📡 Entropía (lead-lag)  `[Avanzado]`
**Para qué sirve:** Descubre qué activo MUEVE a cuál (quién manda y quién sigue), incluso de formas que la correlación no ve.

**Cómo se usa:** Escribe varios activos y pulsa 'Medir flujo info'.

**Qué significa lo que ves:**
- LÍDERES = mandan información. SEGUIDORES = la reciben.

**👉 Qué hago con esto:** Para entender quién marca el ritmo en un grupo de activos.

## Bloque 5 · Extras de análisis

### 5 · Backtest  `[Básico]`
**Para qué sirve:** Prueba una regla simple (medias, RSI) en una empresa y la compara con 'comprar y no tocar'.

**Cómo se usa:** Escribe una empresa, elige estrategia y pulsa Backtest.

**Qué significa lo que ves:**
- Si la regla no gana a 'comprar y mantener', no aporta.

**👉 Qué hago con esto:** Para ver si una idea sencilla habría funcionado. Sin costes: tómalo orientativo.

### 6 · Correlación  `[Básico]`
**Para qué sirve:** Te dice cómo de parecidos se mueven varios activos (para no poner todo en lo mismo).

**Cómo se usa:** Escribe varios activos y pulsa Correlación.

**Qué significa lo que ves:**
- Poco parecidos (correlación baja) = buena diversificación.

**👉 Qué hago con esto:** Antes de añadir una posición, mira que no duplica lo que ya tienes.

### 7 · Cartera  `[Intermedio]`
**Para qué sirve:** Calcula un reparto 'óptimo' clásico entre varios activos (busca el mejor premio por riesgo).

**Cómo se usa:** Escribe varios activos y pulsa Optimizar.

**Qué significa lo que ves:**
- Te da unos pesos sugeridos.

**👉 Qué hago con esto:** OJO: el método clásico es inestable. Para algo más robusto, usa 🧮 HRP.

### 🎯 Alpha (rigor)  `[Avanzado]`
**Para qué sirve:** ¿Tiene esta empresa una ventaja REAL a corto plazo, o es puro azar? Lo mide con honestidad.

**Cómo se usa:** Escribe una empresa y pulsa 'Medir ventaja'. Tarda ~1-2 min.

**Qué significa lo que ves:**
- Si el 'p-valor' es alto, NO hay ventaja (es lo normal en grandes empresas).
- La volatilidad sí es algo predecible; la dirección, no.

**👉 Qué hago con esto:** Te baja a la realidad: casi nadie le gana al azar a corto. Eso es bueno saberlo.

### 📏 Conformal  `[Avanzado]`
**Para qué sirve:** Hace una banda de predicción HONESTA: una que de verdad acierta el % de veces que dice.

**Cómo se usa:** Escribe una empresa y pulsa 'Calibrar banda'. Tarda ~30-60 s.

**Qué significa lo que ves:**
- 'Cobertura real' = cuántas veces el precio cayó dentro de la banda. Si ≈80%, la banda no miente.

**👉 Qué hago con esto:** Para cuando quieras una banda de incertidumbre en la que confiar de verdad.

### 🔔 Alertas  `[Básico]`
**Para qué sirve:** Vigila tu lista y te avisa si pasa algo (subida/bajada brusca, cruce, extremos).

**Cómo se usa:** Escribe tu lista y pulsa Escanear.

**Qué significa lo que ves:**
- Solo muestra lo que salta una alarma ahora mismo.

**👉 Qué hago con esto:** Para no estar mirando la pantalla todo el día. Te avisa de lo importante.

### 🏦 Cartera LP  `[Básico]`
**Para qué sirve:** Tu cartera de LARGO PLAZO en 3 botones: reparte tu dinero entre varios activos de forma robusta (HRP), te dice cuántas acciones comprar, y cada mes te da las órdenes de ajuste. También compara aportar cada mes (DCA) vs entrar de golpe.

**Cómo se usa:** 1) Escribe tus activos y tu capital → 'Crear cartera'. 2) Una vez al MES → 'Revisar rebalanceo'. 3) Cuando ejecutes los ajustes → 'He rebalanceado'.

**Qué significa lo que ves:**
- Te da acciones exactas a comprar y el cash que sobra.
- En la revisión, solo pide ajustar lo que se desvía más de 2.5 puntos.
- Dividendos incluidos (total return).

**👉 Qué hago con esto:** La forma tranquila de invertir a años vista: crea, revisa 1 vez al mes, y no toques más. Mezcla bolsa con oro/bonos para diversificar.

### 🏛️ Lab carteras  `[Básico]`
**Para qué sirve:** Compara TU cartera con las carteras más famosas de la historia (60/40, la Permanente de Browne, la All Weather de Dalio, Buffett 90/10, Bogleheads, Golden Butterfly) y te dice cómo habría aguantado el crash de 2008, el COVID y la inflación de 2022.

**Cómo se usa:** Escribe tu cartera con el formato TICKER:peso (ej. `SPY:50, GLD:20, TLT:20, AGG:10`), elige histórico y rebalanceo, y pulsa Analizar.

**Qué significa lo que ves:**
- **Sortino**: como el Sharpe pero solo castiga las caídas (subir mucho no es 'riesgo').
- **Omega**: cuántos euros ganas por cada euro que pierdes (>1 es bueno).
- **Ulcer**: el 'dolor' — mide lo profundas Y lo largas que son las caídas.
- **Calmar**: rentabilidad anual dividida por la peor caída.
- **Contribución al riesgo**: qué activo pone realmente los sustos (¡no es lo mismo que su peso!).
- **Captura alcista/bajista**: cuánto sigues al índice cuando sube y cuánto sufres cuando baja.

**👉 Qué hago con esto:** Mira la tabla de crisis y pregúntate en serio: ¿habrías aguantado sin vender? Si la respuesta es no, tu cartera es demasiado agresiva para ti. Todo viene explicado en cristiano debajo de las tablas.

### 🗞️ Informe  `[Básico]`
**Para qué sirve:** Tu resumen de la semana en un clic: señales de tu watchlist + riesgo de la cesta + titulares, en un documento Word.

**Cómo se usa:** Pulsa 'Generar informe ahora' (o doble clic en Informe_Semanal.bat). El Word aparece en la carpeta del proyecto.

**Qué significa lo que ves:**
- Usa tu watchlist guardada (el desplegable 💾 de arriba del todo).

**👉 Qué hago con esto:** Genera uno cada fin de semana y revísalo con calma antes de que abra el mercado.

## 📚 Diccionario sencillo

| Palabra | Qué significa (fácil) |
|---|---|
| **Acción / Ticker** | Una empresa que cotiza en bolsa. El ticker es su código corto (AAPL = Apple, MSFT = Microsoft). |
| **COMPRAR / MANTENER / VENDER** | El resumen final. COMPRAR = las señales apuntan a posible subida. MANTENER = ni claro sí ni no, mejor esperar. VENDER = apuntan a bajada. |
| **Forecast (predicción)** | Un dibujo de hacia dónde podría ir el precio. Aviso: a medio plazo NO es fiable; tómalo flojo. |
| **Tendencia** | Si el precio viene subiendo o bajando últimamente. |
| **Volatilidad** | Cuánto se mueve un precio. Alta = pega saltos grandes (más nervios, más riesgo). |
| **Sobrecompra / sobreventa** | Ha subido o bajado mucho muy rápido; podría girarse pronto. |
| **Stop** | El precio al que decides salir si la cosa va mal, para no perder de más. Se pone ANTES de entrar. |
| **Riesgo** | Cuánto dinero pierdes si el precio llega a tu stop. |
| **Drawdown** | La peor caída desde un máximo. 'Cuánto llegué a perder en el peor momento.' |
| **VaR** | Una estimación de la pérdida de un día malo (no del peor de todos, pero sí uno feo). |
| **Backtest** | Probar una idea con datos del pasado: '¿esto habría funcionado?'. |
| **Expectancy** | Cuánto ganas de media por operación, según tu historial. Positiva = vas bien. |
| **Paper trading** | Operar de mentira, con dinero ficticio, para practicar sin riesgo. |
| **Diversificación** | No poner todos los huevos en la misma cesta. |
| **Sharpe** | Una nota de 'ganancia ajustada al riesgo'. Más alta = mejor relación premio/sustos. |
| **Bate al mercado** | Hacerlo mejor que comprar el índice (SPY) y no tocar nada. Si no lo bates, mejor el índice. |
