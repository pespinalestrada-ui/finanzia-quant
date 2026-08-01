// Manual SENCILLO de FinanzIA (.docx) — lenguaje claro + capturas
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  LevelFormat, TableOfContents, PageBreak, BorderStyle, ImageRun,
  Table, TableRow, TableCell, WidthType, ShadingType,
} = require("docx");

// ---- capturas -------------------------------------------------------------
const SHOTS_DIR = path.join(__dirname, "manual_shots");
let shots = {};
try { shots = JSON.parse(fs.readFileSync(path.join(SHOTS_DIR, "mapa.json"), "utf8")); } catch (e) {}
const norm = s => (s || "").replace(/[^a-z0-9]/gi, "").toLowerCase();
const shotsNorm = {}; Object.keys(shots).forEach(k => shotsNorm[norm(k)] = shots[k]);
function pngSize(p){ const b=fs.readFileSync(p); return {w:b.readUInt32BE(16), h:b.readUInt32BE(20)}; }
function imagenDe(nombre){
  const fn = shots[nombre] || shotsNorm[norm(nombre)];
  if(!fn) return null;
  const p = path.join(SHOTS_DIR, fn); if(!fs.existsSync(p)) return null;
  const {w,h}=pngSize(p); const W=580, H=Math.round(580*h/w);
  const b={style:BorderStyle.SINGLE,size:4,color:"CCCCCC",space:4};
  return new Paragraph({ spacing:{before:60,after:160}, border:{top:b,bottom:b,left:b,right:b},
    children:[new ImageRun({type:"png",data:fs.readFileSync(p),transformation:{width:W,height:H},
      altText:{title:nombre,description:"Captura "+nombre,name:nombre}})] });
}

// ---- contenido (lenguaje sencillo) ----------------------------------------
// cada pantalla: n=nombre, nv=nivel, q=para qué (en cristiano), u=cómo se usa,
//                ve=[qué significa lo que ves], hacer=qué hago con esto
const grupos = [
{ titulo:"Bloque 1 · Para el día a día (empieza por aquí)", tabs:[
  { n:"📊 Factores", nv:"Básico",
    q:"Te ordena una lista de empresas de mejor a peor, según si están baratas, si vienen subiendo, si la empresa es sólida y si son tranquilas. Como un ranking para elegir candidatas.",
    u:"Escribe varias empresas separadas por comas (ej. AAPL, MSFT, NVDA) y pulsa Rankear.",
    ve:["La de arriba (rank 1) es la 'mejor nota' del grupo.","La columna 'señal' te dice: COMPRAR (las mejores), EVITAR (las peores) o neutral.","Los números z son solo la posición dentro del grupo: + es bueno, − es malo."],
    hacer:"Úsala para PRESELECCIONAR. Quédate con las 2-3 de arriba y luego míralas a fondo en la pantalla ★ Veredicto." },
  { n:"★ Veredicto", nv:"Básico",
    q:"La pantalla estrella. Analiza UNA empresa por todos lados y te da un resumen: COMPRAR, MANTENER o VENDER, con el porqué.",
    u:"Escribe una empresa (ej. AAPL) y pulsa 'Analizar TODO'. Los dos checkbox son opcionales (déjalos sin marcar al principio).",
    ve:["Arriba, el veredicto en grande con un color: 🟢 comprar, 🟡 esperar, 🔴 vender.","La tabla son los 'motivos' (tendencia, fuerza, etc.). No hace falta entender cada fila.","MANTENER = ni claro sí ni claro no, mejor esperar."],
    hacer:"Es tu pantalla principal. Si dice MANTENER, no operes. NO es un consejo garantizado: es un resumen para ayudarte a decidir." },
  { n:"🌡️ Mercado", nv:"Básico",
    q:"El 'tiempo' de la bolsa antes de mirar empresas: ¿hay miedo o codicia?, ¿está nervioso el mercado?",
    u:"Pulsa 'Medir mercado' (opcional: escribe una empresa para ver sus datos básicos).",
    ve:["Miedo extremo suele ser buen momento (la gente vende por pánico). Codicia extrema, cuidado.","El VIX alto = mercado nervioso: conviene ir con menos dinero."],
    hacer:"Míralo al empezar el día. Si hay pánico o euforia extrema, sé más prudente." },
  { n:"💰 Rentabilidad", nv:"Básico",
    q:"Los cuatro números con los que se juzga si una empresa es un buen NEGOCIO (no si la acción va a subir): ROE, ROA, ROIC y BPA.",
    u:"Escribe una empresa y pulsa 'Medir rentabilidad'. En el segundo hueco puedes poner otras para compararlas, pero solo tiene sentido comparar empresas del MISMO sector.",
    ve:["**ROE**: por cada 100 € que han puesto los dueños, cuánto gana al año. Es el más famoso y el más tramposo: sube solo con endeudarse.","**ROA**: por cada 100 € que la empresa mueve (suyos y prestados), cuánto gana. La deuda no lo maquilla.","**ROIC**: el más honesto. Lo que renta el dinero que de verdad trabaja en el negocio. Por encima de ~10% (lo que cuesta el capital) crea valor; por debajo, lo destruye.","**BPA**: cuánto beneficio toca a cada acción. El único en euros, no en %.","El aviso de **'el ROE es X veces el ROA'**: si sale 1-2 el ROE es negocio de verdad; si sale 5 o más, buena parte es deuda.","La gráfica saca 4 ejercicios: un año bueno es suerte, cuatro seguidos es un negocio."],
    hacer:"Úsalo para DESCARTAR, no para acertar. Son datos contables del pasado: dicen cómo ha ido la empresa, no hacia dónde va la acción. En bancos el ROIC no aplica y un ROA del 0,8% es normal, no un desastre." },
  { n:"8 · Sentimiento", nv:"Básico",
    q:"Lee las noticias recientes de una empresa y te dice si suenan buenas, malas o neutras.",
    u:"Escribe una empresa y pulsa 'Analizar noticias'. La primera vez tarda ~1 min (carga el modelo).",
    ve:["Un resumen: POSITIVO / NEGATIVO / NEUTRAL.","Las noticias viejas pesan menos que las de hoy."],
    hacer:"Úsalo como contexto: ¿hay alguna noticia gorda detrás del movimiento? Es una señal más, no la decisión." },
  { n:"2 · Indicadores", nv:"Básico",
    q:"Los 'termómetros' técnicos clásicos de una empresa (si está cara/barata a corto, si cruza medias, etc.).",
    u:"Escribe una empresa y pulsa Calcular.",
    ve:["Cada indicador da una lectura (sobreventa/sobrecompra, cruce al alza, etc.).","Ninguno decide solo; el ★ Veredicto ya los junta por ti."],
    hacer:"Para curiosear el detalle técnico. Si no quieres complicarte, usa directamente el Veredicto." },
  { n:"3 · Screener", nv:"Básico",
    q:"Pasa un filtro rápido a una lista de empresas y te dice cuáles tienen más empuje ahora mismo.",
    u:"Escribe varias empresas y pulsa Escanear.",
    ve:["Ordenadas por fuerza/empuje."],
    hacer:"Para encontrar candidatas rápido. Luego confírmalas en el ★ Veredicto." },
  { n:"4 · Señales", nv:"Básico",
    q:"Busca avisos claros de compra/venta (cruces, rebotes) en una lista de empresas.",
    u:"Escribe varias empresas y pulsa 'Buscar señales'.",
    ve:["Solo muestra lo accionable ahora."],
    hacer:"Un radar de oportunidades. Confirma siempre con el Veredicto antes de nada." },
  { n:"9 · Tamaño posición", nv:"Básico",
    q:"Te dice CUÁNTAS acciones comprar para arriesgar solo lo que tú decidas (p. ej. el 1% de tu dinero).",
    u:"Pon tu capital y el % de riesgo. Pon entrada y stop, o solo el ticker para que los calcule solos.",
    ve:["Te da el número de acciones.","El 'stop' es el precio al que saldrías si va mal, para no perder de más."],
    hacer:"Úsala SIEMPRE antes de comprar: define cuánto puedes perder ANTES de entrar. Regla de oro." },
  { n:"🛡️ Riesgo", nv:"Básico",
    q:"Mide cuánto podrías perder con tu lista de empresas: en un día malo, en el peor momento, y si están muy 'pegadas' entre sí.",
    u:"Escribe tu lista de empresas y pulsa 'Medir riesgo'.",
    ve:["VaR = pérdida fea de un día malo. Drawdown = la peor caída que habrías sufrido.","Si las empresas están poco relacionadas, diversificas bien (menos sustos)."],
    hacer:"Antes de juntar varias posiciones, comprueba que no es una bomba de riesgo." },
  { n:"1 · Forecast", nv:"Básico",
    q:"Dibuja una predicción del precio a futuro con una banda de '¿hasta dónde podría ir?'.",
    u:"Escribe una empresa, elige el motor (deja Prophet) y pulsa Forecast.",
    ve:["La línea es la predicción; la banda, la incertidumbre (cuanto más lejos, más ancha).","AVISO IMPORTANTE: a 90 días esto NO acierta mejor que el azar. Tómalo flojo."],
    hacer:"Míralo como orientación, NUNCA como una bola de cristal. El valor real está en el riesgo, no en adivinar el precio." },
]},
{ titulo:"Bloque 2 · Practicar y operar (sin dinero real)", tabs:[
  { n:"📒 Diario", nv:"Básico",
    q:"Un cuaderno de operaciones DE MENTIRA. Apuntas tus compras/ventas imaginarias y te dice si tu método gana o pierde.",
    u:"Abrir = apuntar una compra (empresa, precio, stop). Cerrar = poner el precio al que 'saliste'. Refrescar para ver tus números.",
    ve:["Con 20-30 operaciones ya sabes si tu método tiene ventaja.","'Expectancy' = cuánto ganas de media por operación. Positivo = bien."],
    hacer:"ÚSALO antes de jugarte un euro real. Si en mentira pierdes, en real también." },
  { n:"🦙 Alpaca Paper", nv:"Intermedio",
    q:"Como el Diario pero conectado a una cuenta de prácticas REAL (Alpaca): dinero ficticio, precios en vivo, órdenes de mentira.",
    u:"Refrescar = ver tu cuenta. Precio = cotización en vivo. Enviar orden = TÚ pulsas el botón para 'comprar' de mentira.",
    ve:["Es dinero FICTICIO (empiezas con ~100.000 $ de prácticas).","Cada orden puede apuntarse sola en tu Diario."],
    hacer:"Practica aquí semanas antes de pensar en real. Las órdenes las disparas tú; el programa nunca opera solo. Necesita una clave gratis de Alpaca." },
  { n:"⏱️ Intradía", nv:"Intermedio",
    q:"Para operar dentro del mismo día. Lo primero: el 🚦 SEMÁFORO te dice si hoy conviene operar al alza, a la baja o no operar. Y el backtest prueba estrategias DESCONTANDO los costes.",
    u:"'🚦 Semáforo de HOY' = ¿opero o no? (usa datos EN VIVO de Alpaca si tienes clave). '📡 Snapshot EN VIVO' = foto en tiempo real. 'Backtest' = prueba una estrategia (orb/vwap/ema9) con costes. 'Escanear varios' = compara empresas.",
    ve:["El semáforo: 🟢 sesgo largo · 🔴 sesgo corto · 🟡 no operes hoy (la respuesta más frecuente y a menudo la correcta).","En el backtest, lo importante es la ganancia NETA (tras costes), no la bruta.","Hay 3 estrategias para comparar: rotura de apertura (orb), retorno al VWAP (vwap) y pullback a la media (ema9)."],
    hacer:"Empieza el día por el semáforo. Si está 🟡, no operes: no operar también es ganar. Si operas, respeta el stop que te sugiere." },
  { n:"🪙 Veredicto Cripto", nv:"Intermedio",
    q:"El mismo ★ Veredicto pero para criptomonedas (Bitcoin, Ethereum…).",
    u:"Escribe el cripto en formato BTC-USD, ETH-EUR… y pulsa Analizar.",
    ve:["Igual que el Veredicto normal: COMPRAR / MANTENER / VENDER.","Incluye el 'miedo y codicia' del mundo cripto."],
    hacer:"Recuerda que la cripto se mueve muchísimo más: ve con más cuidado y menos dinero." },
]},
{ titulo:"Bloque 3 · El sistema automático (avanzado)", tabs:[
  { n:"📡 Señales", nv:"Intermedio",
    q:"Te dice 'qué tocaría operar hoy' de tu lista, según el método del Veredicto.",
    u:"Escribe tu lista y pulsa 'Generar señales'.",
    ve:["Un ranking con COMPRAR / MANTENER / VENDER."],
    hacer:"Es el primer paso del 'piloto automático'. Trátalo como prácticas, no como dinero seguro." },
  { n:"⚖️ Plan / Riesgo", nv:"Intermedio",
    q:"Convierte esas señales en un plan concreto: cuántas acciones de cada una, con su stop y su riesgo.",
    u:"Pon tu capital y pulsa 'Generar plan'.",
    ve:["Te dice acciones, coste y riesgo de cada posición.","Avisa si te pasas de riesgo."],
    hacer:"Revisa que el riesgo total te parece asumible antes de nada." },
  { n:"🤖 Sistema", nv:"Avanzado",
    q:"Todo junto y de un clic: genera el plan y, si TÚ confirmas, lo manda a la cuenta de prácticas (Alpaca) y lo apunta en el Diario.",
    u:"1) 'Generar plan'. 2) Marca 'Confirmo enviar a PAPER' y pulsa 'Ejecutar'.",
    ve:["Sin marcar la casilla, no envía nada.","Es dinero ficticio."],
    hacer:"El 'robot' de prácticas. Empléalo solo cuando entiendas los pasos anteriores." },
  { n:"📈 Rendimiento", nv:"Intermedio",
    q:"¿Lo estás haciendo mejor que comprar el mercado y no tocar nada? Te lo compara con el índice (SPY).",
    u:"Pon tu capital y pulsa 'Medir rendimiento'.",
    ve:["Tu ganancia vs SPY (comprar y mantener).","Si no le ganas, indexarte (comprar el índice) sería mejor."],
    hacer:"La prueba de la verdad. Si tu sistema no bate a SPY, mejor algo simple." },
  { n:"🎲 Monte Carlo", nv:"Avanzado",
    q:"Simula MILES de futuros posibles para ver el abanico de lo que podría pasar (no una predicción, sino el rango).",
    u:"Precio: una empresa. Sistema: pon tu win-rate/payoff (o usa tu diario).",
    ve:["En 'sistema', lo clave es la PROBABILIDAD DE RUINA (acabar reventado) y la peor caída.","Un sistema con buena media pero alta ruina NO sirve."],
    hacer:"Úsalo para ver si tu método aguanta la mala suerte, no solo la buena." },
  { n:"🔬 Validar Veredicto", nv:"Avanzado",
    q:"Comprueba con datos del pasado si el Veredicto realmente acierta o solo lo parece.",
    u:"Escribe varias empresas y pulsa Validar. Tarda ~1 min.",
    ve:["El número clave ('Deflated Sharpe') dice si la ventaja es real o suerte.","Resultado honesto habitual: en grandes empresas, no hay ventaja fácil."],
    hacer:"La prueba de honestidad. Te recuerda que esto mide, no promete milagros." },
  { n:"🛞 Vol objetivo", nv:"Intermedio",
    q:"EL ALGORITMO del proyecto. No intenta adivinar si el mercado sube o baja (eso ya medimos que no se puede): ajusta CUÁNTO estás invertido según la tormenta prevista. En calma, dentro; cuando la volatilidad se dispara, se retira solo y el resto queda en efectivo.",
    u:"Escribe un activo (SPY, QQQ, tu ETF...), elige la volatilidad que quieres soportar (12% es un buen punto de partida) y pulsa Probar algoritmo. El botón 🔬 Robustez comprueba si funciona con cualquier configuración o solo con una afortunada.",
    ve:["**Qué hace hoy**: el % que deberías tener invertido ahora mismo.","La tabla compara el algoritmo contra comprar y mantener: fíjate sobre todo en la **peor caída** y el **Sharpe**.","**En las crisis**: cuánto habrías perdido en 2008, COVID y 2022 frente a estar siempre dentro.","El **coste de operar ya está descontado** de todos los números."],
    hacer:"Lo que busca no es ganar más, sino **ganar parecido pasando mucho menos miedo**. Si la peor caída baja a la mitad y el Sharpe sube, está haciendo su trabajo. Practícalo en paper antes de aplicarlo con dinero." },
  { n:"🧪 CPCV", nv:"Avanzado",
    q:"La validación más dura de todas: parte la historia en bloques y prueba la estrategia en MUCHAS combinaciones distintas fuera de muestra, con cortafuegos para que el futuro no se cuele.",
    u:"Escribe tu universo y pulsa Validar. Tarda 1-2 minutos.",
    ve:["**PBO**: probabilidad de sobreajuste. Alto (>50%) = elegir 'la mejor configuración' del pasado NO funciona después.","**Sharpe ± margen de error**: si el intervalo cruza 0, no puedes afirmar que ganas.","**Deflated Sharpe**: descuenta el azar de haber probado muchas veces. Necesita >95% para creerse."],
    hacer:"Es el juez más severo. Si una idea sobrevive aquí, es seria; si no, mejor saberlo antes de arriesgar dinero." },
  { n:"📊 Backtest sist.", nv:"Avanzado",
    q:"Prueba la estrategia completa sobre años pasados (con costes) y la compara con el índice.",
    u:"Escribe un universo de empresas y pulsa Backtestear.",
    ve:["Ganancia, riesgo y peor caída vs SPY.","Que ganara en el pasado NO garantiza el futuro."],
    hacer:"Mídelo, pero no te lo creas a pies juntillas: un buen pasado puede ser casualidad." },
]},
{ titulo:"Bloque 4 · Matemática de fondo (solo si te pica la curiosidad)", tabs:[
  { n:"🔗 Pairs (cointegración)", nv:"Avanzado",
    q:"Busca PAREJAS de activos que se mueven juntos. Cuando se separan mucho, apuestas a que vuelven a juntarse.",
    u:"Escribe varias empresas y pulsa 'Buscar pares'.",
    ve:["Te da parejas y cuándo están 'estiradas' (z alto).","Es una técnica neutral (no dependes de que la bolsa suba)."],
    hacer:"Avanzado. Es de lo poco con ventaja real, pero requiere cuidado. Para más adelante." },
  { n:"🧮 HRP Cartera", nv:"Avanzado",
    q:"Reparte tu dinero entre varios activos de forma más robusta que el método clásico (que es inestable).",
    u:"Escribe varios activos y pulsa 'Comparar asignación'.",
    ve:["Compara métodos: el robusto suele dar menos sustos (menos volatilidad)."],
    hacer:"Si algún día gestionas una cartera de varios activos, usa este reparto en vez del clásico." },
  { n:"📉 EVT Colas", nv:"Avanzado",
    q:"Mide bien el riesgo de CRASH (las caídas gordas), que los métodos normales subestiman.",
    u:"Escribe una empresa/índice y pulsa 'Medir cola'.",
    ve:["Compara el riesgo extremo 'normal' vs el real: el normal casi siempre se queda corto."],
    hacer:"Para entender cuánto puede doler de verdad un día negro." },
  { n:"🌀 Régimen (HMM)", nv:"Avanzado",
    q:"Detecta en qué 'modo' está el mercado: calma alcista, nervios, o lateral.",
    u:"Escribe un índice (ej. SPY) y pulsa 'Detectar régimen'.",
    ve:["🟢 calma = bien para seguir tendencia. 🔴 nervios = baja el tamaño. 🟡 lateral = cautela."],
    hacer:"Úsalo como semáforo general antes de operar tendencia." },
  { n:"🎯 Meta-labeling", nv:"Avanzado",
    q:"Un segundo filtro de inteligencia artificial que decide SI hacer caso a una señal (menos operaciones, pero mejores).",
    u:"Escribe una empresa y pulsa 'Medir'. Tarda ~1-2 min.",
    ve:["Compara: con filtro aciertas más, pero operas menos.","A veces ayuda, a veces no: te lo dice con números."],
    hacer:"Curiosidad técnica. Muestra cómo afinar señales sin engañarte." },
  { n:"🧲 RMT (correlación)", nv:"Avanzado",
    q:"Limpia el 'ruido' de cómo se relacionan los activos entre sí, usando una idea de la física.",
    u:"Escribe varios activos (4 o más) y pulsa 'Limpiar correlación'.",
    ve:["Te dice cuántas relaciones son señal de verdad y cuántas son ruido."],
    hacer:"Mejora el reparto de cartera cuando manejas muchos activos." },
  { n:"🛰️ Kalman (pairs)", nv:"Avanzado",
    q:"Versión 'lista' de las parejas: la relación entre dos activos cambia con el tiempo, y esto la sigue al día.",
    u:"Escribe dos activos (A y B) y pulsa 'Kalman dinámico'. Ej: KO y PEP.",
    ve:["Compara la relación 'fija' vs la 'que se adapta'. La que se adapta es más fiable."],
    hacer:"Si haces pares, usa esta versión: no se queda anclada al pasado." },
  { n:"🎲 Opciones", nv:"Avanzado",
    q:"Valora una OPCIÓN (el derecho a comprar o vender algo a un precio fijado) con la fórmula Black-Scholes, y te da sus 'Griegas': a qué es sensible tu posición.",
    u:"Escribe un ticker de EEUU para ver la cadena real de opciones, o déjalo vacío y juega con strike/días/volatilidad para una opción teórica.",
    ve:["**Delta**: cuánto gana la opción si el activo sube 1 € (y, aproximadamente, la probabilidad de acabar valiendo algo).","**Gamma**: cómo se acelera el Delta. Alto = la posición se mueve cada vez más rápido.","**Vega**: cuánto gana si sube la volatilidad. Comprar opciones es apostar a que habrá movimiento.","**Theta**: lo que te quita el reloj cada día. El comprador pierde solo con que pase el tiempo."],
    hacer:"Antes de tocar una opción, mira Theta: si es alta, el tiempo juega en tu contra. Es el único apartado que trata derivados; el resto de la app son acciones." },
  { n:"⏳ OU óptimo", nv:"Avanzado",
    q:"Para pares (dos activos que van de la mano): calcula el 'muelle' que los une y te dice el momento EXACTO de cerrar, con matemáticas en vez de a ojo.",
    u:"Escribe dos activos (ej. KO y PEP, o EWA y EWC), tu capital y el coste de operar. Pulsa Calibrar.",
    ve:["**θ (fuerza del muelle)** y **half-life**: cómo de rápido vuelve el spread a su sitio. Corto = tradeable.","**σ_eq**: cuánto se mueve el spread por puro azar. El stop se pone MÁS ALLÁ de eso.","**SALIDA ÓPTIMA**: el nivel donde compensa cerrar, descontando lo que cuesta operar. No es 'cerrar en la media': suele compensar esperar algo más."],
    hacer:"Si operas pares, usa este umbral en vez de decidir por intuición. Cuanto más caro sea operar, más lejos se pone la salida (la herramienta ya lo calcula)." },
  { n:"📡 Entropía (lead-lag)", nv:"Avanzado",
    q:"Descubre qué activo MUEVE a cuál (quién manda y quién sigue), incluso de formas que la correlación no ve.",
    u:"Escribe varios activos y pulsa 'Medir flujo info'.",
    ve:["LÍDERES = mandan información. SEGUIDORES = la reciben."],
    hacer:"Para entender quién marca el ritmo en un grupo de activos." },
]},
{ titulo:"Bloque 5 · Extras de análisis", tabs:[
  { n:"5 · Backtest", nv:"Básico",
    q:"Prueba una regla simple (medias, RSI) en una empresa y la compara con 'comprar y no tocar'.",
    u:"Escribe una empresa, elige estrategia y pulsa Backtest.",
    ve:["Si la regla no gana a 'comprar y mantener', no aporta."],
    hacer:"Para ver si una idea sencilla habría funcionado. Sin costes: tómalo orientativo." },
  { n:"6 · Correlación", nv:"Básico",
    q:"Te dice cómo de parecidos se mueven varios activos (para no poner todo en lo mismo).",
    u:"Escribe varios activos y pulsa Correlación.",
    ve:["Poco parecidos (correlación baja) = buena diversificación."],
    hacer:"Antes de añadir una posición, mira que no duplica lo que ya tienes." },
  { n:"7 · Cartera", nv:"Intermedio",
    q:"Calcula un reparto 'óptimo' clásico entre varios activos (busca el mejor premio por riesgo).",
    u:"Escribe varios activos y pulsa Optimizar.",
    ve:["Te da unos pesos sugeridos."],
    hacer:"OJO: el método clásico es inestable. Para algo más robusto, usa 🧮 HRP." },
  { n:"🎯 Alpha (rigor)", nv:"Avanzado",
    q:"¿Tiene esta empresa una ventaja REAL a corto plazo, o es puro azar? Lo mide con honestidad.",
    u:"Escribe una empresa y pulsa 'Medir ventaja'. Tarda ~1-2 min.",
    ve:["Si el 'p-valor' es alto, NO hay ventaja (es lo normal en grandes empresas).","La volatilidad sí es algo predecible; la dirección, no."],
    hacer:"Te baja a la realidad: casi nadie le gana al azar a corto. Eso es bueno saberlo." },
  { n:"📏 Conformal", nv:"Avanzado",
    q:"Hace una banda de predicción HONESTA: una que de verdad acierta el % de veces que dice.",
    u:"Escribe una empresa y pulsa 'Calibrar banda'. Tarda ~30-60 s.",
    ve:["'Cobertura real' = cuántas veces el precio cayó dentro de la banda. Si ≈80%, la banda no miente."],
    hacer:"Para cuando quieras una banda de incertidumbre en la que confiar de verdad." },
  { n:"🔔 Alertas", nv:"Básico",
    q:"Vigila tu lista y te avisa si pasa algo (subida/bajada brusca, cruce, extremos).",
    u:"Escribe tu lista y pulsa Escanear.",
    ve:["Solo muestra lo que salta una alarma ahora mismo."],
    hacer:"Para no estar mirando la pantalla todo el día. Te avisa de lo importante." },
  { n:"🏦 Cartera LP", nv:"Básico",
    q:"Tu cartera de LARGO PLAZO en 3 botones: reparte tu dinero entre varios activos de forma robusta (HRP), te dice cuántas acciones comprar, y cada mes te da las órdenes de ajuste. También compara aportar cada mes (DCA) vs entrar de golpe.",
    u:"1) Escribe tus activos y tu capital → 'Crear cartera'. 2) Una vez al MES → 'Revisar rebalanceo'. 3) Cuando ejecutes los ajustes → 'He rebalanceado'.",
    ve:["Te da acciones exactas a comprar y el cash que sobra.","En la revisión, solo pide ajustar lo que se desvía más de 2.5 puntos.","Dividendos incluidos (total return)."],
    hacer:"La forma tranquila de invertir a años vista: crea, revisa 1 vez al mes, y no toques más. Mezcla bolsa con oro/bonos para diversificar." },
  { n:"🏛️ Lab carteras", nv:"Básico",
    q:"Compara TU cartera con las carteras más famosas de la historia (60/40, la Permanente de Browne, la All Weather de Dalio, Buffett 90/10, Bogleheads, Golden Butterfly) y te dice cómo habría aguantado el crash de 2008, el COVID y la inflación de 2022.",
    u:"Escribe tu cartera con el formato TICKER:peso (ej. `SPY:50, GLD:20, TLT:20, AGG:10`), elige histórico y rebalanceo, y pulsa Analizar.",
    ve:["**Sortino**: como el Sharpe pero solo castiga las caídas (subir mucho no es 'riesgo').","**Omega**: cuántos euros ganas por cada euro que pierdes (>1 es bueno).","**Ulcer**: el 'dolor' — mide lo profundas Y lo largas que son las caídas.","**Calmar**: rentabilidad anual dividida por la peor caída.","**Contribución al riesgo**: qué activo pone realmente los sustos (¡no es lo mismo que su peso!).","**Captura alcista/bajista**: cuánto sigues al índice cuando sube y cuánto sufres cuando baja."],
    hacer:"Mira la tabla de crisis y pregúntate en serio: ¿habrías aguantado sin vender? Si la respuesta es no, tu cartera es demasiado agresiva para ti. Todo viene explicado en cristiano debajo de las tablas." },
  { n:"🗞️ Informe", nv:"Básico",
    q:"Tu resumen de la semana en un clic: señales de tu watchlist + riesgo de la cesta + titulares, en un documento Word.",
    u:"Pulsa 'Generar informe ahora' (o doble clic en Informe_Semanal.bat). El Word aparece en la carpeta del proyecto.",
    ve:["Usa tu watchlist guardada (el desplegable 💾 de arriba del todo)."],
    hacer:"Genera uno cada fin de semana y revísalo con calma antes de que abra el mercado." },
]},
];

// ---- glosario sencillo ----------------------------------------------------
const glos = [
  ["Acción / Ticker","Una empresa que cotiza en bolsa. El ticker es su código corto (AAPL = Apple, MSFT = Microsoft)."],
  ["COMPRAR / MANTENER / VENDER","El resumen final. COMPRAR = las señales apuntan a posible subida. MANTENER = ni claro sí ni no, mejor esperar. VENDER = apuntan a bajada."],
  ["Forecast (predicción)","Un dibujo de hacia dónde podría ir el precio. Aviso: a medio plazo NO es fiable; tómalo flojo."],
  ["Tendencia","Si el precio viene subiendo o bajando últimamente."],
  ["Volatilidad","Cuánto se mueve un precio. Alta = pega saltos grandes (más nervios, más riesgo)."],
  ["Sobrecompra / sobreventa","Ha subido o bajado mucho muy rápido; podría girarse pronto."],
  ["Stop","El precio al que decides salir si la cosa va mal, para no perder de más. Se pone ANTES de entrar."],
  ["Riesgo","Cuánto dinero pierdes si el precio llega a tu stop."],
  ["Drawdown","La peor caída desde un máximo. 'Cuánto llegué a perder en el peor momento.'"],
  ["VaR","Una estimación de la pérdida de un día malo (no del peor de todos, pero sí uno feo)."],
  ["Backtest","Probar una idea con datos del pasado: '¿esto habría funcionado?'."],
  ["Expectancy","Cuánto ganas de media por operación, según tu historial. Positiva = vas bien."],
  ["Paper trading","Operar de mentira, con dinero ficticio, para practicar sin riesgo."],
  ["Diversificación","No poner todos los huevos en la misma cesta."],
  ["Sharpe","Una nota de 'ganancia ajustada al riesgo'. Más alta = mejor relación premio/sustos."],
  ["Bate al mercado","Hacerlo mejor que comprar el índice (SPY) y no tocar nada. Si no lo bates, mejor el índice."],
];

// ---- tabla 'según lo que quieras' -----------------------------------------
const tareas = [
  ["Saber si comprar/vender UNA empresa","★ Veredicto"],
  ["Elegir entre VARIAS empresas","📊 Factores  (y luego ★ Veredicto)"],
  ["Ver el ambiente general del mercado","🌡️ Mercado"],
  ["Saber cuántas acciones comprar","9 · Tamaño posición"],
  ["Ver el riesgo de mi lista","🛡️ Riesgo"],
  ["Practicar sin dinero real","📒 Diario  +  🦙 Alpaca Paper"],
  ["Que me avisen de movimientos","🔔 Alertas"],
  ["Ver noticias de una empresa","8 · Sentimiento"],
  ["Saber si mi método de verdad funciona","📈 Rendimiento  +  🔬 Validar Veredicto"],
];

// ---- helpers de formato ----------------------------------------------------
const azul="1F4E79", verde="2E7D32";
const h1=t=>new Paragraph({heading:HeadingLevel.HEADING_1,children:[new TextRun(t)]});
function h2(nombre,nivel){
  const col = nivel==="Básico"?verde : nivel==="Intermedio"?"B26A00":"B00020";
  return new Paragraph({heading:HeadingLevel.HEADING_2,spacing:{before:200,after:60},children:[
    new TextRun(nombre+"  "), new TextRun({text:"["+nivel+"]",bold:true,size:18,color:col})]});
}
const sub=(label,txt)=>new Paragraph({spacing:{before:60,after:30},children:[
  new TextRun({text:label+": ",bold:true,color:azul}), new TextRun(txt)]});
const bullet=t=>new Paragraph({numbering:{reference:"b",level:0},children:[new TextRun(t)]});
function tabla2col(filas,c1,c2,w1,w2){
  const bd={style:BorderStyle.SINGLE,size:1,color:"CCCCCC"}; const bs={top:bd,bottom:bd,left:bd,right:bd};
  const cab=new TableRow({tableHeader:true,children:[c1,c2].map((t,i)=>new TableCell({borders:bs,width:{size:i?w2:w1,type:WidthType.DXA},
    shading:{fill:"D5E8F0",type:ShadingType.CLEAR},margins:{top:60,bottom:60,left:120,right:120},
    children:[new Paragraph({children:[new TextRun({text:t,bold:true})]})]}))});
  const rows=[cab].concat(filas.map(f=>new TableRow({children:f.map((t,i)=>new TableCell({borders:bs,width:{size:i?w2:w1,type:WidthType.DXA},
    margins:{top:60,bottom:60,left:120,right:120},children:[new Paragraph({children:[new TextRun(t)]})]}))})));
  return new Table({width:{size:w1+w2,type:WidthType.DXA},columnWidths:[w1,w2],rows});
}

// ---- montaje ---------------------------------------------------------------
const children=[];
children.push(new Paragraph({spacing:{before:2200},alignment:AlignmentType.CENTER,children:[new TextRun({text:"FinanzIA",bold:true,size:60,color:azul})]}));
children.push(new Paragraph({alignment:AlignmentType.CENTER,spacing:{before:160},children:[new TextRun({text:"Manual fácil — explicado en cristiano",size:30})]}));
children.push(new Paragraph({alignment:AlignmentType.CENTER,spacing:{before:120},children:[new TextRun({text:"Qué es cada pantalla, cómo se usa y qué hacer con ella",italics:true,size:24})]}));
children.push(new Paragraph({alignment:AlignmentType.CENTER,spacing:{before:600},children:[new TextRun({text:"Aviso: esto es una herramienta de aprendizaje. No da consejos de inversión ni mueve dinero. Practica siempre primero sin dinero real.",italics:true,size:20,color:"888888"})]}));
children.push(new Paragraph({children:[new PageBreak()]}));

// Empieza aquí
children.push(h1("Empieza aquí (léelo, son 2 minutos)"));
children.push(new Paragraph({children:[new TextRun({text:"¿Qué es esto? ",bold:true,color:azul}),new TextRun("Un panel con muchas herramientas para analizar la bolsa. No tienes que usarlas todas. Cada pantalla lleva una etiqueta: ")]}));
children.push(new Paragraph({numbering:{reference:"b",level:0},children:[new TextRun({text:"[Básico] ",bold:true,color:verde}),new TextRun("empieza por estas.")]}));
children.push(new Paragraph({numbering:{reference:"b",level:0},children:[new TextRun({text:"[Intermedio] ",bold:true,color:"B26A00"}),new TextRun("cuando te sientas cómodo.")]}));
children.push(new Paragraph({numbering:{reference:"b",level:0},children:[new TextRun({text:"[Avanzado] ",bold:true,color:"B00020"}),new TextRun("sáltatelas al principio; son de matemática fina.")]}));
children.push(new Paragraph({spacing:{before:160},children:[new TextRun({text:"La regla de oro: ",bold:true,color:azul}),new TextRun("nunca pongas dinero real hasta haberlo practicado de mentira (pantallas Diario y Alpaca Paper). Y recuerda: predecir el precio exacto NO se puede; el valor está en controlar el riesgo y no engañarte.")]}));

children.push(new Paragraph({heading:HeadingLevel.HEADING_2,children:[new TextRun("Tus primeros 15 minutos")]}));
["Abre 🌡️ Mercado y mira si hay miedo o codicia (el 'tiempo' de hoy).",
 "Escribe una empresa en ★ Veredicto y pulsa 'Analizar TODO'. Lee el color y el motivo.",
 "Si dice COMPRAR, ve a 9 · Tamaño posición para saber cuántas acciones (y tu stop).",
 "Apunta esa compra de mentira en 📒 Diario. Repite. Tras 20-30, mira si ganarías."]
 .forEach(s=>children.push(new Paragraph({numbering:{reference:"n",level:0},children:[new TextRun(s)]})));

children.push(new Paragraph({heading:HeadingLevel.HEADING_2,spacing:{before:160},children:[new TextRun("Según lo que quieras hacer")]}));
children.push(tabla2col(tareas,"Quiero…","Ve a esta pantalla",4680,4680));
children.push(new Paragraph({children:[new PageBreak()]}));

// Diccionario
children.push(h1("Diccionario sencillo"));
children.push(new Paragraph({spacing:{after:120},children:[new TextRun("Palabras que verás, en lenguaje normal:")]}));
children.push(tabla2col(glos,"Palabra","Qué significa (fácil)",2600,6760));
children.push(new Paragraph({children:[new PageBreak()]}));

// Índice
children.push(h1("Índice de pantallas"));
children.push(new TableOfContents("Tabla de contenidos",{hyperlink:true,headingStyleRange:"1-2"}));

// Pantallas
grupos.forEach(g=>{
  children.push(new Paragraph({children:[new PageBreak()]}));
  children.push(h1(g.titulo));
  g.tabs.forEach(t=>{
    children.push(h2(t.n,t.nv));
    children.push(sub("Para qué sirve (en cristiano)",t.q));
    children.push(sub("Cómo se usa",t.u));
    children.push(new Paragraph({spacing:{before:60,after:20},children:[new TextRun({text:"Qué significa lo que ves:",bold:true,color:azul})]}));
    t.ve.forEach(x=>children.push(bullet(x)));
    children.push(new Paragraph({spacing:{before:40,after:20},children:[new TextRun({text:"Qué hago con esto: ",bold:true,color:verde}),new TextRun(t.hacer)]}));
    const img=imagenDe(t.n);
    if(img){ children.push(new Paragraph({spacing:{before:100,after:10},children:[new TextRun({text:"La pantalla:",italics:true,color:"888888",size:18})]})); children.push(img); }
  });
});

const doc=new Document({
  styles:{ default:{document:{run:{font:"Arial",size:22}}}, paragraphStyles:[
    {id:"Heading1",name:"Heading 1",basedOn:"Normal",next:"Normal",quickFormat:true,
      run:{size:30,bold:true,font:"Arial",color:azul},paragraph:{spacing:{before:240,after:160},outlineLevel:0}},
    {id:"Heading2",name:"Heading 2",basedOn:"Normal",next:"Normal",quickFormat:true,
      run:{size:25,bold:true,font:"Arial"},paragraph:{spacing:{before:200,after:80},outlineLevel:1,
        border:{bottom:{style:BorderStyle.SINGLE,size:4,color:"BBBBBB",space:2}}}},
  ]},
  numbering:{config:[
    {reference:"b",levels:[{level:0,format:LevelFormat.BULLET,text:"•",alignment:AlignmentType.LEFT,style:{paragraph:{indent:{left:520,hanging:260}}}}]},
    {reference:"n",levels:[{level:0,format:LevelFormat.DECIMAL,text:"%1.",alignment:AlignmentType.LEFT,style:{paragraph:{indent:{left:520,hanging:260}}}}]},
  ]},
  sections:[{properties:{page:{size:{width:12240,height:15840},margin:{top:1440,right:1440,bottom:1440,left:1440}}},children}],
});
Packer.toBuffer(doc).then(buf=>{ fs.writeFileSync("Manual_FinanzIA.docx",buf); console.log("OK -> Manual_FinanzIA.docx ("+buf.length+" bytes)"); });

// ---- ADEMÁS: emite la guía en Markdown para la pestaña 📖 del dashboard -----
(function(){
  const L = [];
  L.push("# 📖 Guía de usuario — FinanzIA");
  L.push("");
  L.push("_Herramienta de análisis y educación. No es recomendación de inversión. Practica siempre primero sin dinero real._");
  L.push("");
  L.push("## 🚀 Empieza aquí (2 minutos)");
  L.push("Cada pantalla lleva una etiqueta: **[Básico]** empieza por estas · **[Intermedio]** cuando te sientas cómodo · **[Avanzado]** matemática fina, sáltatelas al principio.");
  L.push("");
  L.push("**La regla de oro:** nunca pongas dinero real hasta haberlo practicado de mentira (Diario y Alpaca Paper). Predecir el precio exacto no se puede; el valor está en controlar el riesgo y no engañarse.");
  L.push("");
  L.push("**Tus primeros 15 minutos:**");
  L.push("1. Abre 🌡️ Mercado y mira si hay miedo o codicia.");
  L.push("2. Escribe una empresa en ★ Veredicto y pulsa 'Analizar TODO'.");
  L.push("3. Si dice COMPRAR, mira el plan sugerido (acciones y stop).");
  L.push("4. Apúntalo en 📒 Diario. Tras 20-30 operaciones sabrás si ganarías.");
  L.push("");
  L.push("**Según lo que quieras hacer:**");
  L.push("");
  L.push("| Quiero… | Ve a |");
  L.push("|---|---|");
  tareas.forEach(f=>L.push(`| ${f[0]} | ${f[1]} |`));
  L.push("");
  grupos.forEach(g=>{
    L.push(`## ${g.titulo}`);
    L.push("");
    g.tabs.forEach(t=>{
      L.push(`### ${t.n}  \`[${t.nv}]\``);
      L.push(`**Para qué sirve:** ${t.q}`);
      L.push("");
      L.push(`**Cómo se usa:** ${t.u}`);
      L.push("");
      L.push("**Qué significa lo que ves:**");
      t.ve.forEach(x=>L.push(`- ${x}`));
      L.push("");
      L.push(`**👉 Qué hago con esto:** ${t.hacer}`);
      L.push("");
    });
  });
  L.push("## 📚 Diccionario sencillo");
  L.push("");
  L.push("| Palabra | Qué significa (fácil) |");
  L.push("|---|---|");
  glos.forEach(f=>L.push(`| **${f[0]}** | ${f[1]} |`));
  L.push("");
  const destino = path.join(__dirname, "..", "cuant_trading", "dashboard", "guia_usuario.md");
  fs.writeFileSync(destino, L.join("\n"), "utf8");
  console.log("OK -> guia_usuario.md (" + L.length + " lineas)");
})();
