"""
Tema Nocturne para la Mesa cuantitativa (Gradio).

Sustituye el tema teal + banner navy por el sistema Nocturne:
fondo #161826, superficie #232532, acento blurple #9184d9, Inter,
densidad compacta y estados (hover / pressed / focus-visible) temáticos.

Uso en cuant_trading/dashboard/dashboard.py — dentro de build(), sustituye
el bloque _head / _theme / _css y el gr.HTML del banner por:

    from cuant_trading.dashboard.finanzia_theme import HEAD, THEME, CSS, TOPBAR_HTML
    ...
    with gr.Blocks(title="FinanzIA — Mesa cuantitativa", head=HEAD,
                   theme=THEME, css=CSS) as app:
        gr.HTML(TOPBAR_HTML(ticker="AAPL", precio="—", capital="10.000 €"))
"""

# --- tokens Nocturne (los mismos del sistema; no inventar hexes fuera de aquí) --
BG          = "#161826"
SURFACE     = "#232532"
PANEL       = "#20222f"   # superficie de las gráficas, un paso por debajo
BAR         = "#13151f"   # barra superior, un paso por debajo del fondo
TEXT        = "#e9e9ed"
ACCENT      = "#9184d9"
ACCENT_200  = "#e7e5fe"
ACCENT_300  = "#d2cefd"
ACCENT_400  = "#b5abfc"
ACCENT_700  = "#5d5294"
ACCENT_800  = "#423a6a"
ACCENT_900  = "#2b2741"
N300        = "#cfd3e5"
N400        = "#b2b6ca"
N500        = "#9397ab"
N600        = "#75798c"
N700        = "#595d6c"
N800        = "#3f424d"
N900        = "#292b31"
UP          = "#63b58e"   # verde desaturado: solo donde hay semántica
DOWN        = "#d9736b"
GOLD        = "#c9b273"

HEAD = ('<meta name="google" content="notranslate">'
        '<script>document.documentElement.lang="es";</script>'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        # Plus Jakarta Sans para la interfaz (cifras tabulares, buen peso 500/600)
        # y JetBrains Mono para los números de las tablas: alinean en columna.
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        'family=Plus+Jakarta+Sans:wght@400;500;600;700&'
        'family=JetBrains+Mono:wght@400;500&display=swap">'
        '<link rel="stylesheet" href="https://unpkg.com/@phosphor-icons/web@2.1.1'
        '/src/light/style.css">')


def _build_theme():
    """Tema Nocturne. Si la API de temas de Gradio cambia, cae al tema por
    defecto sin romper el panel (el CSS de abajo ya hace el 90 % del trabajo)."""
    import gradio as gr
    try:
        fonts = dict(
            font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "system-ui", "sans-serif"],
            font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "Consolas", "monospace"],
        )
    except Exception:
        fonts = {}
    try:
        base = gr.themes.Base(primary_hue="violet", secondary_hue="slate",
                              neutral_hue="slate", **fonts)
    except Exception:
        try:
            return gr.themes.Base()
        except Exception:
            return None

    # Cada token se fija en claro Y en oscuro con el mismo valor: el panel se ve
    # igual sin depender de la preferencia del sistema del usuario.
    vals = {}
    for name, value in [
        ("body_background_fill", BG),
        ("background_fill_primary", BG),
        ("background_fill_secondary", SURFACE),
        ("block_background_fill", SURFACE),
        ("block_border_color", N900),
        ("block_label_background_fill", SURFACE),
        ("block_label_text_color", N500),
        ("block_title_text_color", N400),
        ("body_text_color", TEXT),
        ("body_text_color_subdued", N500),
        ("border_color_primary", N800),
        ("border_color_accent", ACCENT_700),
        ("color_accent", ACCENT),
        ("color_accent_soft", ACCENT_900),
        ("input_background_fill", BG),
        ("input_border_color", N800),
        ("input_border_color_focus", ACCENT),
        ("input_placeholder_color", N600),
        ("button_primary_background_fill", "transparent"),
        ("button_primary_background_fill_hover", ACCENT_900),
        ("button_primary_border_color", ACCENT),
        ("button_primary_text_color", ACCENT_300),
        ("button_secondary_background_fill", "transparent"),
        ("button_secondary_background_fill_hover", N900),
        ("button_secondary_border_color", N800),
        ("button_secondary_text_color", N300),
        ("checkbox_background_color", BG),
        ("checkbox_background_color_selected", ACCENT),
        ("checkbox_border_color", N700),
        ("checkbox_border_color_focus", ACCENT),
        ("checkbox_border_color_selected", ACCENT),
        ("checkbox_label_background_fill", "transparent"),
        ("checkbox_label_background_fill_selected", ACCENT_900),
        ("checkbox_label_text_color", N300),
        ("slider_color", ACCENT),
        ("table_border_color", N900),
        ("table_even_background_fill", SURFACE),
        ("table_odd_background_fill", SURFACE),
        ("table_row_focus", ACCENT_900),
        ("panel_background_fill", SURFACE),
        ("panel_border_color", N900),
        ("accordion_text_color", N300),
        ("block_radius", "8px"),
        ("container_radius", "8px"),
        ("input_radius", "6px"),
        ("button_large_radius", "6px"),
        ("button_small_radius", "6px"),
        ("layout_gap", "10px"),
        ("block_padding", "11px"),
        ("section_header_text_weight", "500"),
    ]:
        vals[name] = value
        vals[name + "_dark"] = value

    try:
        return base.set(**vals)
    except TypeError:
        # alguna versión de Gradio no conoce todos los tokens: los aplicamos
        # uno a uno y descartamos los que no existan
        theme = base
        for k, v in vals.items():
            try:
                theme = theme.set(**{k: v})
            except Exception:
                pass
        return theme
    except Exception:
        return base


THEME = _build_theme()


CSS = """
/* ---- Mesa cuantitativa · Nocturne ------------------------------------- */
:root {
  --nb: #12141f; --ns: #1b1d2a; --nbar: #0e101a; --nt: #e9e9ed;
  --na: #9184d9; --na3: #d2cefd; --na7: #5d5294; --na9: #2b2741;
  --n4: #b2b6ca; --n5: #9397ab; --n6: #75798c; --n7: #595d6c;
  --n8: #3f424d; --n9: #292b31; --ndiv: rgba(233,233,237,.16);
  /* curva de movimiento: masa y frenada, no una rampa lineal */
  --eas: cubic-bezier(.32,.72,0,1);
  /* hairlines: luz por arriba, sombra por abajo. Nada de bordes grises planos */
  --hair: rgba(255,255,255,.07);
  --hair-fuerte: rgba(255,255,255,.11);
  --realce: inset 0 1px 0 rgba(255,255,255,.055);
}
.gradio-container { width: 100%; max-width: 1580px !important; margin: 0 auto !important;
  padding: 16px 18px 28px !important; box-sizing: border-box;
  font-variant-numeric: tabular-nums;
  font-feature-settings: "tnum" 1, "cv05" 1, "ss01" 1; }
.gradio-container, body { background: var(--nb) !important; }
.gradio-container .row, .gradio-container .column, .gradio-container .block {
  min-width: 0; }

/* Profundidad: dos halos de acento MUY tenues fijos al fondo. Dan volumen a la
   página sin coste de repintado (elemento fijo, sin eventos). */
body::before { content: ""; position: fixed; inset: 0; z-index: 0;
  pointer-events: none;
  background:
    linear-gradient(180deg, rgba(145,132,217,.08), rgba(18,20,31,0) 360px),
    linear-gradient(90deg, rgba(145,132,217,.055), transparent 42%, rgba(99,181,142,.032)),
    repeating-linear-gradient(90deg, rgba(255,255,255,.018) 0 1px, transparent 1px 96px); }
.gradio-container { position: relative; z-index: 1; }

/* Grano finísimo: rompe las bandas de los degradados y quita el plano digital.
   Fijo y sin eventos, como manda el rendimiento. */
body::after { content: ""; position: fixed; inset: 0; z-index: 2;
  pointer-events: none; opacity: .022;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='3'/%3E%3C/filter%3E%3Crect width='140' height='140' filter='url(%23n)'/%3E%3C/svg%3E"); }

/* barra superior fina: marca, estado de mercado, ticker, capital ---------- */
#topbar { display: flex; align-items: center; gap: 14px; height: 52px;
  min-height: 52px; padding: 0 18px; margin: 0 0 14px 0; border-radius: 8px;
  background: linear-gradient(180deg, #1a1d2b, #131622);
  border: 1px solid var(--hair);
  box-shadow: var(--realce), 0 14px 34px -26px #000;
  overflow-x: auto; overflow-y: hidden; scrollbar-width: none; }
#topbar::-webkit-scrollbar { display: none; }
#topbar > * { flex: none; white-space: nowrap; }
#topbar .brand { display: flex; align-items: baseline; gap: 9px; }
#topbar .brand b { font-weight: 500; font-size: 15px; letter-spacing: -.01em;
  color: var(--nt); }
#topbar .brand i.sep { width: 1px; height: 12px; background: var(--n7); }
#topbar .brand span { font-size: 11px; letter-spacing: .14em;
  text-transform: uppercase; color: var(--n5); }
#topbar .chip { display: flex; align-items: center; gap: 7px; padding: 5px 11px;
  border-radius: 999px; background: rgba(255,255,255,.045);
  border: 1px solid var(--hair); box-shadow: var(--realce);
  font-size: 11.5px; color: var(--n4); }
#topbar .chip .dot { width: 6px; height: 6px; border-radius: 50%;
  background: var(--n6); }
#topbar .chip.mkt-open .dot { background: #63b58e;
  animation: nbpulse 2.4s ease-in-out infinite; }
#topbar .chip .muted { color: var(--n6); }
#topbar .chip.tk { border: 1px solid var(--na7); background: var(--na9);
  font-size: 12.5px; }
#topbar .chip.tk b { font-weight: 500; letter-spacing: .02em; color: var(--nt); }
#topbar .chip.tk .up { color: #63b58e; }
#topbar .chip.tk .dn { color: #d9736b; }
#topbar .right { display: flex; align-items: center; gap: 8px; margin-left: auto; }
#topbar .disc { font-size: 10.5px; line-height: 52px; color: var(--n6);
  max-width: 400px; overflow: hidden; text-overflow: ellipsis; text-align: right; }
@media (max-width: 1200px) { #topbar .disc { display: none; } }
@keyframes nbpulse { 0%,100% { opacity: 1 } 50% { opacity: .35 } }

/* pestañas: grupo con subrayado de acento, subpestañas como chips --------- */
div[role='tablist'] { gap: 2px; border-bottom: 1px solid var(--hair);
  flex-wrap: wrap; }
div[role='tablist'] > button[role='tab'] { font-weight: 500; font-size: 13.5px;
  letter-spacing: -.005em; padding: 0 14px; height: 42px; background: transparent;
  border: 0; border-bottom: 2px solid transparent; border-radius: 0;
  color: var(--n5); white-space: nowrap;
  transition: color .34s var(--eas), border-color .34s var(--eas); }
div[role='tablist'] > button[role='tab']:hover { color: var(--nt);
  border-bottom-color: var(--n7); background: transparent; }
div[role='tablist'] > button[role='tab'].selected,
div[role='tablist'] > button[role='tab'][aria-selected='true'] { color: var(--nt);
  border-bottom-color: var(--na); background: transparent; }
/* segundo nivel: chips con bisel, no rectángulos planos */
.tabitem div[role='tablist'] { border-bottom: 1px solid var(--hair);
  padding: 8px 0; gap: 4px; }
.tabitem div[role='tablist'] > button[role='tab'] { height: 30px; padding: 0 12px;
  font-size: 12.5px; border-radius: 999px; border: 1px solid transparent;
  border-bottom: 1px solid transparent; color: var(--n5);
  transition: background .34s var(--eas), color .34s var(--eas),
              border-color .34s var(--eas); }
.tabitem div[role='tablist'] > button[role='tab']:hover {
  background: rgba(255,255,255,.04); color: var(--nt); }
.tabitem div[role='tablist'] > button[role='tab'].selected,
.tabitem div[role='tablist'] > button[role='tab'][aria-selected='true'] {
  background: linear-gradient(180deg, rgba(145,132,217,.26), rgba(145,132,217,.14));
  border: 1px solid rgba(145,132,217,.42); color: #efedff;
  box-shadow: var(--realce), 0 1px 10px -4px rgba(145,132,217,.7); }

/* Bisel anidado: cada bloque es una bandeja con una pieza dentro. Es lo que
   separa "div con borde gris" de "componente físico". ---------------------- */
.block { background: linear-gradient(180deg, #1e2130, #191b27) !important;
  border: 1px solid var(--hair) !important; border-radius: 8px !important;
  box-shadow: var(--realce), 0 12px 28px -22px rgba(0,0,0,.9) !important; }
.block:has(> .block), .form { background: transparent !important;
  border-color: transparent !important; box-shadow: none !important; }
.block:hover { border-color: var(--hair-fuerte) !important; }

/* botones: píldora, con luz propia arriba y hundimiento al pulsar --------- */
button.primary, button.secondary { border-radius: 999px !important;
  font-weight: 600 !important; letter-spacing: -.005em; white-space: nowrap;
  padding: 10px 22px !important;
  /* una píldora de 1.500 px de ancho no es un botón, es una barra: se limita
     para que el que va solo en su fila siga pareciendo un botón */
  max-width: 360px !important;
  transition: background .34s var(--eas), border-color .34s var(--eas),
              color .34s var(--eas), transform .18s var(--eas),
              box-shadow .34s var(--eas); }
button.primary { color: #f2f0ff !important;
  background: linear-gradient(180deg, rgba(145,132,217,.30), rgba(145,132,217,.16)) !important;
  border: 1px solid rgba(145,132,217,.5) !important;
  box-shadow: var(--realce), 0 6px 18px -10px rgba(145,132,217,.85) !important; }
button.primary:hover {
  background: linear-gradient(180deg, rgba(145,132,217,.44), rgba(145,132,217,.24)) !important;
  box-shadow: var(--realce), 0 10px 26px -10px rgba(145,132,217,1) !important; }
button.primary:active { transform: scale(.978); }
button.secondary { background: rgba(255,255,255,.035) !important;
  border: 1px solid var(--hair-fuerte) !important; color: var(--n4) !important;
  box-shadow: var(--realce) !important; }
button.secondary:hover { background: rgba(255,255,255,.07) !important;
  color: var(--nt) !important; }
button.secondary:active { transform: scale(.978); }
button:focus-visible, input:focus-visible, select:focus-visible,
textarea:focus-visible, [role='tab']:focus-visible {
  outline: 2px solid var(--na) !important; outline-offset: 2px; }
::selection { background: var(--na9); color: var(--nt); }

/* formularios: etiqueta microscópica arriba, campo hundido --------------- */
.block label > span, label > span[data-testid] { font-size: 10px !important;
  letter-spacing: .14em; text-transform: uppercase; font-weight: 600 !important;
  color: var(--n6) !important; }
input[type='text'], input[type='number'], textarea, select,
.gradio-container input:not([type='range']):not([type='checkbox']):not([type='radio']) {
  background: rgba(0,0,0,.28) !important; border: 1px solid var(--hair) !important;
  color: var(--nt) !important; border-radius: 10px !important;
  font-size: 13.5px !important; padding: 9px 12px !important;
  box-shadow: inset 0 1px 3px rgba(0,0,0,.35) !important;
  transition: border-color .34s var(--eas), box-shadow .34s var(--eas); }
input[type='text']:focus, input[type='number']:focus, textarea:focus {
  border-color: rgba(145,132,217,.55) !important;
  box-shadow: inset 0 1px 3px rgba(0,0,0,.35),
              0 0 0 3px rgba(145,132,217,.14) !important; }
input::placeholder, textarea::placeholder { color: var(--n6) !important; }
input[type='range'] { accent-color: var(--na); }
input[type='checkbox'], input[type='radio'] { accent-color: var(--na); }

/* datos: la tabla es el producto, así que respira y se lee ---------------- */
table { font-size: 13px; font-variant-numeric: tabular-nums;
  border-collapse: separate !important; border-spacing: 0 !important; }
.wrap table, .table-wrap table { min-width: 100%; }
div[data-testid='dataframe'] { border-radius: 8px; overflow: hidden;
  background: rgba(0,0,0,.12); }
table thead th { font-size: 9.5px !important; letter-spacing: .16em;
  text-transform: uppercase; color: var(--n6) !important; font-weight: 600 !important;
  background: rgba(255,255,255,.028) !important; padding: 11px 14px !important;
  border-bottom: 1px solid var(--hair-fuerte) !important; }
table td { padding: 10px 14px !important; border-bottom: 1px solid rgba(255,255,255,.045) !important;
  border-left: 0 !important; border-right: 0 !important; border-top: 0 !important; }
table td:not(:first-child) { font-family: "JetBrains Mono", ui-monospace, monospace;
  font-size: 12.5px; letter-spacing: -.02em; }
tbody tr { transition: background .22s var(--eas); }
tbody tr:hover { background: rgba(145,132,217,.10) !important; }
tbody tr:last-child td { border-bottom: 0 !important; }

/* markdown: jerarquía por tamaño y aire, no por negritas ------------------ */
.prose { color: #c3c8dc; font-size: 13.5px; line-height: 1.62; text-wrap: pretty; }
.prose strong { color: var(--nt); font-weight: 600; }
.prose h1, .prose h2, .prose h3, .prose h4 { letter-spacing: -.022em;
  font-weight: 600; color: #f4f4f8; }
.prose h1 { font-size: 25px; margin: 2px 0 10px; }
.prose h2 { font-size: 19px; margin: 22px 0 8px; }
.prose h3 { font-size: 15.5px; margin: 18px 0 6px; }
.prose a { color: var(--na3); text-decoration: none;
  border-bottom: 1px solid rgba(210,206,253,.3); }
.prose a:hover { color: var(--na); }
.prose code { background: rgba(255,255,255,.06); color: var(--na3);
  border: 1px solid var(--hair); border-radius: 5px; padding: 1px 6px;
  font-size: 12px; }
/* la cita es el aviso honesto: que se note como una franja, no como texto suelto */
.prose blockquote { border-left: 2px solid rgba(145,132,217,.55);
  background: rgba(145,132,217,.06); border-radius: 0 10px 10px 0;
  padding: 9px 14px; margin: 12px 0; color: var(--n4); }
.prose ul, .prose ol { padding-left: 20px; }
.prose li { margin: 3px 0; }

/* gráficas y acordeones -------------------------------------------------- */
.plot, .plot > div, div[data-testid='plot'] { background: transparent !important;
  border-radius: 8px; overflow: hidden; }
.plot img, div[data-testid='plot'] img { border-radius: 8px; display: block; }
.label-wrap { padding: 2px 2px !important; min-height: 0 !important; }
.label-wrap > span { color: var(--n4) !important; font-weight: 600 !important;
  font-size: 13px !important; letter-spacing: -.01em; }
.label-wrap:hover > span { color: var(--nt) !important; }
/* acordeón cerrado: una línea, no una banda vacía de 60 px */
.block:has(> .label-wrap) { padding: 10px 14px !important; }

/* Ritmo: los bloques respiran, pero es un panel de datos, no una landing.
   El aire va en el hueco ENTRE piezas, no en padding interior gigante. */
.gap, .form > .block { margin-bottom: 0 !important; }
.tabitem > .column, .tabitem > div { gap: 12px !important; }
.tabitem { padding-top: 14px !important; }

@media (max-width: 760px) {
  .gradio-container { padding: 10px !important; }
  #topbar { height: auto; flex-wrap: wrap; align-items: flex-start;
    gap: 8px; padding: 10px 12px; }
  #topbar .brand { width: 100%; }
  #topbar .right { width: 100%; margin-left: 0; overflow-x: auto; padding-bottom: 2px; }
  div[role='tablist'] { flex-wrap: nowrap; overflow-x: auto; padding-bottom: 4px;
    scrollbar-width: none; }
  div[role='tablist']::-webkit-scrollbar { display: none; }
  div[role='tablist'] > button[role='tab'] { height: 36px; padding: 0 10px;
    font-size: 12.5px; }
  button.primary, button.secondary { width: 100% !important; max-width: none !important; }
  table thead th, table td { padding: 8px 10px !important; }
}

/* Scrollbar: en un panel oscuro, la barra clara del sistema canta mucho */
::-webkit-scrollbar { width: 11px; height: 11px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,.10); border-radius: 999px;
  border: 3px solid transparent; background-clip: content-box; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,.20);
  border: 3px solid transparent; background-clip: content-box; }

footer { display: none !important; }
"""


def TOPBAR_HTML(ticker="AAPL", precio="—", cambio="", capital="10.000 €",
                mercado="NYSE abierto", cierre="", mercado2="BME cerrado",
                abierto=True, kicker="Mesa cuantitativa", mercados=True,
                nota=""):
    """Barra superior. Sustituye al banner navy: misma información
    (marca, kicker, disclaimer) en una sola línea de 46 px.

    El color del cambio sale del signo (verde sube / rojo baja) y el punto solo
    late si el mercado está abierto de verdad: un indicador que miente es peor
    que no tenerlo.

    La usan las tres apps (panel, Space y forecast SAB), de ahí los parámetros:
    `kicker` cambia el subtítulo de marca, `mercados=False` quita los chips de
    apertura (útil cuando la app trabaja sobre CSV, no en vivo), `capital=""`
    quita el chip de cartera y `nota` añade un aviso propio de esa app."""
    txt = str(cambio).strip()
    clase = "dn" if txt.startswith("-") or txt.startswith("−") else "up"
    cambio_html = f'<span class="{clase}">{cambio}</span>' if cambio else ""
    cierre_html = f'<span class="muted">{cierre}</span>' if cierre else ""
    mkt = "mkt-open" if abierto else "mkt-closed"
    chips_mkt = (f'<div class="chip {mkt}"><span class="dot"></span>'
                 f'<span>{mercado}</span>{cierre_html}</div>'
                 f'<div class="chip"><span class="muted">{mercado2}</span></div>'
                 ) if mercados else ""
    chip_tk = (f'<div class="chip tk"><i class="ph ph-crosshair-simple"></i><b>{ticker}</b>'
               f'<span>{precio}</span>{cambio_html}</div>') if ticker else ""
    chip_cap = (f'<div class="chip"><i class="ph ph-wallet"></i><span>{capital}</span></div>'
                ) if capital else ""
    chip_nota = f'<div class="chip"><span class="muted">{nota}</span></div>' if nota else ""
    return f"""<div id="topbar">
  <div class="brand"><b>FinanzIA</b><i class="sep"></i><span>{kicker}</span></div>
  {chips_mkt}{chip_nota}
  <div class="right">
    {chip_tk}{chip_cap}
    <div class="disc">Análisis y educación · no es recomendación de inversión</div>
  </div>
</div>"""


TOPBAR = TOPBAR_HTML(precio="—", cambio="", cierre="")
