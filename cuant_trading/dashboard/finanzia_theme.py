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
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        'family=Inter:wght@400;500;600&display=swap">'
        '<link rel="stylesheet" href="https://unpkg.com/@phosphor-icons/web@2.1.1'
        '/src/regular/style.css">')


def _build_theme():
    """Tema Nocturne. Si la API de temas de Gradio cambia, cae al tema por
    defecto sin romper el panel (el CSS de abajo ya hace el 90 % del trabajo)."""
    import gradio as gr
    try:
        fonts = dict(
            font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
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
  --nb: #161826; --ns: #232532; --nbar: #13151f; --nt: #e9e9ed;
  --na: #9184d9; --na3: #d2cefd; --na7: #5d5294; --na9: #2b2741;
  --n4: #b2b6ca; --n5: #9397ab; --n6: #75798c; --n7: #595d6c;
  --n8: #3f424d; --n9: #292b31; --ndiv: rgba(233,233,237,.16);
}
.gradio-container { max-width: 1580px !important; margin: 0 auto !important;
  font-variant-numeric: tabular-nums; }
.gradio-container, body { background: var(--nb) !important; }

/* barra superior fina: marca, estado de mercado, ticker, capital ---------- */
#topbar { display: flex; align-items: center; gap: 14px; height: 46px;
  padding: 0 16px; margin: 0 0 11px 0; background: var(--nbar);
  border-bottom: 1px solid var(--ndiv); border-radius: 8px 8px 0 0;
  overflow-x: auto; overflow-y: hidden; scrollbar-width: none; }
#topbar::-webkit-scrollbar { display: none; }
#topbar > * { flex: none; white-space: nowrap; }
#topbar .brand { display: flex; align-items: baseline; gap: 9px; }
#topbar .brand b { font-weight: 500; font-size: 15px; letter-spacing: -.01em;
  color: var(--nt); }
#topbar .brand i.sep { width: 1px; height: 12px; background: var(--n7); }
#topbar .brand span { font-size: 11px; letter-spacing: .14em;
  text-transform: uppercase; color: var(--n5); }
#topbar .chip { display: flex; align-items: center; gap: 7px; padding: 4px 9px;
  border-radius: 6px; background: var(--n9); font-size: 11.5px; color: var(--n4); }
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
#topbar .disc { font-size: 10.5px; line-height: 46px; color: var(--n6);
  max-width: 400px; overflow: hidden; text-overflow: ellipsis; text-align: right; }
@media (max-width: 1200px) { #topbar .disc { display: none; } }
@keyframes nbpulse { 0%,100% { opacity: 1 } 50% { opacity: .35 } }

/* pestañas: grupo con subrayado de acento, subpestañas como chips --------- */
div[role='tablist'] { gap: 2px; border-bottom: 1px solid var(--ndiv);
  flex-wrap: wrap; }
div[role='tablist'] > button[role='tab'] { font-weight: 500; font-size: 13.5px;
  padding: 0 13px; height: 40px; background: transparent; border: 0;
  border-bottom: 2px solid transparent; border-radius: 0; color: var(--n4);
  white-space: nowrap; transition: color .15s, border-color .15s; }
div[role='tablist'] > button[role='tab']:hover { color: var(--nt);
  border-bottom-color: var(--n7); background: transparent; }
div[role='tablist'] > button[role='tab'].selected { color: var(--na);
  border-bottom-color: var(--na); background: transparent; }
/* segundo nivel (las subpestañas dentro de cada grupo) */
.tabitem div[role='tablist'] { border-bottom: 1px solid var(--ndiv);
  padding: 7px 0; gap: 3px; }
.tabitem div[role='tablist'] > button[role='tab'] { height: 28px; padding: 0 10px;
  font-size: 12.5px; border-radius: 6px; border-bottom: 0; color: var(--n5); }
.tabitem div[role='tablist'] > button[role='tab']:hover { background: var(--n9);
  color: var(--nt); border-bottom: 0; }
.tabitem div[role='tablist'] > button[role='tab'].selected { background: var(--na9);
  border: 1px solid var(--na7); color: var(--na3); border-bottom: 1px solid var(--na7); }

/* botones: el primario es un contorno de acento, nunca un relleno --------- */
button.primary, button.secondary { border-radius: 6px !important;
  font-weight: 500 !important; white-space: nowrap;
  transition: background .15s, border-color .15s, color .15s; }
button.primary { background: transparent !important;
  border: 1px solid var(--na) !important; color: var(--na3) !important; }
button.primary:hover { background: var(--na9) !important; }
button.primary:active { background: var(--na7) !important; color: #f5f4ff !important; }
button.secondary { background: transparent !important;
  border: 1px solid var(--n8) !important; color: var(--n4) !important; }
button.secondary:hover { background: var(--n9) !important; color: var(--nt) !important; }
button:focus-visible, input:focus-visible, select:focus-visible,
textarea:focus-visible, [role='tab']:focus-visible {
  outline: 2px solid var(--na) !important; outline-offset: 2px; }
::selection { background: var(--na9); color: var(--nt); }

/* formularios compactos -------------------------------------------------- */
.block label > span, label > span[data-testid] { font-size: 11px !important;
  letter-spacing: .06em; text-transform: uppercase; color: var(--n5) !important; }
input[type='text'], input[type='number'], textarea, select,
.gradio-container input:not([type='range']):not([type='checkbox']):not([type='radio']) {
  background: var(--nb) !important; border: 1px solid var(--n8) !important;
  color: var(--nt) !important; border-radius: 6px !important; font-size: 13px !important; }
input[type='range'] { accent-color: var(--na); }
input[type='checkbox'], input[type='radio'] { accent-color: var(--na); }

/* datos: cifras tabulares, cabecera tenue, filas con hover --------------- */
table { font-size: 13px; font-variant-numeric: tabular-nums; }
table thead th { font-size: 11px !important; letter-spacing: .08em;
  text-transform: uppercase; color: var(--n5) !important;
  background: var(--ns) !important; font-weight: 500 !important; }
tbody tr { transition: background .12s; }
tbody tr:hover { background: var(--na9); }
table td, table th { border-color: var(--n9) !important; }

/* markdown didáctico: jerarquía por tamaño y espacio, no por negritas ---- */
.prose { color: #cfd3e5; font-size: 13px; text-wrap: pretty; }
.prose strong { color: var(--nt); font-weight: 500; }
.prose h1, .prose h2, .prose h3 { letter-spacing: -.015em; font-weight: 500;
  color: var(--nt); }
.prose h3 { font-size: 15px; }
.prose a { color: var(--na3); text-decoration: none; }
.prose a:hover { color: var(--na); }
.prose code { background: var(--n9); color: var(--na3); border-radius: 4px;
  padding: 1px 5px; }

/* gráficas y acordeones -------------------------------------------------- */
.plot, .plot > div, div[data-testid='plot'] { background: var(--ns) !important;
  border-radius: 8px; }
.label-wrap > span { color: var(--n4) !important; font-weight: 500 !important; }
footer { display: none !important; }
"""


def TOPBAR_HTML(ticker="AAPL", precio="—", cambio="", capital="10.000 €",
                mercado="NYSE abierto", cierre="", mercado2="BME cerrado",
                abierto=True):
    """Barra superior. Sustituye al banner navy: misma información
    (marca, kicker, disclaimer) en una sola línea de 46 px.

    El color del cambio sale del signo (verde sube / rojo baja) y el punto solo
    late si el mercado está abierto de verdad: un indicador que miente es peor
    que no tenerlo."""
    txt = str(cambio).strip()
    clase = "dn" if txt.startswith("-") or txt.startswith("−") else "up"
    cambio_html = f'<span class="{clase}">{cambio}</span>' if cambio else ""
    cierre_html = f'<span class="muted">{cierre}</span>' if cierre else ""
    mkt = "mkt-open" if abierto else "mkt-closed"
    return f"""<div id="topbar">
  <div class="brand"><b>FinanzIA</b><i class="sep"></i><span>Mesa cuantitativa</span></div>
  <div class="chip {mkt}"><span class="dot"></span><span>{mercado}</span>{cierre_html}</div>
  <div class="chip"><span class="muted">{mercado2}</span></div>
  <div class="right">
    <div class="chip tk"><i class="ph ph-crosshair-simple"></i><b>{ticker}</b>
      <span>{precio}</span>{cambio_html}</div>
    <div class="chip"><i class="ph ph-wallet"></i><span>{capital}</span></div>
    <div class="disc">Análisis y educación · no es recomendación de inversión</div>
  </div>
</div>"""


TOPBAR = TOPBAR_HTML(precio="—", cambio="", cierre="")
