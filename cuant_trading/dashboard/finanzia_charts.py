"""
Estilo Nocturne para las gráficas de matplotlib de la Mesa cuantitativa.

No cambia NINGÚN cálculo: solo el aspecto. Basta con importarlo una vez,
justo después de `matplotlib.use("Agg")`, y las rcParams quedan aplicadas
a todas las figuras del panel:

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cuant_trading.dashboard.finanzia_charts import (
        C, style, band, marker, zonas_rsi, msg_fig, colorbar, CMAP_CORR, CMAP_SEQ)

Después, en cada figura, una línea al final:

    fig, ax = plt.subplots(figsize=(11, 5))
    ...
    style(ax, titulo="AAPL — equity de 1 €", kicker="BACKTEST · COSTES INCLUIDOS")
    fig.tight_layout()

`style()` pinta el panel, la rejilla tenue, quita las cuatro cajas, coloca
el título en dos niveles (kicker de acento + título) y deja la leyenda sin
marco. Los colores salen de `C`: úsalos en vez de "black", "r", "tab:red"…
"""

import matplotlib as _mpl
import matplotlib.pyplot as _plt
from matplotlib.colors import LinearSegmentedColormap as _LSC


class C:
    """Paleta Nocturne, VALIDADA (no elegida a ojo).

    Los cuatro colores categóricos pasan las cinco comprobaciones del validador
    sobre la superficie oscura #171a25: banda de luminosidad OKLCH 0.48-0.67,
    suelo de croma, separación bajo daltonismo (ΔE≥8), suelo de visión normal
    (ΔE≥15) y contraste ≥3:1.

    La paleta anterior FALLABA cuatro de las cinco. El caso más sangrante estaba
    en la pantalla de KPI: el oro del ROE y el neutro del ROA tenían ΔE 11,5 en
    visión NORMAL — dos series de la misma gráfica que costaba distinguir sin
    tener ningún problema de vista.

    Verde y rojo quedan FUERA del set categórico a propósito: entre ellos es
    imposible pasar la prueba de daltonismo (ΔE 4,7 deutan) y no tiene arreglo
    con color. Se usan solo como ESTADO, y siempre con un segundo canal que
    aguante por sí solo: el signo respecto al cero en las barras del MACD, las
    líneas de 70/30 en el RSI, la palabra COMPRAR/VENDER en el veredicto.

    Orden categórico FIJO: violeta → cian → ámbar → rosa. Nunca se cicla ni se
    reasigna: el color pertenece a la serie, no a su puesto en el ranking.
    """
    # el fondo de la figura debe casar con el de los bloques del panel: la
    # gráfica va sin caja propia (.plot es transparente), así que si no coincide
    # se ve la costura del PNG dentro de la tarjeta
    bg        = "#1c1f2c"   # fondo de la figura = color medio del bloque
    panel     = "#171a25"   # área de datos, un paso por DEBAJO: hunde el dato
    grid      = "#2a2d3a"
    grid_soft = "#232635"
    axis      = "#8b8fa3"
    dim       = "#6d7183"
    text      = "#e9e9ed"

    # --- categóricos, en orden fijo (validados) --------------------------
    acc       = "#8271e0"   # 1 · violeta — la serie protagonista
    cian      = "#2d9ec4"   # 2
    ambar     = "#ab7f28"   # 3
    rosa      = "#c25b86"   # 4
    # el 5º no se inventa un tono: se agrupa en "otros" o se parte la gráfica

    # --- referencia recesiva: gris a propósito, no compite con las series --
    neutral   = "#9aa0b5"   # histórico, comprar-y-mantener, benchmark
    neutral_d = "#6f7486"

    # --- estado (NUNCA solos: llevan posición, línea o etiqueta) ----------
    up        = "#3fa87c"
    down      = "#d4615c"

    # apoyos del mismo violeta (bandas, sombreados): misma familia, no serie
    acc_light = "#a99bf0"
    acc_dim   = "#4b3f86"
    gold      = ambar       # nombre viejo, mantenido para no romper llamadas

    # ciclo por defecto = el orden categórico. Gris y estados quedan fuera:
    # se piden por nombre cuando significan algo.
    cycle = [acc, cian, ambar, rosa]


# --- rampas de color para heatmaps y nubes de puntos ----------------------
# Divergente = DOS tonos y un gris neutro en medio. Nunca un tono en el centro
# y nunca un arcoíris: si el punto medio tiene color, el ojo lee un cambio donde
# solo hay cero. Cian (correlación negativa, diversifica) → gris → ámbar
# (positiva, se mueven juntos). Cian/ámbar es frío/cálido y aguanta daltonismo.
CMAP_CORR = _LSC.from_list("nocturne_corr",
                           ["#1d6f8c", C.cian, "#4a4f5e", C.ambar, "#7d5a17"])
# Secuencial = UN tono, de claro a oscuro. Magnitud, no identidad.
CMAP_SEQ = _LSC.from_list("nocturne_seq",
                          ["#2a2740", C.acc_dim, C.acc, C.acc_light, "#ddd6ff"])
try:
    _mpl.colormaps.register(CMAP_CORR, name="nocturne_corr", force=True)
    _mpl.colormaps.register(CMAP_SEQ, name="nocturne_seq", force=True)
except Exception:
    pass


RC = {
    "figure.facecolor": C.bg,
    "figure.edgecolor": C.bg,
    "figure.dpi": 110,
    "savefig.facecolor": C.bg,
    "savefig.edgecolor": C.bg,
    "axes.facecolor": C.panel,
    "axes.edgecolor": C.grid,
    "axes.labelcolor": C.axis,
    "axes.titlecolor": C.text,
    "axes.titlesize": 12.5,
    "axes.titleweight": "medium",
    "axes.titlelocation": "left",
    "axes.titlepad": 12,
    "axes.labelsize": 10.5,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.prop_cycle": _mpl.cycler(color=C.cycle),
    # rejilla RECESIVA: tiene que estar, pero por detrás. Si compite con el
    # dato, deja de ser referencia y pasa a ser ruido.
    "grid.color": C.grid_soft,
    "grid.linewidth": 0.9,
    "grid.alpha": 0.55,
    "xtick.color": C.dim,
    "ytick.color": C.dim,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.minor.size": 0,
    "ytick.minor.size": 0,
    "text.color": C.text,
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
    "font.size": 11,
    "legend.frameon": False,
    "legend.fontsize": 10.5,
    "legend.labelcolor": C.neutral,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.6,
    "legend.columnspacing": 1.8,
    "legend.loc": "upper left",
    "lines.linewidth": 2.0,          # trazo fino pero con presencia
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "lines.markersize": 8,           # el marcador tiene que poder pincharse
    "lines.markeredgewidth": 2.0,
    "patch.edgecolor": "none",
    # OJO: nada de savefig.bbox="tight" aquí. Las etiquetas al final de cada
    # línea caen FUERA del eje, y con bbox tight matplotlib agranda el lienzo
    # sin pintar de fondo la zona nueva: sale un rectángulo negro en la esquina.
    # El aire se reserva con ax.margins() en cada gráfica.
    "image.cmap": "nocturne_seq",
    "date.autoformatter.month": "%b",
    "date.autoformatter.day": "%d %b",
}


def apply(rc=None):
    """Aplica el estilo globalmente. Se llama solo al importar el módulo."""
    for k, v in (rc or RC).items():
        try:
            _mpl.rcParams[k] = v
        except Exception:
            pass          # una clave que esta versión de matplotlib no conoce


apply()


def style(ax, titulo=None, kicker=None, ylabel=None, xlabel=None,
          legend=None, y_money=False, y_pct=False):
    """Remate final de un eje: rejilla tenue, sin cajas, título en dos niveles.

    titulo  — título de la figura (tamaño 12.5, peso medio)
    kicker  — línea de acento en mayúsculas sobre el título (opcional)
    legend  — True/False para forzar la leyenda; None = solo si hay etiquetas
    """
    ax.set_facecolor(C.panel)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(C.grid)
    ax.grid(True, axis="y", color=C.grid_soft, lw=0.8)
    ax.grid(False, axis="x")
    ax.tick_params(colors=C.dim, length=0, labelsize=10)

    if y_money:
        ax.yaxis.set_major_formatter(
            _mpl.ticker.FuncFormatter(lambda v, _p: f"{v:,.0f}".replace(",", ".")))
    if y_pct:
        ax.yaxis.set_major_formatter(
            _mpl.ticker.FuncFormatter(lambda v, _p: f"{v:.0f}%"))

    if ylabel:
        ax.set_ylabel(ylabel, color=C.axis, fontsize=10.5)
    if xlabel:
        ax.set_xlabel(xlabel, color=C.axis, fontsize=10.5)

    if kicker:
        ax.set_title(kicker.upper(), loc="left", color=C.acc, fontsize=9.5,
                     fontweight="semibold", pad=26)
        if titulo:
            ax.annotate(titulo, xy=(0, 1.0), xytext=(0, 12),
                        xycoords="axes fraction", textcoords="offset points",
                        color=C.text, fontsize=12.5, fontweight="medium",
                        va="bottom", ha="left", annotation_clip=False)
    elif titulo:
        ax.set_title(titulo, loc="left", color=C.text, fontsize=12.5,
                     fontweight="medium", pad=12)

    quiere_leyenda = legend
    if quiere_leyenda is None:
        etiquetas = [l for l in ax.get_legend_handles_labels()[1]
                     if l and not l.startswith("_")]
        quiere_leyenda = bool(etiquetas)
    if quiere_leyenda:
        leg = ax.legend(frameon=False, fontsize=10.5, loc="upper left",
                        borderaxespad=0.2)
        for t in leg.get_texts():
            t.set_color(C.neutral)
    return ax


def band(ax, x, lo, hi, label=None, color=None, alpha=0.17):
    """Banda de confianza (forecast, percentiles)."""
    return ax.fill_between(x, lo, hi, color=color or C.acc, alpha=alpha,
                           linewidth=0, label=label, zorder=1)


def area(ax, x, y, color=None, alpha=0.34, base=None):
    """Relleno DEGRADADO bajo una línea: denso pegado al trazo, disuelto abajo.

    Es lo que separa una línea suelta de una gráfica con cuerpo. Truco correcto:
    se pinta una imagen con degradado vertical y se RECORTA con el polígono de
    la curva. Apilar franjas con `fill_between`, que es lo que se suele hacer,
    deja escalones y una mancha turbia en la base — probado y descartado.

    Va por debajo del trazo (zorder bajo) y muy transparente: aporta volumen,
    no compite con el dato ni ensucia la rejilla."""
    import numpy as _np
    from matplotlib.patches import Polygon as _Poly
    import matplotlib.colors as _mc
    import matplotlib.dates as _mdates

    y = _np.asarray(y, dtype=float)
    xs = _np.asarray(x)
    if xs.dtype.kind in "Mm" or hasattr(xs.flat[0] if xs.size else None, "toordinal"):
        xn = _mdates.date2num(xs)                    # fechas → número
    else:
        xn = xs.astype(float)

    ok = _np.isfinite(y) & _np.isfinite(xn)
    if ok.sum() < 2:
        return
    xn, y = xn[ok], y[ok]
    b = float(_np.nanmin(y)) if base is None else float(base)
    tope = float(_np.nanmax(y))
    if tope <= b:
        return

    rgb = _mc.to_rgb(color or C.acc)
    grad = _np.empty((256, 1, 4))
    grad[:, :, :3] = rgb
    # opacidad al cuadrado: se apaga rápido hacia abajo, no deja bloque de color
    grad[:, :, 3] = (_np.linspace(0.0, 1.0, 256) ** 2 * alpha)[:, None]

    im = ax.imshow(grad, extent=[xn.min(), xn.max(), b, tope], origin="lower",
                   aspect="auto", zorder=0.6, interpolation="bilinear")
    verts = _np.column_stack([_np.r_[xn, xn[::-1]],
                              _np.r_[y, _np.full_like(y, b)]])
    im.set_clip_path(_Poly(verts, closed=True, transform=ax.transData))


def punto_final(ax, x, y, texto=None, color=None, ha="left"):
    """Marca el último dato: punto con anillo del color del fondo + etiqueta.

    El anillo (`mec` del color de la superficie) es el 'surface ring' que separa
    el punto de lo que tenga debajo. La etiqueta va al lado, no encima: una cifra
    en cada punto satura, una en el que importa informa."""
    col = color or C.acc
    ax.plot([x], [y], "o", ms=8, mfc=col, mec=C.panel, mew=2.2, zorder=7,
            clip_on=False)
    if texto:
        ax.annotate(texto, xy=(x, y), xytext=(9 if ha == "left" else -9, 0),
                    textcoords="offset points", ha=ha, va="center",
                    color=col, fontsize=11, fontweight="semibold",
                    zorder=8, annotation_clip=False)


def eje_miles(ax, eje="y", sufijo=""):
    """Miles con punto y decimales con coma, como se escriben aquí."""
    def _f(v, _p):
        s = f"{v:,.0f}".replace(",", ".")
        return s + sufijo
    (ax.yaxis if eje == "y" else ax.xaxis).set_major_formatter(
        _mpl.ticker.FuncFormatter(_f))


def marker(ax, x, y, texto, color=None):
    """Punto anotado con su valor en una cajita (P50, 90 d, cierre…)."""
    color = color or C.acc_light
    ax.plot([x], [y], "o", ms=5.5, mfc=C.bg, mec=color, mew=2, zorder=6)
    ax.annotate(texto, xy=(x, y), xytext=(0, 11), textcoords="offset points",
                ha="center", va="bottom", fontsize=10.5, color=C.text,
                zorder=7, bbox=dict(boxstyle="round,pad=0.32", fc=C.bg,
                                    ec=C.grid, lw=0.9))


def zonas_rsi(ax, alto=70, bajo=30):
    """Sombreado de sobrecompra / sobreventa del RSI."""
    ax.axhspan(alto, 100, color=C.down, alpha=0.12, lw=0)
    ax.axhspan(0, bajo, color=C.up, alpha=0.12, lw=0)
    ax.axhline(alto, color=C.down, ls="--", lw=0.9)
    ax.axhline(bajo, color=C.up, ls="--", lw=0.9)
    ax.axhline(50, color=C.grid, lw=0.8)
    ax.set_ylim(0, 100)
    ax.set_yticks([bajo, 50, alto])


def msg_fig(msg, figsize=(8, 1.9)):
    """Sustituye al hueco gris de las gráficas vacías: mensaje sobre panel."""
    fig, ax = _plt.subplots(figsize=figsize)
    ax.set_facecolor(C.panel)
    ax.text(0.5, 0.5, msg, ha="center", va="center", wrap=True,
            color=C.neutral, fontsize=11.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    return fig


def colorbar(fig, mappable, ax, label=None):
    """Barra de color integrada (sin caja blanca ni etiqueta chillona)."""
    cb = fig.colorbar(mappable, ax=ax, fraction=0.032, pad=0.02)
    cb.outline.set_edgecolor(C.grid)
    cb.ax.tick_params(colors=C.dim, length=0, labelsize=10)
    if label:
        cb.set_label(label, color=C.axis, fontsize=10)
    return cb
