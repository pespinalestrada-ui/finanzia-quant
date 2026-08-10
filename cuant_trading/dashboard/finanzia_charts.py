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
    """Paleta Nocturne. Un acento; verde/rojo solo donde hay semántica."""
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
    acc       = "#9184d9"   # serie principal
    acc_light = "#b5abfc"   # serie secundaria del mismo rol
    acc_dim   = "#5d5294"   # líneas de apoyo, bandas
    neutral   = "#b2b6ca"   # histórico, benchmark
    neutral_d = "#75798c"
    up        = "#63b58e"   # verde desaturado
    down      = "#d9736b"
    gold      = "#c9b273"   # tercera serie (SMA rápida, señal MACD)
    # ciclo por defecto: acento, neutro, oro, verde, rojo, acento claro
    cycle = [acc, neutral, gold, up, down, acc_light]


# --- rampas de color para heatmaps y nubes de puntos ----------------------
# correlación: verde (negativa, protege) → superficie (0) → violeta (positiva)
CMAP_CORR = _LSC.from_list("nocturne_corr",
                           [C.up, "#3c7a63", C.panel, "#6b60a8", C.acc_light])
# secuencial (Sharpe, densidad): del gris de superficie al acento claro
CMAP_SEQ = _LSC.from_list("nocturne_seq", [C.neutral_d, C.acc_dim, C.acc, "#d2cefd"])
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
    "grid.color": C.grid_soft,
    "grid.linewidth": 0.8,
    "grid.alpha": 1.0,
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
    "lines.linewidth": 1.7,
    "lines.solid_capstyle": "round",
    "patch.edgecolor": "none",
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
