"""
cobertura — poner un suelo a una posición con opciones, como una aerolínea con
el queroseno: no para ganar más, sino para saber cuánto es lo máximo que puedes
perder.

Tres formas de cubrirse, y las tres cuestan algo
------------------------------------------------
1. **Put protectora**: compras el derecho a vender a un precio suelo. Pagas prima.
   Suelo garantizado, techo intacto. Es un seguro, y como todo seguro, se paga.
2. **Collar**: compras la put y VENDES una call por encima para financiarla.
   Casi gratis, pero renuncias a la subida por encima del strike de la call. Es
   lo que hacen de verdad las aerolíneas y las mineras.
3. **Reducir exposición**: vender parte y quedarse en efectivo. No cuesta prima,
   pero renuncias a la subida en proporción.

La comparación es el punto
--------------------------
La prima es un coste CIERTO contra un beneficio INCIERTO. Una put del 10% a tres
meses puede costar un 2% de la posición; renovada todo el año, un 8% anual, que
se come casi entera la rentabilidad esperada de una bolsa normal. Por eso este
módulo pone SIEMPRE al lado la alternativa de bajar exposición: muchas veces te
da un suelo parecido sin pagar prima. Nadie te lo dice cuando te vende el seguro.

Un contrato = 100 acciones (convención de EEUU). Las opciones se valoran con el
Black-Scholes-Merton de `options_greeks`, con la volatilidad histórica del activo
o la que tú pongas.

Uso:
    python cobertura.py AAPL --acciones 200 --suelo 10 --dias 90
"""
import argparse
import sys
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

try:                                      # dentro del panel ya está en el path
    import options_greeks as OG
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "options_greeks"))
    import options_greeks as OG

# paleta Nocturne (la misma de cuant_trading/dashboard/finanzia_charts.py)
_NEU, _ACC, _ACC2 = "#b2b6ca", "#9184d9", "#b5abfc"
_UP, _DOWN, _GOLD, _DIM = "#63b58e", "#d9736b", "#c9b273", "#75798c"

CONTRATO = 100          # acciones por contrato


def plan(ticker, acciones=100, suelo_pct=10.0, dias=90, techo_pct=10.0,
         tasa=0.04, vol=None, sesgo=None):
    """Calcula las tres coberturas para una posición concreta.

    suelo_pct: cuánto estás dispuesto a perder antes de que entre el seguro.
    techo_pct: a partir de qué subida renuncias a ganar (solo para el collar).
    sesgo: puntos de volatilidad EXTRA que se le suman a la put. En bolsa las
        puts fuera de dinero siempre cotizan más caras que las calls simétricas
        (la gente paga por protegerse, no por especular al alza). Si se valoran
        ambas patas con la misma σ, el collar sale gratis o cobrando, que es
        justo lo que no pasa en el mercado. Se mide del mercado si se puede;
        si no, se usa el valor de `sesgo`.
    """
    tk = ticker.strip().upper()
    S, vol_hist = OG._spot_y_vol(tk)
    sigma = float(vol) if vol else vol_hist
    tau = max(dias, 1) / 365.0
    sesgo_medido = None
    if sesgo is None:
        sesgo_medido = _sesgo_mercado(tk, suelo_pct, techo_pct)
        sesgo = sesgo_medido if sesgo_medido is not None else 0.05   # 5 pts, típico en índices
    sigma_put = max(1e-4, sigma + float(sesgo))

    n_contratos = max(1, int(round(acciones / CONTRATO)))
    cubiertas = n_contratos * CONTRATO
    valor = acciones * S

    K_put = S * (1 - suelo_pct / 100.0)
    K_call = S * (1 + techo_pct / 100.0)
    put = OG.bsm(S, K_put, tau, tasa, sigma_put, tipo="put")
    call = OG.bsm(S, K_call, tau, tasa, sigma, tipo="call")

    prima_put = put["precio"] * cubiertas
    prima_call = call["precio"] * cubiertas
    coste_collar = prima_put - prima_call

    # equivalente sin opciones: ¿qué exposición deja la misma pérdida máxima?
    # perder como mucho suelo_pct del total estando 'w' invertido en un activo
    # que puede caer 'caida' => w = suelo_pct / caida. Se usa la peor caída a
    # ese plazo que de verdad ha tenido el activo, no una supuesta.
    peor = _peor_caida(tk, dias)
    w_equiv = min(1.0, (suelo_pct / 100.0) / peor) if peor > 0 else np.nan

    return {
        "ticker": tk, "S": S, "sigma": sigma, "sigma_put": sigma_put,
        "sesgo": float(sesgo), "sesgo_medido": sesgo_medido, "vol_hist": vol_hist,
        "acciones": acciones, "cubiertas": cubiertas, "contratos": n_contratos,
        "valor": valor, "dias": dias, "tau": tau,
        "K_put": K_put, "K_call": K_call, "suelo_pct": suelo_pct, "techo_pct": techo_pct,
        "put": put, "call": call,
        "prima_put": prima_put, "prima_call": prima_call, "coste_collar": coste_collar,
        "coste_pct": prima_put / valor * 100 if valor else np.nan,
        "coste_anual_pct": (prima_put / valor) * (365.0 / max(dias, 1)) * 100 if valor else np.nan,
        "collar_pct": coste_collar / valor * 100 if valor else np.nan,
        "suelo_eur": (K_put * cubiertas + (acciones - cubiertas) * 0) - prima_put,
        "subida_para_empatar": prima_put / acciones if acciones else np.nan,
        "peor_caida": peor, "w_equiv": w_equiv,
    }


def _sesgo_mercado(ticker, suelo_pct, techo_pct):
    """Mide del mercado cuánta volatilidad extra cotiza la put frente a la call.

    Devuelve puntos de σ (0.05 = 5 puntos) o None si no hay cadena. Es preferible
    a inventarse un número: el sesgo depende del activo y del momento."""
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        vencs = tk.options
        if not vencs:
            return None
        ch = tk.option_chain(vencs[min(2, len(vencs) - 1)])
        S = float(tk.history(period="5d")["Close"].astype(float).dropna().iloc[-1])
        kp, kc = S * (1 - suelo_pct / 100.0), S * (1 + techo_pct / 100.0)

        def _iv(df, k):
            d = df.dropna(subset=["impliedVolatility"])
            d = d[d["impliedVolatility"] > 0.01]
            if d.empty:
                return None
            fila = d.iloc[(d["strike"] - k).abs().argmin()]
            return float(fila["impliedVolatility"])

        iv_p, iv_c = _iv(ch.puts, kp), _iv(ch.calls, kc)
        if iv_p is None or iv_c is None:
            return None
        return max(0.0, iv_p - iv_c)
    except Exception:
        return None


def _peor_caida(ticker, dias):
    """Peor caída real del activo en una ventana de `dias` (histórico largo)."""
    try:
        import yfinance as yf
        c = yf.Ticker(ticker).history(period="10y", auto_adjust=True)["Close"]
        c = c.astype(float).dropna()
        if len(c) < dias + 30:
            return np.nan
        rel = c / c.shift(dias) - 1.0
        return float(-rel.min())
    except Exception:
        return np.nan


def curva(p, rango=0.45, n=241):
    """Valor de la posición al vencimiento, con y sin cobertura."""
    S0 = p["S"]
    precios = np.linspace(S0 * (1 - rango), S0 * (1 + rango), n)
    acc, cub = p["acciones"], p["cubiertas"]
    desnuda = precios * acc
    # put: protege 'cub' acciones por debajo del strike
    payoff_put = np.maximum(p["K_put"] - precios, 0) * cub
    protegida = desnuda + payoff_put - p["prima_put"]
    # collar: además, la call vendida corta la subida
    payoff_call = -np.maximum(precios - p["K_call"], 0) * cub
    collar = desnuda + payoff_put + payoff_call - p["coste_collar"]
    # reducir exposición: w invertido, el resto quieto en efectivo
    w = p["w_equiv"] if np.isfinite(p["w_equiv"]) else 1.0
    reducida = precios * acc * w + S0 * acc * (1 - w)
    return {"precios": precios, "desnuda": desnuda, "protegida": protegida,
            "collar": collar, "reducida": reducida, "w": w}


def explicar(p):
    e = lambda v: f"{v:,.0f} €".replace(",", ".")
    L = [f"### Proteger {p['acciones']} acciones de {p['ticker']} durante {p['dias']} días",
         f"Precio actual **{p['S']:.2f}** · posición **{e(p['valor'])}** · "
         f"volatilidad usada **{p['sigma']*100:.0f}%**"
         + ("" if p["sigma"] == p["vol_hist"] else f" (histórica {p['vol_hist']*100:.0f}%)"),
         "",
         "#### 1 · Put protectora (el seguro clásico)",
         f"- Compra **{p['contratos']} contratos** de put con strike **{p['K_put']:.2f}** "
         f"(−{p['suelo_pct']:.0f}%). Cubren {p['cubiertas']} acciones.",
         f"- Cuesta **{e(p['prima_put'])}**, el **{p['coste_pct']:.2f}%** de tu posición.",
         f"- El precio tiene que subir **{p['subida_para_empatar']:.2f} €** "
         f"(**{p['subida_para_empatar']/p['S']*100:.1f}%**) solo para recuperar la prima."]
    ca = p["coste_anual_pct"]
    L.append(f"- ⚠️ Renovándolo todo el año son **{ca:.1f}% anual**. "
             + ("Eso se come casi entera la rentabilidad esperada de una bolsa normal "
                "(~8-10%): protegerse siempre sale carísimo."
                if ca > 6 else
                "Asumible como seguro puntual, no como costumbre permanente."))
    L.append("")
    L.append("#### 2 · Collar (lo que hace de verdad una aerolínea)")
    L.append(f"- La misma put, financiada vendiendo **{p['contratos']} calls** con strike "
             f"**{p['K_call']:.2f}** (+{p['techo_pct']:.0f}%).")
    if p["coste_collar"] <= 0:
        L.append(f"- Coste neto **{e(p['coste_collar'])}**: la call paga la put y **te sobra**.")
    else:
        L.append(f"- Coste neto **{e(p['coste_collar'])}** ({p['collar_pct']:.2f}% de la "
                 f"posición), frente al {p['coste_pct']:.2f}% de la put sola.")
    L.append(f"- El precio: si sube por encima de **{p['K_call']:.2f}**, esa subida ya no es "
             f"tuya. Cambias techo por suelo.")
    if p["sesgo_medido"] is not None:
        L.append(f"- La put se valora con **{p['sigma_put']*100:.0f}%** de volatilidad y la call "
                 f"con {p['sigma']*100:.0f}%: son **{p['sesgo']*100:.1f} puntos** de sesgo "
                 f"medidos hoy en la cadena real. En bolsa la protección siempre cotiza más "
                 f"cara que la especulación al alza.")
    else:
        L.append(f"- ⚠️ Sin cadena real para este activo: se aplica un sesgo supuesto de "
                 f"**{p['sesgo']*100:.0f} puntos** a la put. Sin ese ajuste el collar saldría "
                 f"gratis o cobrando, que es justo lo que **no** pasa en el mercado.")
    L.append("")
    L.append("#### 3 · Sin opciones: llevar menos")
    if np.isfinite(p["w_equiv"]):
        L.append(f"- La peor caída real de {p['ticker']} en {p['dias']} días fue "
                 f"**−{p['peor_caida']*100:.1f}%** (10 años).")
        L.append(f"- Para no perder más del {p['suelo_pct']:.0f}% en una caída así, bastaría "
                 f"con tener **{p['w_equiv']*100:.0f}%** invertido y el resto quieto. "
                 f"**Coste en primas: cero.**")
        L.append("- A cambio, te pierdes esa misma proporción de la subida. No hay comida "
                 "gratis, pero tampoco hay que pagar peaje al vendedor de seguros.")
    else:
        L.append("- Sin histórico suficiente para calcular el equivalente sin opciones.")
    L.append("\n> La prima es un coste **cierto**; la protección, un beneficio **incierto**. "
             "Cubrirse tiene sentido cuando no puedes permitirte la caída, no cuando crees "
             "que va a pasar. Valorado con Black-Scholes sobre volatilidad histórica: el "
             "precio real del mercado será distinto (mira 🎲 Opciones para la cadena real). "
             "No es recomendación de inversión.")
    return "\n".join(L)


def tabla(p):
    e = lambda v: round(v, 2)
    return pd.DataFrame([
        {"Estrategia": "Sin cubrir", "Coste hoy": 0.0,
         "Pérdida máxima": "ilimitada (hasta −100%)",
         "Techo de ganancia": "sin límite"},
        {"Estrategia": f"Put {p['K_put']:.2f}", "Coste hoy": e(p["prima_put"]),
         "Pérdida máxima": f"−{p['suelo_pct']:.0f}% − prima ({p['coste_pct']:.1f}%)",
         "Techo de ganancia": "sin límite"},
        {"Estrategia": f"Collar {p['K_put']:.0f}/{p['K_call']:.0f}",
         "Coste hoy": e(p["coste_collar"]),
         "Pérdida máxima": f"−{p['suelo_pct']:.0f}% − coste neto",
         "Techo de ganancia": f"+{p['techo_pct']:.0f}%"},
        {"Estrategia": (f"Llevar {p['w_equiv']*100:.0f}% invertido"
                        if np.isfinite(p["w_equiv"]) else "Reducir exposición"),
         "Coste hoy": 0.0,
         "Pérdida máxima": f"≈ −{p['suelo_pct']:.0f}% en la peor caída histórica",
         "Techo de ganancia": (f"{p['w_equiv']*100:.0f}% de la subida"
                               if np.isfinite(p["w_equiv"]) else "proporcional")},
    ])


def _plot(p, c):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from finanzia_charts import style
    except Exception:
        style = None
    fig, ax = plt.subplots(figsize=(10, 5))
    x = c["precios"]
    base = p["valor"]
    ax.plot(x, c["desnuda"] - base, color=_NEU, lw=1.6, label="Sin cubrir")
    ax.plot(x, c["protegida"] - base, color=_ACC, lw=2.1, label=f"Con put {p['K_put']:.0f}")
    ax.plot(x, c["collar"] - base, color=_GOLD, lw=1.7, ls="--",
            label=f"Collar {p['K_put']:.0f}/{p['K_call']:.0f}")
    ax.plot(x, c["reducida"] - base, color=_UP, lw=1.5, ls=":",
            label=f"Llevar {c['w']*100:.0f}% invertido")
    ax.axhline(0, color=_DIM, lw=1)
    ax.axvline(p["S"], color=_DIM, ls=":", lw=1)
    ax.annotate("precio hoy", xy=(p["S"], (c["desnuda"] - base).min()), xytext=(5, 6),
                textcoords="offset points", color=_DIM, fontsize=9)
    if style:
        style(ax, titulo=f"{p['ticker']} — resultado al vencimiento ({p['dias']} días)",
              kicker="GANANCIA O PERDIDA DE LA POSICION SEGUN DONDE ACABE EL PRECIO",
              xlabel="Precio al vencimiento →", ylabel="Resultado (€)")
    else:
        ax.legend()
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description="Cobertura de una posición con opciones.")
    ap.add_argument("ticker")
    ap.add_argument("--acciones", type=int, default=100)
    ap.add_argument("--suelo", type=float, default=10.0, help="%% de caída que aceptas.")
    ap.add_argument("--techo", type=float, default=10.0, help="%% de subida a la que renuncias.")
    ap.add_argument("--dias", type=int, default=90)
    ap.add_argument("--vol", type=float, help="Volatilidad anual (0.30). Por defecto, histórica.")
    a = ap.parse_args()
    p = plan(a.ticker, a.acciones, a.suelo, a.dias, a.techo, vol=a.vol)
    print(explicar(p))
    print()
    print(tabla(p).to_string(index=False))


if __name__ == "__main__":
    main()
