"""
options_greeks — Black-Scholes-Merton y las Griegas (guía §2 y §3).

Valora una opción europea y calcula sus sensibilidades. Es la pieza que faltaba
en la suite: todo lo demás mide acciones/carteras; esto mide DERIVADOS.

Fórmulas (con dividendo continuo q):
    d1 = [ln(S/K) + (r − q + σ²/2)·τ] / (σ√τ)
    d2 = d1 − σ√τ
    Call = S·e^(−qτ)·N(d1) − K·e^(−rτ)·N(d2)
    Put  = K·e^(−rτ)·N(−d2) − S·e^(−qτ)·N(−d1)

Las Griegas (sensibilidad del precio de la opción a cada factor):
    Delta Δ  ∂V/∂S   cuánto se mueve la opción si el activo sube 1 €.
    Gamma Γ  ∂²V/∂S² cómo cambia Delta (curvatura): el riesgo que no ves.
    Vega  ν   ∂V/∂σ   sensibilidad a la volatilidad (por punto de σ).
    Theta Θ  ∂V/∂τ   lo que pierde por el paso del tiempo (por día).
    Rho   ρ   ∂V/∂r   sensibilidad a los tipos de interés.

También calcula la VOLATILIDAD IMPLÍCITA: dado el precio de mercado, qué σ lo
justifica (se resuelve por bisección, robusto).

Datos de mercado: yfinance trae cadenas de opciones reales (solo EEUU).
No es recomendación de inversión.

Uso:
    python options_greeks.py AAPL
    python options_greeks.py --precio 100 --strike 105 --dias 30 --vol 0.25 --tipo call
"""
import argparse
import sys
from math import log, sqrt, exp
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import norm


def bsm(S, K, tau, r=0.04, sigma=0.25, q=0.0, tipo="call"):
    """Precio Black-Scholes-Merton + Griegas. tau en años. Devuelve dict."""
    S, K, tau, sigma = float(S), float(K), max(float(tau), 1e-9), max(float(sigma), 1e-9)
    d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * tau) / (sigma * sqrt(tau))
    d2 = d1 - sigma * sqrt(tau)
    Nd1, Nd2 = norm.cdf(d1), norm.cdf(d2)
    nd1 = norm.pdf(d1)
    disc_q, disc_r = exp(-q * tau), exp(-r * tau)
    if tipo == "call":
        precio = S * disc_q * Nd1 - K * disc_r * Nd2
        delta = disc_q * Nd1
        theta = (-S * disc_q * nd1 * sigma / (2 * sqrt(tau))
                 - r * K * disc_r * Nd2 + q * S * disc_q * Nd1)
        rho = K * tau * disc_r * Nd2
    else:
        precio = K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
        delta = -disc_q * norm.cdf(-d1)
        theta = (-S * disc_q * nd1 * sigma / (2 * sqrt(tau))
                 + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1))
        rho = -K * tau * disc_r * norm.cdf(-d2)
    gamma = disc_q * nd1 / (S * sigma * sqrt(tau))
    vega = S * disc_q * nd1 * sqrt(tau)
    return {"precio": precio, "delta": delta, "gamma": gamma,
            "vega": vega / 100.0,        # por punto de volatilidad (1%)
            "theta": theta / 365.0,      # por día natural
            "rho": rho / 100.0,          # por punto de tipos (1%)
            "d1": d1, "d2": d2}


def vol_implicita(precio_mkt, S, K, tau, r=0.04, q=0.0, tipo="call"):
    """σ que reproduce el precio de mercado (bisección: robusta, sin derivadas)."""
    lo, hi = 1e-4, 5.0
    f = lambda s: bsm(S, K, tau, r, s, q, tipo)["precio"] - float(precio_mkt)
    if f(lo) > 0 or f(hi) < 0:
        return float("nan")            # precio fuera del rango alcanzable
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _spot_y_vol(ticker):
    """Precio actual y volatilidad histórica anualizada (1 año)."""
    import yfinance as yf
    h = yf.Ticker(ticker).history(period="1y", auto_adjust=True)
    if h.empty:
        raise ValueError(f"'{ticker}' sin datos.")
    c = h["Close"].astype(float).dropna()      # yfinance puede devolver la última fila vacía
    if c.empty:
        raise ValueError(f"'{ticker}' sin precios válidos.")
    r = np.log(c / c.shift(1)).dropna()
    return float(c.iloc[-1]), float(r.std() * np.sqrt(252))


def cadena_real(ticker, tipo="call", max_filas=12):
    """Cadena de opciones REAL de yfinance con Griegas y vol implícita."""
    import yfinance as yf
    tk = yf.Ticker(ticker)
    vencs = tk.options
    if not vencs:
        raise ValueError(f"{ticker}: sin cadena de opciones (solo EEUU).")
    venc = vencs[min(2, len(vencs) - 1)]        # ~3er vencimiento (evita el semanal)
    ch = tk.option_chain(venc)
    df = (ch.calls if tipo == "call" else ch.puts).copy()
    S, vol_hist = _spot_y_vol(ticker)
    tau = max((pd.Timestamp(venc) - pd.Timestamp.today()).days, 1) / 365.0
    df = df.reindex((df["strike"] - S).abs().sort_values().index).head(max_filas)
    filas = []
    for _, o in df.iterrows():
        K = float(o["strike"])
        # precio de mercado: punto medio bid/ask (vivo); lastPrice puede estar rancio
        bid, ask = float(o.get("bid") or 0), float(o.get("ask") or 0)
        mkt = (bid + ask) / 2 if bid > 0 and ask > 0 else float(o.get("lastPrice") or 0)
        iv = float(o.get("impliedVolatility") or 0) or vol_hist
        g = bsm(S, K, tau, 0.04, iv, 0.0, tipo)
        iv_calc = vol_implicita(mkt, S, K, tau, 0.04, 0.0, tipo) if mkt > 0 else float("nan")
        filas.append({
            "Strike": round(K, 2), "Mercado": round(mkt, 2),
            "Teórico BSM": round(g["precio"], 2),
            "Vol implícita %": round(iv * 100, 1),
            "IV calculada %": round(iv_calc * 100, 1) if not np.isnan(iv_calc) else None,
            "Delta": round(g["delta"], 3), "Gamma": round(g["gamma"], 4),
            "Vega": round(g["vega"], 3), "Theta/día": round(g["theta"], 3),
        })
    tabla = pd.DataFrame(filas).sort_values("Strike").reset_index(drop=True)
    # coherencia: si la IV que implica el precio cotizado dobla a la IV del mercado,
    # las cotizaciones vienen de otra sesión (mercado cerrado) → avisar, no fingir
    ivc = pd.to_numeric(tabla["IV calculada %"], errors="coerce").median()
    ivm = pd.to_numeric(tabla["Vol implícita %"], errors="coerce").median()
    desfase = bool(pd.notna(ivc) and pd.notna(ivm) and ivm > 0 and ivc > 2 * ivm)
    meta = {"spot": round(S, 2), "vencimiento": venc, "dias": int(tau * 365),
            "vol_hist_pct": round(vol_hist * 100, 1), "tipo": tipo,
            "desfase_datos": desfase, "iv_mkt_med": round(float(ivm), 1) if pd.notna(ivm) else None,
            "iv_calc_med": round(float(ivc), 1) if pd.notna(ivc) else None}
    return tabla, meta


def explicar(g, S, K, tau, tipo):
    """Lectura en cristiano de las Griegas."""
    moneyness = ("dentro del dinero (ITM)" if (tipo == "call" and S > K) or (tipo == "put" and S < K)
                 else "fuera del dinero (OTM)")
    return "\n".join([
        f"**Precio teórico: {g['precio']:.2f} €** · la opción está {moneyness} · quedan {tau*365:.0f} días.",
        "",
        f"- **Delta {g['delta']:+.3f}** — si el activo sube 1 €, la opción gana ~{abs(g['delta']):.2f} €. "
        f"También se lee como probabilidad aproximada de acabar con valor: ~{abs(g['delta'])*100:.0f}%.",
        f"- **Gamma {g['gamma']:.4f}** — cuánto cambia el Delta por cada euro de movimiento. "
        f"Alto = la posición se acelera (bueno si aciertas, peligroso si no).",
        f"- **Vega {g['vega']:.3f}** — si la volatilidad sube 1 punto, la opción gana ~{g['vega']:.2f} €. "
        f"Comprar opciones es apostar a que la volatilidad sube.",
        f"- **Theta {g['theta']:+.3f}/día** — el reloj te quita ~{abs(g['theta']):.3f} € cada día. "
        f"El comprador de opciones pierde por el mero paso del tiempo.",
        f"- **Rho {g['rho']:+.3f}** — efecto de que los tipos suban 1 punto (suele ser el menos relevante).",
    ])


def main():
    ap = argparse.ArgumentParser(description="Black-Scholes-Merton y las Griegas.")
    ap.add_argument("ticker", nargs="?", default=None, help="Ticker EEUU para cadena real.")
    ap.add_argument("--precio", type=float, default=100.0)
    ap.add_argument("--strike", type=float, default=105.0)
    ap.add_argument("--dias", type=int, default=30)
    ap.add_argument("--vol", type=float, default=0.25)
    ap.add_argument("--tasa", type=float, default=0.04)
    ap.add_argument("--tipo", choices=["call", "put"], default="call")
    a = ap.parse_args()

    if a.ticker:
        tabla, meta = cadena_real(a.ticker.upper(), a.tipo)
        print(f"\n=== {a.ticker.upper()} · cadena de {a.tipo.upper()}s · vencimiento {meta['vencimiento']} "
              f"({meta['dias']} días) ===")
        print(f"Spot {meta['spot']} · volatilidad histórica {meta['vol_hist_pct']}%\n")
        print(tabla.to_string(index=False))
        if meta["desfase_datos"]:
            print(f"\n⚠️ Cotizaciones DESFASADAS: la IV que implican los precios ({meta['iv_calc_med']}%) dobla")
            print(f"   a la IV publicada ({meta['iv_mkt_med']}%) → las opciones cotizan de otra sesión que el spot.")
            print("   Fíate de Delta/Gamma/Vega/Theta (dependen de σ, no del precio cotizado), no del 'teórico vs mercado'.")
        print("\n> 'Vol implícita' = la que descuenta el mercado; si es muy superior a la histórica,")
        print("> las opciones están caras. No es recomendación de inversión.\n")
    else:
        tau = a.dias / 365.0
        g = bsm(a.precio, a.strike, tau, a.tasa, a.vol, 0.0, a.tipo)
        print(f"\n=== {a.tipo.upper()} · S={a.precio} K={a.strike} {a.dias}d σ={a.vol*100:.0f}% r={a.tasa*100:.0f}% ===\n")
        print(explicar(g, a.precio, a.strike, tau, a.tipo).replace("**", ""))
        print()


if __name__ == "__main__":
    main()
