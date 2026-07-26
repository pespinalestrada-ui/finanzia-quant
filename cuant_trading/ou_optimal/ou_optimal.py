"""
ou_optimal — calibración Ornstein-Uhlenbeck y PARADA ÓPTIMA (guía §6.3, §6.4 y §7).

Completa el pairs trading: `pairs_trading` detecta el par y su half-life; esto
calibra el "muelle" y calcula CUÁNDO SALIR de forma matemática, no a ojo.

1) CALIBRACIÓN (§6.3) — el OU continuo dS = θ(μ−S)dt + σdW se estima con su
   gemelo discreto AR(1): S(t+1) = a + φ·S(t) + ε. Equivalencias exactas:
       θ = −ln(φ)/Δt        μ = a/(1−φ)        σ² = σ²_ε·2θ/(1−φ²)
   Desviación estacionaria (cuánto oscila el spread por puro azar):
       σ_eq = √(σ²/(2θ))     ·     half-life = ln2/θ

2) OPERATIVA (§6.4) — tamaño proporcional al estiramiento y stop más allá del
   vaivén normal:
       Tamaño ∝ (S−μ)/σ_eq   ·   Stop = μ ± k·σ_eq
   (usamos σ_eq en ambos, la "unificación de vara" que el propio PDF recomienda)

3) PARADA ÓPTIMA (§7) — ¿en qué nivel exacto cerrar, descontando el coste c_s?
   La función de valor cumple L·u = r·u con el generador infinitesimal
       L = (σ²/2)·d²/dx² + θ(μ−x)·d/dx
   La solución creciente tiene forma integral (Alili-Patie-Pedersen):
       F(x) = ∫₀^∞ u^(r/θ−1)·exp( √(2θ)/σ·(x−μ)·u − u²/2 ) du
   y el umbral óptimo b* sale del acoplamiento suave (smooth-pasting):
       F(b*) − (b*−c_s)·F'(b*) = 0
   Se resuelve por integración numérica + búsqueda de raíz. No es aproximación
   heurística: es la frontera exacta del problema.

No es recomendación de inversión.

Uso:
    python ou_optimal.py KO PEP
    python ou_optimal.py EWA EWC --coste 0.02 --tasa 0.05
"""
import argparse
import sys
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq

_SUITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SUITE / "pairs_trading"))


def calibrar_ou(spread, dt=1.0 / 252.0):
    """AR(1) → parámetros del muelle. Devuelve dict(theta, mu, sigma, sigma_eq, half_life, phi)."""
    s = pd.Series(spread).dropna().values.astype(float)
    if len(s) < 30:
        raise ValueError("Serie demasiado corta para calibrar (mínimo 30 puntos).")
    x, y = s[:-1], s[1:]
    # regresión y = a + phi·x  (mínimos cuadrados)
    phi, a = np.polyfit(x, y, 1)
    resid = y - (a + phi * x)
    sd_eps = float(np.std(resid, ddof=2))
    if not (0 < phi < 1):
        # phi>=1 → no revierte; phi<=0 → oscilación extrema
        return {"theta": float("nan"), "mu": float(np.mean(s)), "sigma": float("nan"),
                "sigma_eq": float(np.std(s)), "half_life": float("nan"), "phi": float(phi),
                "revierte": False}
    theta = -np.log(phi) / dt
    mu = a / (1.0 - phi)
    sigma = sd_eps * np.sqrt(2.0 * theta / (1.0 - phi ** 2))
    sigma_eq = sigma / np.sqrt(2.0 * theta)      # desviación estacionaria
    return {"theta": float(theta), "mu": float(mu), "sigma": float(sigma),
            "sigma_eq": float(sigma_eq), "half_life": float(np.log(2) / theta),
            "phi": float(phi), "revierte": True}


def _F(y, theta, sigma, r):
    """Solución creciente de L·u = r·u sobre el spread CENTRADO (media 0)."""
    z = np.sqrt(2.0 * theta) / sigma * y
    p = r / theta
    f = lambda u: u ** (p - 1.0) * np.exp(z * u - 0.5 * u * u)
    val, _ = quad(f, 0, np.inf, limit=200)
    return val


def _dF(y, theta, sigma, r):
    """Derivada dF/dy (se deriva dentro de la integral)."""
    k = np.sqrt(2.0 * theta) / sigma
    z = k * y
    p = r / theta
    f = lambda u: u ** p * np.exp(z * u - 0.5 * u * u)
    val, _ = quad(f, 0, np.inf, limit=200)
    return k * val


def umbral_optimo(par, coste=0.0, tasa=0.05):
    """
    Umbral de salida óptimo por acoplamiento suave: F(b) − (b−c_s)·F'(b) = 0,
    resuelto sobre el spread CENTRADO (y = S − μ), que es la formulación correcta:
    el pago de cerrar es el BENEFICIO (distancia recorrida), no el nivel absoluto
    del spread. Con el nivel absoluto (μ grande, p. ej. logs) el descuento domina
    y el problema degenera.

    Devuelve dict(b_estrella=nivel absoluto, en_sigmas=distancia a μ en σ_eq, metodo).
    """
    theta, mu, sigma, se = par["theta"], par["mu"], par["sigma"], par["sigma_eq"]
    if not par.get("revierte") or not np.isfinite(theta) or theta <= 0:
        return {"b_estrella": float("nan"), "en_sigmas": float("nan"), "metodo": "no revierte"}
    g = lambda y: _F(y, theta, sigma, tasa) - (y - coste) * _dF(y, theta, sigma, tasa)
    try:
        lo, hi = 1e-9, 6.0 * se
        glo, ghi = g(lo), g(hi)
        intentos = 0
        while glo * ghi > 0 and intentos < 3:      # ampliar si no hay cambio de signo
            hi *= 2.5
            ghi = g(hi)
            intentos += 1
        if glo * ghi > 0:
            raise ValueError("sin raíz")
        y_b = brentq(g, lo, hi, xtol=1e-12, maxiter=300)
        return {"b_estrella": float(mu + y_b), "en_sigmas": float(y_b / se),
                "metodo": "smooth-pasting"}
    except Exception as e:
        return {"b_estrella": float("nan"), "en_sigmas": float("nan"),
                "metodo": f"no resuelto ({str(e)[:40]})"}


def plan_operativo(par, spread_actual, capital=10000.0, factor_riesgo=0.25, k_stop=2.0,
                   coste=0.0, tasa=0.05):
    """Tamaño (§6.4), stop estacionario y salida óptima (§7) para el estado actual."""
    mu, se = par["mu"], par["sigma_eq"]
    z = (spread_actual - mu) / se if se > 0 else 0.0
    lado = "CORTO spread" if z > 0 else "LARGO spread"
    # tamaño ∝ estiramiento (misma vara: sigma_eq), acotado al 100% del capital
    frac = min(1.0, abs(z) * factor_riesgo)
    # el stop se pone AL OTRO LADO: si vas corto (z>0) y sigue subiendo, te sacan
    stop = mu + (k_stop * se if z > 0 else -k_stop * se)
    opt = umbral_optimo(par, coste, tasa)
    b = opt["b_estrella"]
    # el umbral se calcula "por encima de μ"; para un CORTO es su espejo
    if np.isfinite(b):
        salida = b if z < 0 else 2 * mu - b
        salida_sig = opt["en_sigmas"] if z < 0 else -opt["en_sigmas"]
    else:
        salida, salida_sig = float("nan"), float("nan")
    return {"z": z, "lado": lado, "fraccion_capital": frac, "importe": capital * frac,
            "stop": float(stop), "salida_optima": float(salida),
            "salida_en_sigmas": float(salida_sig), "metodo": opt["metodo"]}


def analizar_par(a, b, period="5y", coste=0.0, tasa=0.05, capital=10000.0):
    """Calibra el par (spread log con hedge OLS), calcula umbrales y plan."""
    import pairs_trading as PT
    px = PT._precios([a.upper(), b.upper()], period)
    if px.shape[1] < 2:
        raise ValueError("Sin datos suficientes para el par.")
    la, lb = np.log(px[a.upper()]), np.log(px[b.upper()])
    beta, _ = np.polyfit(la.values, lb.values, 1)     # lb = beta·la + c
    spread = lb - beta * la                            # spread en logs (guía §6.1)
    par = calibrar_ou(spread.values)
    plan = plan_operativo(par, float(spread.iloc[-1]), capital, coste=coste, tasa=tasa)
    par["beta"] = float(beta)
    return par, plan, spread


def informe(a, b, par, plan):
    if not par.get("revierte"):
        return (f"=== {a}/{b} ===\n  El spread NO revierte (φ={par['phi']:.3f}): el muelle no existe.\n"
                "  Sin reversión no hay estrategia: busca otro par en 🔗 Pairs.")
    L = [f"=== {a}/{b} · calibración Ornstein-Uhlenbeck ===\n",
         f"  β (cobertura)        : {par['beta']:.3f}",
         f"  θ (fuerza del muelle): {par['theta']:.2f} /año   → half-life {par['half_life']*252:.1f} días",
         f"  μ (nivel de reposo)  : {par['mu']:.4f}",
         f"  σ_eq (vaivén normal) : {par['sigma_eq']:.4f}\n",
         f"  Spread hoy           : {plan['z']:+.2f} σ_eq  →  {plan['lado']}",
         f"  Tamaño sugerido      : {plan['fraccion_capital']*100:.0f}% del capital "
         f"({plan['importe']:,.0f} €)",
         f"  Stop (±2σ_eq)        : {plan['stop']:.4f}"]
    if np.isfinite(plan["salida_optima"]):
        L.append(f"  SALIDA ÓPTIMA        : {plan['salida_optima']:.4f} "
                 f"({plan['salida_en_sigmas']:+.2f} σ_eq desde μ) — {plan['metodo']}")
        L.append("\n> La salida óptima NO es 'cerrar en la media': el modelo dice que compensa")
        L.append("> esperar hasta ese nivel, descontando el coste de operar. Es la frontera exacta.")
    else:
        L.append(f"  SALIDA ÓPTIMA        : no resuelta ({plan['metodo']}) → usa la media μ.")
    L.append("> No es recomendación de inversión.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Calibración OU + parada óptima para pairs trading.")
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--period", default="5y")
    ap.add_argument("--coste", type=float, default=0.0, help="Coste fijo por operar (en unidades de spread).")
    ap.add_argument("--tasa", type=float, default=0.05, help="Tasa de descuento (impaciencia).")
    ap.add_argument("--capital", type=float, default=10000.0)
    x = ap.parse_args()
    par, plan, _ = analizar_par(x.a, x.b, x.period, x.coste, x.tasa, x.capital)
    print("\n" + informe(x.a.upper(), x.b.upper(), par, plan) + "\n")


if __name__ == "__main__":
    main()
