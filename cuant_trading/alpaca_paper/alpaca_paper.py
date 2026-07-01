"""
alpaca_paper — conector a Alpaca PAPER trading (dinero ficticio, real-time).

El salto a "vivo" del flujo intradía, SIN riesgo: cuenta de papel con datos en
tiempo real (feed IEX gratis) y órdenes simuladas. Usa la API REST directamente
con `requests` (NINGUNA dependencia nueva).

  - cuenta()        : equity, cash, buying power, estado.
  - posiciones()    : posiciones abiertas (papel).
  - cotizacion(sym) : último precio real-time (IEX).
  - ordenes()       : últimas órdenes.
  - enviar_orden()  : MANDA una orden paper. Pensada para que la dispare el USUARIO
                      (botón del dashboard / CLI con --confirmar). Es simulada, pero
                      el asistente nunca la ejecuta por su cuenta.

Claves: se leen del .env de la raíz (ALPACA_KEY, ALPACA_SECRET) con os.getenv.
Nunca se hardcodean ni se suben a GitHub.

Uso:
    python alpaca_paper.py cuenta
    python alpaca_paper.py precio AAPL
    python alpaca_paper.py comprar AAPL 1 --confirmar     (orden paper; requiere --confirmar)
"""
import argparse
import os
import sys
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import warnings
warnings.filterwarnings("ignore")

import requests

API = "https://paper-api.alpaca.markets"          # cuenta/órdenes (PAPER)
DATA = "https://data.alpaca.markets"              # datos de mercado
_ENV_CARGADO = False


def _cargar_env():
    global _ENV_CARGADO
    if _ENV_CARGADO:
        return
    _ENV_CARGADO = True
    env = Path(__file__).resolve().parents[2] / ".env"
    if not env.exists():
        return
    try:
        for ln in env.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#") or "=" not in ln:
                continue
            k, v = ln.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v and k not in os.environ:
                os.environ[k] = v
    except Exception:
        pass


def _headers():
    _cargar_env()
    k, s = os.getenv("ALPACA_KEY"), os.getenv("ALPACA_SECRET")
    if not k or not s:
        raise RuntimeError("Faltan ALPACA_KEY / ALPACA_SECRET en el .env de la raíz.")
    return {"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": s}


def configurada():
    """True si hay claves de Alpaca disponibles."""
    _cargar_env()
    return bool(os.getenv("ALPACA_KEY") and os.getenv("ALPACA_SECRET"))


def cuenta():
    r = requests.get(f"{API}/v2/account", headers=_headers(), timeout=12)
    r.raise_for_status()
    d = r.json()
    return {
        "estado": d.get("status"),
        "moneda": d.get("currency"),
        "equity": float(d.get("equity", 0)),
        "cash": float(d.get("cash", 0)),
        "buying_power": float(d.get("buying_power", 0)),
        "valor_posiciones": float(d.get("long_market_value", 0)),
        "pnl_dia": float(d.get("equity", 0)) - float(d.get("last_equity", 0)),
        "bloqueada": d.get("trading_blocked"),
    }


def posiciones():
    r = requests.get(f"{API}/v2/positions", headers=_headers(), timeout=12)
    r.raise_for_status()
    out = []
    for p in r.json():
        out.append({
            "Símbolo": p.get("symbol"),
            "Lado": p.get("side"),
            "Cant.": float(p.get("qty", 0)),
            "Precio medio": round(float(p.get("avg_entry_price", 0)), 3),
            "Precio actual": round(float(p.get("current_price", 0)), 3),
            "P&L $": round(float(p.get("unrealized_pl", 0)), 2),
            "P&L %": round(float(p.get("unrealized_plpc", 0)) * 100, 2),
        })
    return out


def barras_intradia(symbol, timeframe="5Min", horas=30):
    """Barras intradía REAL-TIME (feed IEX gratis). Devuelve DataFrame OHLCV
    con índice datetime en hora de Nueva York. Solo tickers de EEUU."""
    import pandas as pd
    from datetime import datetime, timedelta, timezone
    sym = symbol.strip().upper()
    start = (datetime.now(timezone.utc) - timedelta(hours=horas)).strftime("%Y-%m-%dT%H:%M:%SZ")
    filas, token = [], None
    for _ in range(5):                                   # paginación
        params = {"timeframe": timeframe, "start": start, "feed": "iex", "limit": 10000}
        if token:
            params["page_token"] = token
        r = requests.get(f"{DATA}/v2/stocks/{sym}/bars", headers=_headers(), params=params, timeout=15)
        r.raise_for_status()
        d = r.json()
        filas.extend(d.get("bars") or [])
        token = d.get("next_page_token")
        if not token:
            break
    if not filas:
        return None
    df = pd.DataFrame(filas).rename(columns={"o": "Open", "h": "High", "l": "Low",
                                             "c": "Close", "v": "Volume"})
    df.index = pd.to_datetime(df["t"]).dt.tz_convert("America/New_York")
    return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)


def cotizacion(symbol):
    """Último trade real-time (feed IEX gratis)."""
    h = _headers()
    sym = symbol.strip().upper()
    r = requests.get(f"{DATA}/v2/stocks/{sym}/trades/latest", headers=h, params={"feed": "iex"}, timeout=12)
    r.raise_for_status()
    t = r.json().get("trade", {})
    return {"symbol": sym, "precio": float(t.get("p", 0)), "hora": t.get("t")}


def ordenes(estado="all", limite=20):
    r = requests.get(f"{API}/v2/orders", headers=_headers(),
                     params={"status": estado, "limit": limite, "direction": "desc"}, timeout=12)
    r.raise_for_status()
    out = []
    for o in r.json():
        out.append({
            "Símbolo": o.get("symbol"), "Lado": o.get("side"), "Cant.": o.get("qty"),
            "Tipo": o.get("type"), "Estado": o.get("status"),
            "Precio medio": o.get("filled_avg_price"), "Creada": (o.get("created_at") or "")[:19],
        })
    return out


def enviar_orden(symbol, qty, side, tipo="market", tif="day"):
    """
    MANDA una orden PAPER (simulada). La dispara el usuario (botón/--confirmar);
    el asistente no la ejecuta de forma autónoma. Devuelve el resumen de la orden.
    """
    body = {"symbol": symbol.strip().upper(), "qty": str(qty), "side": side,
            "type": tipo, "time_in_force": tif}
    r = requests.post(f"{API}/v2/orders", headers=_headers(), json=body, timeout=12)
    if r.status_code >= 300:
        raise RuntimeError(f"Alpaca rechazó la orden ({r.status_code}): {r.text}")
    o = r.json()
    return {"id": o.get("id"), "symbol": o.get("symbol"), "side": o.get("side"),
            "qty": o.get("qty"), "estado": o.get("status")}


def main():
    ap = argparse.ArgumentParser(description="Conector Alpaca PAPER (dinero ficticio).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cuenta")
    sub.add_parser("posiciones")
    sub.add_parser("ordenes")
    pp = sub.add_parser("precio"); pp.add_argument("symbol")
    for nombre in ("comprar", "vender"):
        s = sub.add_parser(nombre)
        s.add_argument("symbol"); s.add_argument("qty", type=float)
        s.add_argument("--confirmar", action="store_true", help="Necesario para enviar la orden.")
    a = ap.parse_args()

    if a.cmd == "cuenta":
        c = cuenta()
        print(f"\nCuenta PAPER · estado {c['estado']} · {c['moneda']}")
        print(f"  Equity        : {c['equity']:.2f}")
        print(f"  Cash          : {c['cash']:.2f}")
        print(f"  Buying power  : {c['buying_power']:.2f}")
        print(f"  Posiciones    : {c['valor_posiciones']:.2f}")
        print(f"  P&L del día   : {c['pnl_dia']:+.2f}\n")
    elif a.cmd == "posiciones":
        import pandas as pd
        p = posiciones()
        print("\n" + (pd.DataFrame(p).to_string(index=False) if p else "Sin posiciones abiertas.") + "\n")
    elif a.cmd == "ordenes":
        import pandas as pd
        o = ordenes()
        print("\n" + (pd.DataFrame(o).to_string(index=False) if o else "Sin órdenes.") + "\n")
    elif a.cmd == "precio":
        q = cotizacion(a.symbol)
        print(f"\n{q['symbol']}: {q['precio']:.4f}  (IEX · {q['hora']})\n")
    elif a.cmd in ("comprar", "vender"):
        side = "buy" if a.cmd == "comprar" else "sell"
        if not a.confirmar:
            print(f"\n[SIMULACRO] {side} {a.qty} {a.symbol.upper()} (orden PAPER). "
                  f"Añade --confirmar para enviarla de verdad.\n")
            return
        o = enviar_orden(a.symbol, a.qty, side)
        print(f"\nOrden PAPER enviada: {o['side']} {o['qty']} {o['symbol']} · estado {o['estado']} · id {o['id']}\n")


if __name__ == "__main__":
    main()
