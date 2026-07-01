"""
intraday — análisis y backtest INTRADÍA con costes (desarrollo, sin arriesgar).

Intradía es otra disciplina (no es swing): barras de minutos, VWAP, rango de
apertura, y sobre todo COSTES (comisión + spread + slippage) que se comen el edge.
Este módulo deja DESARROLLAR y VALIDAR un método intradía gratis, antes de pasar a
datos en tiempo real / paper trading (Alpaca).

Datos: yfinance intradía (gratis pero con retraso ~15 min y poco histórico:
1m → 7 días, 5m/15m → 60 días). Sirve para backtestear, NO para ejecutar en vivo.

Contenido:
  - descargar(): barras intradía de la sesión regular.
  - indicadores_intradia(): VWAP por sesión, rango de apertura (ORB), ATR intradía, RSI.
  - snapshot(): foto del estado intradía de hoy (precio vs VWAP / rango de apertura).
  - backtest_orb(): Opening Range Breakout con MODELO DE COSTES (bruto vs neto).
    Sin coste el backtest miente; aquí se ve cuánto edge sobrevive a los costes.

Uso:
    python intraday.py AAPL
    python intraday.py SAB.MC --interval 15m --or-min 30 --coste-bps 8
"""
import argparse
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

_MIN = {"1m": 1, "2m": 2, "5m": 5, "15m": 15, "30m": 30, "60m": 60, "90m": 90}
_MAX_PERIOD = {"1m": "7d", "2m": "60d", "5m": "60d", "15m": "60d", "30m": "60d", "60m": "60d", "90m": "60d"}


def descargar(ticker, interval="5m", period=None):
    """Barras intradía (sesión regular). Ajusta el periodo al límite de yfinance."""
    if interval not in _MIN:
        raise ValueError(f"Intervalo no soportado: {interval}. Usa {list(_MIN)}.")
    period = period or _MAX_PERIOD[interval]
    h = yf.Ticker(ticker).history(period=period, interval=interval, prepost=False)
    if h.empty:
        raise ValueError(f"'{ticker}' sin datos intradía {interval}. ¿Mercado cerrado o ticker inválido?")
    h = h[["Open", "High", "Low", "Close", "Volume"]].dropna()
    h.index = pd.to_datetime(h.index)
    h["sesion"] = h.index.date
    return h


def indicadores_intradia(df, or_min=30, interval="5m"):
    """Añade VWAP (reinicio por sesión), rango de apertura, ATR intradía y RSI."""
    bar = _MIN.get(interval, 5)
    or_bars = max(1, or_min // bar)
    out = df.copy()
    tp = (out["High"] + out["Low"] + out["Close"]) / 3.0          # precio típico
    pv = tp * out["Volume"]
    # VWAP acumulado dentro de cada sesión
    out["cum_pv"] = pv.groupby(out["sesion"]).cumsum()
    out["cum_v"] = out["Volume"].groupby(out["sesion"]).cumsum()
    out["VWAP"] = out["cum_pv"] / out["cum_v"].replace(0, np.nan)
    # rango de apertura por sesión (primeras or_bars barras)
    def _or(g):
        o = g.iloc[:or_bars]
        g = g.copy()
        g["OR_high"] = o["High"].max()
        g["OR_low"] = o["Low"].min()
        return g
    out = out.groupby("sesion", group_keys=False).apply(_or)
    # ATR intradía (True Range medio, 14 barras)
    pc = out["Close"].shift(1)
    tr = pd.concat([out["High"] - out["Low"], (out["High"] - pc).abs(), (out["Low"] - pc).abs()], axis=1).max(axis=1)
    out["ATR"] = tr.rolling(14).mean()
    # RSI intradía
    d = out["Close"].diff()
    up = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    out["RSI"] = 100 - 100 / (1 + up / (dn + 1e-9))
    return out


def snapshot(ticker, interval="5m", or_min=30):
    """Foto del estado intradía de HOY (última sesión disponible). (fig, tabla, md)."""
    df = descargar(ticker, interval)
    ind = indicadores_intradia(df, or_min, interval)
    ult_sesion = ind["sesion"].iloc[-1]
    hoy = ind[ind["sesion"] == ult_sesion]
    last = hoy.iloc[-1]
    px = float(last["Close"]); vwap = float(last["VWAP"])
    orh, orl = float(last["OR_high"]), float(last["OR_low"])
    rsi = float(last["RSI"])

    pos_vwap = "ENCIMA" if px > vwap else "DEBAJO"
    if px > orh:
        zona = "ROTURA ALCISTA del rango de apertura"
    elif px < orl:
        zona = "ROTURA BAJISTA del rango de apertura"
    else:
        zona = "DENTRO del rango de apertura"
    sesgo = "alcista" if (px > vwap and px > orh) else "bajista" if (px < vwap and px < orl) else "mixto/sin sesgo claro"

    tabla = pd.DataFrame([
        {"Métrica": "Precio", "Valor": f"{px:.3f}"},
        {"Métrica": "VWAP", "Valor": f"{vwap:.3f}  ({pos_vwap})"},
        {"Métrica": "Rango apertura", "Valor": f"{orl:.3f} – {orh:.3f}"},
        {"Métrica": "RSI intradía", "Valor": f"{rsi:.0f}"},
        {"Métrica": "Rango de hoy", "Valor": f"{float(hoy['Low'].min()):.3f} – {float(hoy['High'].max()):.3f}"},
    ])
    md = (f"### {ticker.upper()} · intradía {interval} · sesión {ult_sesion}\n"
          f"**{zona}** · precio **{pos_vwap}** del VWAP · sesgo intradía: **{sesgo}**.\n\n"
          f"> Datos yfinance con retraso ~15 min: para DESARROLLO, no para ejecutar en vivo.")
    fig = _plot_snapshot(hoy, ticker, interval, ult_sesion)
    return fig, tabla, md


def snapshot_alpaca(ticker, or_min=30):
    """Snapshot intradía con datos EN VIVO de Alpaca (IEX). Solo tickers EEUU.
    Devuelve (fig, tabla, md). Lanza ValueError con mensaje claro si no procede."""
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "alpaca_paper"))
    import alpaca_paper as AP
    if not AP.configurada():
        raise ValueError("Faltan claves de Alpaca en el .env (ALPACA_KEY/ALPACA_SECRET).")
    tk = ticker.strip().upper()
    if "." in tk or "-" in tk:
        raise ValueError(f"Alpaca solo cubre EEUU ({tk} no). Usa el Snapshot normal (yfinance).")
    df = AP.barras_intradia(tk, "5Min", horas=30)
    if df is None or len(df) < 8:
        raise ValueError(f"Sin barras intradía de {tk} ahora mismo (¿mercado cerrado?). Usa el Snapshot normal.")
    df = df.copy()
    df["sesion"] = df.index.date
    ult = df["sesion"].iloc[-1]
    df = df[df["sesion"] == ult]                       # solo la sesión más reciente
    if len(df) < 3:
        raise ValueError("Sesión recién abierta: aún hay pocas barras. Reintenta en unos minutos.")
    ind = indicadores_intradia(df, or_min, "5m")
    last = ind.iloc[-1]
    # precio del ÚLTIMO trade (aún más fresco que la última barra)
    try:
        px = float(AP.cotizacion(tk)["precio"]) or float(last["Close"])
    except Exception:
        px = float(last["Close"])
    vwap = float(last["VWAP"]); orh = float(last["OR_high"]); orl = float(last["OR_low"])
    rsi = float(last["RSI"])
    pos_vwap = "ENCIMA" if px > vwap else "DEBAJO"
    zona = ("ROTURA ALCISTA del rango de apertura" if px > orh else
            "ROTURA BAJISTA del rango de apertura" if px < orl else
            "DENTRO del rango de apertura")
    sesgo = ("alcista" if (px > vwap and px > orh) else
             "bajista" if (px < vwap and px < orl) else "mixto/sin sesgo claro")
    tabla = pd.DataFrame([
        {"Métrica": "Precio (último trade, EN VIVO)", "Valor": f"{px:.3f}"},
        {"Métrica": "VWAP", "Valor": f"{vwap:.3f}  ({pos_vwap})"},
        {"Métrica": "Rango apertura", "Valor": f"{orl:.3f} – {orh:.3f}"},
        {"Métrica": "RSI intradía", "Valor": f"{rsi:.0f}"},
        {"Métrica": "Rango de hoy", "Valor": f"{float(ind['Low'].min()):.3f} – {float(ind['High'].max()):.3f}"},
        {"Métrica": "Barras 5m de la sesión", "Valor": f"{len(ind)}"},
    ])
    md = (f"### 📡 {tk} · intradía EN VIVO (Alpaca/IEX) · sesión {ult}\n"
          f"**{zona}** · precio **{pos_vwap}** del VWAP · sesgo intradía: **{sesgo}**.\n\n"
          f"> Datos en tiempo real del feed IEX (gratis). Sin el retraso de ~15 min de yfinance.")
    fig = _plot_snapshot(ind, tk + " (EN VIVO)", "5m", ult)
    return fig, tabla, md


def _sesion_actual(ticker, or_min=30):
    """Barras de la sesión más reciente con indicadores. Intenta Alpaca EN VIVO
    (EEUU) y cae a yfinance (~15 min retraso). Devuelve (ind, px, fuente)."""
    tk = ticker.strip().upper()
    ind = None; px = None; fuente = "yfinance (~15 min retraso)"
    if "." not in tk and "-" not in tk:
        try:
            import sys as _sys
            from pathlib import Path as _P
            _sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "alpaca_paper"))
            import alpaca_paper as AP
            if AP.configurada():
                df = AP.barras_intradia(tk, "5Min", horas=30)
                if df is not None and len(df) >= 8:
                    df = df.copy(); df["sesion"] = df.index.date
                    df = df[df["sesion"] == df["sesion"].iloc[-1]]
                    if len(df) >= 3:
                        ind = indicadores_intradia(df, or_min, "5m")
                        try:
                            px = float(AP.cotizacion(tk)["precio"]) or float(ind["Close"].iloc[-1])
                        except Exception:
                            px = float(ind["Close"].iloc[-1])
                        fuente = "Alpaca/IEX (EN VIVO)"
        except Exception:
            ind = None
    if ind is None:
        df = descargar(tk, "5m")
        ind_all = indicadores_intradia(df, or_min, "5m")
        ult = ind_all["sesion"].iloc[-1]
        ind = ind_all[ind_all["sesion"] == ult]
        px = float(ind["Close"].iloc[-1])
    return ind, px, fuente


def semaforo(ticker, or_min=30):
    """Semáforo del DÍA: 🟢 opera largo / 🔴 opera corto / 🟡 no operes hoy,
    con razones en lenguaje claro. Devuelve (fig, tabla, md)."""
    ind, px, fuente = _sesion_actual(ticker, or_min)
    last = ind.iloc[-1]
    vwap = float(last["VWAP"]); orh = float(last["OR_high"]); orl = float(last["OR_low"])
    rsi = float(last["RSI"]); atr = float(last["ATR"]) if not np.isnan(last["ATR"]) else px * 0.005
    c = ind["Close"]
    ema9 = c.ewm(span=9, adjust=False).mean().iloc[-1]

    razones, votos = [], 0
    # 1. VWAP: el precio de referencia del día
    if px > vwap:
        votos += 1; razones.append("✅ Precio por ENCIMA del VWAP (los compradores mandan hoy)")
    else:
        votos -= 1; razones.append("❌ Precio por DEBAJO del VWAP (los vendedores mandan hoy)")
    # 2. rango de apertura
    if px > orh:
        votos += 1; razones.append("✅ Rotura ALCISTA del rango de apertura")
    elif px < orl:
        votos -= 1; razones.append("❌ Rotura BAJISTA del rango de apertura")
    else:
        razones.append("➖ Sigue DENTRO del rango de apertura (sin dirección aún)")
    # 3. micro-tendencia (EMA9 de 5m)
    if px > ema9:
        votos += 1; razones.append("✅ Por encima de la media rápida de la sesión (EMA9)")
    else:
        votos -= 1; razones.append("❌ Por debajo de la media rápida de la sesión (EMA9)")
    # 4. ¿hay rango suficiente para que compense operar? (vs coste típico)
    rango_dia = float(ind["High"].max() - ind["Low"].min())
    if rango_dia < 4 * (px * 0.0006):                 # rango < ~4x coste ida+vuelta
        razones.append("⚠️ Día MUY estrecho: el movimiento apenas cubre los costes")
        estrecho = True
    else:
        estrecho = False
    # 5. RSI extremo = tarde para perseguir
    tarde = ""
    if rsi > 75 and votos > 0:
        tarde = " (RSI muy alto: tarde para perseguir, espera un retroceso)"
    if rsi < 25 and votos < 0:
        tarde = " (RSI muy bajo: tarde para perseguir, espera un rebote)"

    if estrecho or votos == 0 or abs(votos) == 1:
        verd, emoji = "NO OPERES HOY (o espera)", "🟡"
        consejo = "Señales mezcladas o día estrecho: lo rentable hoy es no operar."
    elif votos >= 2:
        verd, emoji = "SESGO LARGO", "🟢"
        consejo = f"Si operas, mejor al alza. Stop de referencia: {px - 1.5*atr:.2f} (1.5×ATR)." + tarde
    else:
        verd, emoji = "SESGO CORTO", "🔴"
        consejo = f"Si operas, mejor a la baja. Stop de referencia: {px + 1.5*atr:.2f} (1.5×ATR)." + tarde

    tabla = pd.DataFrame([
        {"Métrica": "Precio", "Valor": f"{px:.3f}"},
        {"Métrica": "VWAP", "Valor": f"{vwap:.3f}"},
        {"Métrica": "Rango apertura", "Valor": f"{orl:.3f} – {orh:.3f}"},
        {"Métrica": "RSI intradía", "Valor": f"{rsi:.0f}"},
        {"Métrica": "Rango del día", "Valor": f"{rango_dia:.3f}"},
        {"Métrica": "Fuente de datos", "Valor": fuente},
    ])
    md = (f"# {emoji} {verd}\n\n**{ticker.upper()}** · hoy · datos: {fuente}\n\n"
          + "\n".join("- " + r for r in razones)
          + f"\n\n**Consejo:** {consejo}\n\n"
          "> Semáforo del día (VWAP + apertura + micro-tendencia). El intradía es la liga "
          "difícil: la mayoría de días la respuesta correcta es 🟡. No es recomendación.")
    fig = _plot_snapshot(ind, ticker.upper() + " (semáforo)", "5m", ind["sesion"].iloc[-1])
    return fig, tabla, md


def backtest_orb(df, or_min=30, interval="5m", coste_bps=6.0):
    """
    Opening Range Breakout con MODELO DE COSTES. Una operación por sesión: entra al
    primer cierre que rompe el rango de apertura (long arriba / short abajo), sale al
    cierre de sesión. Resta 'coste_bps' (comisión+spread+slippage) por ida y vuelta.
    Devuelve DataFrame de operaciones (bruto y neto).
    """
    bar = _MIN.get(interval, 5)
    or_bars = max(1, or_min // bar)
    coste = coste_bps / 10000.0
    trades = []
    for fecha, g in df.groupby("sesion"):
        if len(g) < or_bars + 3:
            continue
        o = g.iloc[:or_bars]
        hi, lo = float(o["High"].max()), float(o["Low"].min())
        resto = g.iloc[or_bars:]
        pos, entrada = None, None
        for _, row in resto.iterrows():
            c = float(row["Close"])
            if c > hi:
                pos, entrada = "LONG", c; break
            if c < lo:
                pos, entrada = "SHORT", c; break
        if pos is None:
            continue
        salida = float(g["Close"].iloc[-1])
        bruto = (salida / entrada - 1.0) * (1 if pos == "LONG" else -1)
        neto = bruto - coste
        trades.append({"sesion": str(fecha), "dir": pos, "entrada": round(entrada, 3),
                       "salida": round(salida, 3), "bruto_%": round(bruto * 100, 3),
                       "neto_%": round(neto * 100, 3)})
    return pd.DataFrame(trades)


def backtest_estrategia(df, estrategia="orb", or_min=30, interval="5m", coste_bps=6.0):
    """Backtest intradía CON COSTES de 3 estrategias (1 trade/sesión, salida al cierre):
      - orb : rotura del rango de apertura (la de siempre).
      - vwap: retorno al VWAP — tras estar claramente por debajo/encima, el cruce
              de vuelta del VWAP marca la entrada (reversión a la media del día).
      - ema9: pullback a la EMA9 — en micro-tendencia (EMA9>EMA21), tocar la EMA9
              y cerrar por encima marca la entrada (seguir tendencia tras descanso).
    Devuelve DataFrame de trades (bruto y neto)."""
    if estrategia == "orb":
        return backtest_orb(df, or_min, interval, coste_bps)
    bar = _MIN.get(interval, 5)
    or_bars = max(1, or_min // bar)
    coste = coste_bps / 10000.0
    trades = []
    for fecha, g in df.groupby("sesion"):
        if len(g) < or_bars + 6:
            continue
        c = g["Close"].astype(float)
        tp = (g["High"] + g["Low"] + g["Close"]) / 3.0
        vwap = (tp * g["Volume"]).cumsum() / g["Volume"].cumsum().replace(0, np.nan)
        ema9 = c.ewm(span=9, adjust=False).mean()
        ema21 = c.ewm(span=21, adjust=False).mean()
        pos, entrada = None, None
        for i in range(or_bars + 1, len(g)):
            px = float(c.iloc[i]); pxa = float(c.iloc[i - 1])
            vw = float(vwap.iloc[i]); vwa = float(vwap.iloc[i - 1])
            if estrategia == "vwap":
                # cruce de VUELTA al VWAP tras desviación de al menos 0.15%
                if pxa < vwa * 0.9985 and px > vw:
                    pos, entrada = "LONG", px; break
                if pxa > vwa * 1.0015 and px < vw:
                    pos, entrada = "SHORT", px; break
            else:  # ema9 pullback
                e9, e21 = float(ema9.iloc[i]), float(ema21.iloc[i])
                lo_i, hi_i = float(g["Low"].iloc[i]), float(g["High"].iloc[i])
                if e9 > e21 and lo_i <= e9 and px > e9:
                    pos, entrada = "LONG", px; break
                if e9 < e21 and hi_i >= e9 and px < e9:
                    pos, entrada = "SHORT", px; break
        if pos is None:
            continue
        salida = float(c.iloc[-1])
        bruto = (salida / entrada - 1.0) * (1 if pos == "LONG" else -1)
        trades.append({"sesion": str(fecha), "dir": pos, "entrada": round(entrada, 3),
                       "salida": round(salida, 3), "bruto_%": round(bruto * 100, 3),
                       "neto_%": round((bruto - coste) * 100, 3)})
    return pd.DataFrame(trades)


def metricas_backtest(bt, coste_bps):
    if bt.empty:
        return {"n": 0, "mensaje": "Sin operaciones (no hubo roturas o histórico corto)."}
    b = bt["bruto_%"] / 100.0
    n = bt["neto_%"] / 100.0
    from math import erf, sqrt
    N = len(n)
    win = float((n > 0).mean())
    exp_neto = float(n.mean())
    z = (win - 0.5) / sqrt(0.25 / N) if N > 0 else float("nan")
    pval = 1 - 0.5 * (1 + erf(z / sqrt(2))) if not np.isnan(z) else float("nan")
    return {
        "n": N,
        "win_rate": round(win * 100, 1),
        "exp_bruto_pct": round(float(b.mean()) * 100, 3),
        "exp_neto_pct": round(exp_neto * 100, 3),
        "coste_pct": round(coste_bps / 100.0, 3),
        "total_neto_pct": round(float(n.sum()) * 100, 2),
        "pval": round(pval, 3) if not np.isnan(pval) else None,
        "edge": exp_neto > 0 and (pval is not None and pval < 0.05),
    }


def escanear(tickers, interval="15m", or_min=30, coste_bps=6.0, estrategia="orb"):
    """Backtest con costes sobre varios tickers → tabla rankeada por expectancy NETA."""
    filas = []
    for tk in tickers:
        tk = tk.strip().upper()
        if not tk:
            continue
        try:
            df = descargar(tk, interval)
            bt = backtest_estrategia(df, estrategia, or_min, interval, coste_bps)
            mt = metricas_backtest(bt, coste_bps)
            if mt["n"] == 0:
                continue
            filas.append({
                "Ticker": tk, "Ops": mt["n"], "Win %": mt["win_rate"],
                "Exp bruta %": mt["exp_bruto_pct"], "Exp neta %": mt["exp_neto_pct"],
                "Total neto %": mt["total_neto_pct"], "p": mt["pval"],
                "Edge neto": "SÍ ✅" if mt["edge"] else "no",
            })
        except Exception:
            continue
    df = pd.DataFrame(filas)
    if not df.empty:
        df = df.sort_values("Exp neta %", ascending=False).reset_index(drop=True)
    return df


def _plot_snapshot(hoy, ticker, interval, sesion):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(hoy.index, hoy["Close"], color="black", lw=1.1, label="Precio")
    ax.plot(hoy.index, hoy["VWAP"], color="tab:blue", lw=1.3, label="VWAP")
    ax.axhline(float(hoy["OR_high"].iloc[-1]), color="tab:green", ls="--", lw=1, label="Rango apertura")
    ax.axhline(float(hoy["OR_low"].iloc[-1]), color="tab:red", ls="--", lw=1)
    ax.set_title(f"{ticker.upper()} · intradía {interval} · sesión {sesion}")
    ax.set_xlabel("Hora"); ax.set_ylabel("Precio"); ax.legend(loc="best")
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description="Análisis y backtest intradía con costes.")
    ap.add_argument("tickers", nargs="*", default=["AAPL"], help="Uno = detalle; varios = escaneo rankeado.")
    ap.add_argument("--interval", default="5m", choices=list(_MIN))
    ap.add_argument("--or-min", type=int, default=30, help="Minutos del rango de apertura.")
    ap.add_argument("--coste-bps", type=float, default=6.0, help="Coste ida+vuelta (comisión+spread+slippage) en bps.")
    a = ap.parse_args()
    tickers = a.tickers or ["AAPL"]

    # varios tickers → escaneo rankeado por expectancy neta
    if len(tickers) > 1:
        print(f"\nEscaneando {len(tickers)} tickers · ORB {a.or_min}min · {a.interval} · coste {a.coste_bps} bps...")
        tabla = escanear(tickers, a.interval, a.or_min, a.coste_bps)
        if tabla.empty:
            print("Ningún ticker con operaciones (mercado cerrado, histórico corto o sin roturas).")
            return
        print("\n" + tabla.to_string(index=False))
        ganan = tabla[tabla["Edge neto"].str.startswith("SÍ")]["Ticker"].tolist()
        print(f"\n> Rankeado por expectancy NETA (tras costes). Con edge significativo: "
              f"{', '.join(ganan) if ganan else 'ninguno'}.")
        print("> Si ninguno supera el azar tras costes, es la realidad del intradía líquido. No es recomendación.\n")
        return

    a.ticker = tickers[0]
    print(f"\nDescargando {a.ticker.upper()} intradía {a.interval}...")
    df = descargar(a.ticker, a.interval)
    ind = indicadores_intradia(df, a.or_min, a.interval)
    print(f"{len(df)} barras · {df['sesion'].nunique()} sesiones.\n")

    bt = backtest_orb(df, a.or_min, a.interval, a.coste_bps)
    mt = metricas_backtest(bt, a.coste_bps)
    print(f"=== Backtest Opening Range Breakout ({a.or_min}min) · coste {a.coste_bps} bps ===")
    if mt["n"] == 0:
        print("  " + mt["mensaje"]); return
    print(f"  Operaciones      : {mt['n']}  (1 por sesión con rotura)")
    print(f"  Win rate         : {mt['win_rate']}%")
    print(f"  Expectancy BRUTA : {mt['exp_bruto_pct']:+.3f}% por operación")
    print(f"  Coste por op.    : -{mt['coste_pct']:.3f}%")
    print(f"  Expectancy NETA  : {mt['exp_neto_pct']:+.3f}% por operación  ← la que importa")
    print(f"  Total neto       : {mt['total_neto_pct']:+.2f}%  ·  p={mt['pval']}")
    if mt["edge"]:
        print("  → Edge NETO positivo y significativo. Candidato a paper trading (Alpaca).")
    else:
        print("  → Tras costes NO hay edge significativo. No operes real con esto.")
    print("\n> yfinance intradía: retraso ~15 min, histórico corto. Para desarrollo, no ejecución.")
    print("> El coste se come el edge: por eso el backtest intradía SIN costes miente. No es recomendación.\n")


if __name__ == "__main__":
    main()
