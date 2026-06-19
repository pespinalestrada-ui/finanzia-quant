"""
dashboard — UI Gradio única que agrupa toda la suite cuant_trading + el
forecast del proyecto. 7 pestañas:

  1. Forecast 30/90/120  (motor del proyecto, Prophet + confianza)
  2. Indicadores         (RSI/MACD/Bollinger/ATR/SMA)
  3. Screener            (ranking de watchlist)
  4. Señales             (signal_scanner)
  5. Backtest            (SMA/RSI/Bollinger vs buy&hold)
  6. Correlación         (matriz + heatmap)
  7. Cartera             (Markowitz máx Sharpe + frontera)

Lanzar:
    cd cuant_trading/dashboard
    python dashboard.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
SUITE = HERE.parent                       # cuant_trading/
PROJ = SUITE.parent                       # raíz del proyecto
for p in [PROJ, PROJ / "app", SUITE / "indicators", SUITE / "screener",
          SUITE / "backtester", SUITE / "correlation", SUITE / "portfolio_optimizer",
          SUITE / "signal_scanner", SUITE / "sentiment", SUITE / "position_sizer",
          SUITE / "journal", SUITE / "autogluon_forecast", SUITE / "market_context",
          SUITE / "lstm_forecast", SUITE / "neuralprophet_forecast"]:
    sys.path.insert(0, str(p))

import yfinance as yf
import forecast_tool                       # app/forecast_tool.py
import indicators as IND
import screener as SCR
import backtester as BT
import signal_scanner as SS
import portfolio_optimizer as PO
import sentiment_news as SN                # FinBERT (modelos se cargan en el 1er uso)
import position_sizer as PS
import journal as JR
import market_context as MC


def _dl(ticker, period="1y"):
    h = yf.Ticker(ticker).history(period=period, auto_adjust=False)
    if h.empty:
        raise ValueError(f"Ticker '{ticker}' sin datos.")
    h = h.reset_index()
    h["Date"] = pd.to_datetime(h["Date"]).dt.tz_localize(None)
    return h


def _err_fig(msg):
    f, ax = plt.subplots(figsize=(8, 2)); ax.text(0.5, 0.5, msg, ha="center", va="center", wrap=True)
    ax.axis("off"); return f


# ---- 1. Forecast ----------------------------------------------------------
def tab_forecast(ticker, period, motor="Prophet (rápido)"):
    try:
        if str(motor).startswith("AutoGluon"):
            import autogluon_forecast as AG   # import perezoso: AutoGluon pesa
            ticker = ticker.strip().upper()
            df = AG.descargar(ticker, period)
            preds, lb = AG.entrenar_y_predecir(df, horizon=120, preset="fast_training", time_limit=120)
            tabla, px = AG.resumen(preds, df, horizontes=(30, 90, 120))
            fig = AG.plot(df, preds, ticker)
            top = ""
            if lb is not None and len(lb):
                mejores = lb.sort_values("score_val", ascending=False).head(3)
                top = "\n\n**Top modelos (MASE):** " + " · ".join(
                    f"{r['model']} ({r['score_val']:.2f})" for _, r in mejores.iterrows())
            md = (f"### {ticker} — AutoGluon TimeSeries (cuantiles P10-P90)\n\n"
                  f"Cierre actual {px:.3f} · {len(df)} sesiones · preset fast_training (120 s máx)."
                  f"{top}\n\n> Confianza por ancho de banda: <20% ALTA · 20-45% MEDIA · >45% BAJA. "
                  f"No es recomendación de inversión.")
            return fig, tabla, md
        if str(motor).startswith("LSTM"):
            import lstm_forecast as LF          # red neuronal, torch ya cargado
            fig, tabla, meta = LF.forecast(ticker.strip().upper(), period, horizon=120)
            md = (f"### {ticker.upper()} — Red neuronal LSTM (PyTorch)\n\n"
                  f"Cierre {meta['precio_actual']:.3f} · {meta['n']} sesiones · error residual ~{meta['resid_pct']}%. "
                  f"Red entrenada desde cero en cada consulta.\n\n"
                  f"> Las redes recurrentes sobre precio tienden a 'seguir' el último valor; bandas "
                  f"amplias = incertidumbre real. No es recomendación de inversión.")
            return fig, tabla, md
        if str(motor).startswith("NeuralProphet"):
            import neuralprophet_forecast as NPF
            fig, tabla, meta = NPF.forecast(ticker.strip().upper(), period, horizon=120, epochs=60)
            md = (f"### {ticker.upper()} — NeuralProphet (AR-Net + estacionalidad)\n\n"
                  f"Cierre {meta['precio_actual']:.3f} · {meta['n']} sesiones. Sucesor de Prophet sobre "
                  f"PyTorch: interpretable + memoria autorregresiva.\n\n"
                  f"> Confianza por ancho de banda 80%. No es recomendación de inversión.")
            return fig, tabla, md
        fig, tabla, informe, meta = forecast_tool.forecast(ticker, period=period)
        head = f"**{ticker.upper()}** · {meta['n_sesiones']} sesiones · cierre {meta['precio_actual']:.3f}"
        return fig, tabla, head + "\n\n" + informe
    except Exception as e:
        return _err_fig(f"Error: {e}"), pd.DataFrame(), f"**Error:** {e}"


# ---- 2. Indicadores -------------------------------------------------------
def tab_indicadores(ticker, period):
    try:
        df = _dl(ticker, period)
        df = IND.calcular_todos(df)             # 11 indicadores
        tabla = pd.DataFrame(IND.señales_dict(df), columns=["Indicador", "Valor", "Señal"])
        d = df.iloc[-260:]
        fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios":[3,1,1]})
        ax[0].plot(d["Date"], d["Close"], "k", lw=1); ax[0].plot(d["Date"], d["BB_up"], "b--", lw=.6, alpha=.6)
        ax[0].plot(d["Date"], d["BB_lo"], "b--", lw=.6, alpha=.6); ax[0].plot(d["Date"], d["SMA50"], "tab:orange", lw=.8)
        ax[0].plot(d["Date"], d["SMA200"], "tab:red", lw=.8); ax[0].set_title(f"{ticker.upper()} precio+Bollinger+SMA")
        ax[1].plot(d["Date"], d["RSI"], "tab:purple", lw=1); ax[1].axhline(70,color="r",ls="--",lw=.5); ax[1].axhline(30,color="g",ls="--",lw=.5); ax[1].set_ylabel("RSI")
        ax[2].bar(d["Date"], d["MACD_hist"], color="gray", width=1); ax[2].plot(d["Date"], d["MACD"], "b", lw=.7); ax[2].plot(d["Date"], d["MACD_sig"], "r", lw=.7); ax[2].set_ylabel("MACD")
        fig.tight_layout()
        return fig, tabla
    except Exception as e:
        return _err_fig(f"Error: {e}"), pd.DataFrame()


# ---- 3. Screener ----------------------------------------------------------
def _parse(txt):
    return [t.strip() for t in txt.replace(",", " ").split() if t.strip()]

def tab_screener(txt):
    try:
        rows = [SCR.analizar(t) for t in _parse(txt)]
        rows = [r for r in rows if r]
        df = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
        return df
    except Exception as e:
        return pd.DataFrame([{"Error": str(e)}])


# ---- 4. Señales -----------------------------------------------------------
def tab_signals(txt):
    try:
        return SS.scan(_parse(txt))
    except Exception as e:
        return pd.DataFrame([{"Error": str(e)}])


# ---- 5. Backtest ----------------------------------------------------------
def tab_backtest(ticker, strat, fast, slow, period):
    try:
        h = _dl(ticker, period)
        pos = BT.señales(h, strat, int(fast), int(slow))
        rm = h["Close"].pct_change().fillna(0)
        rs = pos * rm - pos.diff().abs().fillna(0) * 0.001
        es = (1 + rs).cumprod(); eb = (1 + rm).cumprod()
        m = BT.metricas(es, rs, pos); bh = BT.metricas(eb, rm, pd.Series(1.0, index=h.index))
        txt = (f"### {ticker.upper()} · {strat.upper()} · {period}\n\n"
               f"| Métrica | Estrategia | Buy&Hold |\n|---|---|---|\n"
               f"| Retorno | {m['ret_total']*100:.1f}% | {bh['ret_total']*100:.1f}% |\n"
               f"| CAGR | {m['cagr']*100:.1f}% | {bh['cagr']*100:.1f}% |\n"
               f"| Sharpe | {m['sharpe']:.2f} | {bh['sharpe']:.2f} |\n"
               f"| Max DD | {m['max_dd']*100:.1f}% | {bh['max_dd']*100:.1f}% |\n"
               f"| Win rate | {m['win_rate']:.1f}% | — |\n"
               f"| Operaciones | {m['trades']} | 1 |\n\n"
               f"**{'BATE' if m['ret_total']>bh['ret_total'] else 'NO bate'} a buy&hold.**")
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(h["Date"], es, lw=1.4, label=f"{strat.upper()} (x{es.iloc[-1]:.2f})")
        ax.plot(h["Date"], eb, lw=1.1, alpha=.8, label=f"Buy&Hold (x{eb.iloc[-1]:.2f})")
        ax.legend(); ax.set_title(f"{ticker.upper()} equity (1€)"); fig.tight_layout()
        return fig, txt
    except Exception as e:
        return _err_fig(f"Error: {e}"), f"**Error:** {e}"


# ---- 6. Correlación -------------------------------------------------------
def tab_corr(txt, period):
    try:
        tickers = _parse(txt)
        data = {}
        for t in tickers:
            h = yf.Ticker(t).history(period=period, auto_adjust=True)
            if not h.empty:
                s = h["Close"]; s.index = pd.to_datetime(s.index).tz_localize(None); data[t.upper()] = s
        px = pd.DataFrame(data).dropna()
        corr = px.pct_change().dropna().corr()
        fig, ax = plt.subplots(figsize=(1.1*len(corr)+2, 1.0*len(corr)+1.5))
        im = ax.imshow(corr, cmap="RdYlGn_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index)
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(corr.values[i,j])>0.6 else "black")
        plt.colorbar(im); ax.set_title(f"Correlación ({period})"); fig.tight_layout()
        return fig, corr.round(2).reset_index()
    except Exception as e:
        return _err_fig(f"Error: {e}"), pd.DataFrame()


# ---- 7. Cartera (Markowitz) ----------------------------------------------
def tab_cartera(txt, period, rf):
    try:
        from scipy.optimize import minimize
        px = PO.precios(_parse(txt), period)
        rets = px.pct_change().dropna(); mu = rets.mean()*252; cov = rets.cov()*252
        n = len(mu); names = list(mu.index)
        perf = lambda w: (w@mu, np.sqrt(w@cov@w))
        negsh = lambda w: -((w@mu - rf)/np.sqrt(w@cov@w))
        cons = ({"type":"eq","fun":lambda w: w.sum()-1},); bnds = tuple((0,1) for _ in range(n))
        w0 = np.repeat(1/n, n)
        wsh = minimize(negsh, w0, method="SLSQP", bounds=bnds, constraints=cons).x
        wmv = minimize(lambda w: perf(w)[1], w0, method="SLSQP", bounds=bnds, constraints=cons).x
        def blk(nm, w):
            r, v = perf(w); s = (r-rf)/v
            ps = " · ".join(f"{n_}:{wi*100:.0f}%" for n_, wi in sorted(zip(names,w), key=lambda x:-x[1]) if wi>0.005)
            return f"**{nm}** — ret {r*100:.1f}% · vol {v*100:.1f}% · Sharpe {s:.2f}\n\n{ps}"
        txt_out = blk("Máximo Sharpe", wsh) + "\n\n" + blk("Mínima volatilidad", wmv)
        rng = np.random.default_rng(42); N=3000
        W = rng.random((N,n)); W/=W.sum(axis=1,keepdims=True)
        R = W@mu.values; V = np.sqrt((W@cov.values*W).sum(axis=1)); S=(R-rf)/V
        fig, ax = plt.subplots(figsize=(10,6))
        sc = ax.scatter(V*100, R*100, c=S, cmap="viridis", s=7, alpha=.5); plt.colorbar(sc, label="Sharpe")
        rs,vs = perf(wsh); rm,vm = perf(wmv)
        ax.scatter(vs*100, rs*100, marker="*", s=280, color="red", label="Máx Sharpe", zorder=5)
        ax.scatter(vm*100, rm*100, marker="*", s=280, color="blue", label="Mín vol", zorder=5)
        ax.set_xlabel("Volatilidad anual %"); ax.set_ylabel("Retorno anual %"); ax.set_title("Frontera eficiente"); ax.legend()
        fig.tight_layout()
        return fig, txt_out
    except Exception as e:
        return _err_fig(f"Error: {e}"), f"**Error:** {e}"


# ---- 8. Sentimiento (FinBERT) ---------------------------------------------
def tab_sentiment(ticker, max_news):
    try:
        ticker = ticker.strip().upper()
        noticias = SN.extraer_noticias(ticker, int(max_news))
        if not noticias:
            return pd.DataFrame(), f"**{ticker}:** sin noticias disponibles ahora mismo en Yahoo Finance."
        df = SN.analizar(noticias)          # 1ª llamada carga FinBERT+NER (~1 min)
        s, ver = SN.score_global(df)
        pos = (df["Sentimiento"] == "positive").sum()
        neg = (df["Sentimiento"] == "negative").sum()
        neu = (df["Sentimiento"] == "neutral").sum()

        # señal técnica del mismo ticker para veredicto combinado
        tech_txt = ""
        try:
            tech = SS.señales_ticker(ticker)
            if tech:
                combo = {
                    ("POSITIVO", "BUY"):  "✅ Noticias y técnico ALINEADOS al alza.",
                    ("NEGATIVO", "SELL"): "⛔ Noticias y técnico ALINEADOS a la baja.",
                    ("POSITIVO", "SELL"): "⚠️ DIVERGENCIA: noticias positivas pero técnico vendedor.",
                    ("NEGATIVO", "BUY"):  "⚠️ DIVERGENCIA: técnico comprador pero noticias negativas.",
                }.get((ver, tech["Sesgo"]), "Sin lectura combinada clara (algún lado neutral).")
                tech_txt = (f"\n\n**Señal técnica (signal_scanner):** {tech['Sesgo']} "
                            f"({tech['Señales']})\n\n**Lectura combinada:** {combo}")
        except Exception:
            pass

        md = (f"### {ticker} — Sentimiento de noticias: **{ver}** (score {s:+.3f})\n\n"
              f"{pos} positivas · {neg} negativas · {neu} neutrales (FinBERT, {len(df)} titulares)"
              f"{tech_txt}\n\n> Sentimiento ≠ recomendación. Señal complementaria.")
        return df, md
    except Exception as e:
        return pd.DataFrame(), f"**Error:** {e}"


# ---- 9. Position sizer ------------------------------------------------------
def tab_sizer(ticker, capital, riskpct, entry, stop, atr_mult):
    try:
        capital = float(capital); riskpct = float(riskpct)
        entry = float(entry or 0); stop = float(stop or 0); atr_mult = float(atr_mult)
        nota = ""
        if ticker and ticker.strip() and (entry <= 0 or stop <= 0):
            atr, px = PS.atr_actual(ticker.strip().upper())
            if entry <= 0: entry = px
            if stop <= 0: stop = entry - atr_mult * atr
            nota = f"*{ticker.strip().upper()}: precio {px:.3f} · ATR(14) {atr:.3f} · stop {atr_mult}×ATR.*\n\n"
        if entry <= 0 or stop <= 0:
            return "**Faltan datos:** da Entrada y Stop, o un Ticker para calcular stop por ATR."
        if stop >= entry:
            return "**Error:** el stop debe estar por debajo de la entrada (posición larga)."
        riesgo_eur = capital * riskpct / 100
        r_accion = entry - stop
        shares = int(riesgo_eur // r_accion)
        if shares == 0:
            return f"**Riesgo insuficiente:** {riesgo_eur:.2f} € no cubre ni 1 acción (riesgo/acción {r_accion:.2f} €)."
        coste = shares * entry
        md = (nota +
              f"### Resultado\n\n"
              f"| Concepto | Valor |\n|---|---|\n"
              f"| Acciones a comprar | **{shares}** |\n"
              f"| Coste posición | {coste:,.2f} € ({coste/capital*100:.1f}% del capital) |\n"
              f"| Riesgo real | {shares*r_accion:,.2f} € ({riskpct:.2f}% objetivo) |\n"
              f"| Entrada / Stop | {entry:.3f} / {stop:.3f} |\n\n"
              f"**Objetivos:** " +
              " · ".join(f"{r}R={entry + r*r_accion:.3f}" for r in (1, 2, 3)) +
              f"\n\n> Regla: arriesga 0.5–2 % por operación. 1R = pérdida si salta el stop.")
        return md
    except Exception as e:
        return f"**Error:** {e}"


# ---- 10. Veredicto global (agregador) --------------------------------------
def tab_veredicto(ticker, period, con_sentimiento):
    """
    Corre todos los motores sobre un ticker y agrega en un veredicto
    COMPRAR / MANTENER / VENDER. Cada pilar puntúa en [-1, +1] y pondera.
    """
    try:
        ticker = ticker.strip().upper()
        pilares = []  # (nombre, lectura, score, peso)

        # --- 1. Forecast Prophet 90d ponderado por su confianza (peso 0.30)
        fig, tabla_fc, _informe, meta = forecast_tool.forecast(ticker, period=period)
        f90 = tabla_fc.iloc[1]
        var90 = float(f90["Variación %"])
        conf_str = str(f90["Confianza"])                      # "ALTA (76)"
        try:
            conf = int(conf_str.split("(")[1].rstrip(")"))
        except Exception:
            conf = 50
        s_fc = max(-1.0, min(1.0, var90 / 10.0)) * conf / 100.0
        pilares.append(("Forecast 90d (Prophet)", f"{var90:+.1f} % · confianza {conf_str}", s_fc, 0.30))

        # --- 2-4. Técnicos sobre 1 año
        dfh = _dl(ticker, "1y")
        c = dfh["Close"].dropna(); px = float(c.iloc[-1])
        sma50 = float(c.rolling(50).mean().iloc[-1])
        sma200 = float(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else float("nan")
        s_tend = (0.6 if (not np.isnan(sma200) and sma50 > sma200) else -0.6 if not np.isnan(sma200) else 0.0)
        s_tend += 0.4 if px > sma50 else -0.4
        lect_tend = ("ALCISTA" if s_tend > 0 else "BAJISTA" if s_tend < 0 else "MIXTA")
        pilares.append(("Tendencia (SMA50/200 + precio)", lect_tend, s_tend, 0.20))

        rsi = float(IND.rsi(c).iloc[-1])
        s_rsi = 0.6 if rsi < 30 else (-0.6 if rsi > 70 else 0.0)
        pilares.append(("RSI(14)", f"{rsi:.0f} ({'sobreventa' if rsi<30 else 'sobrecompra' if rsi>70 else 'neutral'})", s_rsi, 0.10))

        macd_l, macd_s, _h = IND.macd(c)
        s_macd = 0.5 if float(macd_l.iloc[-1]) > float(macd_s.iloc[-1]) else -0.5
        pilares.append(("MACD", "alcista" if s_macd > 0 else "bajista", s_macd, 0.10))

        # --- 5. Momentum 3m
        mom3 = (px / float(c.iloc[-63]) - 1) * 100 if len(c) > 63 else 0.0
        if np.isnan(mom3):
            mom3 = 0.0
        s_mom = max(-1.0, min(1.0, mom3 / 15.0))
        pilares.append(("Momentum 3 meses", f"{mom3:+.1f} %", s_mom, 0.15))

        # --- 6. Señales del scanner
        tech = SS.señales_ticker(ticker)
        if tech:
            s_sig = max(-1.0, min(1.0, tech["Fuerza"] / 2.0))
            pilares.append(("Señales técnicas (scanner)", tech["Señales"], s_sig, 0.15))

        # --- 7. Sentimiento FinBERT (opcional, lento la 1ª vez)
        if con_sentimiento:
            noticias = SN.extraer_noticias(ticker, 10)
            if noticias:
                dfn = SN.analizar(noticias)
                s_sent, ver_sent = SN.score_global(dfn)
                pilares.append(("Sentimiento noticias (FinBERT)", f"{ver_sent} ({len(dfn)} titulares)", s_sent, 0.15))

        # --- agregación ponderada
        wsum = sum(p[3] for p in pilares)
        total = sum(p[2] * p[3] for p in pilares) / wsum
        if total >= 0.35:
            verd, emoji = "COMPRAR", "🟢"
        elif total <= -0.35:
            verd, emoji = "VENDER", "🔴"
        else:
            verd, emoji = "MANTENER", "🟡"

        tabla = pd.DataFrame(
            [{"Pilar": n, "Lectura": l, "Score": round(s, 2),
              "Peso": f"{w/wsum*100:.0f}%", "Aporte": round(s * w / wsum, 3)}
             for n, l, s, w in pilares]
        )

        md = (f"# {emoji} {verd}\n\n"
              f"**{ticker}** · precio {px:.3f} · score total **{total:+.3f}** "
              f"(umbral: ≥+0.35 comprar · ≤−0.35 vender)\n\n"
              f"Pilares evaluados: {len(pilares)}"
              + ("" if con_sentimiento else " · *(sentimiento desactivado — actívalo para incluir noticias)*")
              + "\n\n> ⚠️ Estimación estadística automática agregando forecast, técnico"
              + (", momentum y sentimiento" if con_sentimiento else " y momentum")
              + ". **NO es recomendación de inversión.** Úsala como resumen, no como orden.")
        return fig, tabla, md
    except Exception as e:
        return _err_fig(f"Error: {e}"), pd.DataFrame(), f"**Error:** {e}"


# ---- 11. Diario de operaciones (paper trading) ------------------------------
def tab_journal_abrir(ticker, entrada, stop, acciones, nota, es_short):
    try:
        nid = JR.abrir(ticker, float(entrada), float(stop), int(acciones),
                       nota or "", "SHORT" if es_short else "LONG")
        return f"✅ Operación **#{nid}** abierta ({ticker.upper()}).", JR.lista(), JR.stats_texto()
    except Exception as e:
        return f"**Error:** {e}", JR.lista(), JR.stats_texto()


def tab_journal_cerrar(op_id, salida):
    try:
        pnl, r = JR.cerrar(int(op_id), float(salida))
        emoji = "🟢" if pnl > 0 else "🔴"
        return f"{emoji} Operación **#{int(op_id)}** cerrada: P&L **{pnl:+.2f} €** ({r:+.2f}R).", JR.lista(), JR.stats_texto()
    except Exception as e:
        return f"**Error:** {e}", JR.lista(), JR.stats_texto()


def tab_journal_refrescar():
    return JR.lista(), JR.stats_texto()


# ---- 12. Termómetro de mercado ----------------------------------------------
def tab_mercado(ticker):
    try:
        fg = MC.fear_greed_bolsa()
        cr = MC.fear_greed_cripto()
        vx = MC.vix_regimen()
        lineas = ["## 🌡️ Termómetro del mercado\n"]
        if fg and fg.get("score") is not None:
            lineas.append(f"### Fear & Greed BOLSA: **{fg['score']}/100 — {MC.etiqueta_fg(fg['score'])}**")
        else:
            lineas.append("### Fear & Greed bolsa: no disponible ahora")
        if cr:
            lineas.append(f"### Fear & Greed CRIPTO: **{cr['score']}/100 — {cr['etiqueta']}**")
        if vx:
            lineas.append(f"### VIX: **{vx['vix']}** — {vx['regimen']}")
        lineas.append(f"\n**Lectura conjunta:** {MC.lectura_conjunta(fg, vx)}")
        md = "\n\n".join(lineas)

        comp = pd.DataFrame(fg["componentes"]).rename(columns={
            "name": "Componente", "val": "Valor", "wt": "Peso %", "desc": "Qué mide", "raw": "Dato"
        }) if fg and fg.get("componentes") else pd.DataFrame()

        fund = pd.DataFrame()
        if ticker and ticker.strip():
            f = MC.fundamentales(ticker.strip().upper())
            fund = pd.DataFrame(list(f.items()), columns=["Métrica", "Valor"])
        return md, comp, fund
    except Exception as e:
        return f"**Error:** {e}", pd.DataFrame(), pd.DataFrame()


# ---- UI -------------------------------------------------------------------
def build():
    import gradio as gr
    # head: bloquea el traductor automático de Chrome (rompe la reactividad de Gradio)
    _head = '<meta name="google" content="notranslate"><script>document.documentElement.lang="es";</script>'
    with gr.Blocks(title="FinanzIA — Mesa cuantitativa", head=_head) as app:
        gr.Markdown("# FinanzIA — Mesa cuantitativa")
        gr.Markdown("Suite de trading algorítmico. Datos Yahoo Finance (retardo ~15 min). "
                    "Análisis y educación — **no es recomendación de inversión**.")
        with gr.Tab("★ Veredicto"):
            gr.Markdown("**Análisis completo en un clic**: forecast + tendencia + RSI + MACD + "
                        "momentum + señales (+ sentimiento opcional) → estimación "
                        "**COMPRAR / MANTENER / VENDER** con desglose por pilar. Tarda ~30-60 s "
                        "(forecast Prophet incluido).")
            with gr.Row():
                tv = gr.Textbox(value="AAPL", label="Ticker", scale=3)
                pv = gr.Dropdown(["2y", "3y", "5y"], value="3y", label="Histórico")
                sv = gr.Checkbox(value=False, label="Incluir sentimiento (1ª vez +1 min)")
                bv = gr.Button("Analizar TODO", variant="primary")
            mdv = gr.Markdown()
            tbv = gr.Dataframe(label="Desglose por pilar", wrap=True)
            plv = gr.Plot(label="Forecast 30/90/120d")
            bv.click(tab_veredicto, [tv, pv, sv], [plv, tbv, mdv])
        with gr.Tab("1 · Forecast"):
            with gr.Row():
                t = gr.Textbox(value="SAB.MC", label="Ticker", scale=3)
                p = gr.Dropdown(["2y","3y","5y","10y"], value="3y", label="Histórico")
                motor = gr.Dropdown(["Prophet (rápido)", "LSTM (red neuronal)",
                                     "NeuralProphet (AR-Net)", "AutoGluon (2 min, cuantiles)"],
                                    value="Prophet (rápido)", label="Motor")
                b = gr.Button("Forecast", variant="primary")
            pl = gr.Plot(); tb = gr.Dataframe(label="30/90/120 días"); md = gr.Markdown()
            b.click(tab_forecast, [t, p, motor], [pl, tb, md])
        with gr.Tab("2 · Indicadores"):
            with gr.Row():
                t2 = gr.Textbox(value="AAPL", label="Ticker", scale=3)
                p2 = gr.Dropdown(["6mo","1y","2y"], value="1y", label="Histórico")
                b2 = gr.Button("Calcular", variant="primary")
            pl2 = gr.Plot(); tb2 = gr.Dataframe(label="Señales")
            b2.click(tab_indicadores, [t2, p2], [pl2, tb2])
        with gr.Tab("3 · Screener"):
            t3 = gr.Textbox(value="AAPL MSFT NVDA GOOGL SAB.MC BBVA.MC", label="Tickers (espacio/coma)")
            b3 = gr.Button("Escanear", variant="primary"); tb3 = gr.Dataframe(label="Ranking")
            b3.click(tab_screener, t3, tb3)
        with gr.Tab("4 · Señales"):
            t4 = gr.Textbox(value="AAPL MSFT NVDA GOOGL SAB.MC BBVA.MC", label="Tickers")
            b4 = gr.Button("Buscar señales", variant="primary"); tb4 = gr.Dataframe(label="Señales accionables")
            b4.click(tab_signals, t4, tb4)
        with gr.Tab("5 · Backtest"):
            with gr.Row():
                t5 = gr.Textbox(value="AAPL", label="Ticker", scale=2)
                st = gr.Dropdown(["sma","rsi","bb"], value="sma", label="Estrategia")
                fa = gr.Number(value=50, label="SMA rápida"); sl = gr.Number(value=200, label="SMA lenta")
                p5 = gr.Dropdown(["2y","5y","10y"], value="5y", label="Histórico")
                b5 = gr.Button("Backtest", variant="primary")
            pl5 = gr.Plot(); md5 = gr.Markdown()
            b5.click(tab_backtest, [t5, st, fa, sl, p5], [pl5, md5])
        with gr.Tab("6 · Correlación"):
            with gr.Row():
                t6 = gr.Textbox(value="AAPL MSFT TLT GLD", label="Tickers", scale=3)
                p6 = gr.Dropdown(["1y","2y","3y"], value="2y", label="Histórico")
                b6 = gr.Button("Correlación", variant="primary")
            pl6 = gr.Plot(); tb6 = gr.Dataframe(label="Matriz")
            b6.click(tab_corr, [t6, p6], [pl6, tb6])
        with gr.Tab("7 · Cartera"):
            with gr.Row():
                t7 = gr.Textbox(value="AAPL MSFT NVDA GOOGL", label="Tickers", scale=3)
                p7 = gr.Dropdown(["2y","3y","5y"], value="3y", label="Histórico")
                rf = gr.Number(value=0.0, label="Tasa libre riesgo (0.03=3%)")
                b7 = gr.Button("Optimizar", variant="primary")
            pl7 = gr.Plot(); md7 = gr.Markdown()
            b7.click(tab_cartera, [t7, p7, rf], [pl7, md7])
        with gr.Tab("8 · Sentimiento"):
            gr.Markdown("Noticias del ticker analizadas con **FinBERT** + entidades. "
                        "⏳ La **primera** consulta carga los modelos (~1 min); las siguientes van rápido.")
            with gr.Row():
                t8 = gr.Textbox(value="AAPL", label="Ticker", scale=3)
                n8 = gr.Slider(3, 20, value=10, step=1, label="Nº noticias")
                b8 = gr.Button("Analizar noticias", variant="primary")
            md8 = gr.Markdown(); tb8 = gr.Dataframe(label="Titulares analizados", wrap=True)
            b8.click(tab_sentiment, [t8, n8], [tb8, md8])
        with gr.Tab("9 · Tamaño posición"):
            gr.Markdown("Cuántas acciones comprar arriesgando un % fijo. Da Entrada+Stop, "
                        "o solo Ticker para stop automático por ATR.")
            with gr.Row():
                t9 = gr.Textbox(value="", label="Ticker (opcional)", scale=2)
                c9 = gr.Number(value=10000, label="Capital €")
                r9 = gr.Number(value=1.0, label="Riesgo %")
            with gr.Row():
                e9 = gr.Number(value=0, label="Entrada (0=auto)")
                s9 = gr.Number(value=0, label="Stop (0=auto)")
                a9 = gr.Number(value=2.0, label="Múltiplo ATR")
                b9 = gr.Button("Calcular", variant="primary")
            md9 = gr.Markdown()
            b9.click(tab_sizer, [t9, c9, r9, e9, s9, a9], [md9])
        with gr.Tab("📒 Diario"):
            gr.Markdown("**Paper trading** — registra operaciones SIMULADAS y mide si tu método "
                        "tiene ventaja real antes de arriesgar dinero. Con 20-30 cerradas, el "
                        "win rate y payoff de aquí alimentan el Kelly del Tamaño posición.")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Abrir operación")
                    jt = gr.Textbox(label="Ticker", value="")
                    with gr.Row():
                        je = gr.Number(label="Entrada"); js = gr.Number(label="Stop")
                        ja = gr.Number(label="Acciones", value=10, precision=0)
                    jn = gr.Textbox(label="Nota (ej. veredicto del sistema)", value="")
                    jshort = gr.Checkbox(label="Es venta en corto (SHORT)", value=False)
                    jb1 = gr.Button("Abrir (simulada)", variant="primary")
                with gr.Column():
                    gr.Markdown("#### Cerrar operación")
                    jid = gr.Number(label="ID de la operación", precision=0)
                    jx = gr.Number(label="Precio de salida")
                    jb2 = gr.Button("Cerrar", variant="primary")
                    jb3 = gr.Button("🔄 Refrescar diario")
            jmsg = gr.Markdown()
            jstats = gr.Textbox(label="Estadísticas (incluye Kelly para position sizer)", lines=11)
            jtabla = gr.Dataframe(label="Operaciones", wrap=True)
            jb1.click(tab_journal_abrir, [jt, je, js, ja, jn, jshort], [jmsg, jtabla, jstats])
            jb2.click(tab_journal_cerrar, [jid, jx], [jmsg, jtabla, jstats])
            jb3.click(tab_journal_refrescar, [], [jtabla, jstats])
            app.load(tab_journal_refrescar, [], [jtabla, jstats])
        with gr.Tab("🌡️ Mercado"):
            gr.Markdown("Contexto ANTES de mirar tickers: **Fear & Greed** (bolsa y cripto), "
                        "**VIX** con régimen de volatilidad, y fundamentales del ticker que quieras.")
            with gr.Row():
                tm = gr.Textbox(value="SAB.MC", label="Ticker para fundamentales (opcional)", scale=3)
                bm = gr.Button("Medir mercado", variant="primary")
            mdm = gr.Markdown()
            with gr.Row():
                tbm1 = gr.Dataframe(label="Componentes Fear & Greed", wrap=True)
                tbm2 = gr.Dataframe(label="Fundamentales", wrap=True)
            bm.click(tab_mercado, [tm], [mdm, tbm1, tbm2])
    return app


if __name__ == "__main__":
    build().launch(server_name="127.0.0.1", server_port=7862, inbrowser=False)
