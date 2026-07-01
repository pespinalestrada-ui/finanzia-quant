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
          SUITE / "lstm_forecast", SUITE / "neuralprophet_forecast",
          SUITE / "alpha_forecast", SUITE / "conformal_forecast",
          SUITE / "risk_metrics", SUITE / "alerts", SUITE / "factor_scorer",
          SUITE / "intraday", SUITE / "alpaca_paper", SUITE / "veredicto_backtest",
          SUITE / "signal_engine", SUITE / "risk_manager", SUITE / "orchestrator",
          SUITE / "performance", SUITE / "pairs_trading", SUITE / "hrp_portfolio",
          SUITE / "evt_risk", SUITE / "montecarlo", SUITE / "system_backtest",
          SUITE / "hmm_regime", SUITE / "meta_labeling", SUITE / "rmt_clean",
          SUITE / "kalman_hedge", SUITE / "transfer_entropy", SUITE / "rebalance",
          SUITE / "informe"]:
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
import factor_scorer as FS               # modelo multi-factor (value/momentum/quality/low-vol)


import time as _time
_DL_CACHE = {}                                 # (ticker,period) -> (ts, df). TTL en proceso.
_DL_TTL = 1800                                 # 30 min: evita re-descargar el mismo ticker entre pestañas

def _dl(ticker, period="1y", adjust=False):
    key = (ticker.upper(), period, adjust)
    hit = _DL_CACHE.get(key)
    if hit and (_time.time() - hit[0]) < _DL_TTL:
        return hit[1].copy()
    h = yf.Ticker(ticker).history(period=period, auto_adjust=adjust)
    if h.empty:
        raise ValueError(f"Ticker '{ticker}' sin datos.")
    h = h.reset_index()
    h["Date"] = pd.to_datetime(h["Date"]).dt.tz_localize(None)
    _DL_CACHE[key] = (_time.time(), h.copy())
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
        h = _dl(ticker, period, adjust=True)   # total return: dividendos incluidos
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
            port_rets = rets @ w
            var_95 = np.percentile(port_rets, 5) * 100
            cvar_95 = port_rets[port_rets <= np.percentile(port_rets, 5)].mean() * 100
            ps = " · ".join(f"{n_}:{wi*100:.0f}%" for n_, wi in sorted(zip(names,w), key=lambda x:-x[1]) if wi>0.005)
            return f"**{nm}** — ret {r*100:.1f}% · vol {v*100:.1f}% · Sharpe {s:.2f} · **VaR 95%:** {var_95:+.2f}% · **CVaR:** {cvar_95:+.2f}%\n\n{ps}"
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
def _var90_de_tabla(tab):
    """Saca la 'Variación %' del horizonte 90d de la tabla de cualquier motor."""
    for _, r in tab.iterrows():
        if "90" in str(r.get("Horizonte", "")):
            return float(r["Variación %"])
    return float(tab.iloc[min(1, len(tab) - 1)]["Variación %"])


def _mape_de_tabla(tab):
    """Saca el MAPE del horizonte 90d de la tabla para ponderación OOS."""
    for _, r in tab.iterrows():
        if "90" in str(r.get("Horizonte", "")):
            val = str(r.get("MAPE backtest %", "10.0")).replace("%", "").strip()
            try:
                return float(val) if val != "n/d" else 10.0
            except ValueError:
                return 10.0
    return 10.0


def tab_veredicto(ticker, period, con_sentimiento, con_modelos=False, capital=10000, registrar=True, cripto=False):
    """
    Agrega forecast (consenso multi-modelo opcional) + batería técnica completa
    + volumen + sentimiento en un veredicto COMPRAR / MANTENER / VENDER.
    Cada pilar puntúa en [-1, +1] y pondera. Adapta los motores a lo instalado.

    cripto=True: el forecast usa frecuencia diaria 7d (forecast_tool detecta cripto),
    la ventana de momentum se mide en días naturales y se añade un pilar de
    Fear & Greed cripto (contrarian).
    """
    try:
        ticker = ticker.strip().upper()
        pilares = []  # (nombre, lectura, score, peso)
        notas_modelos = ""
        factor_score_val = None   # nota de factores, para el auto-log del diario

        # --- 1. Forecast: Prophet siempre; consenso multi-modelo si se pide -----
        fig, tabla_fc, _informe, meta = forecast_tool.forecast(ticker, period=period)
        prophet_var = _var90_de_tabla(tabla_fc)
        prophet_mape = _mape_de_tabla(tabla_fc)
        conf_str = str(tabla_fc.iloc[1]["Confianza"])         # "ALTA (76)"
        try:
            conf = int(conf_str.split("(")[1].rstrip(")"))
        except Exception:
            conf = 50
        modelos = [("Prophet", prophet_var, prophet_mape)]

        if con_modelos:
            # LSTM (torch ya cargado)
            try:
                import lstm_forecast as LF
                _f, lf_tab, _m = LF.forecast(ticker, period, horizon=120)
                modelos.append(("LSTM", _var90_de_tabla(lf_tab), _mape_de_tabla(lf_tab)))
            except Exception:
                pass
            # NeuralProphet (solo si instalado)
            import importlib.util as _ilu
            if _ilu.find_spec("neuralprophet") is not None:
                try:
                    import neuralprophet_forecast as NPF
                    _f, np_tab, _m = NPF.forecast(ticker, period, horizon=120, epochs=50)
                    modelos.append(("NeuralProphet", _var90_de_tabla(np_tab), _mape_de_tabla(np_tab)))
                except Exception:
                    pass
            # AutoGluon (solo si instalado)
            if _ilu.find_spec("autogluon") is not None:
                try:
                    import autogluon_forecast as AG
                    dfa = AG.descargar(ticker, period)
                    preds, _lb = AG.entrenar_y_predecir(dfa, horizon=120, preset="fast_training", time_limit=90)
                    ag_tab, _px = AG.resumen(preds, dfa, horizontes=(30, 90, 120))
                    modelos.append(("AutoGluon", _var90_de_tabla(ag_tab), _mape_de_tabla(ag_tab)))
                except Exception:
                    pass

        # Pilar Forecast: SIEMPRE Prophet (estable, modelo de cabecera). El multi-modelo
        # NO cambia el veredicto — solo informa de si los demás coinciden en dirección.
        # (Antes el ensemble 1/MAPE de 4 forecasts ruidosos a 90d movía el veredicto al
        #  azar alrededor del umbral: el forecast a 90d no tiene edge validado.)
        var_fc = prophet_var
        s_fc = max(-1.0, min(1.0, prophet_var / 10.0)) * conf / 100.0
        lect_fc = f"Prophet {prophet_var:+.1f}% · conf {conf_str}"
        if len(modelos) > 1:
            otros = [(n, v) for n, v, _ in modelos if n != "Prophet"]
            acuerdo = sum(1 for _, v in otros if (v > 0) == (prophet_var > 0))
            lect_fc += " · " + " ".join(f"{n[:4]} {v:+.0f}%" for n, v in otros)
            notas_modelos = (f"\n\n**Multi-modelo (informativo):** {acuerdo}/{len(otros)} modelos coinciden "
                             f"en dirección con Prophet. No altera el veredicto — el forecast a 90d no tiene "
                             f"edge validado, así que mandan técnica y factores. Sirve para ver consenso/disenso.")
        # peso del forecast PROPORCIONAL a su convicción: un forecast plano (~0%) casi no
        # participa (evita que dilúya el veredicto); uno fuerte (±5%+) pesa el máximo.
        w_fc = max(0.06, 0.30 * min(1.0, abs(var_fc) / 5.0))
        pilares.append(("Forecast 90d (Prophet)", lect_fc + f"  · peso {w_fc*100:.0f}%", s_fc, w_fc))

        # --- 2. Técnicos sobre 1 año (con OHLCV completo) -----------------------
        dfh = _dl(ticker, "1y")
        df = IND.calcular_todos(dfh)
        last = df.iloc[-1]
        c = df["Close"].dropna(); px = float(c.iloc[-1])

        sma50, sma200 = float(last["SMA50"]), float(last["SMA200"])
        s_tend = (0.6 if (not np.isnan(sma200) and sma50 > sma200) else -0.6 if not np.isnan(sma200) else 0.0)
        s_tend += 0.4 if px > sma50 else -0.4
        pilares.append(("Tendencia (SMA50/200 + precio)", "ALCISTA" if s_tend > 0 else "BAJISTA" if s_tend < 0 else "MIXTA", s_tend, 0.15))

        # ADX: fuerza de tendencia (gate). Solo cuenta si la tendencia es fuerte.
        adx_v = float(last["ADX"]); dir_adx = 1 if last["DI_POS"] > last["DI_NEG"] else -1
        s_adx = dir_adx * 0.7 if adx_v > 25 else 0.0
        pilares.append(("ADX (fuerza tendencia)", f"{adx_v:.0f} ({'FUERTE ' + ('alcista' if dir_adx>0 else 'bajista') if adx_v>25 else 'débil/lateral'})", s_adx, 0.08))

        # Hurst: detección de régimen
        hurst_v = last.get("HURST", float("nan"))

        # Consenso de osciladores: RSI + Estocástico + Williams%R + MFI + CCI
        votos = []
        rsi_v = float(last["RSI"]); votos.append(1 if rsi_v < 30 else -1 if rsi_v > 70 else 0)
        k_v = float(last["STOCH_K"]); votos.append(1 if k_v < 20 else -1 if k_v > 80 else 0)
        wr_v = float(last["WILLR"]); votos.append(1 if wr_v < -80 else -1 if wr_v > -20 else 0)
        mfi_v = float(last["MFI"]); votos.append(1 if mfi_v < 20 else -1 if mfi_v > 80 else 0)
        cci_v = float(last["CCI"]); votos.append(1 if cci_v < -100 else -1 if cci_v > 100 else 0)
        s_osc = float(np.mean(votos))
        n_sv = sum(1 for v in votos if v > 0); n_sc = sum(1 for v in votos if v < 0)
        pilares.append(("Osciladores (RSI/Estoc/W%R/MFI/CCI)", f"{n_sv} sobreventa · {n_sc} sobrecompra de 5", s_osc, 0.14))

        macd_l, macd_s = float(last["MACD"]), float(last["MACD_sig"])
        s_macd = 0.5 if macd_l > macd_s else -0.5
        pilares.append(("MACD", "alcista" if s_macd > 0 else "bajista", s_macd, 0.08))

        win_mom = 90 if cripto else 63        # cripto cotiza 7d: ~90 filas = 3 meses naturales
        mom3 = (px / float(c.iloc[-win_mom]) - 1) * 100 if len(c) > win_mom else 0.0
        if np.isnan(mom3):
            mom3 = 0.0
        # cripto es más volátil → normaliza con divisor mayor para no saturar el score
        s_mom = max(-1.0, min(1.0, mom3 / (25.0 if cripto else 15.0)))
        pilares.append(("Momentum 3 meses", f"{mom3:+.1f} %", s_mom, 0.12))

        # OBV: dirección del flujo de volumen
        obv_up = df["OBV"].iloc[-1] > df["OBV"].iloc[-10]
        pilares.append(("OBV (volumen)", "flujo subiendo" if obv_up else "flujo bajando", 0.4 if obv_up else -0.4, 0.05))

        # Señales del scanner
        tech = SS.señales_ticker(ticker)
        if tech:
            s_sig = max(-1.0, min(1.0, tech["Fuerza"] / 2.0))
            pilares.append(("Señales técnicas (scanner)", tech["Señales"], s_sig, 0.10))

        # Factores fundamentales (value/momentum/quality/low-vol) — institucional.
        # Solo acciones: la cripto no tiene PER/ROE. Best-effort (no rompe el veredicto).
        if not cripto:
            try:
                s_fac, lec_fac, _ = FS.score_absoluto(ticker)
                factor_score_val = s_fac
                pilares.append(("Factores (value/mom/quality/low-vol)", lec_fac, s_fac, 0.12))
            except Exception:
                pass

        # Sentimiento FinBERT (opcional)
        if con_sentimiento:
            noticias = SN.extraer_noticias(ticker, 10)
            if noticias:
                dfn = SN.analizar(noticias)
                s_sent, ver_sent = SN.score_global(dfn)
                pilares.append(("Sentimiento noticias (FinBERT)", f"{ver_sent} ({len(dfn)} titulares)", s_sent, 0.15))

        # Fear & Greed cripto (solo cripto): contrarian → miedo extremo = posible compra
        if cripto:
            cr = MC.fear_greed_cripto()
            if cr:
                sc = cr["score"]
                s_fg = max(-0.6, min(0.6, (50 - sc) / 50.0 * 0.6))   # miedo (+) / codicia (−)
                pilares.append(("Fear & Greed cripto (contrarian)", f"{sc} · {cr['etiqueta']}", s_fg, 0.12))

        # --- agregación ponderada ----------------------------------------------
        wsum = sum(p[3] for p in pilares)
        total = sum(p[2] * p[3] for p in pilares) / wsum

        # --- veredicto por umbral (el régimen NO vetea, solo avisa) -----------
        if total >= 0.35:
            verd, emoji = "COMPRAR", "🟢"
        elif total <= -0.35:
            verd, emoji = "VENDER", "🔴"
        else:
            verd, emoji = "MANTENER", "🟡"

        # Filtro de Régimen (INFORMATIVO): si el mercado es muy lateral (Hurst≈0.5,
        # ADX<20) y la señal es marginal, avisa de menor fiabilidad — pero NO cambia
        # el veredicto. (Antes vetaba a MANTENER y ahogaba señales válidas.)
        if (not np.isnan(hurst_v) and 0.42 < hurst_v < 0.54 and adx_v < 20
                and verd != "MANTENER" and abs(total) < 0.45):
            notas_modelos += (f"\n\n> ⚠️ Mercado lateral (Hurst={hurst_v:.2f}, ADX={adx_v:.0f}<20): "
                              f"señal {verd} MENOS fiable — usa tamaño reducido.")

        # --- Plan sugerido (entrada/stop/nº acciones) + registro opcional ------
        capital = float(capital) if capital else 10000.0
        plan_md, nota_log = "", ""
        if verd in ["COMPRAR", "VENDER"]:
            try:
                vol_anual = PS.garch_volatility(ticker)
                atr, _px_atr = PS.atr_actual(ticker)
                if not (atr and atr > 0):
                    atr = px * 0.02
                stop = px - 2 * atr if verd == "COMPRAR" else px + 2 * atr
                r_accion = abs(px - stop)
                if not np.isnan(vol_anual) and vol_anual > 0:
                    shares = max(1, int((capital * min(0.35, 0.15 / vol_anual)) // px))
                else:
                    shares = max(1, int((capital * 0.01) // r_accion))
                riesgo_eur = r_accion * shares
                coste = shares * px
                accion_txt = "Comprar" if verd == "COMPRAR" else "Vender (en corto)"
                plan_md = (f"\n\n### 📋 Plan sugerido (capital {capital:,.0f} €)\n"
                           f"- **{accion_txt} {shares} acciones** a ~{px:.2f} → coste {coste:,.0f} € "
                           f"({coste/capital*100:.0f}% del capital)\n"
                           f"- **Stop en {stop:.2f}**: si llega ahí, sales. Perderías ~{riesgo_eur:,.0f} € "
                           f"({riesgo_eur/capital*100:.1f}% del capital)\n"
                           f"- Practícalo primero: mándalo a paper en 🦙 Alpaca o apúntalo en 📒 Diario.")
                if registrar:
                    nid = JR.abrir(ticker, px, stop, shares, f"Auto-Veredicto: {verd} (Score {total:+.2f})",
                                   "SHORT" if verd == "VENDER" else "LONG", factor_score_val)
                    nota_log = f"\n\n✅ Apuntado en el 📒 Diario como operación **#{nid}** (simulada, para medir tu acierto)."
            except Exception as ej:
                nota_log = f"\n\n⚠️ No se pudo calcular el plan/registrar: {ej}"
        else:
            plan_md = ("\n\n### 📋 Qué hacer\n- **Nada.** La señal no es clara: espera, o busca otra "
                       "empresa en 📊 Factores. No operar también es una decisión (y a menudo la mejor).")

        # --- explicación en cristiano (a favor / en contra) ---------------------
        _AMIGO = [("Forecast", "Proyección a 90 días"), ("Tendencia", "Tendencia de fondo"),
                  ("ADX", "Fuerza de la tendencia"), ("Osciladores", "Termómetros de corto plazo"),
                  ("MACD", "Indicador MACD"), ("Momentum", "Empuje de los últimos 3 meses"),
                  ("OBV", "Dinero entrando/saliendo"), ("Señales", "Radar técnico"),
                  ("Factores", "Salud fundamental"), ("Sentimiento", "Tono de las noticias"),
                  ("Fear & Greed", "Miedo/codicia cripto")]
        def _amigo(n):
            for pref, lab in _AMIGO:
                if n.startswith(pref) or pref in n:
                    return lab
            return n
        contrib = sorted(((n, l, s * w / wsum) for n, l, s, w in pilares), key=lambda x: -x[2])
        favor = [f"- ✅ **{_amigo(n)}** — {l}" for n, l, a in contrib if a > 0.008][:4]
        contra = [f"- ❌ **{_amigo(n)}** — {l}" for n, l, a in reversed(contrib) if a < -0.008][:4]
        razones = ""
        if favor:
            razones += "\n\n**A favor:**\n" + "\n".join(favor)
        if contra:
            razones += "\n\n**En contra:**\n" + "\n".join(contra)

        fuerza = ("FUERTE" if abs(total) >= 0.5 else "moderada" if abs(total) >= 0.35
                  else "débil" if abs(total) >= 0.15 else "sin dirección clara")

        tabla = pd.DataFrame(
            [{"Pilar": n, "Lectura": l, "Score": round(s, 2),
              "Peso": f"{w/wsum*100:.0f}%", "Aporte": round(s * w / wsum, 3)}
             for n, l, s, w in pilares]
        )

        md = (f"# {emoji} {verd}" + ("  🪙 *cripto*" if cripto else "") + "\n\n"
              f"**{ticker}** · precio {px:.3f} · señal {fuerza} (score {total:+.2f}, "
              f"se compra desde +0.35 y se vende desde −0.35)"
              + razones
              + plan_md
              + notas_modelos
              + nota_log
              + "\n\n> ⚠️ Resumen estadístico automático, **no un consejo garantizado**. "
              + "El detalle técnico completo está en la tabla de abajo.")
        return fig, tabla, md
    except Exception as e:
        return _err_fig(f"Error: {e}"), pd.DataFrame(), f"**Error:** {e}"


def tab_veredicto_cripto(ticker, period, con_sentimiento, con_modelos=False, capital=10000, registrar=True):
    """Veredicto para criptomonedas: mismo agregador con forecast diario 7d,
    momentum en días naturales y pilar de Fear & Greed cripto."""
    return tab_veredicto(ticker, period, con_sentimiento, con_modelos, capital, registrar, cripto=True)


# ---- 11. Diario de operaciones (paper trading) ------------------------------
def tab_journal_abrir(ticker, entrada, stop, acciones, nota, es_short):
    try:
        # nota de factores al abrir (best-effort; cripto u otros fallos → None)
        try:
            nf, _, _ = FS.score_absoluto(ticker)
        except Exception:
            nf = None
        nid = JR.abrir(ticker, float(entrada), float(stop), int(acciones),
                       nota or "", "SHORT" if es_short else "LONG", nf)
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


# ---- 13. Alpha: dirección corto plazo (ML) + volatilidad (GARCH) ------------
def tab_alpha(ticker, horizon, period):
    try:
        import alpha_forecast as AF
        ticker = ticker.strip().upper()
        df = AF.descargar(ticker, period)
        d = AF.backtest_direccion(df, int(horizon))
        if d is None:
            return f"**{ticker}:** histórico insuficiente para el backtest direccional."
        bate = "**SÍ** bate al azar ✅" if d["pval"] < 0.05 else "**NO** bate al azar"
        lineas = [f"## {ticker} — dirección a {int(horizon)} días (ML, walk-forward purgado)\n",
                  f"- **Acierto direccional:** {d['da']*100:.1f}%  (N={d['N']})",
                  f"- **Significancia:** z={d['z']:+.2f}, p={d['pval']:.3f} → {bate}",
                  f"- **AUC:** {d['auc']:.3f}  (0.5 = azar)"]
        if not np.isnan(d["da_conv"]):
            lineas.append(f"- **Acierto en señales de alta convicción:** {d['da_conv']*100:.1f}% ({d['n_conv']} señales)")
        g = AF.backtest_volatilidad(df)
        if g:
            lect = ("predecible (modesto)" if g["corr"] > 0.2 else "débilmente predecible" if g["corr"] > 0.08 else "sin predictibilidad clara")
            lineas.append(f"- **Volatilidad GARCH(1,1):** corr(vol_pred, |ret|) = {g['corr']:.2f} → {lect}")
        lineas.append("\n> LightGBM + features leak-free + Hurst + embargo (López de Prado). "
                      "Mide si hay ventaja REAL; si p≥0.05 no la hay (eficiencia de mercado). **No es recomendación.**")
        return "\n".join(lineas)
    except Exception as e:
        return f"**Error:** {e}"


# ---- 14. Conformal: bandas CALIBRADAS --------------------------------------
def tab_conformal(ticker, period):
    try:
        import conformal_forecast as CF
        fig, tabla, meta = CF.forecast(ticker.strip().upper(), period=period, n_origenes=25)
        cob = meta.get("cobertura_media")
        md = (f"### {ticker.upper()} — banda {meta['objetivo']}% CALIBRADA (split conformal)\n"
              f"Precio {meta['precio_actual']:.3f} · {meta['n']} sesiones · "
              f"**cobertura real medida ≈ {cob:.0f}%** (objetivo {meta['objetivo']}%).\n\n"
              f"> La banda se calibra con los errores reales del walk-forward y se MIDE su "
              f"cobertura. Arregla la banda de Prophet, que solo cubría 14-29% real. "
              f"No es recomendación.")
        return fig, tabla, md
    except Exception as e:
        return _err_fig(f"Error: {e}"), pd.DataFrame(), f"**Error:** {e}"


# ---- 16. Riesgo de cartera: VaR/CVaR/drawdown/correlación -------------------
def tab_riesgo(txt, period, conf):
    try:
        import risk_metrics as RM
        tickers = _parse(txt)
        if not tickers:
            return _err_fig("Mete tickers."), pd.DataFrame(), pd.DataFrame()
        fig, tabla, corr, meta = RM.forecast(tickers, period, float(conf))
        return fig, tabla, corr
    except Exception as e:
        return _err_fig(f"Error: {e}"), pd.DataFrame(), pd.DataFrame()


# ---- 17. Alertas de watchlist ----------------------------------------------
def tab_alertas(txt):
    try:
        import alerts as AL
        tickers = _parse(txt)
        if not tickers:
            return pd.DataFrame(), "Mete tickers en la watchlist."
        filas = AL.escanear(tickers)
        if not filas:
            return pd.DataFrame(), f"Sin alertas ahora mismo en {len(tickers)} tickers."
        df = pd.DataFrame([{"Ticker": t, "Precio": round(p, 3), "Tipo": cat, "Aviso": txt2}
                           for t, p, cat, txt2 in filas])
        return df, f"**{len(df)} alertas** en {len(tickers)} tickers."
    except Exception as e:
        return pd.DataFrame(), f"**Error:** {e}"


# ---- 18. Factores: ranking multi-factor de un universo ---------------------
def tab_factores(txt):
    try:
        tickers = _parse(txt)
        if len(tickers) < 2:
            return pd.DataFrame(), "Mete **al menos 2** tickers: el ranking es relativo al universo."
        df = FS.rankear(tickers)
        cols = ["rank", "ticker", "z_value", "z_momentum", "z_quality", "z_lowvol", "nota", "señal"]
        out = df[cols].copy()
        for c in ["z_value", "z_momentum", "z_quality", "z_lowvol", "nota"]:
            out[c] = out[c].round(2)
        top, bot = df.iloc[0]["ticker"], df.iloc[-1]["ticker"]
        md = (f"### Ranking multi-factor ({len(tickers)} acciones)\n"
              f"**Mejor:** {top} · **Peor:** {bot}. Nota = z-score cruzado ponderado "
              f"(value 30% · momentum 30% · quality 25% · low-vol 15%).\n\n"
              f"> Así rankean los fondos cuant/smart-beta. Factores = premios de riesgo de "
              f"LARGO plazo (Fama-French), no timing. Fundamentales faltantes = neutros. "
              f"No es recomendación.")
        return out, md
    except Exception as e:
        return pd.DataFrame(), f"**Error:** {e}"


# ---- 19. Intradía: snapshot + backtest ORB con costes ----------------------
def tab_intraday_snapshot(ticker, interval, or_min):
    try:
        import intraday as IN
        fig, tabla, md = IN.snapshot(ticker.strip().upper(), interval, int(or_min))
        return fig, tabla, md
    except Exception as e:
        return _err_fig(f"Error: {e}"), pd.DataFrame(), f"**Error:** {e}"


def tab_informe_semanal():
    try:
        import informe_semanal as INF
        ruta = INF.generar()
        return (f"✅ Informe generado: **{ruta}**\n\n"
                f"(Señales de tu watchlist + riesgo + titulares. Ábrelo con Word.)")
    except Exception as e:
        return f"**Error:** {e}"


def tab_cartera_lp_crear(txt, capital):
    try:
        import rebalance as RB
        tickers = _parse(txt)
        if len(tickers) < 3:
            return pd.DataFrame(), "Mete al menos 3 activos."
        plan, meta = RB.crear(tickers, float(capital))
        md = (f"**Cartera creada y guardada** · invertido {meta['invertido']:,.0f} € de "
              f"{meta['capital']:,.0f} € (cash restante {meta['cash']:,.0f} €).\n\n"
              f"> Compra esas acciones (o pruébalo en paper). Vuelve ~1 vez al MES y pulsa "
              f"'Revisar rebalanceo': te dirá qué ajustar. Dividendos incluidos (total return).")
        return plan, md
    except Exception as e:
        return pd.DataFrame(), f"**Error:** {e}"


def tab_cartera_lp_revisar():
    try:
        import rebalance as RB
        tabla, meta = RB.revisar()
        md = (f"**Revisión de rebalanceo** · valor actual {meta['valor_total']:,.0f} € · "
              f"**{meta['n_ordenes']} órdenes de ajuste** (solo desvíos > 2.5 pp).\n\n"
              f"> Ejecuta las órdenes marcadas (o en paper) y pulsa 'He rebalanceado' para "
              f"actualizar la cartera guardada.")
        return tabla, md
    except Exception as e:
        return pd.DataFrame(), f"**Error:** {e}"


def tab_cartera_lp_aplicar():
    try:
        import rebalance as RB
        tabla, _ = RB.revisar()
        RB.aplicar_ordenes(tabla)
        tabla2, meta = RB.revisar()
        return tabla2, f"✅ Cartera actualizada con las órdenes. Valor {meta['valor_total']:,.0f} €."
    except Exception as e:
        return pd.DataFrame(), f"**Error:** {e}"


def tab_dca(txt, total, anios):
    try:
        import rebalance as RB
        tickers = _parse(txt)
        if not tickers:
            return _err_fig("Mete activos."), pd.DataFrame(), "Mete activos."
        tabla, meta, curvas = RB.comparar_dca(tickers, float(total), int(anios))
        fig = RB._plot_dca(curvas, tickers)
        md = (f"**Gana: {meta['gana']}** (en este periodo). {meta['n_meses']} aportes de "
              f"~{meta['aporte_mes']:,.0f} €/mes vs todo al inicio.\n\n"
              f"> Históricamente la entrada única gana ~2 de cada 3 veces (el mercado sube más "
              f"tiempo del que baja), pero el DCA reduce el riesgo de entrar en el peor momento "
              f"y es más llevadero psicológicamente. Dividendos incluidos. No es recomendación.")
        return fig, tabla, md
    except Exception as e:
        return _err_fig(f"Error: {e}"), pd.DataFrame(), f"**Error:** {e}"


def tab_intraday_semaforo(ticker, or_min):
    """Semáforo del día: opera largo / corto / no operes hoy."""
    try:
        import intraday as IN
        fig, tabla, md = IN.semaforo(ticker.strip().upper(), int(or_min))
        return fig, tabla, md
    except Exception as e:
        return _err_fig(f"{e}"), pd.DataFrame(), f"**No disponible:** {e}"


def tab_intraday_live(ticker, or_min):
    """Snapshot intradía EN VIVO con Alpaca (IEX). Fallback con mensaje claro."""
    try:
        import intraday as IN
        fig, tabla, md = IN.snapshot_alpaca(ticker.strip().upper(), int(or_min))
        return fig, tabla, md
    except Exception as e:
        return _err_fig(f"{e}"), pd.DataFrame(), f"**No disponible:** {e}"


def tab_intraday_backtest(ticker, interval, or_min, coste_bps, estrategia="orb"):
    try:
        import intraday as IN
        df = IN.descargar(ticker.strip().upper(), interval)
        bt = IN.backtest_estrategia(df, estrategia, int(or_min), interval, float(coste_bps))
        mt = IN.metricas_backtest(bt, float(coste_bps))
        if mt["n"] == 0:
            return pd.DataFrame(), f"**{ticker.upper()}:** {mt['mensaje']}"
        verd = ("✅ **Edge NETO positivo y significativo** → candidato a paper trading (Alpaca)."
                if mt["edge"] else
                "⛔ **Tras costes NO hay edge significativo.** No operes real con esto.")
        md = (f"### {ticker.upper()} · ORB {or_min}min · {interval} · coste {coste_bps} bps\n"
              f"- Operaciones: **{mt['n']}** · Win rate: **{mt['win_rate']}%** · p={mt['pval']}\n"
              f"- Expectancy **bruta** {mt['exp_bruto_pct']:+.3f}% → coste −{mt['coste_pct']:.3f}% → "
              f"**neta {mt['exp_neto_pct']:+.3f}%** por operación\n"
              f"- Total neto acumulado: **{mt['total_neto_pct']:+.2f}%**\n\n{verd}\n\n"
              f"> El coste se come el edge: un backtest intradía SIN costes miente. "
              f"yfinance intradía = desarrollo (retraso ~15 min), no ejecución. No es recomendación.")
        return bt, md
    except Exception as e:
        return pd.DataFrame(), f"**Error:** {e}"


def _alpaca_estado():
    import alpaca_paper as AP
    c = AP.cuenta()
    md = (f"**Cuenta PAPER** · estado {c['estado']} · {c['moneda']}  \n"
          f"Equity **{c['equity']:,.2f}** · Cash {c['cash']:,.2f} · "
          f"Buying power {c['buying_power']:,.2f} · P&L día **{c['pnl_dia']:+.2f}**")
    pos = AP.posiciones()
    return md, pd.DataFrame(pos) if pos else pd.DataFrame()


def tab_alpaca_refrescar():
    try:
        import alpaca_paper as AP
        if not AP.configurada():
            return "⚠️ Faltan ALPACA_KEY / ALPACA_SECRET en el .env.", pd.DataFrame()
        return _alpaca_estado()
    except Exception as e:
        return f"**Error:** {e}", pd.DataFrame()


def tab_alpaca_precio(symbol):
    try:
        import alpaca_paper as AP
        q = AP.cotizacion(symbol)
        return f"**{q['symbol']}**: {q['precio']:.4f}  (real-time IEX · {q['hora']})"
    except Exception as e:
        return f"**Error:** {e}"


def tab_alpaca_orden(symbol, qty, lado, registrar):
    """Envía una orden PAPER. La dispara el USUARIO con este botón (no el asistente).
    Si 'registrar', la apunta en el diario como apertura, con su nota de factores."""
    try:
        import alpaca_paper as AP
        symbol = (symbol or "").strip().upper()
        if not symbol or float(qty) <= 0:
            return "Indica símbolo y cantidad > 0.", "", pd.DataFrame()
        # freno de pérdida diaria: si hoy pierdes más del límite, no más órdenes
        try:
            fr = AP.freno_diario()
            if fr["bloqueado"]:
                return (f"🛑 **FRENO DIARIO ACTIVO**: hoy la cuenta pierde {fr['pct']}% "
                        f"(límite −{fr['limite_pct']}%). No se envían más órdenes hoy — "
                        f"parar a tiempo es la regla nº1. (Ajustable con FRENO_DIARIO_PCT en el .env.)",
                        "", pd.DataFrame())
        except Exception:
            pass
        side = "buy" if lado == "Comprar" else "sell"
        o = AP.enviar_orden(symbol, float(qty), side)
        msg = (f"✅ Orden PAPER enviada: **{o['side']} {o['qty']} {o['symbol']}** · "
               f"estado **{o['estado']}** · id `{o['id']}`")

        # registro en el diario (como apertura) con nota de factores
        if registrar:
            try:
                entrada = AP.cotizacion(symbol)["precio"]
                # stop de referencia: 2×ATR si hay; si no, 2 %
                try:
                    atr, _px = PS.atr_actual(symbol)
                    dist = 2 * atr if atr and not np.isnan(atr) else entrada * 0.02
                except Exception:
                    dist = entrada * 0.02
                direccion = "LONG" if side == "buy" else "SHORT"
                stop = entrada - dist if direccion == "LONG" else entrada + dist
                try:
                    nf, _, _ = FS.score_absoluto(symbol)
                except Exception:
                    nf = None
                acc = max(1, int(round(float(qty))))
                nid = JR.abrir(symbol, entrada, stop, acc,
                               f"Alpaca PAPER {side} (orden {o['id']})", direccion, nf)
                msg += f"\n\n📒 Registrada en el diario como **#{nid}** ({direccion}, entrada {entrada:.3f}, stop {stop:.3f})."
            except Exception as ej:
                msg += f"\n\n⚠️ Orden OK pero no se pudo registrar en el diario: {ej}"

        estado_md, pos = _alpaca_estado()
        return msg, estado_md, pos
    except Exception as e:
        return f"**Error:** {e}", "", pd.DataFrame()


def tab_intraday_scan(txt, interval, or_min, coste_bps, estrategia="orb"):
    try:
        import intraday as IN
        tickers = _parse(txt)
        if len(tickers) < 2:
            return pd.DataFrame(), "Mete **al menos 2** tickers para escanear."
        tabla = IN.escanear(tickers, interval, int(or_min), float(coste_bps), estrategia)
        if tabla.empty:
            return pd.DataFrame(), "Ningún ticker con operaciones (mercado cerrado/histórico corto/sin roturas)."
        ganan = tabla[tabla["Edge neto"].str.startswith("SÍ")]["Ticker"].tolist()
        md = (f"### Escaneo ORB · {len(tickers)} tickers · {interval} · coste {coste_bps} bps\n"
              f"Rankeado por **expectancy NETA** (tras costes). "
              f"Con edge significativo (p<0.05): **{', '.join(ganan) if ganan else 'ninguno'}**.\n\n"
              f"> Si ninguno supera el azar tras costes, es la realidad del intradía líquido — "
              f"no un fallo. Opera en paper solo los que tengan edge neto significativo. No es recomendación.")
        return tabla, md
    except Exception as e:
        return pd.DataFrame(), f"**Error:** {e}"


# ---- 29. Física/Info: RMT + Kalman + Entropía de transferencia --------------
def tab_rmt(txt, period):
    try:
        import rmt_clean as RMT
        tickers = _parse(txt)
        if len(tickers) < 4:
            return _err_fig("≥4 activos."), "Mete al menos 4 activos (RMT necesita una matriz)."
        meta = RMT.comparar(tickers, period)
        if meta is None:
            return _err_fig("Datos insuficientes."), "Datos insuficientes."
        return RMT._plot(meta), "```\n" + RMT.informe(meta) + "\n```"
    except Exception as e:
        return _err_fig(f"Error: {e}"), f"**Error:** {e}"


def tab_kalman(a, b, period):
    try:
        import kalman_hedge as KH
        res = KH.analizar(a.strip().upper(), b.strip().upper(), period)
        return KH._plot(res), "```\n" + KH.informe(res) + "\n```"
    except Exception as e:
        return _err_fig(f"Error: {e}"), f"**Error:** {e}"


def tab_te(txt, bins, period):
    try:
        import transfer_entropy as TE
        tickers = _parse(txt)
        if len(tickers) < 3:
            return _err_fig("≥3 activos."), "Mete al menos 3 activos."
        M, neto, _ = TE.matriz(tickers, period, int(bins))
        return TE._plot(M), "```\n" + TE.informe(M, neto) + "\n```"
    except Exception as e:
        return _err_fig(f"Error: {e}"), f"**Error:** {e}"


# ---- 28. Régimen HMM + Meta-labeling ---------------------------------------
def tab_hmm(ticker, estados, period):
    try:
        import hmm_regime as HM
        res = HM.analizar(ticker.strip().upper(), int(estados), period)
        return HM._plot(res), "```\n" + HM.informe(res) + "\n```"
    except Exception as e:
        return _err_fig(f"Error: {e}"), f"**Error:** {e}"


def tab_meta(ticker, horizon, umbral, period):
    try:
        import meta_labeling as ML
        res = ML.backtest(ticker.strip().upper(), int(horizon), float(umbral), period)
        return "```\n" + ML.informe(ticker.strip().upper(), res) + "\n```"
    except Exception as e:
        return f"**Error:** {e}"


# ---- 27. Backtest de la estrategia completa --------------------------------
def tab_system_backtest(txt, top_n, rebal, coste_bps, period):
    try:
        import system_backtest as SB
        tickers = _parse(txt)
        if len(tickers) < 2:
            return _err_fig("Mete ≥2 tickers."), "Mete al menos 2 tickers."
        res = SB.backtest(tickers, period, int(top_n), int(rebal), float(coste_bps))
        if res is None:
            return _err_fig("Datos insuficientes."), "Datos insuficientes."
        return SB._plot(res), "```\n" + SB.informe(res) + "\n```"
    except Exception as e:
        return _err_fig(f"Error: {e}"), f"**Error:** {e}"


# ---- 26. Monte Carlo (precio + sistema) ------------------------------------
def tab_mc_precio(ticker, horizon, metodo):
    try:
        import montecarlo as MC
        paths, fin, meta = MC.simular_precio(ticker.strip().upper(), int(horizon), 3000, metodo)
        return MC._plot_precio(paths, meta, ticker.strip().upper()), "```\n" + MC.informe_precio(ticker.strip().upper(), meta) + "\n```"
    except Exception as e:
        return _err_fig(f"Error: {e}"), f"**Error:** {e}"


def tab_mc_sistema(winrate, payoff, riesgo, trades, usar_diario):
    try:
        import montecarlo as MC
        ret = MC._retornos_trade_diario() if usar_diario else None
        if usar_diario and ret is None:
            return _err_fig("Diario sin suficientes operaciones cerradas (≥5)."), \
                   "Diario sin suficientes operaciones cerradas (≥5). Usa win-rate/payoff o cierra más en paper."
        equity, meta = MC.simular_sistema(ret, float(winrate), float(payoff),
                                          float(riesgo) / 100.0, int(trades), 5000, 10000.0)
        return MC._plot_sistema(equity, meta), "```\n" + MC.informe_sistema(meta) + "\n```"
    except Exception as e:
        return _err_fig(f"Error: {e}"), f"**Error:** {e}"


# ---- 25. Matemática avanzada: pairs / HRP / EVT ----------------------------
def tab_pairs(txt, period):
    try:
        import pairs_trading as PT
        tickers = _parse(txt)
        if len(tickers) < 2:
            return pd.DataFrame(), "Mete al menos 2 tickers."
        df = PT.buscar(tickers, period)
        if df.empty:
            return pd.DataFrame(), "Ningún par cointegrado (p<0.10) con reversión clara."
        return df, ("**Pares cointegrados** (arbitraje estadístico, market-neutral). "
                    "Opera los de p<0.05 cuando |z|>2: largo el barato, corto el caro. "
                    "Half-life corto = revierte rápido. No es recomendación.")
    except Exception as e:
        return pd.DataFrame(), f"**Error:** {e}"


def tab_hrp(txt, period):
    try:
        import hrp_portfolio as HP
        tickers = _parse(txt)
        if len(tickers) < 3:
            return pd.DataFrame(), pd.DataFrame(), "Mete al menos 3 activos."
        out = HP.comparar(tickers, period)
        if out is None:
            return pd.DataFrame(), pd.DataFrame(), "Datos insuficientes."
        tabla, pesos, _ = out
        wdf = (pesos["HRP"] * 100).round(1).reset_index()
        wdf.columns = ["Activo", "Peso HRP %"]
        md = ("**Comparación OOS** (pesos con 1ª mitad, vol/Sharpe en 2ª). HRP y Min-Var "
              "(Ledoit-Wolf) baten a la equiponderada con menos vol. HRP no invierte la "
              "covarianza → robusto. No es recomendación.")
        return tabla, wdf, md
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), f"**Error:** {e}"


def tab_evt(ticker, period, umbral):
    try:
        import evt_risk as EV
        r = EV._retornos(ticker.strip().upper(), period)
        res = EV.evt(r, float(umbral))
        return "```\n" + EV.informe(ticker.strip().upper(), res) + "\n```"
    except Exception as e:
        return f"**Error:** {e}"


# ---- 24. Monitor de rendimiento vs benchmark -------------------------------
def tab_rendimiento(capital, benchmark):
    try:
        import performance as PF
        res = PF.analizar(float(capital), benchmark.strip().upper() or "SPY")
        if res.get("n", 0) == 0:
            return _err_fig(res.get("mensaje", "")), "```\n" + res.get("mensaje", "") + "\n```"
        fig = PF._plot(res)
        return fig, "```\n" + PF.informe(res) + "\n```"
    except Exception as e:
        return _err_fig(f"Error: {e}"), f"**Error:** {e}"


# ---- 23. Orquestador del sistema (plan + ejecución paper) -------------------
def tab_sistema_plan(txt, capital, tv, mp, umbral):
    try:
        import orchestrator as OR, risk_manager as RKM
        tickers = _parse(txt)
        if not tickers:
            return pd.DataFrame(), "Mete tickers."
        plan, meta = OR.plan_de_hoy(tickers, float(capital), float(tv), int(mp), float(umbral))
        if "mensaje" in meta:
            return pd.DataFrame(), meta["mensaje"]
        aviso = "⚠️ EXCEDE límite de riesgo" if meta["excede_riesgo"] else "✓ dentro del límite"
        md = (f"**Plan de hoy** · {meta['n_ordenes']} órdenes · exposición {meta['exposicion_pct']}% · "
              f"riesgo {meta['riesgo_total_pct']}% ({aviso}).\n\n"
              f"> Revisa el plan. Para mandarlo a PAPER usa el botón de abajo (lo disparas TÚ).")
        return plan, md
    except Exception as e:
        return pd.DataFrame(), f"**Error:** {e}"


def tab_sistema_ejecutar(txt, capital, tv, mp, umbral, confirmar):
    """Ejecuta el plan en Alpaca PAPER. Requiere que el usuario marque la confirmación."""
    try:
        if not confirmar:
            return pd.DataFrame(), "☝️ Marca **'Confirmo enviar a PAPER'** para ejecutar. (Es dinero ficticio.)"
        import orchestrator as OR
        tickers = _parse(txt)
        plan, meta = OR.plan_de_hoy(tickers, float(capital), float(tv), int(mp), float(umbral))
        if plan.empty:
            return pd.DataFrame(), "Plan vacío: nada que ejecutar."
        rdf, resumen = OR.ejecutar(plan, registrar=True)
        return rdf, f"✅ {resumen}"
    except Exception as e:
        return pd.DataFrame(), f"**Error:** {e}"


# ---- 22. Gestor de riesgo / plan de órdenes --------------------------------
def tab_plan_riesgo(txt, capital, target_vol, max_pos, umbral):
    try:
        import risk_manager as RKM
        tickers = _parse(txt)
        if not tickers:
            return pd.DataFrame(), "Mete tickers."
        plan, meta = RKM.generar_plan(tickers, float(capital), float(target_vol),
                                      int(max_pos), 2.0, float(umbral))
        if "mensaje" in meta:
            return pd.DataFrame(), meta["mensaje"]
        aviso = "⚠️ **EXCEDE** tu límite de riesgo diario" if meta["excede_riesgo"] else "✓ dentro del límite"
        md = (f"**Exposición:** {meta['exposicion_pct']}% · "
              f"**Riesgo total:** {meta['riesgo_total_eur']:.0f} € ({meta['riesgo_total_pct']}%) "
              f"vs límite {meta['limite_riesgo_pct']:.0f}% → {aviso}\n\n"
              f"> Vol targeting + stop ATR. Entrada del bucle de ejecución (paper). "
              f"Las órdenes las disparas tú.")
        return plan, md
    except Exception as e:
        return pd.DataFrame(), f"**Error:** {e}"


# ---- 21. Generador de señales del sistema ----------------------------------
def tab_senales_sistema(txt, umbral, con_factor):
    try:
        import signal_engine as SE
        tickers = _parse(txt)
        if not tickers:
            return pd.DataFrame(), "Mete tickers."
        df = SE.generar(tickers, float(umbral), bool(con_factor))
        if df.empty:
            return pd.DataFrame(), "Sin señales (datos insuficientes)."
        comprar = df[df["Señal"].str.contains("COMPRAR")]["Ticker"].tolist()
        vender = df[df["Señal"].str.contains("VENDER")]["Ticker"].tolist()
        md = (f"**COMPRAR:** {', '.join(comprar) if comprar else '—'}  ·  "
              f"**VENDER:** {', '.join(vender) if vender else '—'}\n\n"
              f"> Score técnico del Veredicto (validado). En líquidas no supera multiple-testing → "
              f"marco de PAPER, no alfa garantizado. Entrada del bucle de ejecución.")
        return df, md
    except Exception as e:
        return pd.DataFrame(), f"**Error:** {e}"


# ---- 20. Validar Veredicto (backtest honesto del score) --------------------
def tab_validar_veredicto(txt, horizon, trials):
    try:
        import veredicto_backtest as VB
        tickers = _parse(txt)
        if len(tickers) < 3:
            return "Mete **al menos 3** tickers (la validación es cross-section)."
        res = VB.backtest(tickers, int(horizon), trials=int(trials) if trials else None)
        return "```\n" + VB.informe(res) + "\n```"
    except Exception as e:
        return f"**Error:** {e}"


# ---- Watchlist única persistente (compartida por todas las pestañas) -------
_WL_FILE = PROJ / "watchlist.txt"
_WL_DEFAULT = "AAPL, MSFT, NVDA, GOOGL, AMZN, META, JPM, XOM, KO, SAB.MC"

def _wl_load():
    try:
        if _WL_FILE.exists():
            txt = _WL_FILE.read_text(encoding="utf-8").strip()
            if txt:
                return ", ".join(t.strip().upper() for t in txt.replace("\n", ",").split(",") if t.strip())
    except Exception:
        pass
    return _WL_DEFAULT

def _wl_save(txt):
    limpio = ", ".join(t.strip().upper() for t in (txt or "").replace("\n", ",").split(",") if t.strip())
    if not limpio:
        actual = _wl_load()
        return ("⚠️ Watchlist vacía: mantengo la anterior.", *([actual] * 11))
    _WL_FILE.write_text(limpio, encoding="utf-8")
    return (f"✅ Watchlist guardada ({limpio.count(',') + 1} valores). Aplicada a todas las pestañas.",
            *([limpio] * 11))


# ---- UI -------------------------------------------------------------------
def build():
    import gradio as gr
    # head: bloquea el traductor automático de Chrome (rompe la reactividad de Gradio)
    _head = '<meta name="google" content="notranslate"><script>document.documentElement.lang="es";</script>'
    with gr.Blocks(title="FinanzIA — Mesa cuantitativa", head=_head) as app:
        gr.Markdown("# FinanzIA — Mesa cuantitativa")
        gr.Markdown("Suite de trading algorítmico. Datos Yahoo Finance (retardo ~15 min). "
                    "Análisis y educación — **no es recomendación de inversión**.")
        WL = _wl_load()
        with gr.Accordion("💾 Mi watchlist (compartida por todas las pestañas)", open=False):
            with gr.Row():
                wtxt = gr.Textbox(value=WL, label="Tus valores (separados por coma)", scale=4)
                wbtn = gr.Button("Guardar watchlist", variant="primary")
            wmd = gr.Markdown()
        with gr.Tab("🎯 Operar"):
            with gr.Tabs():
                with gr.Tab("📊 Factores"):
                    gr.Markdown("**Modelo multi-factor (smart beta)** — cómo deciden los fondos cuant "
                                "qué comprar. Rankea un universo por **value + momentum + quality + low-vol** "
                                "(z-score cruzado). Compra el top, evita el fondo. Tarda ~1-3 s/acción "
                                "(descarga fundamentales).")
                    with gr.Row():
                        tf = gr.Textbox(value=WL,
                                        label="Universo de acciones (coma)", scale=4)
                        bf = gr.Button("Rankear", variant="primary")
                    mdf = gr.Markdown()
                    tblf = gr.Dataframe(label="Ranking multi-factor", wrap=True)
                    bf.click(tab_factores, [tf], [tblf, mdf])
                with gr.Tab("★ Veredicto"):
                    gr.Markdown("**Análisis completo en un clic**: forecast + tendencia + ADX + "
                                "**consenso de 5 osciladores** + MACD + momentum + volumen (OBV) + señales "
                                "(+ consenso multi-modelo y sentimiento opcionales) → estimación "
                                "**COMPRAR / MANTENER / VENDER** con desglose por pilar.")
                    with gr.Row():
                        tv = gr.Textbox(value="AAPL", label="Ticker", scale=3)
                        pv = gr.Dropdown(["2y", "3y", "5y"], value="3y", label="Histórico")
                        cv = gr.Number(value=10000, label="Tu capital €")
                        bv = gr.Button("Analizar TODO", variant="primary")
                    with gr.Row():
                        rv = gr.Checkbox(value=True, label="Si sale COMPRAR/VENDER, apuntarlo en el 📒 Diario (simulado)")
                        sv = gr.Checkbox(value=False, label="Incluir noticias (sentimiento; 1ª vez +1 min)")
                        mv = gr.Checkbox(value=False, label="Consenso multi-modelo (informativo, +2 min)")
                    mdv = gr.Markdown()
                    tbv = gr.Dataframe(label="Detalle técnico (opcional)", wrap=True)
                    plv = gr.Plot(label="Forecast 30/90/120d")
                    bv.click(tab_veredicto, [tv, pv, sv, mv, cv, rv], [plv, tbv, mdv])
                with gr.Tab("🪙 Veredicto Cripto"):
                    gr.Markdown("**Veredicto para criptomonedas** (BTC-USD, ETH-EUR, SOL-USD…). "
                                "Mismo agregador pero con **forecast diario 7d**, momentum en días "
                                "naturales y pilar de **Fear & Greed cripto** (contrarian) → "
                                "**COMPRAR / MANTENER / VENDER**. Usa tickers Yahoo tipo `BTC-USD`.")
                    with gr.Row():
                        tvc = gr.Textbox(value="BTC-USD", label="Ticker cripto (BASE-FIAT)", scale=3)
                        pvc = gr.Dropdown(["1y", "2y", "3y", "5y"], value="3y", label="Histórico")
                        cvc = gr.Number(value=10000, label="Tu capital €")
                        bvc = gr.Button("Analizar cripto", variant="primary")
                    with gr.Row():
                        rvc = gr.Checkbox(value=True, label="Si sale COMPRAR/VENDER, apuntarlo en el 📒 Diario (simulado)")
                        svc = gr.Checkbox(value=False, label="Incluir noticias (sentimiento; 1ª vez +1 min)")
                        mvc = gr.Checkbox(value=False, label="Consenso multi-modelo (informativo, +2 min)")
                    mdvc = gr.Markdown()
                    tbvc = gr.Dataframe(label="Detalle técnico (opcional)", wrap=True)
                    plvc = gr.Plot(label="Forecast cripto 30/90/120d")
                    bvc.click(tab_veredicto_cripto, [tvc, pvc, svc, mvc, cvc, rvc], [plvc, tbvc, mdvc])
                with gr.Tab("⏱️ Intradía"):
                    gr.Markdown("**Intradía (desarrollo, sin arriesgar)**: VWAP + rango de apertura + "
                                "**backtest Opening Range Breakout con COSTES** (comisión+spread+slippage). "
                                "Datos yfinance intradía (retraso ~15 min, histórico corto) → para *validar* "
                                "un método, no para ejecutar en vivo. El coste se come el edge: aquí se ve.")
                    with gr.Row():
                        ti = gr.Textbox(value="AAPL", label="Ticker", scale=3)
                        ii = gr.Dropdown(["5m", "15m", "30m", "60m"], value="15m", label="Intervalo")
                        ori = gr.Dropdown([15, 30, 60], value=30, label="Rango apertura (min)")
                        ci = gr.Number(value=6.0, label="Coste ida+vuelta (bps)")
                        esti = gr.Dropdown(["orb", "vwap", "ema9"], value="orb",
                                           label="Estrategia (backtest)")
                    with gr.Row():
                        bsem = gr.Button("🚦 Semáforo de HOY (¿opero o no?)", variant="primary")
                        bi0 = gr.Button("📡 Snapshot EN VIVO (Alpaca, EEUU)", variant="secondary")
                        bi1 = gr.Button("📷 Snapshot yfinance (~15 min retraso)", variant="secondary")
                        bi2 = gr.Button("🧪 Backtest ORB con costes", variant="secondary")
                    mdi = gr.Markdown()
                    figi = gr.Plot()
                    tbli = gr.Dataframe(wrap=True)
                    bsem.click(tab_intraday_semaforo, [ti, ori], [figi, tbli, mdi])
                    bi0.click(tab_intraday_live, [ti, ori], [figi, tbli, mdi])
                    bi1.click(tab_intraday_snapshot, [ti, ii, ori], [figi, tbli, mdi])
                    bi2.click(tab_intraday_backtest, [ti, ii, ori, ci, esti], [tbli, mdi])
                    gr.Markdown("---\n**Escaneo multi-ticker**: ¿dónde (si en algún sitio) sobrevive el edge al coste?")
                    with gr.Row():
                        tis = gr.Textbox(value=WL,
                                         label="Universo (coma)", scale=4)
                        bi3 = gr.Button("🔭 Escanear varios (rank por exp. neta)", variant="primary")
                    bi3.click(tab_intraday_scan, [tis, ii, ori, ci, esti], [tbli, mdi])
                with gr.Tab("🦙 Alpaca Paper"):
                    gr.Markdown("**Paper trading en vivo** (Alpaca · dinero FICTICIO, datos real-time IEX). "
                                "El salto a 'vivo' sin riesgo. **Las órdenes las envías TÚ** con el botón — "
                                "el asistente nunca opera por su cuenta. Configura `ALPACA_KEY`/`ALPACA_SECRET` en `.env`.")
                    with gr.Row():
                        bap = gr.Button("🔄 Refrescar cuenta y posiciones", variant="secondary")
                    mdap = gr.Markdown()
                    tblap = gr.Dataframe(label="Posiciones abiertas (paper)", wrap=True)
                    gr.Markdown("**Cotización real-time**")
                    with gr.Row():
                        tapq = gr.Textbox(value="AAPL", label="Símbolo", scale=3)
                        bapq = gr.Button("Precio")
                    mdapq = gr.Markdown()
                    gr.Markdown("---\n**Enviar orden PAPER** (la disparas tú; es dinero ficticio)")
                    with gr.Row():
                        taps = gr.Textbox(value="AAPL", label="Símbolo", scale=2)
                        tapn = gr.Number(value=1, label="Cantidad")
                        tapl = gr.Radio(["Comprar", "Vender"], value="Comprar", label="Lado")
                        bapo = gr.Button("📨 Enviar orden PAPER", variant="primary")
                    tapr = gr.Checkbox(value=True, label="Registrar en el diario como apertura (con nota de factores). "
                                                         "Desmárcalo si esta orden CIERRA una posición.")
                    mdapo = gr.Markdown()
                    bap.click(tab_alpaca_refrescar, [], [mdap, tblap])
                    bapq.click(tab_alpaca_precio, [tapq], [mdapq])
                    bapo.click(tab_alpaca_orden, [taps, tapn, tapl, tapr], [mdapo, mdap, tblap])
                    app.load(tab_alpaca_refrescar, [], [mdap, tblap])
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
        with gr.Tab("⚙️ Sistema algo"):
            with gr.Tabs():
                with gr.Tab("📡 Señales"):
                    gr.Markdown("**Qué operar hoy**: corre el score del Veredicto sobre tu watchlist → "
                                "ranking COMPRAR/MANTENER/VENDER. Entrada del bucle de ejecución (sizing "
                                "→ Alpaca paper → diario). Rápido (sin Prophet).")
                    with gr.Row():
                        tse = gr.Textbox(value=WL,
                                         label="Watchlist (coma)", scale=4)
                        use = gr.Slider(0.1, 0.6, value=0.35, step=0.05, label="Umbral señal")
                        fse = gr.Checkbox(value=False, label="Añadir factores")
                        bse = gr.Button("Generar señales", variant="primary")
                    mdse = gr.Markdown()
                    tbse = gr.Dataframe(label="Ranking de señales", wrap=True)
                    bse.click(tab_senales_sistema, [tse, use, fse], [tbse, mdse])
                with gr.Tab("⚖️ Plan / Riesgo"):
                    gr.Markdown("**Cuánto operar de cada señal**: vol targeting + máx posiciones + "
                                "stop ATR + tope de riesgo. Convierte las señales en un plan de órdenes "
                                "concreto (acciones, coste, stop, riesgo €). Entrada del bucle de ejecución.")
                    with gr.Row():
                        trk = gr.Textbox(value=WL,
                                         label="Watchlist (coma)", scale=3)
                        crk = gr.Number(value=10000, label="Capital €")
                        vrk = gr.Slider(0.08, 0.30, value=0.15, step=0.01, label="Vol objetivo")
                        mrk = gr.Dropdown([3, 4, 5, 6, 8], value=5, label="Máx posiciones")
                        urk = gr.Slider(0.1, 0.6, value=0.30, step=0.05, label="Umbral señal")
                        brk = gr.Button("Generar plan", variant="primary")
                    mdrk = gr.Markdown()
                    tbrk = gr.Dataframe(label="Plan de órdenes", wrap=True)
                    brk.click(tab_plan_riesgo, [trk, crk, vrk, mrk, urk], [tbrk, mdrk])
                with gr.Tab("🤖 Sistema"):
                    gr.Markdown("**El sistema completo de un clic**: señales → plan/riesgo → ejecución "
                                "PAPER → diario. Genera el plan, revísalo, y si quieres lo mandas a Alpaca "
                                "paper (dinero ficticio). **Las órdenes las disparas TÚ** con confirmación.")
                    with gr.Row():
                        tsy = gr.Textbox(value=WL,
                                         label="Watchlist (coma)", scale=3)
                        csy = gr.Number(value=10000, label="Capital €")
                        vsy = gr.Slider(0.08, 0.30, value=0.15, step=0.01, label="Vol objetivo")
                        msy = gr.Dropdown([3, 4, 5, 6, 8], value=5, label="Máx pos.")
                        usy = gr.Slider(0.1, 0.6, value=0.30, step=0.05, label="Umbral")
                    with gr.Row():
                        bsy1 = gr.Button("1) Generar plan de hoy", variant="secondary")
                        conf = gr.Checkbox(value=False, label="Confirmo enviar a PAPER (dinero ficticio)")
                        bsy2 = gr.Button("2) ▶️ Ejecutar en PAPER", variant="primary")
                    mdsy = gr.Markdown()
                    tbsy = gr.Dataframe(label="Plan / Resultado", wrap=True)
                    bsy1.click(tab_sistema_plan, [tsy, csy, vsy, msy, usy], [tbsy, mdsy])
                    bsy2.click(tab_sistema_ejecutar, [tsy, csy, vsy, msy, usy, conf], [tbsy, mdsy])
                with gr.Tab("📈 Rendimiento"):
                    gr.Markdown("**¿Bates al mercado?** Curva de equity del diario + métricas "
                                "(win rate, profit factor, expectancy, drawdown) y comparación **vs SPY** "
                                "(comprar y mantener). El listón honesto: si no bates a indexarte, mejor indexarse.")
                    with gr.Row():
                        cpf = gr.Number(value=10000, label="Capital inicial €")
                        bpf_t = gr.Textbox(value="SPY", label="Benchmark", scale=2)
                        bpf = gr.Button("Medir rendimiento", variant="primary")
                    mdpf = gr.Markdown()
                    figpf = gr.Plot()
                    bpf.click(tab_rendimiento, [cpf, bpf_t], [figpf, mdpf])
                with gr.Tab("🎲 Monte Carlo"):
                    gr.Markdown("**El abanico de lo posible** (no predicción). **Precio**: miles de "
                                "trayectorias futuras (bootstrap de retornos). **Sistema**: bootstrap de "
                                "operaciones → prob. de acabar positivo, drawdown y **prob. de ruina**.")
                    gr.Markdown("#### Monte Carlo de PRECIO")
                    with gr.Row():
                        tmc = gr.Textbox(value="AAPL", label="Ticker", scale=3)
                        hmc = gr.Dropdown([30, 60, 90, 120, 252], value=90, label="Horizonte (días)")
                        emc = gr.Dropdown(["bootstrap", "gbm"], value="bootstrap", label="Método")
                        bmc1 = gr.Button("Simular precio", variant="primary")
                    mdmc1 = gr.Markdown()
                    figmc1 = gr.Plot()
                    bmc1.click(tab_mc_precio, [tmc, hmc, emc], [figmc1, mdmc1])
                    gr.Markdown("---\n#### Monte Carlo del SISTEMA (robustez)")
                    with gr.Row():
                        wmc = gr.Slider(0.3, 0.7, value=0.55, step=0.01, label="Win rate")
                        pmc = gr.Slider(0.5, 3.0, value=1.5, step=0.1, label="Payoff (G/P)")
                        rmc = gr.Slider(0.25, 3.0, value=1.0, step=0.25, label="Riesgo/op %")
                        nmc = gr.Dropdown([30, 50, 100, 200], value=50, label="Nº operaciones")
                        dmc = gr.Checkbox(value=False, label="Usar mi diario real")
                        bmc2 = gr.Button("Simular sistema", variant="primary")
                    mdmc2 = gr.Markdown()
                    figmc2 = gr.Plot()
                    bmc2.click(tab_mc_sistema, [wmc, pmc, rmc, nmc, dmc], [figmc2, mdmc2])
                with gr.Tab("📊 Backtest Sistema"):
                    gr.Markdown("**Backtest de la estrategia completa** sobre el histórico: compra el "
                                "**top-N** por score del Veredicto, rebalancea, **resta costes**, y compara "
                                "con **SPY** buy & hold. Un backtest que bate a SPY mide, no promete.")
                    with gr.Row():
                        tsb = gr.Textbox(value=WL,
                                         label="Universo (coma)", scale=3)
                        nsb = gr.Dropdown([2, 3, 4, 5], value=3, label="Top-N")
                        rsb = gr.Dropdown([5, 10, 21, 42], value=21, label="Rebal (días)")
                        csb = gr.Number(value=10, label="Coste bps")
                        psb = gr.Dropdown(["3y", "5y", "8y"], value="5y", label="Histórico")
                        bsb = gr.Button("Backtestear", variant="primary")
                    mdsb = gr.Markdown()
                    figsb = gr.Plot()
                    bsb.click(tab_system_backtest, [tsb, nsb, rsb, csb, psb], [figsb, mdsb])
                with gr.Tab("🔬 Validar Veredicto"):
                    gr.Markdown("**¿El Veredicto predice de verdad?** Backtest honesto del score técnico "
                                "point-in-time: **IC** (score↔retorno futuro), retornos por **quintil**, "
                                "Sharpe long-short y **Deflated Sharpe** (corrige multiple-testing). "
                                "El paso obligatorio antes de automatizar. Tarda ~1 min.")
                    with gr.Row():
                        tvb = gr.Textbox(value=WL,
                                         label="Universo (coma)", scale=4)
                        hvb = gr.Dropdown([5, 10, 20], value=10, label="Horizonte (días)")
                        trvb = gr.Number(value=20, label="Nº pruebas (deflación)")
                        bvb = gr.Button("Validar", variant="primary")
                    mdvb = gr.Markdown()
                    bvb.click(tab_validar_veredicto, [tvb, hvb, trvb], [mdvb])
        with gr.Tab("🔬 Cuant avanzado"):
            with gr.Tabs():
                with gr.Tab("🔗 Pairs (cointegración)"):
                    gr.Markdown("**Arbitraje estadístico market-neutral**: busca pares COINTEGRADOS "
                                "(Engle-Granger) con reversión a la media (half-life Ornstein-Uhlenbeck). "
                                "Largo el barato / corto el caro cuando el z-score del spread es extremo.")
                    with gr.Row():
                        tpr = gr.Textbox(value="KO, PEP, XOM, CVX, V, MA, JPM, BAC, GLD, SLV",
                                         label="Universo (coma)", scale=4)
                        ppr = gr.Dropdown(["2y", "3y", "5y"], value="3y", label="Histórico")
                        bpr = gr.Button("Buscar pares", variant="primary")
                    mdpr = gr.Markdown()
                    tbpr = gr.Dataframe(label="Pares cointegrados", wrap=True)
                    bpr.click(tab_pairs, [tpr, ppr], [tbpr, mdpr])
                with gr.Tab("🧮 HRP Cartera"):
                    gr.Markdown("**Asignación robusta**: HRP (López de Prado) + Min-Var con **Ledoit-Wolf** "
                                "vs equiponderada, medido **fuera de muestra**. Arregla la inestabilidad de "
                                "Markowitz (no invierte la covarianza).")
                    with gr.Row():
                        thr = gr.Textbox(value="AAPL, MSFT, NVDA, GOOGL, AMZN, JPM, XOM, KO, GLD, TLT",
                                         label="Activos (coma)", scale=4)
                        phr = gr.Dropdown(["3y", "4y", "5y"], value="4y", label="Histórico")
                        bhr = gr.Button("Comparar asignación", variant="primary")
                    mdhr = gr.Markdown()
                    with gr.Row():
                        tbhr = gr.Dataframe(label="HRP vs Min-Var vs Equal (OOS)", wrap=True)
                        whr = gr.Dataframe(label="Pesos HRP", wrap=True)
                    bhr.click(tab_hrp, [thr, phr], [tbhr, whr, mdhr])
                with gr.Tab("📉 EVT Colas"):
                    gr.Markdown("**Riesgo de cola (crash)** con Teoría de Valores Extremos: ajusta una "
                                "Pareto Generalizada a la cola (POT) → VaR/ES extremos. Compara con histórico "
                                "y normal: se ve cuánto **subestima la normal** el riesgo de crash.")
                    with gr.Row():
                        tev = gr.Textbox(value="SPY", label="Ticker", scale=3)
                        pev = gr.Dropdown(["5y", "8y", "10y", "max"], value="10y", label="Histórico")
                        uev = gr.Slider(0.90, 0.98, value=0.95, step=0.01, label="Umbral cola (cuantil u)")
                        bev = gr.Button("Medir cola", variant="primary")
                    mdev = gr.Markdown()
                    bev.click(tab_evt, [tev, pev, uev], [mdev])
                with gr.Tab("🌀 Régimen (HMM)"):
                    gr.Markdown("**¿En qué régimen estamos?** Hidden Markov Model (gaussiano) identifica "
                                "estados ocultos: 🟢 calma alcista, 🔴 alta volatilidad, 🟡 lateral. Úsalo "
                                "como GATE: opera tendencia en 🟢, recorta en 🔴.")
                    with gr.Row():
                        thm = gr.Textbox(value="SPY", label="Ticker", scale=3)
                        nhm = gr.Dropdown([2, 3, 4], value=3, label="Nº regímenes")
                        phm = gr.Dropdown(["5y", "8y", "10y", "max"], value="8y", label="Histórico")
                        bhm = gr.Button("Detectar régimen", variant="primary")
                    mdhm = gr.Markdown()
                    fighm = gr.Plot()
                    bhm.click(tab_hmm, [thm, nhm, phm], [fighm, mdhm])
                with gr.Tab("🎯 Meta-labeling"):
                    gr.Markdown("**¿Actuar o no sobre la señal?** Meta-labeling (López de Prado): un 2º "
                                "modelo ML filtra las señales primarias malas (tendencia) → menos trades, "
                                "mejor precisión. La probabilidad sirve para dimensionar. Tarda ~1-2 min.")
                    with gr.Row():
                        tme = gr.Textbox(value="AAPL", label="Ticker", scale=3)
                        hme = gr.Dropdown([3, 5, 10], value=5, label="Horizonte (días)")
                        ume = gr.Slider(0.5, 0.7, value=0.55, step=0.01, label="Umbral prob.")
                        pme = gr.Dropdown(["5y", "8y", "10y"], value="8y", label="Histórico")
                        bme = gr.Button("Medir meta-labeling", variant="primary")
                    mdme = gr.Markdown()
                    bme.click(tab_meta, [tme, hme, ume, pme], [mdme])
                with gr.Tab("🧲 RMT (correlación)"):
                    gr.Markdown("**Limpia la matriz de correlación** con Random Matrix Theory (econofísica). "
                                "Marchenko-Pastur separa señal de ruido: solo los autovalores sobre λ+ son "
                                "reales. Mejora la asignación (no optimiza sobre ruido). Compara OOS.")
                    with gr.Row():
                        trm = gr.Textbox(value="AAPL, MSFT, NVDA, GOOGL, AMZN, META, JPM, XOM, KO, WMT, GLD, TLT, XLE, XLF",
                                         label="Activos (coma, ≥4)", scale=4)
                        prm = gr.Dropdown(["3y", "4y", "5y"], value="4y", label="Histórico")
                        brm = gr.Button("Limpiar correlación", variant="primary")
                    mdrm = gr.Markdown()
                    figrm = gr.Plot()
                    brm.click(tab_rmt, [trm, prm], [figrm, mdrm])
                with gr.Tab("🛰️ Kalman (pairs)"):
                    gr.Markdown("**Hedge ratio DINÁMICO** para pairs trading con filtro de Kalman. El β entre "
                                "dos activos deriva en el tiempo; Kalman lo estima día a día (mejor que el β "
                                "fijo OLS). Operas el z-score del spread. Pares clásicos: KO/PEP, EWA/EWC, V/MA.")
                    with gr.Row():
                        tka = gr.Textbox(value="EWA", label="Activo A", scale=2)
                        tkb = gr.Textbox(value="EWC", label="Activo B", scale=2)
                        pka = gr.Dropdown(["3y", "5y", "8y"], value="5y", label="Histórico")
                        bka = gr.Button("Kalman dinámico", variant="primary")
                    mdka = gr.Markdown()
                    figka = gr.Plot()
                    bka.click(tab_kalman, [tka, tkb, pka], [figka, mdka])
                with gr.Tab("📡 Entropía (lead-lag)"):
                    gr.Markdown("**¿Qué activo lidera a cuál?** Entropía de transferencia (teoría de la "
                                "información): flujo de información direccional y NO lineal que la correlación "
                                "no ve. Líderes vs seguidores. Útil para lead-lag y selección de features.")
                    with gr.Row():
                        tte = gr.Textbox(value="SPY, QQQ, TLT, GLD, HYG, XLF, XLE",
                                         label="Activos (coma, ≥3)", scale=4)
                        bte_n = gr.Dropdown([3, 4, 5], value=3, label="Bins")
                        pte = gr.Dropdown(["2y", "3y", "5y"], value="3y", label="Histórico")
                        bte = gr.Button("Medir flujo info", variant="primary")
                    mdte = gr.Markdown()
                    figte = gr.Plot()
                    bte.click(tab_te, [tte, bte_n, pte], [figte, mdte])
        with gr.Tab("🛠️ Análisis"):
            with gr.Tabs():
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
                    t3 = gr.Textbox(value=WL, label="Tickers (espacio/coma)")
                    b3 = gr.Button("Escanear", variant="primary"); tb3 = gr.Dataframe(label="Ranking")
                    b3.click(tab_screener, t3, tb3)
                with gr.Tab("4 · Señales"):
                    t4 = gr.Textbox(value=WL, label="Tickers")
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
        with gr.Tab("📚 Contexto y riesgo"):
            with gr.Tabs():
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
                with gr.Tab("🎯 Alpha (rigor)"):
                    gr.Markdown("**¿Hay ventaja REAL?** Dirección a corto plazo con ML (LightGBM + features "
                                "leak-free + Hurst, walk-forward purgado) + volatilidad GARCH. Mide con test de "
                                "significancia si el modelo bate al azar. Tarda ~1-2 min (reentrena en cada corte).")
                    with gr.Row():
                        ta = gr.Textbox(value="MSFT", label="Ticker", scale=3)
                        ha = gr.Dropdown([3, 5, 10], value=5, label="Días (dirección)")
                        pa = gr.Dropdown(["5y", "8y", "10y"], value="8y", label="Histórico")
                        ba = gr.Button("Medir ventaja", variant="primary")
                    mda = gr.Markdown()
                    ba.click(tab_alpha, [ta, ha, pa], [mda])
                with gr.Tab("📏 Conformal"):
                    gr.Markdown("**Banda CALIBRADA** (split conformal). Arregla la banda de Prophet "
                                "(que solo cubría 14-29% real): aquí el radio se calibra con los errores "
                                "del walk-forward y se MIDE la cobertura. Tarda ~30-60 s.")
                    with gr.Row():
                        tc = gr.Textbox(value="SAB.MC", label="Ticker", scale=3)
                        pc = gr.Dropdown(["3y", "5y", "8y"], value="5y", label="Histórico")
                        bc = gr.Button("Calibrar banda", variant="primary")
                    mdc = gr.Markdown()
                    figc = gr.Plot()
                    tblc = gr.Dataframe(wrap=True)
                    bc.click(tab_conformal, [tc, pc], [figc, tblc, mdc])
                with gr.Tab("🛡️ Riesgo"):
                    gr.Markdown("**Riesgo de cartera**: VaR/CVaR históricos, máximo drawdown y "
                                "correlación. Mide lo que SÍ es estimable (riesgo), no la dirección.")
                    with gr.Row():
                        tr = gr.Textbox(value=WL, label="Watchlist (coma)", scale=3)
                        pr = gr.Dropdown(["1y", "3y", "5y"], value="3y", label="Histórico")
                        cr = gr.Dropdown([0.95, 0.99], value=0.95, label="Confianza VaR")
                        br = gr.Button("Medir riesgo", variant="primary")
                    figr = gr.Plot()
                    with gr.Row():
                        tblr = gr.Dataframe(label="VaR / CVaR / Drawdown", wrap=True)
                        corrr = gr.Dataframe(label="Correlación", wrap=True)
                    br.click(tab_riesgo, [tr, pr, cr], [figr, tblr, corrr])
                with gr.Tab("🔔 Alertas"):
                    gr.Markdown("**Vigilancia de watchlist**: RSI extremo, pico de volatilidad, "
                                "movimiento brusco, cruce de SMA50, proximidad a máx/mín 52s.")
                    with gr.Row():
                        tal = gr.Textbox(value=WL, label="Watchlist (coma)", scale=4)
                        bal = gr.Button("Escanear", variant="primary")
                    mdal = gr.Markdown()
                    tblal = gr.Dataframe(wrap=True)
                    bal.click(tab_alertas, [tal], [tblal, mdal])
                with gr.Tab("🏦 Cartera LP"):
                    gr.Markdown("**Cartera de LARGO PLAZO guiada**: reparte tu capital con HRP "
                                "(asignación robusta), te dice cuántas acciones comprar, y cada mes "
                                "'Revisar rebalanceo' te da las órdenes de ajuste. Dividendos incluidos.")
                    with gr.Row():
                        tlp = gr.Textbox(value="AAPL, MSFT, JPM, KO, GLD, TLT",
                                         label="Activos (coma, ≥3; mezcla bolsa/oro/bonos)", scale=3)
                        clp = gr.Number(value=10000, label="Capital €")
                    with gr.Row():
                        blp1 = gr.Button("1) Crear cartera (hoy)", variant="primary")
                        blp2 = gr.Button("2) Revisar rebalanceo (mensual)", variant="secondary")
                        blp3 = gr.Button("3) He rebalanceado (actualizar)", variant="secondary")
                    mdlp = gr.Markdown()
                    tbllp = gr.Dataframe(wrap=True)
                    blp1.click(tab_cartera_lp_crear, [tlp, clp], [tbllp, mdlp])
                    blp2.click(tab_cartera_lp_revisar, [], [tbllp, mdlp])
                    blp3.click(tab_cartera_lp_aplicar, [], [tbllp, mdlp])
                    gr.Markdown("---\n**¿Aportar cada mes (DCA) o entrar de golpe?** Compara con datos reales.")
                    with gr.Row():
                        tdca = gr.Number(value=12000, label="Dinero total €")
                        adca = gr.Dropdown([3, 5, 8, 10], value=5, label="Años")
                        bdca = gr.Button("Comparar DCA vs entrada única", variant="secondary")
                    mddca = gr.Markdown()
                    figdca = gr.Plot()
                    tbldca = gr.Dataframe(wrap=True)
                    bdca.click(tab_dca, [tlp, tdca, adca], [figdca, tbldca, mddca])
                with gr.Tab("🗞️ Informe semanal"):
                    gr.Markdown("**Tu resumen en un clic**: señales de tu watchlist guardada + "
                                "riesgo de la cesta + titulares recientes → un Word en la raíz "
                                "del proyecto. Tarda ~10-20 s. También por doble clic: Informe_Semanal.bat.")
                    binf = gr.Button("🗞️ Generar informe ahora", variant="primary")
                    mdinf = gr.Markdown()
                    binf.click(tab_informe_semanal, [], [mdinf])
        # guardar watchlist -> actualiza el fichero y los 11 campos que la usan
        wbtn.click(_wl_save, [wtxt], [wmd, t3, t4, tse, trk, tsy, tvb, tsb, tr, tal, tis, tf])
    return app


if __name__ == "__main__":
    build().launch(server_name="127.0.0.1", server_port=7862, inbrowser=False)
