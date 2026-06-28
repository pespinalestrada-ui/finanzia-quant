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
          SUITE / "signal_engine", SUITE / "risk_manager"]:
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

def _dl(ticker, period="1y"):
    key = (ticker.upper(), period)
    hit = _DL_CACHE.get(key)
    if hit and (_time.time() - hit[0]) < _DL_TTL:
        return hit[1].copy()
    h = yf.Ticker(ticker).history(period=period, auto_adjust=False)
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


def tab_veredicto(ticker, period, con_sentimiento, con_modelos=False, cripto=False):
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

        # Ensemble ponderado por Skill OOS (1/MAPE)
        if len(modelos) == 1:
            prophet_score = max(-1.0, min(1.0, prophet_var / 10.0)) * conf / 100.0
            s_fc = prophet_score
            lect_fc = f"{prophet_var:+.1f} % · confianza {conf_str}"
        else:
            pesos = []
            var_pond = []
            for n, v, mape in modelos:
                peso = 1.0 / max(0.5, mape)  # evitar división por cero
                pesos.append(peso)
                var_pond.append(v * peso)
            
            suma_pesos = sum(pesos)
            var_ensemble = sum(var_pond) / suma_pesos
            
            s_fc = max(-1.0, min(1.0, var_ensemble / 10.0)) * conf / 100.0
            lect_fc = " · ".join(f"{n} {v:+.1f}%" for n, v, _ in modelos) + f" → Ensemble: {var_ensemble:+.1f}%"
            notas_modelos = f"\n\n**Ensemble OOS ({len(modelos)} modelos):** Ponderado inversamente por el error reciente de cada modelo (1/MAPE). Predicción combinada direccional: {var_ensemble:+.2f}%."
        pilares.append(("Forecast 90d" + (" (OOS Ensemble)" if len(modelos) > 1 else " (Prophet)"), lect_fc, s_fc, 0.30))

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

        # Filtro de Régimen: bloquea señales en mercados laterales
        en_rango = False
        razon_rango = ""
        if not np.isnan(hurst_v) and 0.40 < hurst_v < 0.55 and adx_v < 25:
            en_rango = True
            razon_rango = f"Filtro de Régimen ACTIVO: Hurst={hurst_v:.2f} (paseo aleatorio) y ADX={adx_v:.0f} (<25). "

        if en_rango:
            verd, emoji = "MANTENER", "🟡"
            notas_modelos += f"\n\n> ⚠️ **{razon_rango}** Forzando veredicto a MANTENER por falta de tendencia direccional."
        elif total >= 0.35:
            verd, emoji = "COMPRAR", "🟢"
        elif total <= -0.35:
            verd, emoji = "VENDER", "🔴"
        else:
            verd, emoji = "MANTENER", "🟡"

        # --- Auto-Logging -----------------------------------------------------
        nota_log = ""
        if verd in ["COMPRAR", "VENDER"]:
            try:
                # Usa posición base 10k, objetivo 15% volatilidad
                vol_anual = PS.garch_volatility(ticker)
                atr, px_atr = PS.atr_actual(ticker)
                stop = px - 2 * atr if verd == "COMPRAR" else px + 2 * atr
                r_accion = abs(px - stop)
                
                if not np.isnan(vol_anual) and vol_anual > 0:
                    weight = 0.15 / vol_anual
                    coste_obj = 10000 * weight
                    shares = max(1, int(coste_obj // px))
                else:
                    riesgo_eur = 10000 * 0.01
                    shares = max(1, int(riesgo_eur // r_accion))
                
                nid = JR.abrir(ticker, px, stop, shares, f"Auto-Veredicto: {verd} (Score {total:+.2f})", "SHORT" if verd == "VENDER" else "LONG", factor_score_val)
                nota_log = f"\n\n✅ **Auto-Logging:** Operación #{nid} registrada automáticamente en el Diario (simulada) para medir expectancy."
            except Exception as ej:
                nota_log = f"\n\n⚠️ Error al auto-registrar en Diario: {ej}"

        tabla = pd.DataFrame(
            [{"Pilar": n, "Lectura": l, "Score": round(s, 2),
              "Peso": f"{w/wsum*100:.0f}%", "Aporte": round(s * w / wsum, 3)}
             for n, l, s, w in pilares]
        )

        extras = []
        if not con_modelos:
            extras.append("consenso multi-modelo OFF")
        if not con_sentimiento:
            extras.append("sentimiento OFF")
        md = (f"# {emoji} {verd}" + ("  🪙 *cripto (forecast diario 7d)*" if cripto else "") + "\n\n"
              f"**{ticker}** · precio {px:.3f} · score total **{total:+.3f}** "
              f"(umbral: ≥+0.35 comprar · ≤−0.35 vender)\n\n"
              f"{len(pilares)} pilares"
              + (f" · *({', '.join(extras)})*" if extras else "")
              + notas_modelos
              + nota_log
              + "\n\n> ⚠️ Estimación estadística automática (forecast + batería técnica + volumen"
              + (" + sentimiento" if con_sentimiento else "")
              + "). **NO es recomendación de inversión.** Resumen, no orden.")
        return fig, tabla, md
    except Exception as e:
        return _err_fig(f"Error: {e}"), pd.DataFrame(), f"**Error:** {e}"


def tab_veredicto_cripto(ticker, period, con_sentimiento, con_modelos=False):
    """Veredicto para criptomonedas: mismo agregador con forecast diario 7d,
    momentum en días naturales y pilar de Fear & Greed cripto."""
    return tab_veredicto(ticker, period, con_sentimiento, con_modelos, cripto=True)


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


def tab_intraday_backtest(ticker, interval, or_min, coste_bps):
    try:
        import intraday as IN
        df = IN.descargar(ticker.strip().upper(), interval)
        bt = IN.backtest_orb(df, int(or_min), interval, float(coste_bps))
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


def tab_intraday_scan(txt, interval, or_min, coste_bps):
    try:
        import intraday as IN
        tickers = _parse(txt)
        if len(tickers) < 2:
            return pd.DataFrame(), "Mete **al menos 2** tickers para escanear."
        tabla = IN.escanear(tickers, interval, int(or_min), float(coste_bps))
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


# ---- UI -------------------------------------------------------------------
def build():
    import gradio as gr
    # head: bloquea el traductor automático de Chrome (rompe la reactividad de Gradio)
    _head = '<meta name="google" content="notranslate"><script>document.documentElement.lang="es";</script>'
    with gr.Blocks(title="FinanzIA — Mesa cuantitativa", head=_head) as app:
        gr.Markdown("# FinanzIA — Mesa cuantitativa")
        gr.Markdown("Suite de trading algorítmico. Datos Yahoo Finance (retardo ~15 min). "
                    "Análisis y educación — **no es recomendación de inversión**.")
        with gr.Tab("📊 Factores"):
            gr.Markdown("**Modelo multi-factor (smart beta)** — cómo deciden los fondos cuant "
                        "qué comprar. Rankea un universo por **value + momentum + quality + low-vol** "
                        "(z-score cruzado). Compra el top, evita el fondo. Tarda ~1-3 s/acción "
                        "(descarga fundamentales).")
            with gr.Row():
                tf = gr.Textbox(value="AAPL, MSFT, NVDA, JPM, XOM, KO, SAB.MC, ITX.MC",
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
                bv = gr.Button("Analizar TODO", variant="primary")
            with gr.Row():
                mv = gr.Checkbox(value=False, label="Consenso multi-modelo (LSTM + pesados si están, +tiempo)")
                sv = gr.Checkbox(value=False, label="Incluir sentimiento FinBERT (1ª vez +1 min)")
            mdv = gr.Markdown()
            tbv = gr.Dataframe(label="Desglose por pilar", wrap=True)
            plv = gr.Plot(label="Forecast 30/90/120d")
            bv.click(tab_veredicto, [tv, pv, sv, mv], [plv, tbv, mdv])
        with gr.Tab("🪙 Veredicto Cripto"):
            gr.Markdown("**Veredicto para criptomonedas** (BTC-USD, ETH-EUR, SOL-USD…). "
                        "Mismo agregador pero con **forecast diario 7d**, momentum en días "
                        "naturales y pilar de **Fear & Greed cripto** (contrarian) → "
                        "**COMPRAR / MANTENER / VENDER**. Usa tickers Yahoo tipo `BTC-USD`.")
            with gr.Row():
                tvc = gr.Textbox(value="BTC-USD", label="Ticker cripto (BASE-FIAT)", scale=3)
                pvc = gr.Dropdown(["1y", "2y", "3y", "5y"], value="3y", label="Histórico")
                bvc = gr.Button("Analizar cripto", variant="primary")
            with gr.Row():
                mvc = gr.Checkbox(value=False, label="Consenso multi-modelo (LSTM…, +tiempo)")
                svc = gr.Checkbox(value=False, label="Incluir sentimiento FinBERT (1ª vez +1 min)")
            mdvc = gr.Markdown()
            tbvc = gr.Dataframe(label="Desglose por pilar", wrap=True)
            plvc = gr.Plot(label="Forecast cripto 30/90/120d")
            bvc.click(tab_veredicto_cripto, [tvc, pvc, svc, mvc], [plvc, tbvc, mdvc])
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
            with gr.Row():
                bi1 = gr.Button("📷 Snapshot de hoy", variant="secondary")
                bi2 = gr.Button("🧪 Backtest ORB con costes", variant="primary")
            mdi = gr.Markdown()
            figi = gr.Plot()
            tbli = gr.Dataframe(wrap=True)
            bi1.click(tab_intraday_snapshot, [ti, ii, ori], [figi, tbli, mdi])
            bi2.click(tab_intraday_backtest, [ti, ii, ori, ci], [tbli, mdi])
            gr.Markdown("---\n**Escaneo multi-ticker**: ¿dónde (si en algún sitio) sobrevive el edge al coste?")
            with gr.Row():
                tis = gr.Textbox(value="AAPL, MSFT, NVDA, TSLA, JPM, SAB.MC",
                                 label="Universo (coma)", scale=4)
                bi3 = gr.Button("🔭 Escanear varios (rank por exp. neta)", variant="primary")
            bi3.click(tab_intraday_scan, [tis, ii, ori, ci], [tbli, mdi])
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
        with gr.Tab("📡 Señales"):
            gr.Markdown("**Qué operar hoy**: corre el score del Veredicto sobre tu watchlist → "
                        "ranking COMPRAR/MANTENER/VENDER. Entrada del bucle de ejecución (sizing "
                        "→ Alpaca paper → diario). Rápido (sin Prophet).")
            with gr.Row():
                tse = gr.Textbox(value="AAPL, MSFT, NVDA, GOOGL, AMZN, META, JPM, XOM, KO, WMT",
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
                trk = gr.Textbox(value="AAPL, MSFT, NVDA, GOOGL, AMZN, META, JPM, XOM, KO, WMT",
                                 label="Watchlist (coma)", scale=3)
                crk = gr.Number(value=10000, label="Capital €")
                vrk = gr.Slider(0.08, 0.30, value=0.15, step=0.01, label="Vol objetivo")
                mrk = gr.Dropdown([3, 4, 5, 6, 8], value=5, label="Máx posiciones")
                urk = gr.Slider(0.1, 0.6, value=0.30, step=0.05, label="Umbral señal")
                brk = gr.Button("Generar plan", variant="primary")
            mdrk = gr.Markdown()
            tbrk = gr.Dataframe(label="Plan de órdenes", wrap=True)
            brk.click(tab_plan_riesgo, [trk, crk, vrk, mrk, urk], [tbrk, mdrk])
        with gr.Tab("🔬 Validar Veredicto"):
            gr.Markdown("**¿El Veredicto predice de verdad?** Backtest honesto del score técnico "
                        "point-in-time: **IC** (score↔retorno futuro), retornos por **quintil**, "
                        "Sharpe long-short y **Deflated Sharpe** (corrige multiple-testing). "
                        "El paso obligatorio antes de automatizar. Tarda ~1 min.")
            with gr.Row():
                tvb = gr.Textbox(value="AAPL, MSFT, NVDA, GOOGL, AMZN, META, JPM, XOM, KO, WMT",
                                 label="Universo (coma)", scale=4)
                hvb = gr.Dropdown([5, 10, 20], value=10, label="Horizonte (días)")
                trvb = gr.Number(value=20, label="Nº pruebas (deflación)")
                bvb = gr.Button("Validar", variant="primary")
            mdvb = gr.Markdown()
            bvb.click(tab_validar_veredicto, [tvb, hvb, trvb], [mdvb])
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
                tr = gr.Textbox(value="SAB.MC, BBVA.MC, IBE.MC", label="Watchlist (coma)", scale=3)
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
                tal = gr.Textbox(value="SAB.MC, BBVA.MC, AAPL, MSFT, NVDA", label="Watchlist (coma)", scale=4)
                bal = gr.Button("Escanear", variant="primary")
            mdal = gr.Markdown()
            tblal = gr.Dataframe(wrap=True)
            bal.click(tab_alertas, [tal], [tblal, mdal])
    return app


if __name__ == "__main__":
    build().launch(server_name="127.0.0.1", server_port=7862, inbrowser=False)
