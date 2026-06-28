"""
meta_labeling — meta-etiquetado (López de Prado): un 2º modelo decide SI actuar.

Idea: un modelo PRIMARIO da el LADO (aquí: seguir tendencia, largo si SMA20>SMA50).
Suele tener recall alto pero precisión baja (muchas señales falsas). Un modelo
SECUNDARIO (ML) aprende a distinguir las señales primarias BUENAS de las malas y
filtra → menos operaciones, pero mejor precisión y mayor retorno por trade. También
da la PROBABILIDAD para dimensionar la apuesta (bet sizing).

Valida con walk-forward PURGADO (embargo del horizonte, sin leakage) y compara:
  - Primario solo (todas las señales) vs Meta-filtrado (solo prob>umbral).

Features leak-free de alpha_forecast. Clasificador LightGBM. No es recomendación.

Uso:
    python meta_labeling.py AAPL
    python meta_labeling.py SPY --horizon 5 --umbral 0.55 --period 8y
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

_SUITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SUITE / "alpha_forecast"))
import alpha_forecast as AF


def backtest(ticker, horizon=5, umbral=0.55, period="8y", step=5, train_min=500):
    import lightgbm as lgb
    df = AF.descargar(ticker, period)
    f, r = AF.features(df)
    c = df["Close"]
    sma20, sma50 = c.rolling(20).mean(), c.rolling(50).mean()
    primario_long = (sma20 > sma50)                          # señal primaria: tendencia
    fwd = c.shift(-horizon) / c - 1.0
    meta_y = (fwd > 0).astype(int)                            # ¿la señal primaria acierta?

    data = f.copy()
    data["y"] = meta_y; data["fwd"] = fwd; data["prim"] = primario_long.astype(int)
    data = data.dropna()
    X = data.drop(columns=["y", "fwd", "prim"])
    Y, FWD, PRIM = data["y"], data["fwd"], data["prim"]
    n = len(X)
    if n < train_min + horizon + 50:
        return None

    probs, reales, fwds = [], [], []
    t = train_min
    while t < n - 1:
        tr_end = t - horizon                                 # embargo (purga etiquetas que miran al test)
        if tr_end < 100:
            t += step; continue
        # entrena el secundario SOLO con señales primarias del pasado
        mask_tr = PRIM.iloc[:tr_end] == 1
        Xtr, Ytr = X.iloc[:tr_end][mask_tr.values], Y.iloc[:tr_end][mask_tr.values]
        if len(Xtr) < 80 or Ytr.nunique() < 2:
            t += step; continue
        # test: solo días con señal primaria activa
        sl = slice(t, t + step)
        prim_te = PRIM.iloc[sl] == 1
        Xte = X.iloc[sl][prim_te.values]
        if Xte.empty:
            t += step; continue
        clf = lgb.LGBMClassifier(n_estimators=120, num_leaves=15, max_depth=4,
                                 learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                                 min_child_samples=30, reg_lambda=1.0, random_state=42, verbosity=-1)
        clf.fit(Xtr, Ytr)
        p = clf.predict_proba(Xte)[:, 1]
        probs.extend(p)
        reales.extend(Y.iloc[sl][prim_te.values].values)
        fwds.extend(FWD.iloc[sl][prim_te.values].values)
        t += step

    probs, reales, fwds = np.array(probs), np.array(reales), np.array(fwds)
    if len(probs) < 30:
        return None
    # Primario solo (todas las señales de tendencia)
    prec_prim = float(reales.mean())
    ret_prim = float(fwds.mean()) * 100
    # Meta-filtrado (solo prob>umbral)
    sel = probs > umbral
    if sel.sum() < 5:
        return {"n_prim": len(probs), "prec_prim": prec_prim, "ret_prim": ret_prim,
                "mensaje": "El secundario filtró casi todo; baja el umbral."}
    prec_meta = float(reales[sel].mean())
    ret_meta = float(fwds[sel].mean()) * 100
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(reales, probs)) if len(set(reales)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    return {"horizon": horizon, "umbral": umbral, "auc": auc,
            "n_prim": len(probs), "prec_prim": prec_prim, "ret_prim": ret_prim,
            "n_meta": int(sel.sum()), "prec_meta": prec_meta, "ret_meta": ret_meta,
            "mejora_prec": (prec_meta - prec_prim) * 100,
            "mejora_ret": ret_meta - ret_prim}


def informe(ticker, res):
    if res is None:
        return f"{ticker}: histórico insuficiente para el meta-etiquetado."
    if "mensaje" in res:
        return f"{ticker}: {res['mensaje']}"
    util = res["mejora_prec"] > 1 and res["mejora_ret"] > 0
    L = [f"=== Meta-labeling · {ticker} · horizonte {res['horizon']}d · umbral {res['umbral']} ===\n",
         f"  AUC del secundario        : {res['auc']:.3f}  (0.5 = no distingue)\n",
         f"  {'':<22}{'PRIMARIO':>10}{'META-FILTRADO':>15}",
         f"  {'Nº señales':<22}{res['n_prim']:>10}{res['n_meta']:>15}",
         f"  {'Precisión (aciertos)':<22}{res['prec_prim']*100:>9.1f}%{res['prec_meta']*100:>14.1f}%",
         f"  {'Retorno medio/trade':<22}{res['ret_prim']:>+9.2f}%{res['ret_meta']:>+14.2f}%",
         f"\n  Mejora con meta : precisión {res['mejora_prec']:+.1f}pp · retorno {res['mejora_ret']:+.2f}%/trade",
         f"  → El meta-labeling {'APORTA ✅ (menos trades, mejores)' if util else 'no aporta claramente aquí'}."]
    L.append("\n> Primario = tendencia (SMA20>SMA50). Secundario = LightGBM filtra las señales malas.")
    L.append("> Walk-forward purgado con embargo. La probabilidad sirve también para dimensionar. No es recomendación.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Meta-labeling (López de Prado): filtra la señal primaria con ML.")
    ap.add_argument("tickers", nargs="*", default=["AAPL"])
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--umbral", type=float, default=0.55)
    ap.add_argument("--period", default="8y")
    a = ap.parse_args()
    for tk in (a.tickers or ["AAPL"]):
        res = backtest(tk.upper(), a.horizon, a.umbral, a.period)
        print("\n" + informe(tk.upper(), res) + "\n")


if __name__ == "__main__":
    main()
