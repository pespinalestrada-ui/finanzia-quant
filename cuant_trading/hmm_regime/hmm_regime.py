"""
hmm_regime — detección de RÉGIMEN de mercado con un Hidden Markov Model gaussiano.

Los mercados cambian de régimen (calma alcista / calma bajista / alta volatilidad)
y una estrategia buena en uno falla en otro. El HMM modela estados ocultos con
transiciones de Markov y emisiones gaussianas, los infiere de los datos y dice en
qué régimen estás HOY → para gatear estrategias (operar tendencia solo en régimen
tendencial, recortar en alta-vol).

Implementación propia en numpy (Baum-Welch + Viterbi, en espacio log), SIN
dependencias nuevas. Features: [retorno, |retorno|] estandarizados (capta dirección
y volatilidad). Estados etiquetados por su media/vol.

No es recomendación de inversión.

Uso:
    python hmm_regime.py SPY
    python hmm_regime.py AAPL --estados 3 --period 8y
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
from scipy.special import logsumexp

np.random.seed(42)


def _log_gauss_diag(X, mean, var):
    """log N(x; mean, diag(var)) por fila. X (T,d), mean (d,), var (d,)."""
    var = np.maximum(var, 1e-6)
    return -0.5 * (np.log(2 * np.pi * var).sum() + ((X - mean) ** 2 / var).sum(axis=1))


class GaussianHMM:
    def __init__(self, n_estados=3, n_iter=30):
        self.K = n_estados
        self.n_iter = n_iter

    def _emis_log(self, X):
        return np.column_stack([_log_gauss_diag(X, self.means[k], self.vars[k]) for k in range(self.K)])

    def fit(self, X):
        from sklearn.cluster import KMeans
        T, d = X.shape
        km = KMeans(self.K, n_init=5, random_state=42).fit(X)
        self.means = km.cluster_centers_.copy()
        self.vars = np.array([X[km.labels_ == k].var(0) + 1e-3 if (km.labels_ == k).sum() > 1
                              else X.var(0) + 1e-3 for k in range(self.K)])
        self.start = np.full(self.K, 1.0 / self.K)
        self.trans = np.full((self.K, self.K), 1.0 / self.K)

        prev = -np.inf
        for _ in range(self.n_iter):
            logB = self._emis_log(X)                       # (T,K)
            # forward-backward en log
            la = np.zeros((T, self.K)); lb = np.zeros((T, self.K))
            la[0] = np.log(self.start + 1e-12) + logB[0]
            ltrans = np.log(self.trans + 1e-12)
            for t in range(1, T):
                la[t] = logB[t] + logsumexp(la[t-1][:, None] + ltrans, axis=0)
            for t in range(T-2, -1, -1):
                lb[t] = logsumexp(ltrans + logB[t+1][None, :] + lb[t+1][None, :], axis=1)
            ll = logsumexp(la[-1])
            gamma = la + lb
            gamma -= logsumexp(gamma, axis=1, keepdims=True)
            g = np.exp(gamma)                              # (T,K) posteriores
            # xi (transiciones)
            xi = np.zeros((self.K, self.K))
            for t in range(T-1):
                m = la[t][:, None] + ltrans + logB[t+1][None, :] + lb[t+1][None, :]
                m -= logsumexp(m)
                xi += np.exp(m)
            # M-step
            self.start = g[0] / g[0].sum()
            self.trans = xi / xi.sum(1, keepdims=True)
            for k in range(self.K):
                w = g[:, k]
                self.means[k] = (w[:, None] * X).sum(0) / w.sum()
                self.vars[k] = (w[:, None] * (X - self.means[k]) ** 2).sum(0) / w.sum() + 1e-4
            if ll - prev < 1e-4:
                break
            prev = ll
        self._logB_cache = self._emis_log(X)
        return self

    def viterbi(self, X):
        T = len(X)
        logB = self._emis_log(X); ltrans = np.log(self.trans + 1e-12)
        delta = np.zeros((T, self.K)); psi = np.zeros((T, self.K), int)
        delta[0] = np.log(self.start + 1e-12) + logB[0]
        for t in range(1, T):
            m = delta[t-1][:, None] + ltrans
            psi[t] = m.argmax(0); delta[t] = m.max(0) + logB[t]
        path = np.zeros(T, int); path[-1] = delta[-1].argmax()
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
        return path


def _etiqueta(media_ret, vol):
    """Nombre legible del régimen por su media de retorno y volatilidad."""
    if vol > 0:  # se decide relativo fuera; aquí texto base
        pass
    return None


def analizar(ticker, n_estados=3, period="8y"):
    h = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if h.empty:
        raise ValueError(f"'{ticker}' sin datos.")
    c = h["Close"].astype(float)
    r = np.log(c / c.shift(1)).dropna()
    # features rolling → regímenes PERSISTENTES (no ruido diario): tendencia 5d + vol 10d
    feat = pd.DataFrame({"trend": r.rolling(5).mean(), "vol": r.rolling(10).std()}).dropna()
    fechas = feat.index
    rr = r.reindex(fechas).values
    X = feat.values
    X = (X - X.mean(0)) / X.std(0)
    hmm = GaussianHMM(n_estados).fit(X)
    path = hmm.viterbi(X)

    # estadística por régimen (retornos reales)
    info = []
    for k in range(n_estados):
        mask = path == k
        if mask.sum() == 0:
            continue
        info.append({"estado": k, "n": int(mask.sum()),
                     "ret_dia_pct": float(rr[mask].mean() * 100),
                     "vol_anual": float(rr[mask].std() * np.sqrt(252) * 100),
                     "frac": float(mask.mean())})
    df = pd.DataFrame(info)
    # Sharpe anualizado del régimen (señal de calidad, ya tradeable)
    df["sharpe"] = (df["ret_dia_pct"] / 100 * 252) / (df["vol_anual"] / 100).replace(0, np.nan)
    nombres = {}
    df_sorted = df.sort_values("sharpe", ascending=False).reset_index()
    for rank, row in df_sorted.iterrows():
        if rank == 0:
            nombres[int(row["estado"])] = "🟢 Alcista/calma"
        elif row["vol_anual"] == df["vol_anual"].max():
            nombres[int(row["estado"])] = "🔴 Alta volatilidad/estrés"
        else:
            nombres[int(row["estado"])] = "🟡 Bajista/lateral"
    df["Régimen"] = df["estado"].map(nombres)

    estado_hoy = int(path[-1])
    # persistencia esperada del régimen actual (1/(1-p_ii))
    p_ii = hmm.trans[estado_hoy, estado_hoy]
    dur = 1.0 / (1 - p_ii) if p_ii < 1 else float("inf")
    res = {"ticker": ticker.upper(), "df": df, "nombres": nombres, "path": path,
           "precio": c, "fechas": fechas, "estado_hoy": estado_hoy,
           "regimen_hoy": nombres.get(estado_hoy, f"Estado {estado_hoy}"),
           "duracion_esperada": dur, "trans": hmm.trans}
    return res


def _plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    c = res["precio"].reindex(res["fechas"])
    cols = {"🟢 Alcista/calma": "tab:green", "🔴 Alta volatilidad/estrés": "tab:red",
            "🟡 Bajista/lateral": "tab:orange"}
    for k, nom in res["nombres"].items():
        m = res["path"] == k
        ax.scatter(np.array(res["fechas"])[m], c.values[m], s=4,
                   color=cols.get(nom, "gray"), label=nom)
    ax.set_title(f"{res['ticker']} · régimen HMM (hoy: {res['regimen_hoy']})")
    ax.set_xlabel("Fecha"); ax.set_ylabel("Precio"); ax.legend(loc="best", markerscale=3, fontsize=8)
    fig.tight_layout()
    return fig


def informe(res):
    L = [f"=== Régimen de mercado (HMM) · {res['ticker']} ===\n",
         f"  RÉGIMEN HOY : {res['regimen_hoy']}  · duración esperada ~{res['duracion_esperada']:.0f} sesiones\n"]
    d = res["df"][["Régimen", "frac", "ret_dia_pct", "vol_anual", "sharpe"]].copy()
    d["frac"] = (d["frac"] * 100).round(0).astype(int).astype(str) + "%"
    d["ret_dia_pct"] = d["ret_dia_pct"].round(3).astype(str) + "%"
    d["vol_anual"] = d["vol_anual"].round(1).astype(str) + "%"
    d["sharpe"] = d["sharpe"].round(2)
    d.columns = ["Régimen", "% tiempo", "Ret/día", "Vol anual", "Sharpe"]
    L.append(d.to_string(index=False))
    L.append("\n> Úsalo como GATE: opera tendencia en 🟢, recorta tamaño / evita en 🔴.")
    L.append("> HMM gaussiano (Baum-Welch). No es recomendación de inversión.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Detección de régimen de mercado con HMM gaussiano.")
    ap.add_argument("ticker", nargs="?", default="SPY")
    ap.add_argument("--estados", type=int, default=3)
    ap.add_argument("--period", default="8y")
    a = ap.parse_args()
    res = analizar(a.ticker, a.estados, a.period)
    print("\n" + informe(res) + "\n")


if __name__ == "__main__":
    main()
