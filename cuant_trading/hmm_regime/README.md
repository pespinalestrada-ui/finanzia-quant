# hmm_regime

Detección de **régimen de mercado** con un Hidden Markov Model gaussiano. Los
mercados cambian de régimen (calma alcista / alta volatilidad / lateral) y una
estrategia buena en uno falla en otro. El HMM infiere los estados ocultos y dice en
cuál estás **hoy** → úsalo como **gate**: opera tendencia en 🟢, recorta/evita en 🔴.

- Implementación propia en **numpy** (Baum-Welch + Viterbi, espacio log) → **sin
  dependencias nuevas**.
- Features rolling (tendencia 5d + volatilidad 10d) → regímenes persistentes.
- Estados etiquetados por su retorno/volatilidad.

## Uso
```bash
python hmm_regime.py SPY
python hmm_regime.py AAPL --estados 3 --period 8y
```
En el dashboard: pestaña **🌀 Régimen (HMM)**.

## Resultado típico (SPY)
🟢 Calma alcista (~47% del tiempo, Sharpe alto) · 🔴 Alta volatilidad (~15%, vol ~36%,
retorno negativo) · 🟡 Lateral. Más la duración esperada del régimen actual.
No es recomendación de inversión.
