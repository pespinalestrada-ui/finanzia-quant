# meta_labeling

**Meta-etiquetado** (López de Prado): un 2º modelo decide **SI actuar** sobre una
señal primaria. El modelo PRIMARIO da el lado (aquí: seguir tendencia, largo si
SMA20>SMA50) — recall alto, precisión baja. El SECUNDARIO (ML) aprende a distinguir
las señales primarias buenas de las malas y filtra → **menos operaciones, mejor
precisión**. La probabilidad también sirve para **dimensionar** la apuesta.

- Features leak-free de `alpha_forecast`. Clasificador **LightGBM**.
- Validación **walk-forward purgada** con embargo (sin leakage).
- Compara primario solo vs meta-filtrado (precisión, retorno/trade, nº señales).

## Uso
```bash
python meta_labeling.py AAPL
python meta_labeling.py SPY --horizon 5 --umbral 0.55 --period 8y
```
En el dashboard: pestaña **🎯 Meta-labeling**.

## Honesto
A veces aporta (p.ej. AAPL: precisión 54%→56% con menos trades), a veces no
(p.ej. NVDA: AUC<0.5, el secundario no distingue). La herramienta lo MIDE — no asume
que siempre mejora. No es recomendación de inversión.
