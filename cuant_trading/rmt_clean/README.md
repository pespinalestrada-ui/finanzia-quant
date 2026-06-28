# rmt_clean

Limpieza de la matriz de correlación con **Random Matrix Theory** (econofísica:
Bouchaud, Potters). La correlación de N activos con T datos está dominada por **ruido**.

Marchenko-Pastur: sin señal, los autovalores caerían en el bulk [λ−, λ+] con
`λ± = (1 ± √(N/T))²`. Los que están **por encima de λ+** son señal (modos de
mercado/sector); el resto es ruido. Se aplanan los del bulk a su media → correlación
estable para construir cartera.

Compara min-variance con correlación **cruda vs limpia (RMT) vs Ledoit-Wolf**, fuera
de muestra, y dibuja el espectro de autovalores vs la densidad de Marchenko-Pastur.

## Uso
```bash
python rmt_clean.py AAPL MSFT NVDA GOOGL AMZN META JPM XOM KO WMT GLD TLT
```
En el dashboard: pestaña **🧲 RMT (correlación)**.

## Nota honesta
La ventaja de RMT crece cuando N/T es grande (muchos activos, pocos datos), donde la
matriz cruda es más ruidosa. Con pocos activos y mucho histórico aporta menos. El
diagnóstico (cuántos autovalores son señal) ya es valioso. No es recomendación.
