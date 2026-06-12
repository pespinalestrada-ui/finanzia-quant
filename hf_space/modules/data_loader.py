"""
Descarga y caché local de datos financieros.

Usa yfinance para SAB.MC y pares macro. Cachea en data/ para no golpear
Yahoo en cada ejecución.
"""

from pathlib import Path
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def get_ticker_history(ticker: str, period: str = "5y", force_refresh: bool = False) -> pd.DataFrame:
    """
    Descarga (o lee de caché local) el histórico de un ticker de Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Símbolo de Yahoo Finance (ej. 'SAB.MC', '^IBEX', 'EURUSD=X').
    period : str
        Periodo soportado por yfinance: '1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max'.
    force_refresh : bool
        Si True, ignora la caché y vuelve a descargar.

    Returns
    -------
    pd.DataFrame con columnas [Date, Open, High, Low, Close, Volume, Dividends, Stock Splits]
    """
    safe_name = ticker.replace("^", "").replace("=", "_").replace(".", "_")
    cache_path = DATA_DIR / f"{safe_name}_{period}.csv"

    if cache_path.exists() and not force_refresh:
        df = pd.read_csv(cache_path, parse_dates=["Date"])
        return df

    t = yf.Ticker(ticker)
    hist = t.history(period=period, auto_adjust=False)
    hist = hist.reset_index()
    # Normalizar zona horaria a naive UTC
    if pd.api.types.is_datetime64tz_dtype(hist["Date"]):
        hist["Date"] = hist["Date"].dt.tz_localize(None)
    hist["Date"] = pd.to_datetime(hist["Date"]).dt.date
    hist["Date"] = pd.to_datetime(hist["Date"])
    hist.to_csv(cache_path, index=False)
    return hist


def get_sab(period: str = "5y", force_refresh: bool = False) -> pd.DataFrame:
    """Atajo para Banco Sabadell (BME)."""
    return get_ticker_history("SAB.MC", period=period, force_refresh=force_refresh)


def get_ibex(period: str = "5y", force_refresh: bool = False) -> pd.DataFrame:
    """Atajo para IBEX 35."""
    return get_ticker_history("^IBEX", period=period, force_refresh=force_refresh)


def get_eurusd(period: str = "5y", force_refresh: bool = False) -> pd.DataFrame:
    """Atajo para EUR/USD."""
    return get_ticker_history("EURUSD=X", period=period, force_refresh=force_refresh)


# Hitos clave para la OPA hostil de BBVA sobre Sabadell
OPA_BBVA_EVENTS = pd.DataFrame({
    "holiday": "opa_bbva",
    "ds": pd.to_datetime([
        "2024-04-30",  # BBVA presenta oferta amistosa
        "2024-05-09",  # SAB rechaza la oferta
        "2024-05-09",  # BBVA convierte la oferta en OPA hostil
        "2024-09-04",  # BdE autoriza la OPA
        "2025-05-06",  # CNMC autoriza con compromisos
        "2025-07-22",  # Gobierno impone condiciones adicionales
        "2025-10-10",  # Fin del periodo de aceptación (resultado)
    ]),
    "lower_window": 0,
    "upper_window": 2,  # efecto persiste un par de días tras la noticia
})
