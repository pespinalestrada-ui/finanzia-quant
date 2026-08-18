"""
tanda_programada — una tanda de 10 configuraciones, pensada para el programador
de tareas de Windows (cada 5 horas).

Qué hace cada vez:
  1. Prueba 10 configuraciones nuevas y las suma al recuento acumulado.
  2. Rehace los perfiles elegibles con su validación en un universo independiente.
  3. Escribe una línea de veredicto en el registro.

Por qué el recuento acumulado importa AQUÍ más que en ninguna parte: a 10 por
tanda y 5 tandas al día, en una semana van 350 configuraciones probadas. Con ese
número, encontrar una que parezca buena por azar deja de ser posible y pasa a ser
seguro. Lo único que impide que eso se convierta en un autoengaño es que el
listón (Deflated Sharpe y Šidák) suba con cada prueba, y que la palabra final la
tenga la validación en valores que no participaron en elegir nada.

Registro: data/tune_veredicto.log
Quitar la tarea:  schtasks /Delete /TN "FinanzIA - busqueda Veredicto" /F
"""
import sys
import traceback
from datetime import datetime
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent / "veredicto_backtest"))
sys.path.insert(0, str(AQUI.parent / "indicators"))

LOG = AQUI.parent.parent / "data" / "tune_veredicto.log"


def apunta(linea):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(linea.rstrip() + "\n")
    print(linea)


def main():
    sello = datetime.now().strftime("%Y-%m-%d %H:%M")
    try:
        import numpy as np
        import veredicto_tune as VT

        res = VT.tanda(n=10)
        est = VT.cargar_estado()
        n_tot = len(est["pruebas"])

        mejor = res.get("mejor") or {}
        apunta(f"[{sello}] tanda {est['tandas']} · {n_tot} configuraciones acumuladas "
               f"· mejor de la tanda: Sharpe {mejor.get('sr_oos', float('nan')):+.2f} "
               f"· PBO {res.get('pbo', float('nan'))*100:.0f}% "
               f"· listón por azar {res.get('sr0', float('nan')):+.3f}")

        # el marcador resuelve aqui lo que haya vencido: asi no depende de que
        # abras la pestaña para ponerse al dia
        try:
            sys.path.insert(0, str(AQUI.parent / "marcador"))
            import marcador as MC
            n_res = MC.resolver()
            if n_res:
                apunta(f"[{sello}] marcador: {n_res} plazos resueltos")
        except Exception as e:
            apunta(f"[{sello}] marcador no disponible: {str(e)[:60]}")

        # los perfiles se rehacen para que el desplegable del panel no se quede viejo
        perfiles = VT.construir_perfiles(top=5)
        validas = [p for p in perfiles if p["etiqueta"].startswith("✅") and p.get("busqueda")]
        if validas:
            v = validas[0]
            apunta(f"[{sello}] *** CANDIDATA QUE AGUANTA FUERA: {v['nombre']} · "
                   f"acierto fuera {v['validacion']['tasa']*100:.1f}% · "
                   f"Sharpe {v['validacion']['sharpe']:+.2f}. REVÍSALA A MANO antes de usarla.")
        else:
            peor = min((p["validacion"]["tasa"] for p in perfiles
                        if p.get("busqueda") and p.get("validacion")), default=float("nan"))
            apunta(f"[{sello}] ninguna aguanta fuera de su universo (la peor baja al "
                   f"{peor*100:.1f}% de acierto). Sigue mandando la configuración ACTUAL.")
    except Exception:
        apunta(f"[{sello}] ERROR en la tanda:\n{traceback.format_exc()}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
