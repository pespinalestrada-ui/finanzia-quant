"""
actualizar_space.py — sube el dashboard al Space y verifica que queda LIVE.

Hace todo por API (fiable), sin depender de ningún navegador:
  1. Pide tu token Write (no se guarda).
  2. Sube app.py + modules cambiados forzando un commit NUEVO (marca de tiempo
     única) para que HF reconstruya seguro y rompa cualquier caché.
  3. Lanza un factory reboot (reconstruye la imagen desde cero).
  4. Espera a que el Space esté RUNNING y comprueba que el código nuevo está vivo
     (busca el checkbox "Consenso multi-modelo" en la config real de la app).
  5. Te dice si quedó bien y te recuerda revocar el token.

Uso:
    cd hf_space
    python actualizar_space.py
Token Write: https://huggingface.co/settings/tokens  (tipo Write)
"""
import sys
import time
from datetime import datetime, timezone
from getpass import getpass
from pathlib import Path

USUARIO = "Pespinal0312"
SPACE = f"{USUARIO}/finanzia-dashboard"
HERE = Path(__file__).resolve().parent


def main():
    print("=" * 62)
    print("  Actualizar y verificar el Space FinanzIA (vía API)")
    print("=" * 62)
    print(f"\nSpace: {SPACE}")
    print("Token Write: https://huggingface.co/settings/tokens (tipo Write)\n")

    token = getpass("Pega tu token (no se ve al escribir) + Enter: ").strip()
    if not token.startswith("hf_"):
        print("\n[X] Eso no parece un token (empiezan por 'hf_'). Abortado.")
        sys.exit(1)

    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi(token=token)
    try:
        quien = api.whoami()["name"]
    except Exception as e:
        print(f"\n[X] Token invalido: {e}")
        sys.exit(1)
    print(f"\n[OK] Autenticado como: {quien}")

    # --- 1. preparar app.py con marca unica (garantiza commit nuevo) ---
    app_local = (HERE / "app.py").read_text(encoding="utf-8")
    if "Consenso multi-modelo" not in app_local:
        print("[X] El app.py local NO tiene la version nueva. Aborto por seguridad.")
        sys.exit(1)
    sello = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    app_marcado = app_local.rstrip() + f"\n# deploy: {sello}\n"

    ops = [CommitOperationAdd(path_in_repo="app.py", path_or_fileobj=app_marcado.encode("utf-8"))]
    # TODOS los modulos, no una lista fija: al añadir el tema Nocturne
    # (finanzia_theme / finanzia_charts) una lista cerrada dejaba fuera ficheros
    # nuevos y el Space arrancaba con ImportError.
    for p in sorted((HERE / "modules").glob("*.py")):
        ops.append(CommitOperationAdd(path_in_repo=f"modules/{p.name}",
                                      path_or_fileobj=str(p)))
    for extra in ("requirements.txt", "README.md"):
        p = HERE / extra
        if p.exists():
            ops.append(CommitOperationAdd(path_in_repo=extra, path_or_fileobj=str(p)))

    print(f"[..] Subiendo {len(ops)} ficheros (commit forzado: {sello}) ...")
    api.create_commit(repo_id=SPACE, repo_type="space", operations=ops,
                      commit_message=f"Veredicto multi-modelo + bateria tecnica ({sello})")
    print("[OK] Commit subido.")

    # --- 2. factory reboot (imagen desde cero) ---
    print("[..] Lanzando factory reboot (reconstruccion limpia) ...")
    api.restart_space(repo_id=SPACE, factory_reboot=True)

    # --- 3. esperar a RUNNING ---
    print("[..] Esperando a que el Space arranque (puede tardar 5-12 min):")
    running = False
    for i in range(40):  # ~20 min
        try:
            rt = api.get_space_runtime(SPACE)
            stage = rt.stage
        except Exception:
            stage = "?"
        print(f"     t+{i*30:4d}s  estado={stage}")
        if stage == "RUNNING":
            running = True
            break
        time.sleep(30)

    if not running:
        print("\n[!] No llego a RUNNING en el tiempo de espera. Revisa los logs en la web.")
        print(f"    https://huggingface.co/spaces/{SPACE}?logs=container")
        return

    # --- 4. verificar que el codigo NUEVO esta vivo ---
    print("[..] Verificando que el Veredicto nuevo esta en la app viva ...")
    import requests
    ok_live = False
    for _ in range(6):
        try:
            r = requests.get(f"https://{USUARIO.lower()}-finanzia-dashboard.hf.space/config",
                             headers={"Authorization": f"Bearer {token}"}, timeout=20)
            if r.status_code == 200 and "Consenso multi-modelo" in r.text:
                ok_live = True
                break
        except Exception:
            pass
        time.sleep(20)

    print("\n" + "=" * 62)
    if ok_live:
        print("  [OK] LISTO. El Veredicto nuevo (consenso multi-modelo + osciladores")
        print("       + ADX + OBV) esta VIVO en el Space.")
    else:
        print("  [!] El Space arranco pero la verificacion no confirmo el codigo nuevo")
        print("      (puede ser cache de la web; recarga en incognito en 1-2 min).")
    print(f"  URL: https://huggingface.co/spaces/{SPACE}")
    print("  >> Revoca el token en https://huggingface.co/settings/tokens")
    print("=" * 62)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelado.")
    except Exception as e:
        print(f"\n[X] Error: {e}")
        sys.exit(1)
    input("\nPulsa Enter para cerrar...")
