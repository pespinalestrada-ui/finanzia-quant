"""
notify — avisos a tu móvil por Telegram (gratis, sin dependencias nuevas).

Configuración (una vez):
  1. En Telegram habla con @BotFather → /newbot → te da un TOKEN.
  2. Pega en el .env de la raíz:  TELEGRAM_TOKEN=123456:ABC...
  3. Envía cualquier mensaje a tu bot (p. ej. "hola").
  4. python notify.py chatid   → te dice tu TELEGRAM_CHAT_ID; pégalo en el .env.
  5. python notify.py test     → debe llegarte el mensaje.

Uso:
    python notify.py test [texto]
    python notify.py chatid
    python notify.py alertas            (escanea tu watchlist y manda los avisos)
    python notify.py semaforo AAPL      (te manda el semáforo del día)
"""
import os
import sys
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import warnings
warnings.filterwarnings("ignore")

import requests

_PROJ = Path(__file__).resolve().parents[2]
_SUITE = Path(__file__).resolve().parents[1]
_ENV_CARGADO = False


def _cargar_env():
    global _ENV_CARGADO
    if _ENV_CARGADO:
        return
    _ENV_CARGADO = True
    env = _PROJ / ".env"
    if not env.exists():
        return
    try:
        for ln in env.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#") or "=" not in ln:
                continue
            k, v = ln.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v and k not in os.environ:
                os.environ[k] = v
    except Exception:
        pass


def configurado():
    _cargar_env()
    return bool(os.getenv("TELEGRAM_TOKEN"))


def _api(metodo):
    _cargar_env()
    tok = os.getenv("TELEGRAM_TOKEN")
    if not tok:
        raise RuntimeError("Falta TELEGRAM_TOKEN en el .env (créalo con @BotFather).")
    return f"https://api.telegram.org/bot{tok}/{metodo}"


def descubrir_chat_id():
    """Tras enviarle un mensaje a tu bot, esto encuentra tu chat_id."""
    r = requests.get(_api("getUpdates"), timeout=15)
    r.raise_for_status()
    for upd in reversed(r.json().get("result", [])):
        chat = (upd.get("message") or upd.get("edited_message") or {}).get("chat")
        if chat and chat.get("id"):
            return str(chat["id"])
    return None


def enviar(texto):
    """Manda un mensaje a tu Telegram. Devuelve True/False. Nunca lanza excepción."""
    try:
        _cargar_env()
        chat = os.getenv("TELEGRAM_CHAT_ID") or descubrir_chat_id()
        if not chat:
            return False
        r = requests.post(_api("sendMessage"),
                          json={"chat_id": chat, "text": texto[:4000]}, timeout=15)
        return r.status_code == 200
    except Exception:
        return False


def _watchlist():
    f = _PROJ / "watchlist.txt"
    if f.exists():
        txt = f.read_text(encoding="utf-8").strip()
        if txt:
            return [t.strip().upper() for t in txt.replace("\n", ",").split(",") if t.strip()]
    return ["AAPL", "MSFT", "NVDA", "SAB.MC"]


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    if cmd == "chatid":
        cid = descubrir_chat_id()
        if cid:
            print(f"Tu TELEGRAM_CHAT_ID es: {cid}")
            print("Pégalo en el .env como: TELEGRAM_CHAT_ID=" + cid)
        else:
            print("No encontrado. ¿Le has enviado ya un mensaje a tu bot? Mándale 'hola' y reintenta.")
    elif cmd == "test":
        txt = " ".join(sys.argv[2:]) or "✅ FinanzIA conectado a Telegram."
        print("Enviado ✅" if enviar(txt) else "FALLO: revisa TELEGRAM_TOKEN/TELEGRAM_CHAT_ID en el .env.")
    elif cmd == "alertas":
        sys.path.insert(0, str(_SUITE / "alerts"))
        import alerts as AL
        filas = AL.escanear(_watchlist())
        if not filas:
            print("Sin alertas ahora mismo (no se envía nada)."); return
        lineas = [f"🔔 FinanzIA · {len(filas)} alertas:"] + \
                 [f"• {t}: {txt} ({px:.2f})" for t, px, _c, txt in filas[:15]]
        ok = enviar("\n".join(lineas))
        print("Alertas enviadas ✅" if ok else "FALLO al enviar.")
    elif cmd == "semaforo":
        tk = sys.argv[2].upper() if len(sys.argv) > 2 else "SPY"
        sys.path.insert(0, str(_SUITE / "intraday"))
        import intraday as IN
        _f, _t, md = IN.semaforo(tk, 30)
        cabecera = md.split("\n")[0].replace("#", "").strip()
        consejo = next((l for l in md.split("\n") if l.startswith("**Consejo:**")), "")
        ok = enviar(f"🚦 {tk} hoy: {cabecera}\n{consejo.replace('**','')}")
        print("Semáforo enviado ✅" if ok else "FALLO al enviar.")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
