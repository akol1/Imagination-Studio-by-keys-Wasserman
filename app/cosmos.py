"""The vision brain: NVIDIA Cosmos3-Edge on TensorFold, the eyes AutoPilot watches each segment with.

Cosmos3-Edge (nvidia/Cosmos3-Edge, OpenMDW 1.1) is a small vision-language reasoner, strong at physical, spatial and
"what happened" questions. TensorFold's port (branch ``cosmos3-cuda-115`` on a DGX Spark, ``cosmos3-metal-115`` on a
Mac) serves it as an OpenAI-compatible chat API with ``--vision``: images as data-URL ``image_url`` parts, several in
one request (``--vision-max-images``, 4 unless set), PNG always (JPEG needs libturbojpeg beside the binary).

Where the vision brain is (``vision.json`` in the AutoPilot folder, set in Settings and on the AutoPilot panel):

  ``studio:cosmos``   the Studio's own instance (the default): started here on a loopback port when a segment is
                      watched, under a kill-guard (``COSMOS3_GUARD_GB``, 15: MemAvailable, vm_stat on a Mac), unloaded
                      after ``COSMOS3_IDLE_S`` (120) idle seconds and **before any other local job of the queue**
                      (``release``, called at the top of ``server.perform``). It loads in about 3 s and holds about 3.5 GB.
                      It never takes ``~/studio-gpu.lock.d``. Never point the Studio at another program's Cosmos (a voice
                      agent's eyes): give the Studio its own.
  ``http://host:port``  any OpenAI-compatible vision server the user runs (nearby addresses only, slate.nearby_endpoint)
  empty               no vision server: AutoPilot watches through the QA gate's vision path instead (qa.look: the film's
                      brain, Claude Code, Codex or a local vision model) on one time-ordered frame sheet

The binary is ``COSMOS3_BIN`` (default ``$PREFIX/cosmos3/bin/tensorfold-native``) and the converted weights
``COSMOS3_MODEL`` (default ``$PREFIX/cosmos3/model``).
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

STUDIO = "studio:cosmos"
SERVED = "cosmos3"
PREFIX = Path(os.environ.get("PREFIX", Path.home() / ".local/opt/tensorfold-studio")).expanduser()
BIN = Path(os.environ.get("COSMOS3_BIN", PREFIX / "cosmos3" / "bin" / "tensorfold-native")).expanduser()
MODEL = Path(os.environ.get("COSMOS3_MODEL", PREFIX / "cosmos3" / "model")).expanduser()
PORT = int(os.environ.get("COSMOS3_PORT", "8362"))  # not 8099: that is where a voice agent's own Cosmos usually listens
ENDPOINT = f"http://127.0.0.1:{PORT}/v1"
GUARD_GB = float(os.environ.get("COSMOS3_GUARD_GB", "15"))  # the serve script's own floor on a DGX Spark
IDLE_S = float(os.environ.get("COSMOS3_IDLE_S", "120"))
MAX_IMAGES = 9  # eight frames of a segment and its last frame, at most
# the port reads at most 1200 image rows a request at its default context; a 448x256 frame is 112 rows (16 px
# patches merged 2 x 2), so nine frames fit with room to spare
FRAME = (448, 256)
TIMEOUT = 240

HOME: Path | None = None
lock = threading.RLock()
state = {"proc": None, "guard": None, "used": 0.0, "inflight": 0, "log": None, "why": ""}


def attach(home: Path) -> None:
    global HOME
    HOME = home


# ---------- where the vision brain is ----------

def settings_file() -> Path:
    return (HOME or Path.home()) / "autopilot" / "vision.json"


def settings() -> dict:
    """{endpoint, model, frames}: the saved vision brain, the Studio's own Cosmos when nothing is saved."""

    try:
        saved = json.loads(settings_file().read_text())
    except (OSError, ValueError):
        saved = {}
    endpoint = saved.get("endpoint") if isinstance(saved.get("endpoint"), str) else STUDIO
    model = saved.get("model") if isinstance(saved.get("model"), str) and saved.get("model") else SERVED
    try:
        frames = max(3, min(MAX_IMAGES - 1, int(saved.get("frames") or 7)))
    except (TypeError, ValueError):
        frames = 7
    return {"endpoint": endpoint.strip(), "model": model.strip(), "frames": frames}


def save_settings(body: dict) -> dict:
    now = settings()
    for key in ("endpoint", "model"):
        if isinstance(body.get(key), str):
            now[key] = body[key].strip()
    if body.get("frames") is not None:
        now["frames"] = body["frames"]
    target = settings_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    scratch = target.with_suffix(".tmp")
    scratch.write_text(json.dumps(now, indent=1))
    scratch.replace(target)
    return settings()


def installed() -> bool:
    return BIN.is_file() and os.access(BIN, os.X_OK) and (MODEL / "config.json").is_file()


def summary() -> dict:
    said = settings()
    managed = said["endpoint"].lower() == STUDIO
    return {**said, "managed": managed, "installed": installed(), "loaded": loaded(), "port": PORT,
            "why": "" if not managed or installed() else f"Cosmos3-Edge is not installed here ({BIN} and {MODEL}/config.json): "
                                                          "AutoPilot watches through the QA gate's vision path until it is, or name a vision server"}


# ---------- the Studio's own instance ----------

def mem_available_gb() -> float:
    try:
        import voice  # the same reading the Voice engine's guard uses (MemAvailable, vm_stat on a Mac, a card's free memory)
        return voice.mem_available_gb()
    except Exception:  # noqa: BLE001
        return -1.0


def loaded() -> bool:
    proc = state["proc"]
    return bool(proc and proc.poll() is None)


def answers(endpoint: str = ENDPOINT, timeout: float = 2) -> bool:
    try:
        with urllib.request.urlopen(endpoint + "/models", timeout=timeout) as reply:
            return reply.status == 200
    except Exception:  # noqa: BLE001
        return False


def guard(proc: subprocess.Popen) -> None:
    """The kill-guard beside the server: stopped when the machine's free memory falls under the floor."""

    while proc.poll() is None:
        free = mem_available_gb()
        if 0 <= free < GUARD_GB:
            proc.kill()
            state["why"] = f"stopped by its memory guard at {free:.1f} GB available (floor {GUARD_GB:g} GB)"
            print(f"[cosmos] {state['why']}", flush=True)
            return
        if state["inflight"] == 0 and state["used"] and time.time() - state["used"] > IDLE_S:
            stop("idle")
            return
        time.sleep(1.0)


def load() -> str:
    """The Studio's Cosmos, started if it is not running: its address."""

    with lock:
        if loaded() and answers():
            return ENDPOINT
        if not installed():
            raise LookupError(summary()["why"])
        free = mem_available_gb()
        if 0 <= free < GUARD_GB + 6:
            raise LookupError(f"Cosmos3-Edge needs about 6 GB above its {GUARD_GB:g} GB floor; {free:.1f} GB is available")
        log_path = (HOME or Path.home()) / "autopilot" / "cosmos3.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(log_path, "ab")
        command = [str(BIN), "serve", str(MODEL), "--name", SERVED, "--host", "127.0.0.1", "--port", str(PORT),
                   "--no-update-check", "--vision", "--vision-max-images", str(MAX_IMAGES), "--context", "16384",
                   "--parallel", "1", "--keep-warm", "0"]
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                cwd=str(BIN.parent), start_new_session=True)
        state.update(proc=proc, log=log, why="", used=time.time())
        threading.Thread(target=guard, args=(proc,), daemon=True, name="cosmos-guard").start()
        for _ in range(240):
            if proc.poll() is not None:
                raise LookupError("Cosmos3-Edge stopped while loading: " + (state["why"] or tail(log_path)))
            if answers():
                return ENDPOINT
            time.sleep(0.5)
        stop("did not answer in 120 s")
        raise LookupError("Cosmos3-Edge did not answer in 120 s: " + tail(log_path))


def tail(path: Path) -> str:
    try:
        return path.read_text(errors="replace")[-400:]
    except OSError:
        return ""


def stop(why: str = "") -> bool:
    with lock:
        proc = state["proc"]
        if not proc or proc.poll() is not None:
            state["proc"] = None
            return False
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
        state["proc"] = None
        if state["log"]:
            state["log"].close()
            state["log"] = None
        print(f"[cosmos] unloaded ({why})", flush=True)
        return True


def release(why: str = "") -> bool:
    """Unloaded before another local job of the queue (server.perform), after the request in flight."""

    for _ in range(600):
        if state["inflight"] == 0:
            break
        time.sleep(0.1)
    return stop(why)


@contextlib.contextmanager
def request():
    """(endpoint, model) of the vision brain for one request, the Studio's Cosmos loaded for it when that is the one."""

    said = settings()
    if not said["endpoint"]:
        raise LookupError("no vision server is set")
    if said["endpoint"].lower() != STUDIO:
        yield said["endpoint"].rstrip("/") if said["endpoint"].rstrip("/").endswith("/v1") else said["endpoint"].rstrip("/") + "/v1", said["model"]
        return
    with lock:
        endpoint = load()
        state["inflight"] += 1
    try:
        yield endpoint, SERVED
    finally:
        state["inflight"] -= 1
        state["used"] = time.time()


# ---------- asking it ----------

def data_url(picture: Path) -> str:
    kind = "png" if picture.suffix.lower() == ".png" else "jpeg"
    return f"data:image/{kind};base64," + base64.b64encode(picture.read_bytes()).decode()


def chat(endpoint: str, model: str, system: str, frames: list[tuple[str, Path]], text: str, max_tokens: int = 2000) -> str:
    """One request: the frames in time order, each after its caption ("0.00 s"), then the question."""

    content = []
    for caption, picture in frames:
        content.append({"type": "text", "text": caption})
        content.append({"type": "image_url", "image_url": {"url": data_url(picture)}})
    content.append({"type": "text", "text": text})
    body = json.dumps({"model": model, "temperature": 0.2, "max_tokens": max_tokens,
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]}).encode()
    call = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(call, timeout=TIMEOUT) as reply:
            return json.loads(reply.read())["choices"][0]["message"]["content"] or ""
    except urllib.error.HTTPError as error:
        said = error.read().decode(errors="replace")[:300]
        raise TooMany(said) if error.code in (400, 413, 422) and "image" in said.lower() else RuntimeError(f"{error.code}: {said}")


class TooMany(RuntimeError):
    """The server refused the images (more than it takes, or too many rows): ask again with one frame sheet."""


def look(system: str, frames: list[tuple[str, Path]], sheet: Path | None, text: str) -> tuple[str, str]:
    """(answer, who) from the vision server: every frame in one request, or, when it takes fewer images, the one
    time-ordered sheet of the same frames."""

    with request() as (endpoint, model):
        who = "Cosmos3-Edge" if model == SERVED else model or "the vision server"
        try:
            if len(frames) > MAX_IMAGES:  # more than the Studio's Cosmos is started to take (--vision-max-images)
                raise TooMany(f"{len(frames)} images, {MAX_IMAGES} at most")
            return chat(endpoint, model, system, frames, text), who
        except TooMany:
            if sheet is None:
                raise
            note = ("The picture is a contact sheet of the frames, left to right then top to bottom, in time order: "
                    + "; ".join(caption for caption, _ in frames) + ".\n\n")
            return chat(endpoint, model, system, [("Frame sheet", sheet)], note + text), who + " (frame sheet)"
