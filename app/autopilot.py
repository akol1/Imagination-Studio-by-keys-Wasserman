"""AutoPilot: open-ended, self-directing long-form video in the Production tab.

A story seed (a world, where it starts, the characters with their looks and pictures, a visual style and the user's
direction) becomes a film that keeps going until it is stopped. Each turn of the loop:

  plan      the Director (the film's text brain: the Studio's Bolt, Claude Code, Codex or a local server) reads the
            scene graph and the last segment's ending and writes the next segment as the Studio's own prompt fields;
            an answer that restages an earlier beat, repeats the segment before, keeps a scene past its budget, or
            names what must not be seen is turned back with the reason and asked again (``MAX_ATTEMPTS``)
  render    the segment goes through the Studio's render queue like any clip: the same engines, presets, LoRAs,
            memory guard and Spark routing. It opens on the last frame of the segment before (image to video); a new
            scene opens on the cast's reference pictures (Ref2VA) where that model is installed, else from text
  check     the QA gate (qa.py: the Whisper line check and the identity sheet) with the vision brain as its eyes
  watch     the vision brain (cosmos.py: NVIDIA Cosmos3-Edge by default) looks at eight frames of the segment in time
            order with the Whisper transcript and reports only what changed: the scene graph's delta, the state the
            segment ends in, and continuity problems. Identity drift, confirmed by a second look at the face alone,
            fails the segment like the QA gate does
  continuity  the same frames are checked against the continuity bible (continuity.py: everything the film has
            established, versioned by era and scene) with a checklist a category; a major finding is looked at again,
            closer, and a confirmed one renders the segment again with the finding in the Director's hands and a new
            seed (``retakes``, two by default; the best take is kept when they run out); a minor one becomes a
            correction in the Director's next prompt
  commit    the delta is merged into the scene graph; past ``COMPACT_AT`` events the oldest are folded into a chapter;
            the kept take's check is written into the bible, which is never folded
  preview   every ``preview_every`` segments the film is joined again (``film.mp4``): a continuation drops the frame it
            repeats and its sound cross-fades over the same 1/24 s, a new scene dissolves in a quarter second

Pause stops the loop after the segment at work; resume picks it up where it stopped, a restart of the app too.
Branch makes a new run that shares the first N segments and their memory and continues with new direction.

The idea of an endless film that watches itself and decides what happens next is Echo-Infinity's (Yuxuan Bian,
Nan Duan, Qiang Xu et al., CUHK and Joy Future Academy, JD). The loop's design, the scene-graph memory (its delta schema, the merge and id folding, the chapter
fold), the Director's rules against repetition and stagnation and the branch are adapted from Video Rebirth's
HyperFlow-Infinity (github.com/Video-Rebirth/hyperflow-infinity), Copyright Video Rebirth, Apache License 2.0; see
NOTICE and LICENSES/HyperFlow-Infinity-LICENSE. The changes: the Studio's engines, presets and queue instead of the
HyperFlow 8-step adapter and its latent hand-off (docs/autopilot.md has the design of a latent-tail continuation), the
Studio's prompt fields and content rules, a vision brain of the user's choice, and the QA gate on every segment.

Run files live in ``<studio folder>/autopilot/<run id>/``: ``run.json`` (settings, seed, segments, state), ``memory.json``
(the scene graph, schema in ``scene-graph.schema.json`` beside it), ``memory/S0001.json`` (the memory after each
segment), ``bible.json`` and ``bible/S0001.json`` (the continuity bible now and after each segment), ``segments/S0001/``
(takes, frames, the Director's and the watcher's answers, the continuity checks), ``film.mp4`` and ``film-check.json``
(the film-level pass).
"""

from __future__ import annotations

import copy
import difflib
import json
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

import checks  # the dense yes/no checks, decided in code, and the pace measured from the pixels
import continuity  # the continuity bible and what its checks mean
import cosmos  # the vision brain (NVIDIA Cosmos3-Edge on TensorFold), beside this file
import lengths  # which frame counts the model takes
import negation  # what a prompt forbids by name
import qa  # Sam Wasserman's QA gate: the Whisper transcript, the frame sheet and its vision path
import slate  # the brains (slate.ask) and the Slate bible a seed can come from

router = APIRouter()
HOME: Path = Path()
FPS = 24
TICK = 2.0
LOG = 200
MAX_ATTEMPTS = 3  # a Director or watcher answer that does not hold up is asked again, this many tries in all
STATE_WORDS, EVENTS_PER_SEGMENT = 12, 2
COMPACT_AT, FOLD = 22, 10  # above this many events the oldest FOLD become one chapter
RECENT_EVENTS = 8
SIMILAR_ACTION, SIMILAR_BEAT = 0.9, 0.85  # difflib ratios above which a new segment repeats an old one
LIMIT = 500  # segments a run may have when it is asked for a number (0 runs until it is stopped)
ACTIVE = ("running",)
CONTINUE_FADE, CUT_FADE = 1 / FPS, 0.25  # the joins of the film: the repeated frame dropped, and a dissolve on a new scene

# the defaults hold the lessons of the first long run (the 60 s Viking film): the Sharp Spec (1376x768, FastH3 8
# passes), fewer and longer pieces (10 s), and a keyframe a piece on a run with an outline
DEFAULTS = {"style": "", "guidance": "", "preset": "sharp", "quality": "fh8", "width": None, "height": None, "seconds": 10,
            "loras": [], "segments": 6, "seed": 1, "max_scene_segments": 8, "beat_history": 12, "qa": True, "retakes": 2,
            "preview_every": 3, "styles": [], "keyframes": "auto", "identity": "auto", "ref_turbo": True, "music": "", "director": "brain",
            "brain": {"backend": slate.AUTO, "endpoint": "", "model": ""}, "continuity": copy.deepcopy(continuity.DEFAULT),
            # the Director reasons before it answers (Bolt's thinking switch), within this many tokens of reasoning
            "think": True, "think_tokens": 4096,
            # the dense yes/no checks (checks.py)
            "checks": copy.deepcopy(checks.DEFAULT),
            # Infinity: no fixed length; it ends on Stop, a time or length limit (0: none), or an ending the Director chooses.
            # Its film is joined and a checkpoint written after every piece, the last joins checked every film_every pieces
            "infinity": False, "max_minutes": 0, "max_film_seconds": 0, "film_every": 4, "film_window": 6, "story_every": 6,
            # memory first: a piece that does not fit is made shorter at the same size, never smaller; not under this
            "min_seconds": 4}
RETAKES = 4  # the most retakes a segment may be given
MEMORY_RETRIES = 3  # a piece the memory guard stopped is made again shorter, at the same size, this many times at most
INFINITY_RETRIES = 5  # an Infinity run that hits a problem waits and carries on, this many times in a row at most
INFINITY_WAIT_S = 60
KEYFRAMES = ("auto", "on", "off")  # a picture a piece to open on: auto is on for a run with an outline
IDENTITY = ("auto", "frame", "references")  # how a new scene keeps who is who: Ref2VA pictures where installed, or words

# what the server hands in (attach); tests stand in for them
submit = lambda body: (_ for _ in ()).throw(RuntimeError("the queue is not attached"))
job_of = lambda job_id: None
cancel = lambda job_id: None
compose = lambda fields, first_frame, frames=None: fields.get("action", "")
compose_references = lambda fields, count: fields.get("action", "")
ref2va = lambda: False
ref2va_turbo = lambda: False
room = lambda params: None  # memory_of for a piece's params with this machine's memory now (server.autopilot_room); None: no check

runs: dict[str, dict] = {}
guard = threading.RLock()
starting = threading.Lock()


class Stopped(Exception):
    pass


class Failed(Exception):
    pass


def attach(home: Path, *, queue, lookup, stop_job, plain, referenced, references=None, references_turbo=None,
           quality: str = "", fits=None) -> APIRouter:
    global HOME, submit, job_of, cancel, compose, compose_references, ref2va, ref2va_turbo, room
    HOME, submit, job_of, cancel, compose, compose_references = home, queue, lookup, stop_job, plain, referenced
    room = fits or room
    ref2va, ref2va_turbo = references or ref2va, references_turbo or ref2va_turbo
    if quality:
        DEFAULTS["quality"] = quality
    migrate(HOME)
    (HOME / "autopilot").mkdir(parents=True, exist_ok=True)
    cosmos.attach(HOME)
    return router


def migrate(home: Path) -> None:
    """Runs made while the feature was called AutoRL Generation (folder ``autorl/``): moved to ``autopilot/`` once, with
    the paths inside their files."""

    old, new = home / "autorl", home / "autopilot"
    if not old.is_dir() or new.exists():
        return
    old.rename(new)
    for file in list(new.glob("*/run.json")) + list(new.glob("*/memory/*.json")):
        try:
            file.write_text(file.read_text().replace('"autorl/', '"autopilot/'))
        except OSError:
            pass


def resume_later() -> None:
    later = threading.Timer(6.0, resume_all)
    later.daemon = True
    later.start()


# ---------- the scene graph (adapted from HyperFlow-Infinity's memory.py, Apache-2.0) ----------

SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "AutoPilot scene graph",
    "description": "What the film has shown so far: who and what persists (nodes), what changed the story (events), and "
                   "older events folded into chapters. Written only from what the watcher saw, never from what was asked for.",
    "type": "object",
    "required": ["nodes", "events", "chapters"],
    "properties": {
        "nodes": {"type": "array", "items": {"type": "object", "required": ["id", "type", "label", "state"], "properties": {
            "id": {"type": "string", "description": "short unique id, reused for the same person, place or thing"},
            "type": {"enum": ["character", "place", "object"]},
            "label": {"type": "string", "description": "a name, three words at most"},
            "state": {"type": "string", "description": "how it is now, twelve words at most"}}}},
        "events": {"type": "array", "items": {"type": "object", "required": ["t", "subject", "action"], "properties": {
            "t": {"type": "string", "description": "MM:SS on the film's clock"},
            "subject": {"type": "string"}, "action": {"type": "string", "description": "a verb phrase, eight words at most"},
            "target": {"type": "string"}, "place": {"type": "string"}, "segment": {"type": "integer"}}}},
        "chapters": {"type": "array", "items": {"type": "object", "required": ["id", "title", "summary"], "properties": {
            "id": {"type": "string"}, "t_start": {"type": "string"}, "t_end": {"type": "string"},
            "title": {"type": "string"}, "summary": {"type": "string"},
            "carried_forward": {"type": "array", "items": {"type": "string"}, "maxItems": 3}}}},
    },
}
EMPTY = {"graph": {"nodes": [], "events": [], "chapters": []}, "archive": [], "observed_through": 0}


def clip_words(text, limit: int) -> str:
    return " ".join(str(text or "").split()[:limit])


def mmss(seconds: float) -> str:
    value = int(round(seconds))
    return f"{value // 60:02d}:{value % 60:02d}"


def resolve_id(new_id, known) -> str:
    """A proposed id mapped onto an existing node when it is plainly the same thing (``guard`` / ``guard_2``,
    ``ada`` / ``ada_older``, near spellings): vision models do not reliably reuse ids."""

    if not new_id or new_id in known:
        return new_id
    for old in known:
        if str(new_id).startswith(f"{old}_") or str(old).startswith(f"{new_id}_"):
            return old
    match = difflib.get_close_matches(str(new_id), [str(k) for k in known], n=1, cutoff=0.82)
    return match[0] if match else new_id


def merge(graph: dict, delta: dict, segment: int = 0) -> dict:
    """One watcher delta added to the graph: states replace, at most two events a segment, ids folded."""

    graph = copy.deepcopy(graph or EMPTY["graph"])
    graph.setdefault("chapters", [])
    by_id = {n.get("id"): n for n in graph["nodes"]}
    remap = {}
    for node in delta.get("add_nodes") or []:
        if not isinstance(node, dict) or not node.get("id"):
            continue
        raw = str(node["id"])
        target = resolve_id(raw, by_id.keys())
        remap[raw] = target
        state = clip_words(node.get("state"), STATE_WORDS)
        if target in by_id:
            if state:
                by_id[target]["state"] = state
        else:
            kind = node.get("type") if node.get("type") in ("character", "place", "object") else "character"
            graph["nodes"].append({"id": target, "type": kind, "label": clip_words(node.get("label"), 3) or target, "state": state})
            by_id[target] = graph["nodes"][-1]
    for change in delta.get("update_nodes") or []:
        if not isinstance(change, dict):
            continue
        node = by_id.get(remap.get(change.get("id"), resolve_id(change.get("id"), by_id.keys())))
        if node is not None and change.get("state"):
            node["state"] = clip_words(change["state"], STATE_WORDS)
    for event in [e for e in delta.get("add_events") or [] if isinstance(e, dict)][:EVENTS_PER_SEGMENT]:
        event = {k: event.get(k) for k in ("t", "subject", "action", "target", "place") if event.get(k)}
        for key in ("subject", "target", "place"):
            if event.get(key):
                event[key] = remap.get(event[key], resolve_id(event[key], by_id.keys()))
        event["action"] = clip_words(event.get("action"), 8)
        if segment:
            event["segment"] = segment
        if event["action"]:
            graph["events"].append(event)
    return graph


def fold(graph: dict, chapter: dict, folded: list) -> list:
    """The ``folded`` oldest events replaced by one chapter; the removed events are returned for the archive."""

    graph.setdefault("chapters", []).append({
        "id": f"ch{len(graph['chapters']) + 1}", "t_start": folded[0].get("t", ""), "t_end": folded[-1].get("t", ""),
        "title": clip_words(chapter.get("title"), 5), "summary": clip_words(chapter.get("summary"), 30),
        "carried_forward": [clip_words(x, 10) for x in (chapter.get("carried_forward") or [])[:3]]})
    graph["events"] = graph["events"][len(folded):]
    return folded


def as_text(graph: dict, recent: int = RECENT_EVENTS) -> str:
    """A bounded view for the Director: chapters, the cast as they are now, places and things, the last events.
    Bounded on purpose: a view that grew with the film grew the prompts with it."""

    graph, lines = graph or {}, []
    for ch in graph.get("chapters") or []:
        carried = "; ".join(ch.get("carried_forward") or [])
        lines.append(f"[{ch.get('t_start')}-{ch.get('t_end')}] {ch.get('title')}: {ch.get('summary')}" + (f" (still matters: {carried})" if carried else ""))
    nodes = graph.get("nodes") or []
    cast = [n for n in nodes if (n.get("type") or "character") == "character"]
    if cast:
        lines.append("Cast as they now are: " + "; ".join(f"{n.get('label')} -- {n.get('state')}" for n in cast))
    world = [n for n in nodes if (n.get("type") or "character") != "character"]
    if world:
        lines.append("Places and things in play: " + ", ".join(f"{n.get('label')} ({n.get('state')})" if n.get("state") else str(n.get("label")) for n in world))
    tail = (graph.get("events") or [])[-recent:]
    if tail:
        lines.append("Just happened: " + "; ".join(f"{e.get('t')} {e.get('subject')} {e.get('action')}" + (f" -> {e['target']}" if e.get("target") else "") for e in tail))
    return "\n".join(lines)


def scene_length(segments: list) -> int:
    """Segments since, and including, the last one that opened a scene."""

    n = 0
    for done in reversed(segments):
        n += 1
        if done.get("scene_change"):
            break
    return n


def signature(delta: dict):
    """A delta in a comparable form (timestamps left out); None when it carries no state to compare."""

    if not delta or delta.get("add_nodes") or not delta.get("add_events") or not delta.get("update_nodes"):
        return None
    norm = lambda value: " ".join(str(value or "").lower().split())
    events = sorted((norm(e.get("subject")), norm(e.get("action")), norm(e.get("target")), norm(e.get("place"))) for e in delta["add_events"] if isinstance(e, dict))
    states = sorted((norm(n.get("id")), norm(n.get("state"))) for n in delta["update_nodes"] if isinstance(n, dict))
    if any(not e[1] for e in events) or any(not s[1] for s in states):
        return None
    return [events, states]


def stagnation(segments: list):
    """Three segments in a row, in one scene, whose watcher saw the same events and states: the film is circling."""

    recent = segments[-3:]
    if len(recent) < 3 or len({s.get("scene") for s in recent}) != 1:
        return None
    marks = [signature((s.get("watch") or {}).get("delta")) for s in recent]
    if marks[0] is not None and all(m == marks[0] for m in marks):
        return {"segments": [s["index"] for s in recent], "reason": "three segments in a row show the same events and states"}
    return None


# ---------- the prompts (the Director's rules adapted from HyperFlow-Infinity's prompts.py, Apache-2.0) ----------

DIRECTOR_SYSTEM = """You are the director of a long-form film generated one short segment at a time by a \
text-to-video-and-audio model (MiniMax H3). Each segment is a few seconds long with its own synchronized sound.

Answer with a single JSON object and nothing else:
{
  "scene": "<short name of the scene this segment belongs to>",
  "scene_change": true | false,
  "beat": "<one short line: what this segment accomplishes in the story>",
  "new_characters": [{"name": "...", "look": "<concrete visual description: age, face, hair, clothing>"}],
  "era": "<empty, unless this segment opens the film or jumps to another time or era: then a short label like 'Boston, 1773'>",
  "era_rules": ["<with an era label only: what belongs to that era's material culture and what would be out of place, each 12 words max>"],
  "keyframe": "<only when the prompt asks for it: the still picture this segment opens on, 60 words max: who is where, the framing, the place, the light>",
  "shot": {
    "setting": "<the place, the time of day and the light, concrete and visual>",
    "camera": "<framing and camera movement>",
    "action": "<what happens, in temporal order: one legible, achievable action and its visible result>",
    "dialogue": [{"speaker": "<name>", "voice": "<how the voice sounds>", "text": "<a short spoken line>"}],
    "soundscape": "<the diegetic sound: ambience, impacts, movement>",
    "music": "<the score, or N/A>"
  },
  "pace": {"kind": "fight|action|chase|calm|dialogue", "beats": [{"from": <second>, "to": <second>, "speed": "fast|real|slow", "what": "<5 words>"}]}
}

"scene_change" is false while the story stays in the same place with the same people: the segment then continues \
the previous one from its last frame, the action picking up exactly where it stopped, the same light and camera \
language. Set it to true when the story moves to another place or time: the segment then opens fresh. Prefer \
continuous scenes of several segments over rapid cutting, but move on once a scene's purpose is done.

Every segment moves the story one concrete, visible step. Restating the situation in other words is a wasted \
segment: "they are hunted", "they are cornered", "they hide", "they still hide" is one beat, not four. A new \
expression, another reaction, a camera change or restarting the same unfinished action is not a new step.

List a character in "new_characters" only the first time they appear, with a look concrete enough to repeat word \
for word forever after. Name characters by name in "action" and "setting".

Rules that matter:
- Keep it suitable for a general audience: nothing sexual, no graphic gore or lingering on injury; conflict and \
danger are shown without graphic detail. No real public figures. Children are never placed in harm.
- Write only what a camera and a microphone can record: no inner thoughts, backstory or commentary.
- Never ask for on-screen text, subtitles, logos or watermarks.
- Say what is there, never what is not: no "no X", "without X" or "X is absent". The video model has no negative \
prompt and what is named tends to appear.
- Dialogue is optional and short: about two words a second of the segment at most. Leave "dialogue" empty for a \
segment without speech.
- Continue from what the film actually showed (the watcher's report and the scene graph), not from what an earlier \
segment asked for. Previous beats are requests, not facts.
- The continuity bible is canonical: every person keeps their face, marks, wardrobe and its state and what they hold \
until the story changes it on screen; an action in progress continues or resolves; what has happened stays happened. \
Name only the details that matter in this shot (the musket in his right hand), and make every correction it asks \
for. A time-skip is a new scene with an "era" label.
- Write in English.
- Pace. A fight, a chase or an action segment is mostly FAST and in real time: quick strikes, rapid exchanges, \
sharp impacts, a snappy camera (whip pans, quick handheld reframes). Slow motion is kept for one or two key moves \
only (the decisive blow, a signature technique), a third of the segment at most, and its timing is written into \
"action", like "0-6 s rapid real-time exchange of punches and blocks; 6-8 s slow motion on the centerline punch; \
8-10 s back to full speed as he drives forward". "pace.beats" lists those seconds with their speed. Outside the key \
move neither "action" nor "camera" says slow or slowly. A fight "action" opens with the strike already under way (its \
first words are the movement: "Eirik's fist is already driving into ..."), never with a stand-off or a look, and a fight \
segment ends in the middle of a movement, never a still pose: the next one starts from that frame. A calm or dialogue \
segment moves at natural real-time speed; its "beats" may be empty.

You run the film on your own: nobody approves a segment. You choose the story's next step and the camera for every \
segment yourself (framing, angle, movement: whatever shows the action best; there is no fixed camera mode). Favour \
action: each segment shows a physical action with a visible result. Keep the prompt short: "action" at most 40 words, \
"camera" at most 15, "setting" at most 30. The continuity bible is for you to keep things right, not to be copied out.

This is open-ended: there is no prescribed plot, act schedule, ending or total length. Decide the next short step \
from the evidence."""

WATCH_SYSTEM = ("You watch a film as it is generated, one short segment at a time, and keep its scene graph: you report only "
                "what each new segment changes, and anything that breaks continuity. Answer only with JSON.")

COMPACT_SYSTEM = "You compress a film's history into chapters. Answer only with JSON."


def seed_text(seed: dict, style: str) -> str:
    cast = "\n".join(f"- {c.get('name')}: {c.get('look') or 'as first seen'}" + (f" (segments {c.get('from') or 1} to {c.get('to') or 'the end'})" if c.get("from") or c.get("to") else "")
                     for c in seed.get("characters") or [] if c.get("name"))
    return "\n".join(bit for bit in (
        f"Inspiration (a starting point, not a plot to retell): {seed['inspiration']}" if seed.get("inspiration") else "",
        f"World: {seed.get('world') or 'open'}", f"Where the film starts: {seed.get('situation') or 'open'}",
        f"Characters:\n{cast}" if cast else "", f"Visual style: {style}" if style else "") if bit)


def outline_text(run: dict, index: int) -> str:
    """The user's ordered beats, when the seed has them: the whole list, and the one this segment carries."""

    outline = run["seed"].get("outline") or []
    if not outline:
        return ""
    listed = "\n".join(f"  {i}. {b['text']}" for i, b in enumerate(outline, start=1))
    beat = beat_of(run, index)
    if beat is None:
        return (f"\nThe user's outline ({len(outline)} beats) is done:\n{listed}\nFrom here on the film is open-ended: continue from "
                "where it ended.\n")
    return (f"\nThe user's outline, one beat a segment, in order:\n{listed}\nThis is segment {index}: it carries beat {index}"
            f" (\"{beat['text']}\"). Follow it, in your own concrete shot, continuing from what the film shows now."
            + (" It opens on a picture of its own, a new scene: set scene_change to true." if beat.get("image") else "") + "\n")


def direction(settings: dict, branch: dict | None, index: int) -> str:
    text = ""
    if settings.get("guidance"):
        text += f"\nThe user's creative direction: {settings['guidance']}\n"
    if branch and index > int(branch.get("after") or 0):
        text += (f"\nThe user asked for another way the story goes on after segment {branch['after']}:\n"
                 "<user_story_guidance>\n" + branch["guidance"] + "\n</user_story_guidance>\n"
                 "This is direction for what comes next, NOT something that already happened. Develop it from what the film "
                 "shows now; keep the shared past, who is who, what they hold and where they are. Get there through visible, "
                 "achievable actions: no teleporting, no restarting the opening, no pretending it already happened.\n")
    return text


def style_at(settings: dict, index: int) -> str:
    """The visual style of segment ``index``: the run's style (what every segment shares) and the scoped styles whose
    segments include it ({style, from, to}, like the cast sheets). On the 60 s Viking test film one style named the
    sea, the island and 1770s Boston at once, and Boston's streets came into the storm at sea: give each its span."""

    scoped = [str(x.get("style") or "").strip() for x in settings.get("styles") or []
              if isinstance(x, dict) and (x.get("from") or 1) <= index <= (x.get("to") or 10 ** 6)]
    return " ".join(bit for bit in (str(settings.get("style") or "").strip(), *scoped) if bit)


def first_prompt(run: dict, seconds: float) -> str:
    s = run["settings"]
    return (seed_text(run["seed"], style_at(s, 1)) + "\n" + outline_text(run, 1) + direction(s, run.get("branch"), 1)
            + f"\nWrite segment 1, about {seconds:.0f} seconds. It is the opening: establish the world, the look and the "
              "mood, and start the situation. Name its scene and set \"scene_change\" to true. Fill \"era\" and \"era_rules\" "
              "for the film's opening era.")


def next_prompt(run: dict, index: int, seconds: float, memory_text: str, note: str, bible_text: str = "") -> str:
    s, segments = run["settings"], committed(run)
    previous = segments[-1]
    beats = "\n".join(f"  {seg['index']}. {seg.get('beat', '')[:110]}" for seg in segments[-int(s.get("beat_history") or 12):]) or "  (none yet)"
    budget, length = int(s.get("max_scene_segments") or 8), scene_length(segments)
    pacing = f"The current scene is {previous.get('scene')!r} and has run {length} segment(s) of a {budget}-segment budget. "
    pacing += ("That budget is spent: set \"scene_change\" to true and move the story to a new place or situation grounded in what happened."
               if length >= budget else "Advance one new visible consequence without restaging what is done.")
    cast = "\n".join(f"- {name}: {look}" for name, look in cast_at(run, index).items())
    watched = previous.get("watch") or {}
    said = (previous.get("transcript") or "").strip()
    return (seed_text(run["seed"], style_at(s, index)) + "\n"
            + (f"\nCast -- repeat these looks word for word:\n{cast}\n" if cast else "")
            + f"\nBeats asked for so far (requests, not verified events; do not repeat them):\n{beats}\n"
            + f"\nYou are writing segment {index}, about {seconds:.0f} seconds. {pacing}\n"
            + f"\nThe previous segment ({previous.get('scene')}) was asked to show: {previous.get('fields', {}).get('action', '')}\n"
            + f"What the watcher saw it actually show: {watched.get('summary') or 'no report'}\n"
            + f"How it ends, on its last frame: {watched.get('ending') or 'no report'}\n"
            + (f"What was heard: {said}\n" if said else "")
            + (f"\nThe story so far (a running summary):\n{run['story_so_far']['text']}\n" if (run.get("story_so_far") or {}).get("text") else "")
            + (f"\nMemory of the film so far:\n{memory_text}\n" if memory_text else "")
            + (f"\nThe continuity bible (canonical, built from everything the film has shown; never contradict it):\n{bible_text}\n" if bible_text else "")
            + (f"\n{note}\n" if note else "")
            + outline_text(run, index) + direction(s, run.get("branch"), index)
            + (INFINITY_ASK if s.get("infinity") else "")
            + "\nContinue from what really came out, above all the state the last frame leaves things in. Use the memory to keep "
              "people, places and continuity consistent and to avoid repeating what already happened; do not describe the memory itself.")


INFINITY_ASK = ("\nThis film has no fixed length: it goes on while the story does. Add \"end\": true to your answer only when this "
                "segment brings the story to a real ending; otherwise leave it out.\n")


def watch_prompt(run: dict, seg: dict, view: dict, older: int, transcript: str) -> str:
    cast = "\n".join(f"- {name}: {look}" for name, look in cast_at(run, seg["index"]).items()) or "- (no pinned looks)"
    start = float(seg.get("start") or 0)
    return (f"The images are frames of segment {seg['index']} of the film, in time order, covering {mmss(start)} to "
            f"{mmss(start + float(seg.get('seconds') or 0))} on the film's clock; the last image is its final frame.\n"
            + (f"What is heard in it (speech to text, may be rough): {transcript}\n" if transcript else "What is heard in it: no speech.\n")
            + f"\nThe segment was ASKED to show (a request, not a fact): {seg.get('fields', {}).get('action', '')}\n"
            + f"\nThe cast's pinned looks:\n{cast}\n"
            + f"\nThe scene graph so far (every node and the {RECENT_EVENTS} latest events; {older} older events are kept but not shown):\n"
            + json.dumps(view, ensure_ascii=False, indent=1)
            + "\n\nReport ONLY what this segment adds or changes, as this JSON:\n"
            '{"add_nodes": [{"id": "<short unique id>", "type": "character|place|object", "label": "<3 words max>", "state": "<12 words max>"}], '
            '"update_nodes": [{"id": "<existing id>", "state": "<12 words max>"}], '
            '"add_events": [{"t": "<the time on the film\'s clock, like 01:07>", "subject": "<node id>", "action": "<verb phrase, 8 words max>", "target": "<node id or empty>", "place": "<node id>"}], '
            '"summary": "<25 words max: what actually happened>", '
            '"ending": "<40 words max: who and what is where on the final frame, poses, what they hold, the light>", '
            '"continuity": [{"kind": "identity|costume|object|place|other", "who": "<node id or name>", "note": "<15 words max>"}], '
            '"identity_drift": true | false}\n\n'
            "Rules that matter:\n"
            "- A node is a person, a place or a thing that persists. Never a node for an action, a shot or a moment. The "
            "same person older or the same crowd elsewhere keeps its node: reuse the id and update its state.\n"
            "- \"type\": character is only for people and animals; a vehicle, prop or effect is an object.\n"
            "- \"state\" replaces the old state: how it is NOW, twelve words.\n"
            "- At most TWO events, and only events that change the story: something gained or lost, a relationship turns, a "
            "place left for good, a goal met or broken. Glances, smiles, steps and camera moves are not events.\n"
            "- \"continuity\" lists only real problems you can see, never what stayed consistent: a face that changes into another person's, clothing that "
            "differs from the pinned look or from earlier frames, an object that vanishes or changes, a jump in place. Empty when none.\n"
            "- \"identity_drift\" is true only when a pinned character's face, age or build clearly becomes someone else's, "
            "or melts. Turning away, distance, motion blur or new lighting is not drift.\n"
            "- Describe what the frames show, not what was asked for; a voice that is only heard is not a node. Any list may be empty.")


def compact_prompt(events: list, chapters: list) -> str:
    lines = "\n".join(f"  {e.get('t', '--:--')} {e.get('subject', '')} {e.get('action', '')} {e.get('target', '')}".rstrip() for e in events)
    seen = "; ".join(c.get("title", "") for c in chapters) or "(none yet)"
    return (f"These are the events of one stretch of a film, oldest first:\n{lines}\n\nChapters already summarised: {seen}.\n\n"
            'Fold this stretch into a single chapter. Answer with:\n{"title": "<at most 5 words>", "summary": "<at most 30 words, '
            'what this stretch accomplished in the story>", "carried_forward": ["<at most 3 items, each at most 10 words: a '
            'consequence that still matters later>"]}\n\nWrite what changed, not what was shown. Anything left out is gone from the memory.')


def stagnation_note(found: dict) -> str:
    return ("The film is circling: " + found["reason"] + f" (segments {', '.join(map(str, found['segments']))}). The next segment MUST "
            "set scene_change to true, use a different scene name, and complete a visible change of situation chosen by you.")


# ---------- the brains ----------

def ask_text(run: dict, system: str, prompt: str) -> str:
    """The film's brain, reasoning first when the run says so (``think``): on the Studio's Bolt, or any local server,
    its thinking switch with a bounded budget (``think_tokens``); the reasoning is dropped from the answer. A CLI brain
    reasons on its own."""

    b, s = run["settings"].get("brain") or {}, run["settings"]
    think = int(s.get("think_tokens") or 0) if s.get("think", True) else 0
    return slate.ask(b.get("backend") or slate.AUTO, system, prompt, "top", b.get("endpoint") or "", b.get("model") or "", think=think)


def ask_vision(system: str, frames: list, sheet: Path, prompt: str, brain: dict) -> tuple[str, str]:
    """The watcher: the vision server (cosmos.py) with every frame, else the QA gate's vision path on the sheet."""

    if cosmos.settings()["endpoint"]:
        try:
            return cosmos.look(system, frames, sheet, prompt)
        except LookupError:
            pass  # the Studio's Cosmos is not installed or has no room: the QA gate's eyes instead
    note = "The picture is a contact sheet of the frames, left to right then top to bottom, in time order: " + "; ".join(c for c, _ in frames) + ".\n\n"
    return qa.look(sheet, brain, system + "\n\n" + note + prompt)


def ask_json(asker, check, prompt: str, what: str) -> tuple[dict, list[str]]:
    """A brain asked for JSON, again with the reason when the answer does not hold up: (answer, the reasons turned back)."""

    problems, asked = [], prompt
    for _ in range(MAX_ATTEMPTS):
        try:
            data = slate.extract_json(asker(asked))
            problem = check(data) if isinstance(data, dict) else "the answer is not a JSON object"
        except ValueError as error:
            problem = str(error)
        except Exception as error:  # noqa: BLE001 - a brain that does not answer stops the run, to be resumed
            raise Failed(f"the {what} did not answer: {str(error)[:300]}")
        if not problem:
            return data, problems
        problems.append(problem)
        asked = prompt + f"\n\nYOUR LAST ANSWER COULD NOT BE USED: {problem}. Answer again with the JSON only."
    raise Failed(f"the {what}'s answer could not be used {MAX_ATTEMPTS} times: {problems[-1]}")


# ---------- the run's files ----------

def run_dir(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id or ""):
        raise HTTPException(400, "bad run id")
    return HOME / "autopilot" / run_id


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(path.suffix + ".tmp")
    scratch.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    scratch.replace(path)


def load(run_id: str) -> dict:
    run = read_json(run_dir(run_id) / "run.json")
    if not run:
        raise HTTPException(404, "no such run")
    return run


def save(run: dict) -> None:
    with guard:
        run["rev"] = int(run.get("rev") or 0) + 1
        run["updated"] = slate.now()
        write_json(run_dir(run["id"]) / "run.json", run)


def say(run: dict, line: str, **current) -> None:
    run.setdefault("log", []).append({"at": slate.now(), "text": line})
    del run["log"][:-LOG]
    run["current"] = {"text": line, **current}
    save(run)


def memory_of(run: dict) -> dict:
    return read_json(run_dir(run["id"]) / "memory.json", copy.deepcopy(EMPTY))


def bible_of(run: dict) -> dict:
    """The continuity bible of the run; a run made before it existed gets one from its seed and cast."""

    found = read_json(run_dir(run["id"]) / "bible.json")
    return found if isinstance(found, dict) and found.get("version") else continuity.new(run["seed"], run.get("cast"))


def save_bible(run: dict, bible: dict, segment_id: str = "") -> None:
    write_json(run_dir(run["id"]) / "bible.json", bible)
    if segment_id:
        write_json(run_dir(run["id"]) / "bible" / f"{segment_id}.json", bible)


def committed(run: dict) -> list:
    return [s for s in run.get("segments") or [] if s.get("status") == "done"]


def relative(path: Path) -> str:
    return str(path.relative_to(HOME))


def on_disk(path: str | None) -> bool:
    if not path:
        return False
    target = (HOME / path).resolve()
    return HOME.resolve() in target.parents and target.is_file()


# ---------- the seed ----------

def span(value) -> tuple[int | None, int | None]:
    """A character's segments: "1-7", "8-", "-7" or "3" (both ends counted), or nothing for every segment."""

    text = str(value or "").replace(" ", "")
    found = re.fullmatch(r"(\d*)(-?)(\d*)", text)
    if not text or not found or not (found[1] or found[3]):
        return None, None
    first = int(found[1]) if found[1] else None
    last = int(found[3]) if found[3] else (None if found[2] else first)
    return first, last


def cast_at(run: dict, index: int) -> dict:
    """{name: look} of the cast in segment ``index``: a seed character only within its segments (a boy in 1-7 and the
    man he becomes from 8 are two sheets), and everyone the Director brought in."""

    seeded = {c["name"]: c for c in run["seed"].get("characters") or []}
    out = {}
    for name, look in (run.get("cast") or {}).items():
        c = seeded.get(name)
        if c and not ((c.get("from") or 1) <= index <= (c.get("to") or 10 ** 6)):
            continue
        if look:
            out[name] = look
    return out


def beat_of(run: dict, index: int) -> dict | None:
    outline = (run.get("seed") or {}).get("outline") or []
    return outline[index - 1] if 0 < index <= len(outline) else None


def clean_seed(given: dict) -> dict:
    if not isinstance(given, dict):
        raise HTTPException(400, "seed is an object: world, situation, characters")
    characters = []
    for i, c in enumerate(given.get("characters") or []):
        if not isinstance(c, dict) or not str(c.get("name") or "").strip():
            continue
        image = str(c.get("image") or "").strip()
        if image and not on_disk(image):
            raise HTTPException(400, f"the picture of {c['name']} is not a file in the studio's folder: {image}")
        first, last = span(c.get("segments"))
        characters.append({"id": re.sub(r"[^a-z0-9]+", "_", str(c["name"]).lower()).strip("_") or f"c{i + 1}",
                           "name": str(c["name"]).strip()[:80], "look": str(c.get("look") or "").strip()[:600], "image": image or None,
                           **({"wardrobe": str(c["wardrobe"]).strip()[:300]} if str(c.get("wardrobe") or "").strip() else {}),
                           **({"from": first} if first else {}), **({"to": last} if last else {})})
    outline = []
    for i, beat in enumerate((given.get("outline") or [])[:LIMIT]):
        beat = {"text": beat} if isinstance(beat, str) else beat if isinstance(beat, dict) else {}
        text, image = str(beat.get("text") or "").strip()[:600], str(beat.get("image") or "").strip()
        if image and not on_disk(image):
            raise HTTPException(400, f"the opening picture of beat {i + 1} is not a file in the studio's folder: {image}")
        if text or image:
            outline.append({"text": text, **({"image": image} if image else {})})
    rules = given.get("rules") or []
    rules = [line.strip() for line in (rules.splitlines() if isinstance(rules, str) else rules) if isinstance(line, str) and line.strip()][:8]
    seed = {"inspiration": str(given.get("inspiration") or "").strip()[:600], "world": str(given.get("world") or "").strip()[:1500],
            "situation": str(given.get("situation") or "").strip()[:1500], "characters": characters[:9],
            "opening_image": str(given.get("opening_image") or "").strip() or None, "slate": given.get("slate") or None,
            **({"outline": outline} if outline else {}), **({"rules": [r[:300] for r in rules]} if rules else {})}
    if seed["opening_image"] and not on_disk(seed["opening_image"]):
        raise HTTPException(400, "the opening picture is not a file in the studio's folder")
    if not (seed["world"] or seed["situation"] or seed["inspiration"]):
        raise HTTPException(400, "give the story a seed: a world, where it starts, or an inspiration")
    return seed


@router.get("/api/autopilot/seed/slate/{project}")
def seed_from_slate(project: str) -> dict:
    """A seed from a Slate film's bible: its world and logline, its characters' looks and pinned stills, its look."""

    film = slate.get_project(project)
    places = [l for l in film.get("locations") or [] if (l.get("description") or l.get("name"))]
    world = " ".join(bit for bit in (film.get("world") or "", *(slate.sentence(f"{l.get('name')}: {l.get('description') or ''}".strip(': ')) for l in places[:3])) if bit)
    characters = [{"name": c.get("name"), "look": slate.person(c), "image": (slate.stills_of(c) or [None])[0]}
                  for c in film.get("characters") or [] if c.get("name")]
    look = next((slate.sentence("; ".join(bit for bit in (s.get("tone"), s.get("palette"), s.get("lighting")) if bit))
                 for s in film.get("lookbook") or [] if s.get("tone") or s.get("palette")), "")
    return {"seed": {"world": world.strip(), "situation": film.get("logline") or "", "characters": characters,
                     "inspiration": film.get("name") or "", "slate": film.get("id")},
            "style": look, "name": film.get("name") or "Slate film"}


def clean_settings(given: dict, base: dict | None = None) -> dict:
    s = copy.deepcopy(base or DEFAULTS)
    given = given if isinstance(given, dict) else {}
    for key in ("style", "guidance", "music"):
        if given.get(key) is not None:
            s[key] = str(given[key]).strip()[:2000]
    for key in ("preset", "quality"):
        if given.get(key):
            s[key] = str(given[key])
    if given.get("keyframe_model") is not None:  # the picture model an edited keyframe is drawn with ("" for the Studio's default)
        s["keyframe_model"] = str(given["keyframe_model"]).strip()[:40]
    for key, low, high in (("seconds", 1, 15), ("segments", 0, LIMIT), ("seed", 0, 2 ** 31 - 1), ("max_scene_segments", 2, 50),
                           ("beat_history", 3, 50), ("retakes", 0, RETAKES), ("preview_every", 1, 50), ("width", 256, 1536), ("height", 256, 1536),
                           ("think_tokens", 0, 32768), ("max_minutes", 0, 60 * 24 * 30), ("max_film_seconds", 0, 10 ** 6), ("film_every", 1, 100),
                           ("film_window", 2, 50), ("story_every", 1, 100), ("min_seconds", 1, 15)):
        if given.get(key) not in (None, ""):
            try:
                s[key] = max(low, min(high, int(given[key])))
            except (TypeError, ValueError):
                raise HTTPException(400, f"{key} is a whole number from {low} to {high}")
    for key in ("qa", "ref_turbo", "think", "infinity"):
        if given.get(key) is not None:
            s[key] = bool(given[key])
    if given.get("checks") is not None:
        try:
            s["checks"] = checks.options({"checks": {**(s.get("checks") or {}), **(given["checks"] if isinstance(given["checks"], dict) else {})}})
        except ValueError as error:
            raise HTTPException(400, str(error))
    if given.get("keyframes") in KEYFRAMES:
        s["keyframes"] = given["keyframes"]
    if given.get("identity") in IDENTITY:
        s["identity"] = given["identity"]
    if given.get("director") in ("brain", "vision"):
        s["director"] = given["director"]
    if isinstance(given.get("styles"), list):
        styles = []
        for x in given["styles"][:12]:
            text = str((x or {}).get("style") or "").strip()[:1000] if isinstance(x, dict) else ""
            if text:
                first, last = span(x.get("segments")) if x.get("segments") not in (None, "") else (x.get("from"), x.get("to"))
                styles.append({"style": text, **({"from": int(first)} if first else {}), **({"to": int(last)} if last else {})})
        s["styles"] = styles
    if isinstance(given.get("loras"), list):
        s["loras"] = slate.lora_list(given["loras"])[:4]
    if given.get("continuity") is not None:
        try:
            s["continuity"] = continuity.clean(given["continuity"], s.get("continuity"))
        except ValueError as error:
            raise HTTPException(400, str(error))
    if isinstance(given.get("brain"), dict):
        b = given["brain"]
        s["brain"] = {"backend": str(b.get("backend") or slate.AUTO), "endpoint": slate.nearby_endpoint(str(b.get("endpoint") or "")),
                      "model": str(b.get("model") or "")}
    return s


# ---------- the routes ----------

def public(run: dict) -> dict:
    live = runs.get(run["id"])
    return {**run, "running": bool(live and live["thread"].is_alive())}


@router.get("/api/autopilot/runs")
def list_runs() -> list[dict]:
    out = []
    for file in sorted((HOME / "autopilot").glob("*/run.json"), key=lambda f: -f.stat().st_mtime):
        run = read_json(file)
        if not run:
            continue
        done = committed(run)
        out.append({"id": run["id"], "name": run.get("name", ""), "state": run.get("state"), "segments": len(done),
                    "seconds": round(sum(float(s.get("seconds") or 0) for s in done), 2), "updated": run.get("updated"),
                    "branch": run.get("branch"), "running": bool(runs.get(run["id"]) and runs[run["id"]]["thread"].is_alive())})
    return out


@router.post("/api/autopilot/runs")
def create_run(body: dict, request: Request) -> dict:
    """A new run: {name, seed: {world, situation, inspiration, characters: [{name, look, image}], opening_image},
    settings: {...DEFAULTS}, start: true to begin at once}."""

    import agents

    agents.local(request, True)  # the Director and the watcher run the user's brains: the same check as the Director's
    run_id = time.strftime("r%m%d-%H%M%S-") + uuid.uuid4().hex[:5]
    seed = clean_seed(body.get("seed") or {})
    settings = clean_settings(body.get("settings") or {})
    run = {"id": run_id, "name": (str(body.get("name") or "").strip() or "AutoPilot film")[:120], "created": slate.now(),
           "state": "idle", "error": "", "seed": seed, "settings": settings, "segments": [], "branch": None, "film": None,
           "cast": {c["name"]: c["look"] for c in seed["characters"] if c.get("look")}, "log": [], "current": {}, "rev": 0}
    folder = run_dir(run_id)
    folder.mkdir(parents=True)
    write_json(folder / "memory.json", copy.deepcopy(EMPTY))
    write_json(folder / "scene-graph.schema.json", SCHEMA)
    save_bible(run, continuity.new(seed, run["cast"]))
    say(run, "Created.")
    if body.get("start"):
        return public(begin(run))
    return public(run)


@router.get("/api/autopilot/runs/{run_id}")
def get_run(run_id: str) -> dict:
    run = load(run_id)
    return {**public(run), "memory": memory_of(run), "bible": bible_of(run)}


@router.put("/api/autopilot/runs/{run_id}")
def change_settings(run_id: str, body: dict, request: Request) -> dict:
    """New settings or a new name between segments: they apply from the next segment on."""

    import agents

    agents.local(request, True)
    live = runs.get(run_id)
    run = live["run"] if live and live["thread"].is_alive() else load(run_id)  # the loop reads them at its next segment
    with guard:
        if body.get("settings") is not None:
            run["settings"] = clean_settings(body["settings"], run["settings"])
        if body.get("name"):
            run["name"] = str(body["name"]).strip()[:120]
        save(run)
    return public(run)


def begin(run: dict) -> dict:
    with starting:
        live = runs.get(run["id"])
        if live and live["thread"].is_alive():
            raise HTTPException(409, "this run is already going")
        run["state"], run["error"], run["pause"] = "running", "", False
        for seg in run.get("segments") or []:  # a segment that failed is planned again on resume, at its own seed
            if seg.get("status") == "failed":
                seg["status"], seg["job"], seg["error"] = "planned", None, ""
        say(run, "Started." if not committed(run) else f"Resumed after segment {len(committed(run))}.")
        stop = threading.Event()
        thread = threading.Thread(target=work, args=(run, stop), daemon=True, name="autopilot-" + run["id"])
        runs[run["id"]] = {"thread": thread, "stop": stop, "run": run}
        thread.start()
        return run


@router.post("/api/autopilot/runs/{run_id}/start")
def start_run(run_id: str, request: Request) -> dict:
    """Start, or resume a paused, stopped or failed run where it stopped."""

    import agents

    agents.local(request, True)
    return public(begin(load(run_id)))


@router.post("/api/autopilot/runs/{run_id}/pause")
def pause_run(run_id: str, request: Request) -> dict:
    """Pause after the segment at work: it renders, is checked and joins the film; then the loop waits."""

    import agents

    agents.local(request, True)
    live = runs.get(run_id)
    if not live or not live["thread"].is_alive():
        return public(load(run_id))
    live["run"]["pause"] = True
    say(live["run"], "Pausing after this segment.")
    return public(live["run"])


@router.post("/api/autopilot/runs/{run_id}/stop")
def stop_run(run_id: str, request: Request) -> dict:
    """Stop now: a segment still waiting in the queue is taken out (it is planned again on resume); one rendering
    finishes and is picked up on resume."""

    import agents

    agents.local(request, True)
    live = runs.get(run_id)
    if live and live["thread"].is_alive():
        live["stop"].set()
        return {"stopping": True}
    run = load(run_id)
    if run.get("state") in ACTIVE:
        run["state"] = "paused"
        say(run, "Stopped.")
    return public(run)


@router.post("/api/autopilot/runs/{run_id}/branch")
def branch_run(run_id: str, body: dict, request: Request) -> dict:
    """A new run that shares segments 1..``after`` of this one (their clips stay where they are), with the memory as
    it was after segment ``after``, and goes on with new ``guidance``. {after, guidance, name?, segments?, seed?, start?}"""

    import agents

    agents.local(request, True)
    source = load(run_id)
    done = committed(source)
    try:
        after = int(body.get("after"))
    except (TypeError, ValueError):
        raise HTTPException(400, "after is the number of the segment to branch from")
    if not 1 <= after <= len(done):
        raise HTTPException(400, f"branch from a committed segment, 1 to {len(done)}")
    guidance = str(body.get("guidance") or "").strip()
    if not guidance:
        raise HTTPException(400, "give the branch its direction: what should happen from here")
    snapshot = read_json(run_dir(run_id) / "memory" / f"S{after:04d}.json")
    if not snapshot:
        raise HTTPException(409, f"segment {after} has no memory snapshot to branch from")
    new_id = time.strftime("r%m%d-%H%M%S-") + uuid.uuid4().hex[:5]
    settings = clean_settings({k: body[k] for k in ("segments", "seed") if body.get(k) not in (None, "")}, source["settings"])
    if body.get("segments") in (None, "") and settings["segments"]:
        settings["segments"] = after + max(1, settings["segments"] - after)
    elif settings["segments"]:
        settings["segments"] = after + settings["segments"]  # a number given is how many new segments
    shared = [{**copy.deepcopy(s), "shared": run_id} for s in done[:after]]
    run = {"id": new_id, "name": (str(body.get("name") or "").strip() or f"{source.get('name')} · branch at {after}")[:120],
           "created": slate.now(), "state": "idle", "error": "", "seed": copy.deepcopy(source["seed"]), "settings": settings,
           "segments": shared, "branch": {"from": run_id, "after": after, "guidance": guidance[:2000]}, "film": None,
           "cast": copy.deepcopy(snapshot.get("cast") or source.get("cast") or {}), "log": [], "current": {}, "rev": 0}
    folder = run_dir(new_id)
    folder.mkdir(parents=True)
    memory = {key: copy.deepcopy(snapshot.get(key, EMPTY[key])) for key in EMPTY}
    write_json(folder / "memory.json", memory)
    for seg in shared:
        mark = read_json(run_dir(run_id) / "memory" / f"S{seg['index']:04d}.json")
        if mark:
            write_json(folder / "memory" / f"S{seg['index']:04d}.json", mark)
    write_json(folder / "scene-graph.schema.json", SCHEMA)
    bible = read_json(run_dir(run_id) / "bible" / f"S{after:04d}.json")
    save_bible(run, bible if isinstance(bible, dict) and bible.get("version") else continuity.new(run["seed"], run["cast"]))
    for seg in shared:
        mark = read_json(run_dir(run_id) / "bible" / f"S{seg['index']:04d}.json")
        if mark:
            write_json(folder / "bible" / f"S{seg['index']:04d}.json", mark)
    say(run, f"Branched from {source.get('name')} after segment {after}: {guidance[:200]}")
    if body.get("start"):
        return public(begin(run))
    return public(run)


@router.post("/api/autopilot/runs/{run_id}/assemble")
def assemble_now(run_id: str, request: Request) -> dict:
    import agents

    agents.local(request, True)
    live = runs.get(run_id)
    run = live["run"] if live and live["thread"].is_alive() else load(run_id)  # the loop's own copy while it runs
    if not committed(run):
        raise HTTPException(400, "no segment is finished yet")
    publish(run)
    return public(run)


@router.post("/api/autopilot/runs/{run_id}/film-check")
def film_check_now(run_id: str, request: Request) -> dict:
    """The film-level continuity pass (report only), in the background: the run must not be at work."""

    import agents

    agents.local(request, True)
    live = runs.get(run_id)
    if live and live["thread"].is_alive():
        raise HTTPException(409, "the run is at work: the film is checked when it finishes, or pause it first")
    run = load(run_id)
    if len(committed(run)) < 2:
        raise HTTPException(400, "the film needs two segments or more to be compared")
    run["film_check"] = {"state": "running", "at": slate.now()}
    say(run, "Checking the whole film's continuity.", stage="film-check")
    threading.Thread(target=film_check_safely, args=(run,), daemon=True, name="autopilot-film-" + run_id).start()
    return public(run)


@router.get("/api/autopilot/vision")
def vision_state() -> dict:
    return cosmos.summary()


@router.post("/api/autopilot/vision")
def vision_save(body: dict, request: Request) -> dict:
    """{endpoint: "studio:cosmos" | a nearby http://host:port | "" (the QA gate's eyes), model, frames}"""

    import agents

    agents.local(request, True)
    endpoint = str(body.get("endpoint") if body.get("endpoint") is not None else cosmos.settings()["endpoint"]).strip()
    if endpoint and endpoint.lower() != cosmos.STUDIO:
        slate.nearby_endpoint(endpoint)  # 400 unless it is on this computer, this network or the tailnet
    try:
        frames = int(body["frames"]) if body.get("frames") not in (None, "") else None
    except (TypeError, ValueError):
        raise HTTPException(400, "frames is a whole number")
    cosmos.save_settings({"endpoint": endpoint, "model": str(body.get("model") or ""), **({"frames": frames} if frames else {})})
    return cosmos.summary()


def resume_all() -> None:
    for file in (HOME / "autopilot").glob("*/run.json"):
        run = read_json(file)
        if run and run.get("state") in ACTIVE and run["id"] not in runs:
            try:
                say(run, "Picked up after a restart.")
                begin(run)
            except HTTPException:
                pass


# ---------- the loop ----------

def wait(stop: threading.Event, seconds: float | None = None) -> None:
    if stop.wait(TICK if seconds is None else seconds):
        raise Stopped()


def work(run: dict, stop: threading.Event) -> None:
    misses, began = 0, time.time()
    while True:
        try:
            if step(run, stop, began):
                return
            misses = 0
        except Stopped:
            pending = next((s for s in run.get("segments") or [] if s.get("status") == "queued"), None)
            if pending and (job_of(pending.get("job") or "") or {}).get("status") == "queued":
                cancel(pending["job"])
                pending["status"], pending["job"] = "planned", None
            run["state"] = "paused"
            say(run, "Stopped. Resume to carry on from here: every finished segment and the memory are kept.")
            return
        except Exception as error:  # noqa: BLE001 - the reason goes to the page; resume tries again
            why = str(getattr(error, "detail", error))[:600]
            if run["settings"].get("infinity") and misses < INFINITY_RETRIES and not stop.is_set():
                # Infinity goes on by itself: the piece at work is planned again from its checkpoint after a pause
                misses += 1
                for seg in run.get("segments") or []:
                    if seg.get("status") == "failed":
                        seg["status"], seg["job"], seg["error"] = "planned", None, ""
                say(run, f"A problem: {why[:240]}. Infinity carries on from the last checkpoint in {INFINITY_WAIT_S} s "
                         f"(try {misses} of {INFINITY_RETRIES}).")
                if stop.wait(INFINITY_WAIT_S):
                    run["state"] = "paused"
                    say(run, "Stopped. Resume to carry on from here: every finished segment and the memory are kept.")
                    return
                continue
            run["state"], run["error"] = "failed", why
            say(run, "Stopped on a problem: " + run["error"])
            return


def finished(run: dict, began: float) -> str:
    """Why the run is complete ("" while it goes on): its number of segments, or, in Infinity, a time or length limit
    or an ending the Director chose."""

    s, done = run["settings"], committed(run)
    if not s.get("infinity"):
        limit = int(s.get("segments") or 0)
        return f"{limit} segments" if limit and len(done) >= limit else ""
    if done and done[-1].get("end"):
        return "the Director ended the story"
    seconds = sum(float(x.get("seconds") or 0) for x in done)
    if s.get("max_film_seconds") and seconds >= float(s["max_film_seconds"]):
        return f"the film reached {seconds:.0f} s of its {float(s['max_film_seconds']):.0f}"
    minutes = (float(run.get("worked_s") or 0) + time.time() - began) / 60
    if s.get("max_minutes") and minutes >= float(s["max_minutes"]):
        return f"{minutes:.0f} minutes of its {float(s['max_minutes']):.0f}"
    return ""


def step(run: dict, stop: threading.Event, began: float) -> bool:
    """One turn of the loop; True when the run has paused or is complete."""

    if stop.is_set():
        raise Stopped()
    if run.get("pause"):
        run["state"], run["pause"] = "paused", False
        run["worked_s"] = float(run.get("worked_s") or 0) + time.time() - began
        if (run.get("film") or {}).get("segments") != len(committed(run)):
            publish(run)
        say(run, f"Paused after segment {len(committed(run))}. Resume to carry on.")
        return True
    why = finished(run, began)
    if why:
        if (run.get("film") or {}).get("segments") != len(committed(run)):
            publish(run)
        if continuity.options(run["settings"])["film_pass"] and len(committed(run)) >= 2 and not (run.get("film_check") or {}).get("segments") == len(committed(run)):
            film_check_safely(run)
        run["state"] = "done"
        run["worked_s"] = float(run.get("worked_s") or 0) + time.time() - began
        say(run, f"Done ({why}): {len(committed(run))} segments, {run['film']['seconds'] if run.get('film') else 0:.1f} s of film.")
        return True
    segment(run, stop)
    s, n = run["settings"], len(committed(run))
    if s.get("infinity") or n == 1 or n % int(s.get("preview_every") or 3) == 0:
        publish(run)  # Infinity: the film so far is joined after every piece, always playable
        checkpoint(run, committed(run)[-1])
    if s.get("infinity") and continuity.options(s)["film_pass"] and n >= 2 and n % max(1, int(s.get("film_every") or 4)) == 0:
        film_check_safely(run, int(s.get("film_window") or 6))
    if s.get("infinity") and n % max(1, int(s.get("story_every") or 6)) == 0:
        story_so_far(run)
    return False


def story_so_far(run: dict) -> None:
    """The running summary the Director reads in Infinity (bounded: the memory's chapters and the last beats folded
    into 120 words), written again every ``story_every`` pieces; a summary that cannot be made keeps the old one."""

    memory, done = memory_of(run), committed(run)
    beats = "\n".join(f"  {x['index']}. {x.get('beat', '')[:110]}" for x in done[-12:])
    prompt = ((f"The story so far, as last summarised:\n{run['story_so_far']['text']}\n\n" if (run.get("story_so_far") or {}).get("text") else "")
              + f"What the film remembers:\n{as_text(memory['graph'])}\n\nThe latest segments' beats:\n{beats}\n\n"
              'Write the story so far in 120 words at most: who wants what, what has happened that still matters, where things stand now. '
              'Answer: {"text": "<120 words max>"}')
    try:
        data, _ = ask_json(lambda text: ask_text(run, COMPACT_SYSTEM, text), lambda d: "" if str(d.get("text") or "").strip() else '"text" is the summary', prompt, "story summary")
        run["story_so_far"] = {"text": clip_words(data["text"], 130), "through": len(done), "at": slate.now()}
        save(run)
    except Failed as error:
        say(run, "The story so far could not be summarised again: " + str(error)[:160])


def segment(run: dict, stop: threading.Event) -> None:
    """One segment from plan to commit; a segment left half-way (a stop, a restart) is picked up where it was."""

    seg = next((s for s in run["segments"] if s.get("status") not in ("done",)), None)
    if seg is None:
        seg = plan(run)
    while seg["status"] != "done":
        if seg["status"] == "planned":
            render(run, seg)
        elif seg["status"] == "queued":
            follow(run, seg, stop)
        elif seg["status"] == "rendered":
            check(run, seg)
        elif seg["status"] == "keyframe":
            render_keyframe(run, seg)
        elif seg["status"] == "keyframe_queued":
            follow_keyframe(run, seg, stop)
        elif seg["status"] == "keyframe_ready":
            check_keyframe(run, seg)
        elif seg["status"] == "retake":
            fixing = bool(seg.get("fix"))
            if fixing:
                replan(run, seg)  # the Director rewrites the shot with the findings in hand
            seg["seed"] = int(seg["seed"]) + 1
            seg["status"] = "keyframe" if fixing and keyframes_for(run, seg["index"]) else "planned"  # a new keyframe from the new shot
            if seg["status"] == "keyframe":
                seg["keyframe"] = None
            save(run)
        elif seg["status"] == "failed":
            raise Failed(f"segment {seg['index']} could not be rendered: {seg.get('error') or 'no reason'}")


# ---------- plan ----------

def frames_of(settings: dict) -> int:
    return lengths.nearest(float(settings.get("seconds") or 5))


def director_prompt(run: dict, index: int, seconds: float) -> tuple[str, dict | None]:
    """What the Director reads for segment ``index``, and the stagnation it must break (None when there is none)."""

    ask = KEYFRAME_ASK if keyframes_for(run, index) else ""
    if index == 1:
        bible = continuity.director_text(bible_of(run), 1, list(cast_at(run, 1)))
        return first_prompt(run, seconds) + f"\n\nThe continuity bible so far (canonical):\n{bible}" + ask, None
    done = committed(run)
    memory = memory_of(run)
    stuck = stagnation(done)
    bible = continuity.director_text(bible_of(run), index, list(cast_at(run, index)))
    return next_prompt(run, index, seconds, as_text(memory["graph"]), stagnation_note(stuck) if stuck else "", bible) + ask, stuck


def keyframes_for(run: dict, index: int) -> bool:
    """Whether segment ``index`` opens on a keyframe of its own: the run's choice (auto: when it has an outline), and
    never for a beat that brings its own picture."""

    choice = run["settings"].get("keyframes", "auto")
    wanted = choice == "on" or (choice == "auto" and bool((run.get("seed") or {}).get("outline")))
    return wanted and not (beat_of(run, index) or {}).get("image")


# the video sizes of the presets AutoPilot offers (server.PRESETS): an edited keyframe is made at the video's own size
PRESET_SIZES = {"sharp": (1376, 768), "small": (864, 480), "p720": (1280, 736), "native": (1344, 768), "draft": (1344, 768)}

KEYFRAME_ASK = ("\nThis segment opens on a keyframe made first by the picture model: fill \"keyframe\" with that opening still "
                "(who is where, the framing, the place, the light; 60 words at most), and write the action from that picture on. "
                "A fight's keyframe catches a blow already in motion (a fist or a weapon mid-swing, a body leaning into the "
                "strike), never a stand-off: the video model eases slowly out of a still pose.")


def director_asker(run: dict):
    if run["settings"].get("director") == "vision":
        return lambda text: ask_vision_text(run, prompt_with_system(text))
    return lambda text: ask_text(run, DIRECTOR_SYSTEM, text)


def fields_of(shot: dict) -> dict:
    return {"setting": str(shot.get("setting") or ""), "camera": str(shot.get("camera") or ""), "action": str(shot.get("action") or ""),
            "dialogue": [d for d in shot.get("dialogue") or [] if isinstance(d, dict) and str(d.get("text") or "").strip()][:3],
            "soundscape": str(shot.get("soundscape") or ""), "music": str(shot.get("music") or "N/A")}


def plan(run: dict) -> dict:
    s, done = run["settings"], committed(run)
    index = len(done) + 1
    frames = frames_of(s)
    seconds = lengths.seconds_of(frames)
    run["state"] = "running"
    say(run, f"Segment {index}: the Director writes it.", stage="director", segment=index)
    bible = bible_of(run)
    continuity.arrive(bible, index, run["seed"])  # a cast sheet that starts here opens an era before the Director reads it
    save_bible(run, bible)
    prompt, stuck = director_prompt(run, index, seconds)

    def check_answer(data: dict) -> str:
        return director_problem(run, data, index, stuck)

    data, rejected = ask_json(director_asker(run), check_answer, prompt, "Director")
    shot = data["shot"]
    if index == 1 or (beat_of(run, index) or {}).get("image"):
        data["scene_change"] = True  # the opening, or a beat that opens on a picture of its own
    for person in data.get("new_characters") or []:
        if isinstance(person, dict) and person.get("name") and person.get("look"):
            run.setdefault("cast", {}).setdefault(str(person["name"]).strip(), str(person["look"]).strip())
    bible = bible_of(run)
    continuity.planned(bible, index, data, run.get("cast") or {})
    fields = fields_of(shot)
    continuity.voices(bible, index, fields["dialogue"])  # each speaker keeps the voice the bible first heard
    save_bible(run, bible)
    seg = {"index": index, "id": f"S{index:04d}", "status": "planned", "scene": str(data["scene"]).strip()[:80],
           "scene_change": bool(data["scene_change"]), "beat": str(data["beat"]).strip()[:200], "fields": fields,
           "rejected": rejected, "seed": int(s.get("seed") or 0) + index - 1, "frames": frames, "seconds": seconds,
           "start": round(sum(float(x.get("seconds") or 0) for x in done), 3), "takes": [], "job": None, "output": None,
           "era": continuity.era_at(bible, index)["id"], "planned": slate.now(), "pace": checks.pace_of(data.get("pace")),
           **({"end": True} if data.get("end") is True and s.get("infinity") else {})}
    if keyframes_for(run, index):
        seg.update(status="keyframe", keyframe_prompt=clip_words(data.get("keyframe"), 70), keyframes=[])
    folder = run_dir(run["id"]) / "segments" / seg["id"]
    write_json(folder / "director.json", {"prompt": prompt, "answer": data, "rejected": rejected})
    run["segments"].append(seg)
    if rejected:
        say(run, f"Segment {index}: {len(rejected)} answer{'s' if len(rejected) != 1 else ''} of the Director turned back ({rejected[-1][:120]}).", segment=index)
    say(run, f"Segment {index}: {seg['beat']}", stage="planned", segment=index)
    checkpoint(run, next_seg=seg)
    return seg


def replan(run: dict, seg: dict) -> None:
    """A take that failed its continuity check written again by the Director, with each finding: the same story step
    in the same scene, the shot corrected. The next take renders it at a new seed."""

    index, take = seg["index"], len(seg["takes"])
    say(run, f"Segment {index}: the Director rewrites take {take} with the continuity findings.", stage="director", segment=index)
    prompt, stuck = director_prompt(run, index, float(seg["seconds"]))
    found = "\n".join(f"- {f['category']}" + (f" (frame {f['frame']})" if f.get("frame") else "") + f": {f['evidence']}" for f in seg.get("fix") or [])
    if any(f.get("category") == "pace" for f in seg.get("fix") or []):
        found += ("\nThe video model made it slow: write the fight as fast, real-time action (quick strikes, rapid exchanges, sharp "
                  "impacts, whip pans), name the speed plainly, and keep slow motion to the one key move with its seconds.")
    prompt += (f"\n\nTHE TAKE YOU WROTE FOR SEGMENT {index} WAS RENDERED AND FAILED ITS CONTINUITY CHECK:\n{found}\n"
               f"Write segment {index} again: the same story step (\"{seg['beat']}\") in the same scene ({seg['scene']!r}), with the shot "
               "rewritten so that each of these comes out right; say the bible's details plainly where they matter. Keep \"scene\" and "
               "\"scene_change\" as they were.")

    def check_answer(data: dict) -> str:
        return director_problem(run, data, index, stuck if seg["scene_change"] else None)

    data, rejected = ask_json(director_asker(run), check_answer, prompt, "Director")
    fields = fields_of(data["shot"])
    bible = bible_of(run)
    continuity.voices(bible, index, fields["dialogue"])
    save_bible(run, bible)
    if keyframes_for(run, index) and str(data.get("keyframe") or "").strip():
        seg["keyframe_prompt"] = clip_words(data["keyframe"], 70)
    seg.update(fields=fields, beat=str(data["beat"]).strip()[:200] or seg["beat"], rejected=(seg.get("rejected") or []) + rejected, pace=checks.pace_of(data.get("pace")),
               fixed=(seg.get("fixed") or []) + [{"take": take, "findings": seg.get("fix") or []}], fix=[])
    write_json(run_dir(run["id"]) / "segments" / seg["id"] / f"director-take{take + 1}.json", {"prompt": prompt, "answer": data, "rejected": rejected})


def prompt_with_system(text: str) -> str:
    return DIRECTOR_SYSTEM + "\n\n" + text


def ask_vision_text(run: dict, text: str) -> str:
    """The Director on the vision server, from text alone (``director: vision``)."""

    with cosmos.request() as (endpoint, model):
        return cosmos.chat(endpoint, model, "Answer with JSON only.", [], text)


def director_problem(run: dict, data: dict, index: int, stuck) -> str:
    """Why a Director answer cannot be used, or "" when it can."""

    shot = data.get("shot")
    if not isinstance(shot, dict) or not str(shot.get("action") or "").strip():
        return '"shot" is an object with at least "action"'
    if not str(data.get("scene") or "").strip() or not str(data.get("beat") or "").strip():
        return '"scene" and "beat" are short lines of text'
    if not isinstance(data.get("scene_change"), bool):
        return '"scene_change" is true or false'
    fields = {"scene": str(shot.get("setting") or ""), "camera": str(shot.get("camera") or ""), "action": str(shot.get("action") or ""),
              "soundscape": str(shot.get("soundscape") or ""), "dialogue": [d for d in shot.get("dialogue") or [] if isinstance(d, dict)]}
    for key, most in (("action", 60), ("camera", 25), ("scene", 45)):
        if len(fields[key].split()) > most:
            return f'"{"setting" if key == "scene" else key}" is {len(fields[key].split())} words: keep it to {most} at most, short and concrete'
    if keyframes_for(run, index) and not str(data.get("keyframe") or "").strip():
        return '"keyframe" is the still picture this segment opens on'
    if data.get("end") is not None and not isinstance(data.get("end"), bool):
        return '"end" is true or false'
    paced = checks.pace_problem(checks.pace_of(data.get("pace")), fields["action"], fields["camera"], lengths.seconds_of(frames_of(run["settings"])))
    if paced:
        return paced
    named = negation.check_fields(fields)["negations"]
    if named:
        return (f"\"{named[0]['phrase']}\" names what must not be seen; the video model has no negative prompt and what is "
                "named tends to appear. Say what is there instead")
    if index == 1:
        return ""
    done = committed(run)
    previous = done[-1]
    action = fields["action"].strip().lower()
    if difflib.SequenceMatcher(None, action, str(previous["fields"].get("action") or "").strip().lower()).ratio() >= SIMILAR_ACTION:
        return "the action repeats the segment before almost word for word; move the story one visible step on"
    beat = str(data["beat"]).strip().lower()
    for seg in done[-int(run["settings"].get("beat_history") or 12):]:
        if difflib.SequenceMatcher(None, beat, str(seg.get("beat") or "").strip().lower()).ratio() >= SIMILAR_BEAT:
            return f"the beat restages segment {seg['index']} (\"{seg.get('beat')}\"); choose a materially different consequence"
    budget = int(run["settings"].get("max_scene_segments") or 8)
    if scene_length(done) >= budget and not data["scene_change"]:
        return f"the scene {previous.get('scene')!r} has used its {budget}-segment budget: set scene_change to true and move on"
    if stuck and (not data["scene_change"] or str(data["scene"]).strip().lower() == str(previous.get("scene") or "").strip().lower()):
        return "the film is circling: set scene_change to true with a different scene name"
    return ""


# ---------- render ----------

def pictured(run: dict, index: int) -> list[dict]:
    """The cast with a reference picture in segment ``index``."""

    active = cast_at(run, index)
    return [c for c in run["seed"].get("characters") or [] if c.get("image") and on_disk(c["image"])
            and (c["name"] in active or not c.get("look"))
            and (c.get("from") or 1) <= index <= (c.get("to") or 10 ** 6)][:9]


def references_for(run: dict, index: int = 1) -> list[str]:
    return [c["image"] for c in pictured(run, index)]


def mode_of(run: dict, seg: dict) -> str:
    """How a segment opens: on the segment before (``previous``), the cast's pictures (``references``), the opening
    picture (``image``) or from text."""

    if seg.get("keyframe") and on_disk(seg["keyframe"]):
        return "keyframe"  # the piece's own keyframe, made and checked first
    identity = run["settings"].get("identity", "auto")
    pictures = bool(references_for(run, seg["index"])) and identity != "frame" and ref2va()
    if (beat_of(run, seg["index"]) or {}).get("image"):
        return "image"  # the user's own opening picture for this beat
    if seg["index"] > 1 and not seg["scene_change"]:
        return "previous"
    if pictures:
        return "references"
    if seg["index"] == 1 and run["seed"].get("opening_image"):
        return "image"
    return "text"


def wardrobe_at(run: dict, name: str, index: int) -> str:
    """The exact wardrobe line of ``name`` in segment ``index``: the seed sheet's for its segments (continuity.wardrobe_line)."""

    for c in run["seed"].get("characters") or []:
        if c.get("name") == name and (c.get("from") or 1) <= index <= (c.get("to") or 10 ** 6):
            return continuity.wardrobe_line(c)
    return ""


def look_lines(run: dict, fields: dict, index: int) -> str:
    """The pinned looks of the cast this segment names, said every time (the renderer has no memory), each with its
    exact wardrobe line again: "the same clothes in every shot"."""

    text = " ".join(str(fields.get(k) or "") for k in ("setting", "action")).lower()
    out = []
    for name, look in cast_at(run, index).items():
        if look and name.lower().split()[0] in text:
            out.append(f"{name} is {look.rstrip('.')}.")
            wear = wardrobe_at(run, name, index)
            if wear:
                out.append(f"{name} wears exactly this in every shot: {wear.rstrip('.')}.")
    return " ".join(out)


def prompt_of(run: dict, seg: dict, mode: str) -> str:
    s, f = run["settings"], dict(seg["fields"])
    pace = checks.pace_of(seg.get("pace"))
    paced = checks.pace_line(pace, float(seg.get("seconds") or 10))
    if paced:
        # a fight starts in motion: the speed is the first thing the action says, the plan's pace after it
        f["action"] = (checks.pace_opening(pace) + " " + f.get("action", "").strip() + " " + paced).strip()
    style = style_at(s, seg["index"])  # only this segment's era or setting: a style for every era bled Boston into the sea
    music = f.get("music") or "N/A"
    if s.get("music") and music.upper() == "N/A":
        music = s["music"]
    dialogue = [{"speaker": d.get("speaker") or "", "voice": d.get("voice") or "a natural voice", "text": str(d["text"]).strip(), "language": "English"}
                for d in f.get("dialogue") or []]
    if mode == "references":
        names = [c["name"] for c in pictured(run, seg["index"])]
        for line in dialogue:
            line["subject"] = names.index(line["speaker"]) + 1 if line["speaker"] in names else 1
        wear = " ".join(f"{n} wears exactly this in every shot: {wardrobe_at(run, n, seg['index']).rstrip('.')}." for n in names if wardrobe_at(run, n, seg["index"]))
        fields = {"scene": " ".join(bit for bit in (style, wear, f.get("setting", "")) if bit), "camera": f.get("camera", ""),
                  "action": f.get("action", ""), "dialogue": dialogue, "soundscape": f.get("soundscape", ""), "music": music,
                  "subjects": names, "people": True}
        return compose_references(fields, len(names))
    scene = " ".join(bit for bit in (style, look_lines(run, f, seg["index"]), f.get("setting", "")) if bit)
    fields = {"scene": scene, "camera": f.get("camera", ""), "action": f.get("action", ""), "dialogue": dialogue,
              "soundscape": f.get("soundscape", ""), "music": music}
    return compose(fields, mode in ("previous", "image", "keyframe"), frames=seg["frames"])


def render(run: dict, seg: dict) -> None:
    s = run["settings"]
    done = committed(run)
    mode = mode_of(run, seg)
    take = len(seg["takes"]) + 1
    output = f"autopilot/{run['id']}/segments/{seg['id']}/take{take}.mp4"
    prompt = prompt_of(run, seg, mode)
    params = {"prompt": prompt, "preset": s["preset"], "frames": seg["frames"], "seed": int(seg["seed"]), "output": output,
              "qa": False,  # the gate runs here, with the watcher, once per take (check)
              "brain": s.get("brain") or {}, "autopilot": {"run": run["id"], "segment": seg["index"]}}
    if s["preset"] == "custom" and s.get("width") and s.get("height"):
        params.update(width=int(s["width"]), height=int(s["height"]))
    if mode == "references":
        params["references"] = references_for(run, seg["index"])
        if s.get("ref_turbo") and ref2va_turbo():
            params["ref_turbo"] = True
    else:
        params["quality"] = s["quality"]
        if s.get("loras"):
            params["loras"] = s["loras"]
    if mode == "previous":
        params["first_frame_from"] = done[-1]["output"]
    if mode == "keyframe":
        params["image"] = seg["keyframe"]
    if mode == "image":
        params["image"] = (beat_of(run, seg["index"]) or {}).get("image") or run["seed"]["opening_image"]
    fitted = fitting(run, params)
    if fitted != params["frames"]:
        shorten(run, seg, "memory first: at this size only a shorter piece fits this machine now", fitted)
        params["frames"] = seg["frames"]
        params["prompt"] = prompt = prompt_of(run, seg, mode)
    while True:
        try:
            job = submit({"kind": "video", "title": f"AutoPilot · {run['name']} · {seg['index']}: {seg['beat'][:60]}", "params": params})
            break
        except HTTPException as error:
            shorter = re.search(r"What fits: (\d+) frames", str(error.detail))
            if not shorter or not shorten(run, seg, "the queue's memory check: " + str(error.detail)[:200], int(shorter[1])):
                raise Failed(f"the queue refused segment {seg['index']}: {error.detail}") from None
            params["frames"] = seg["frames"]
            params["prompt"] = prompt = prompt_of(run, seg, mode)
    seg.update(status="queued", job=job["id"], mode=mode, prompt=prompt, output=output,
               settings={k: params.get(k) for k in ("preset", "quality", "frames", "width", "height", "loras", "ref_turbo") if params.get(k) is not None})
    say(run, f"Segment {seg['index']}: rendering" + (f", take {take} at seed {seg['seed']}" if take > 1 else "")
        + {"previous": ", from the last frame of the segment before", "references": ", from the cast's pictures (a new scene)",
           "image": ", from the opening picture", "keyframe": ", from its keyframe", "text": ", from text (a new scene)"}[mode] + ".", stage="render", segment=seg["index"], job=job["id"])


def follow(run: dict, seg: dict, stop: threading.Event) -> None:
    while True:
        job = job_of(seg.get("job") or "")
        if job is None:  # the app restarted: the clip is there, or the take is queued again
            seg["status"] = "rendered" if on_disk(seg.get("output")) else "planned"
            save(run)
            return
        status = job.get("status")
        if status == "done":
            seg["output"] = (job.get("outputs") or [seg["output"]])[0]
            seg["status"] = "rendered" if on_disk(seg["output"]) else "failed"
            seg["error"] = "" if seg["status"] == "rendered" else "the job finished without its clip"
            save(run)
            return
        if status == "cancelled":
            seg["status"], seg["job"] = "planned", None
            save(run)
            raise Stopped()
        if status == "failed":
            error = job.get("error") or "the render failed"
            if MEMORY_STOP.search(error) and shorten(run, seg, "the render was stopped for memory: " + clip_words(error, 40)):
                seg["status"], seg["job"] = "planned", None  # the same piece again, shorter, at the same size
                save(run)
                return
            seg["status"], seg["error"] = "failed", error
            save(run)
            return
        wait(stop)


# ---------- memory first ----------

# what a render that ran out of room says (spark_generate.py, memory_guard.py, the queue)
MEMORY_STOP = re.compile(r"memory|protect the machine|refused before anything was loaded|out of memory|\bOOM\b|NV_ERR_NO_MEMORY|killed", re.I)


def shortest(run: dict) -> int:
    return lengths.nearest(float(run["settings"].get("min_seconds") or DEFAULTS["min_seconds"]))


def fitting(run: dict, params: dict) -> int:
    """The longest length, up to the piece's own, that fits this machine's memory now at the piece's size (the memory
    guard's estimate with what is available now); the piece's own when it fits, or when nothing shorter does (the
    queue then says why)."""

    try:
        said = room(params)
    except Exception:  # noqa: BLE001 - no estimate: the queue and the render's own checks still stand
        said = None
    if not said or said.get("fits", True):
        return params["frames"]
    shorter = (said.get("nearest") or {}).get("frames")
    return shorter if shorter and shortest(run) <= shorter < params["frames"] else params["frames"]


def shorten(run: dict, seg: dict, why: str, frames: int | None = None) -> bool:
    """The piece made shorter at the same size: to ``frames``, or the next length the model takes (the memory guard
    stopped it). False when it is at the run's shortest already or has been shortened MEMORY_RETRIES times."""

    events = seg.setdefault("memory", [])
    if sum(1 for e in events if e.get("retry")) >= MEMORY_RETRIES:
        return False
    counts = [c for c in lengths.counts() if shortest(run) <= c < int(seg["frames"])]
    # a stop by the guard: a fifth shorter (one step of 17 frames is 0.7 s, too little to change the peak much)
    goal = frames or int(int(seg["frames"]) * 0.8)
    target = max([c for c in counts if c <= goal], default=None) or (min(counts) if counts and not frames else None)
    if not target:
        return False
    events.append({"at": slate.now(), "from": seg["frames"], "to": target, "why": why[:300], "retry": not why.startswith("memory first")})
    ratio = lengths.seconds_of(target) / max(0.1, float(seg.get("seconds") or lengths.seconds_of(int(seg["frames"]))))
    for beat in (seg.get("pace") or {}).get("beats") or []:  # the pace plan's seconds, scaled to the new length
        beat["from"], beat["to"] = round(beat["from"] * ratio, 2), round(beat["to"] * ratio, 2)
    seg["frames"], seg["seconds"] = target, lengths.seconds_of(target)
    say(run, f"Segment {seg['index']}: {why[:160]}; now {lengths.seconds_of(target):.1f} s ({target} frames) at the same size.", segment=seg["index"])
    return True


# ---------- keyframes ----------

def render_keyframe(run: dict, seg: dict) -> None:
    """The piece's opening still: the Director's keyframe prompt, with this segment's style and the pinned looks of
    whoever it names, as one picture of the queue (a ``scout`` job of one) at the video's size."""

    s, n = run["settings"], len(seg.get("keyframes") or []) + 1
    round_ = len(seg.get("takes") or [])  # the keyframes of this video take (a rewritten shot gets new ones)
    looks = look_lines(run, {"setting": seg.get("keyframe_prompt", ""), "action": seg["fields"].get("action", "")}, seg["index"])
    prompt = " ".join(bit for bit in (style_at(s, seg["index"]), looks, seg.get("keyframe_prompt") or seg["fields"].get("setting", "")) if bit)
    params = {"prompt": prompt, "count": 1, "seed": int(seg["seed"]) + 7919 * (n - 1), "preset": s["preset"], "autopilot": {"run": run["id"], "segment": seg["index"], "keyframe": n},
              **({"image_model": s["keyframe_model"]} if s.get("keyframe_model") else {})}
    if s["preset"] == "custom" and s.get("width") and s.get("height"):
        params.update(width=int(s["width"]) // 16 * 16, height=int(s["height"]) // 16 * 16)
    kind = "scout"
    named = " ".join((seg.get("keyframe_prompt") or "", seg["fields"].get("action", ""), seg["fields"].get("setting", ""))).lower()
    portraits = [c for c in pictured(run, seg["index"]) if c["name"].lower().split()[0] in named][:3]
    if portraits:
        # the era's reference portrait is the picture the keyframe is drawn from (an edit of it, the others beside it):
        # face, hair and clothes come from the picture, not from words the picture model reads its own way
        kind = "edit"
        who = "; ".join(f"image {i + 1} is {c['name']}" for i, c in enumerate(portraits))
        params = {"image": portraits[0]["image"], "images": [c["image"] for c in portraits[1:]], "seed": params["seed"],
                  "prompt": (f"{who}. Keep each one's face, hair, beard and clothes exactly as in their picture. Make this film still: "
                             + prompt), "autopilot": params["autopilot"],
                  **({"image_model": s["keyframe_model"]} if s.get("keyframe_model") else {})}
        size = PRESET_SIZES.get(s["preset"]) or ((int(s["width"]) // 16 * 16, int(s["height"]) // 16 * 16) if s.get("width") and s.get("height") else None)
        if size:
            params.update(width=size[0], height=size[1])
    try:
        job = submit({"kind": kind, "title": f"AutoPilot · {run['name']} · {seg['index']}: keyframe {n}", "params": params})
    except HTTPException as error:
        raise Failed(f"the queue refused the keyframe of segment {seg['index']}: {error.detail}") from None
    seg.setdefault("keyframes", []).append({"take": n, "round": round_, "job": job["id"], "prompt": prompt, "seed": params["seed"], "path": None,
                                            "kind": kind, **({"from": params["image"]} if kind == "edit" else {})})
    seg.update(status="keyframe_queued", job=job["id"])
    say(run, f"Segment {seg['index']}: its keyframe" + (f", take {n}" if n > 1 else "") + ".", stage="keyframe", segment=seg["index"], job=job["id"])


def follow_keyframe(run: dict, seg: dict, stop: threading.Event) -> None:
    last = seg["keyframes"][-1]
    while True:
        job = job_of(last["job"])
        if job is None:  # the app restarted: made again
            seg["keyframes"].pop()
            seg["status"] = "keyframe"
            save(run)
            return
        if job.get("status") == "done":
            picture = next((o for o in job.get("outputs") or [] if str(o).endswith(".png")), None)
            picture = str(Path(picture).relative_to(HOME)) if picture and Path(picture).is_absolute() and HOME.resolve() in Path(picture).resolve().parents else picture
            if not on_disk(picture):
                seg["status"], seg["error"] = "failed", "the keyframe job finished without its picture"
            else:
                last["path"], seg["status"] = picture, "keyframe_ready"
            save(run)
            return
        if job.get("status") == "cancelled":
            seg["keyframes"].pop()
            seg["status"], seg["job"] = "keyframe", None
            save(run)
            raise Stopped()
        if job.get("status") == "failed":
            seg["status"], seg["error"] = "failed", job.get("error") or "the keyframe failed"
            save(run)
            return
        wait(stop)


KEYFRAME_CATEGORIES = ("identity", "wardrobe", "props", "setting", "era", "location", "lighting")


def check_keyframe(run: dict, seg: dict) -> None:
    """The keyframe checked by the eyes against the bible before anything moves: a major finding draws it again
    (the Director rewrites the keyframe prompt from it, at a new seed) while ``retakes`` allow; the take with the
    fewest problems opens the piece otherwise."""

    last = seg["keyframes"][-1]
    opts = continuity.options(run["settings"])
    categories = [c for c in continuity.checked(opts) if c in KEYFRAME_CATEGORIES]
    result = {"findings": [], "categories": categories}
    if categories:
        bible = bible_of(run)
        picture = HOME / last["path"]
        prompt = continuity.keyframe_prompt(bible, seg["index"], seg, last["prompt"], categories, list(cast_at(run, seg["index"])))
        try:
            data, _ = ask_json(lambda text: ask_vision(continuity.CHECK_SYSTEM, [("Frame 1, the keyframe", picture)], picture, text,
                                                       run["settings"].get("brain") or {})[0], continuity.check_problem, prompt, "keyframe check")
            checks = continuity.triage(data, 1, categories)
            for category, entry in checks.items():
                entry["category"] = category
                entry["action"] = continuity.act(entry, opts)
            result.update(checks=checks, findings=[e for e in checks.values() if e["action"] != "none"])
        except Failed as error:
            result["skipped"] = str(error)[:300]
    dense = dense_keyframe(run, seg, HOME / last["path"], run_dir(run["id"]) / "segments" / seg["id"])
    result["findings"] = (result.get("findings") or []) + dense["findings"]
    result["dense"] = {"stats": dense["stats"], "log": dense["log"]}
    add_stats(run, seg, dense["stats"])
    last["continuity"] = result
    majors = [f for f in result["findings"] if f["status"] == "major"]
    this_round = [k for k in seg["keyframes"] if k.get("round", 0) == last.get("round", 0)]
    if majors and len(this_round) <= int(run["settings"].get("retakes") or 0):
        rewrite_keyframe(run, seg, majors)
        seg["status"] = "keyframe"
        say(run, f"Segment {seg['index']}: keyframe {last['take']} failed ({'; '.join(f['category'] + ': ' + f['evidence'] for f in majors)[:200]}); drawn again.", segment=seg["index"])
        return
    kept = min(reversed(this_round), key=lambda k: (sum(1 for f in (k.get("continuity") or {}).get("findings") or [] if f["status"] == "major"),
                                                          len((k.get("continuity") or {}).get("findings") or [])))
    seg.update(keyframe=kept["path"], status="planned")
    save(run)


def rewrite_keyframe(run: dict, seg: dict, findings: list) -> None:
    found = "\n".join(f"- {f['category']}: {f['evidence']}" for f in findings)
    prompt = (f"You wrote this keyframe prompt for segment {seg['index']} of a film: {seg.get('keyframe_prompt')}\n"
              f"The picture made from it failed its continuity check:\n{found}\n"
              f"The continuity bible (canonical):\n{continuity.director_text(bible_of(run), seg['index'], list(cast_at(run, seg['index'])))}\n"
              'Write the keyframe prompt again so that each of these comes out right. Answer: {"keyframe": "<60 words max>"}')
    try:
        data, _ = ask_json(director_asker(run), lambda d: "" if str(d.get("keyframe") or "").strip() else '"keyframe" is text', prompt, "Director")
        seg["keyframe_prompt"] = clip_words(data["keyframe"], 70)
    except Failed:
        pass  # the same prompt again, at the next keyframe seed


# ---------- check and watch ----------

def sample(clip: Path, folder: Path, count: int) -> tuple[list[tuple[str, Path]], Path, float]:
    """``count`` frames evenly through the clip and its last frame, 448x256 PNG, with captions; and one sheet of them."""

    length = qa.duration(clip)
    folder.mkdir(parents=True, exist_ok=True)
    times = [round(i * max(length - 0.1, 0) / max(count - 1, 1), 2) for i in range(count)]
    frames = []
    for i, at in enumerate(times):
        target = folder / f"f{i + 1:02d}.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(at), "-i", str(clip), "-frames:v", "1", "-vf",
                        f"scale={cosmos.FRAME[0]}:{cosmos.FRAME[1]}", str(target)], check=True, env=qa.ENV, timeout=60)
        frames.append((f"Frame {i + 1}, at {at:.2f} s", target))
    last = folder / "last.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-sseof", "-0.1", "-i", str(clip), "-update", "1", "-vf",
                    f"scale={cosmos.FRAME[0]}:{cosmos.FRAME[1]}", str(last)], check=True, env=qa.ENV, timeout=60)
    frames.append((f"Final frame, at {length:.2f} s", last))
    sheet = contact_sheet([p for _, p in frames], folder / "sheet.png")
    return frames, sheet, length


def contact_sheet(paths: list, sheet: Path, across: int = 4) -> Path:
    """The pictures in rows of ``across``, left to right then top to bottom, each scaled to one frame's size."""

    paths = list(paths)
    across = max(1, min(across, len(paths)))
    while len(paths) % across:
        paths.append(paths[-1])
    command = ["ffmpeg", "-v", "error", "-y"]
    for p in paths:
        command += ["-i", str(p)]
    rows = len(paths) // across
    size = f"scale={cosmos.FRAME[0]}:{cosmos.FRAME[1]}:force_original_aspect_ratio=decrease,pad={cosmos.FRAME[0]}:{cosmos.FRAME[1]}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    graph = "".join(f"[{i}:v]{size}[p{i}];" for i in range(len(paths)))
    graph += "".join(f"{''.join(f'[p{r * across + c}]' for c in range(across))}" + (f"hstack={across}" if across > 1 else "copy") + f"[r{r}];" for r in range(rows))
    graph += "".join(f"[r{r}]" for r in range(rows)) + (f"vstack={rows}[out]" if rows > 1 else "copy[out]")
    subprocess.run(command + ["-filter_complex", graph, "-map", "[out]", "-frames:v", "1", str(sheet)], check=True, env=qa.ENV, timeout=120)
    return sheet


def watcher_brain(run: dict) -> dict:
    """The QA gate's vision model: the vision server when one is set (a local backend), else the film's brain."""

    said = cosmos.settings()
    if said["endpoint"]:
        if said["endpoint"].lower() != cosmos.STUDIO:
            return {"backend": "local", "endpoint": said["endpoint"], "model": said["model"]}
        if cosmos.installed():
            return {"backend": "local", "endpoint": cosmos.STUDIO, "model": cosmos.SERVED}
    return run["settings"].get("brain") or {}


def gate(clip: Path, prompt: str, sheet: Path, brain: dict) -> dict:
    """qa.check with the Studio's Cosmos as its eyes when that is the vision brain (loaded for the call)."""

    if brain.get("endpoint") == cosmos.STUDIO:
        try:
            with cosmos.request() as (endpoint, model):
                return qa.check(clip, prompt, sheet, {"backend": "local", "endpoint": endpoint, "model": model})
        except LookupError:
            brain = {}
    return qa.check(clip, prompt, sheet, brain)


def check_watch(data: dict) -> str:
    for key in ("add_nodes", "update_nodes", "add_events", "continuity"):
        if data.get(key) is not None and not isinstance(data.get(key), list):
            return f'"{key}" is a list'
    if not isinstance(data.get("identity_drift"), bool):
        # read leniently: "true"/"false" as words, and left out as false (the identity questions of every window and
        # the drift confirm look again). On the gate's Infinity run Cosmos3-Edge left it out three times in a row
        word = str(data.get("identity_drift") or "").strip().lower()
        data["identity_drift"] = word in ("true", "yes", "1")
    if not str(data.get("ending") or "").strip():
        return '"ending" says what the final frame shows'
    return ""


DRIFT_SYSTEM = "You compare people across frames of one video. Answer only with JSON."


def confirm_drift(run: dict, frames: list, watched: dict, index: int) -> tuple[bool, dict]:
    """A second, narrower look before identity drift fails a take: the first, middle and last frames, and one question
    (is it the same person). Measured on the first real run: Cosmos3-Edge's scene-graph answer called a courier who
    got on her vehicle "identity drift"; asked only about the face, hair and age it answers the question asked."""

    names = "; ".join(f"{n}: {look}" for n, look in cast_at(run, index).items()) or "the main character"
    pick = [frames[0], frames[len(frames) // 2], frames[-1]]
    prompt = (f"The three images are frames of one short video clip, in time order. The pinned looks: {names}.\n"
              "Look only at the main person's FACE, HAIR, AGE and BUILD. Clothing, props, vehicles, pose, distance, motion blur "
              "and lighting do not matter. Is it clearly the same person in all three frames where the person is visible?\n"
              'Answer: {"same": true | false, "why": "<15 words max>"}')
    try:
        data, _ = ask_json(lambda text: ask_vision(DRIFT_SYSTEM, pick, Path(pick[-1][1]), text, run["settings"].get("brain") or {})[0],
                           lambda d: "" if isinstance(d.get("same"), bool) else '"same" is true or false', prompt, "drift check")
    except Failed as error:
        return True, {"error": str(error)[:200]}  # the watcher's word stands when the check cannot be made
    return not data["same"], {"same": data["same"], "why": clip_words(data.get("why"), 20)}


def check(run: dict, seg: dict) -> None:
    """The take judged: the QA gate (Whisper line check, identity sheet), the watcher (scene-graph delta, drift) and
    the continuity check against the bible. A take that fails is rendered again while ``retakes`` allow: at the next
    seed, and, when the continuity check or the drift check found it, with the shot rewritten by the Director from the
    findings. When they run out the best take is kept, the reasons beside it."""

    s = run["settings"]
    clip = HOME / seg["output"]
    folder = run_dir(run["id"]) / "segments" / seg["id"]
    take = len(seg["takes"]) + 1
    say(run, f"Segment {seg['index']}: checking and watching take {take}.", stage="watch", segment=seg["index"])
    report = {"passed": None, "skipped": True, "note": "the QA gate is off for this run"}
    transcript = ""
    if s.get("qa", True) and qa.ready()["runs"]:
        try:
            report = gate(clip, seg.get("prompt") or "", folder / f"qa-take{take}.png", watcher_brain(run))
            transcript = ((report.get("audio") or {}).get("heard") or "").strip()
        except Exception as error:  # noqa: BLE001 - the clip is made; a gate that cannot run does not lose it
            report = {"passed": None, "error": str(error)[-300:]}
    elif qa.ready()["transcript"]:
        try:
            transcript = (qa.transcribe(clip) or "").strip()
        except Exception:  # noqa: BLE001
            transcript = ""
    frames, sheet, length = sample(clip, folder / f"frames-take{take}", cosmos.settings()["frames"])
    memory = memory_of(run)
    graph = memory["graph"]
    view = {"nodes": graph["nodes"], "events": graph["events"][-RECENT_EVENTS:], "chapters": graph.get("chapters") or []}
    prompt = watch_prompt(run, {**seg, "seconds": length}, view, max(0, len(graph["events"]) - RECENT_EVENTS), transcript)
    who = []

    def asker(text: str) -> str:
        answer, by = ask_vision(WATCH_SYSTEM, frames, sheet, text, run["settings"].get("brain") or {})
        who.append(by)
        return answer
    watched, rejected = ask_json(asker, check_watch, prompt, "watcher")
    watched = {"delta": {k: watched.get(k) or [] for k in ("add_nodes", "update_nodes", "add_events")},
               "summary": clip_words(watched.get("summary"), 40), "ending": clip_words(watched.get("ending"), 60),
               # a note that something stayed the same is not a problem (Cosmos3-Edge wrote such notes on the first real run)
               "continuity": [c for c in watched.get("continuity") or [] if isinstance(c, dict) and not continuity.BENIGN.search(str(c.get("note") or ""))][:6],
               "identity_drift": watched["identity_drift"], "by": who[-1] if who else "", "rejected": rejected,
               "sheet": relative(sheet), "frames": [relative(p) for _, p in frames]}
    if watched["identity_drift"]:
        watched["identity_drift"], watched["drift_check"] = confirm_drift(run, frames, watched, seg["index"])
    write_json(folder / f"watch-take{take}.json", watched)
    say(run, f"Segment {seg['index']}: checking take {take} against the continuity bible.", stage="continuity", segment=seg["index"])
    checked = check_continuity(run, seg, frames, sheet, transcript, clip, folder, take, length)
    say(run, f"Segment {seg['index']}: the dense checks of take {take} (windows, the join, the pace).", stage="checks", segment=seg["index"])
    dense = dense_check(run, seg, clip, folder, take, length, bible_of(run))
    passed = set(dense.get("passed") or [])
    for found in checked.get("findings") or []:
        # the checklist's one verdict against the yes/no answers of every window: a major the windows never saw is a
        # correction for the Director, not a retake (the Viking v2 film's false retakes came from single verdicts)
        if found.get("status") == "major" and found["category"] in passed and found.get("source") != "rule":
            found.update(status="minor", demoted="the yes/no checks of every window saw no problem here")
            found["action"] = continuity.act(found, continuity.options(s))
    if watched["identity_drift"] and "identity" in passed:
        # the same for the watcher's drift (its confirm is three frames and one question): on the first live Infinity
        # run (2026-10-26) it rendered a piece again whose identity questions had passed in every window
        watched.update(identity_drift=False, drift_overruled="the identity questions passed in every window")
        write_json(folder / f"watch-take{take}.json", watched)
    checked.setdefault("findings", []).extend(dense["findings"])
    checked["dense"] = {k: dense.get(k) for k in ("stats", "pace", "windows", "passed", "filtered", "file", "by", "skipped") if dense.get(k) is not None}
    add_stats(run, seg, dense["stats"])
    reasons, fix = [], []
    audio, identity = report.get("audio") or {}, report.get("identity") or {}
    if audio.get("result") == "fail" and continuity.heard_alike(audio.get("wanted"), audio.get("heard")):
        # a name heard as a word that sounds the same ("Eirik!" as "Eric!") is a minor note, never a retake
        audio["alike"] = True
        if "audio" in (checked.get("categories") or []) or "audio" in continuity.options(s)["categories"]:
            checked.setdefault("findings", []).append({"category": "audio", "status": "minor", "verdict": "minor", "frame": 0, "source": "gate", "action": "note",
                                                       "subject": "", "evidence": f"the line \"{audio.get('wanted')}\" was heard as \"{audio.get('heard')}\": it sounds alike, kept"})
    if audio.get("result") == "fail" and not audio.get("alike"):
        reasons.append(f"audio gate: {audio.get('note', '')}")
    if identity.get("result") == "fail":
        reasons.append(f"identity gate: {identity.get('note', '')}")
        fix.append({"category": continuity.gate_category(identity.get("note")), "frame": 0, "evidence": clip_words(identity.get("note"), 30)})
    if watched["identity_drift"]:
        drift = "; ".join(c.get("note", "") for c in watched["continuity"] if c.get("kind") == "identity")[:200]
        reasons.append("the watcher saw identity drift: " + drift)
        fix.append({"category": "identity", "frame": 0, "evidence": drift or "the main character's face changes into someone else's"})
    for found in checked.get("findings") or []:
        if found.get("action") == "retake":
            reasons.append(f"continuity, {found['category']}: {found['evidence']}"[:240])
            fix.append(found)
    seg["takes"].append({"take": take, "seed": seg["seed"], "output": seg["output"], "job": seg.get("job"), "passed": not reasons,
                         "reasons": reasons, "qa": {k: report.get(k) for k in ("passed", "audio", "identity", "note", "error") if report.get(k) is not None},
                         "watch": watched, "continuity": checked, "transcript": transcript, "fields": copy.deepcopy(seg["fields"]),
                         "beat": seg["beat"], "prompt": seg.get("prompt"), "mode": seg.get("mode")})
    write_json(folder / f"continuity-take{take}.json", checked)
    if reasons and take <= int(s.get("retakes") or 0):
        seg["status"], seg["fix"] = "retake", fix
        say(run, f"Segment {seg['index']}: take {take} failed ({'; '.join(reasons)[:200]}); again at the next seed"
            + (", the shot rewritten from the findings." if fix else "."), segment=seg["index"])
        return
    kept = seg["takes"][-1]
    if reasons and len(seg["takes"]) > 1:
        kept = continuity.best(seg["takes"])
        if kept is not seg["takes"][-1]:
            seg.update(output=kept["output"], job=kept["job"], fields=kept["fields"], beat=kept["beat"], prompt=kept["prompt"], mode=kept["mode"], seed=kept["seed"])
        kept["reasons"] = kept["reasons"] + [f"no take passed: kept take {kept['take']} of {len(seg['takes'])}, the one with the fewest problems"]
    commit(run, seg, kept["watch"], kept["transcript"], kept["reasons"], kept["continuity"])


# ---------- the continuity check ----------

def check_continuity(run: dict, seg: dict, frames: list, sheet: Path, transcript: str, clip: Path, folder: Path, take: int,
                     length: float) -> dict:
    """The take against the continuity bible: one checklist question to the eyes (the frames, the transcript, the
    bible's slice, a category a line), the answer triaged (continuity.triage), each major finding looked at again
    closer (second_look), and the 180-degree rule measured on a continuation. A check that cannot be made does not
    lose the take: it is marked skipped."""

    opts = continuity.options(run["settings"])
    categories = continuity.checked(opts)
    if not categories:
        return {"skipped": "no category is checked in this run's settings", "categories": [], "findings": []}
    bible = bible_of(run)
    index, names = seg["index"], list(cast_at(run, seg["index"]))
    continuation = seg.get("mode") == "previous"
    start = float(seg.get("start") or 0)
    prompt = continuity.check_prompt(bible, index, seg, len(frames), transcript, categories, f"{mmss(start)} to {mmss(start + length)}",
                                     names, continuation, style_at(run["settings"], index))
    who = []

    def asker(text: str) -> str:
        answer, by = ask_vision(continuity.CHECK_SYSTEM, frames, sheet, text, run["settings"].get("brain") or {})
        who.append(by)
        return answer
    try:
        data, rejected = ask_json(asker, continuity.check_problem, prompt, "continuity check")
    except Failed as error:
        return {"skipped": str(error)[:300], "categories": categories, "findings": []}
    checks = continuity.triage(data, len(frames), categories)
    findings = []
    for category, entry in checks.items():
        entry["category"] = category
        if entry["status"] == "major":
            continuity.settle(entry, second_look(run, entry, frames, clip, folder, take, bible, index), opts)
        entry["action"] = continuity.act(entry, opts)
        if entry["action"] != "none":
            findings.append(entry)
    if continuation and "spatial" in categories:
        for entry in continuity.screen_flip(bible, index, data.get("positions")):
            entry["action"] = continuity.act(entry, opts)
            findings.append(entry)
    present = []
    for p in data.get("present") or []:
        try:
            n = int(p.get("frame") or 0) if isinstance(p, dict) else 0
        except (TypeError, ValueError):
            n = 0
        if isinstance(p, dict) and p.get("who"):
            present.append({"who": clip_words(p["who"], 4), "frame": relative(frames[n - 1][1]) if 1 <= n <= len(frames) else ""})
    return {"by": who[-1] if who else "", "strictness": opts["strictness"], "categories": categories, "checks": checks, "findings": findings,
            "facts": [f for f in data.get("facts") or [] if isinstance(f, dict)][:6], "present": present[:9],
            "positions": [p for p in data.get("positions") or [] if isinstance(p, dict)][:9],
            "open_actions": [a for a in data.get("open_actions") or [] if isinstance(a, dict)][:6],
            "light": clip_words(data.get("light"), 14), "rejected": rejected}


# ---------- the dense checks (checks.py) ----------

def window_asker(run: dict, folder: Path, who: list):
    """ask(system, pictures, text) on the vision brain, each request with a sheet of its own pictures for the QA gate's
    eyes when no vision server answers."""

    count = [0]

    def ask(system: str, pictures: list, text: str) -> str:
        count[0] += 1
        try:
            sheet = contact_sheet([p for _, p in pictures], folder / f"sheet-{count[0]:03d}.png", across=min(4, len(pictures)))
        except (subprocess.SubprocessError, OSError):
            sheet = Path(pictures[-1][1])
        answer, by = ask_vision(system, pictures, sheet, text, run["settings"].get("brain") or {})
        who.append(by)
        return answer
    return ask


def references(run: dict, bible: dict, index: int, seg: dict, folder: Path) -> list:
    """The reference picture of each person the shot names (their cast picture, else their clearest earlier frame)."""

    text = " ".join(str((seg.get("fields") or {}).get(k) or "") for k in ("setting", "action")).lower()
    out = []
    for name in [n for n in cast_at(run, index) if n.lower().split()[0] in text][:2]:
        sid = continuity.subject_of(bible, name, "character", create=False)
        ref = (bible["subjects"].get(sid) or {}).get("ref") if sid else None
        if ref and on_disk(ref):
            out.append((f"Reference picture of {name}", still(HOME / ref, folder / f"ref-{sid}.png", fit=(448, 448))))
    return out


def yes_no(run: dict, asker, pictures: list, header: str, asked: list, opts: dict, log: list, where: dict) -> tuple[list, dict]:
    """One window's questions, each failed major one voted on: the failed questions, as {question, answer}."""

    found = checks.ask_window(asker, pictures, header, asked, slate.extract_json, log, where)
    failed = []
    for q in asked:
        answer = found.get(q["key"])
        if q["key"] == "pace" or not answer or answer == q["expect"]:  # the windows' pace answer is weighed in pace_verdict
            continue
        if q["major"] and opts["votes"] > 1:
            answer = checks.vote(asker, pictures, header, q, answer, opts["votes"], slate.extract_json, log, {**where, "kind": "vote"})
            if not answer or answer == q["expect"]:
                continue
        failed.append({"question": q, "answer": answer})
    return failed, found


def finding_of(q: dict, answer: str, where: str, frame: int, pictures: list) -> dict:
    status = "major" if q["major"] else "minor"
    return {"category": q["category"], "status": status, "verdict": status, "frame": frame, "subject": "", "source": "yes/no",
            "evidence": clip_words(f"{where}: asked \"{q['text']}\", the answer was {answer}", 40), "_pictures": pictures}


def dense_check(run: dict, seg: dict, clip: Path, folder: Path, take: int, length: float, bible: dict) -> dict:
    """The take looked at in windows (checks.py): the bible's yes/no questions a window, the join with the piece before
    on a continuation, and the pace, measured and asked. Failed major questions are voted on, then looked at again
    (second_look). Returns {"findings", "passed" (categories asked and never failed), "log", "stats", "pace"}."""

    s = run["settings"]
    opts, copts = checks.options(s), continuity.options(s)
    categories = continuity.watched(copts)
    if not opts["dense"] or not categories:
        return {"skipped": "the dense checks are off for this run", "findings": [], "passed": [], "log": [], "stats": checks.summary([])}
    index = seg["index"]
    work = folder / f"windows-take{take}"
    work.mkdir(parents=True, exist_ok=True)
    who, log, failed = [], [], []
    asker = window_asker(run, work, who)
    asked_about, failed_about = set(), set()
    refs = references(run, bible, index, seg, work)
    base = checks.questions(bible, index, seg, cast_at(run, index), categories, continuity)
    pace = checks.pace_of(seg.get("pace"))
    fast = opts["pace"] and "pace" in categories and pace["kind"] in checks.FAST_KINDS
    slow_answers = []
    spans = checks.windows(length, opts["every"], opts["frames"])
    for w, times in enumerate(spans, 1):
        pics = [(f"Frame at {t:.2f} s", work / f"w{w:02d}-{i + 1}.png") for i, t in enumerate(times)]
        pics = [(c, p) for (c, p), t in zip(pics, times) if checks.grab(clip, t, p, cosmos.FRAME, qa.ENV)]
        if not pics:
            continue
        mid = sum(times) / len(times)
        asked = list(base) + ([checks.pace_question(checks.planned(pace, mid, length) == "slow")] if fast else [])
        if not asked:
            continue
        header = (f"Window {w} of {len(spans)} of segment {index}: " + (f"the first {len(refs)} image(s) are reference pictures, then " if refs else "")
                  + f"{len(pics)} frames from {times[0]:.1f} to {times[-1]:.1f} s, in time order.")
        bad, found = yes_no(run, asker, refs + pics, header, asked, opts, log, {"kind": "window", "window": w, "at": times})
        if fast:
            slow_answers.append((mid, found.get("pace")))
        asked_about |= {q["category"] for q in asked if found.get(q["key"])}
        for b in bad:
            failed_about.add(b["question"]["category"])
            failed.append(finding_of(b["question"], b["answer"], f"window {w} ({times[0]:.1f}-{times[-1]:.1f} s)", len(pics) // 2 + 1, pics))
    done = committed(run)
    previous = seg_last_frame(done[-1]) if done and seg.get("mode") == "previous" else None
    if opts["join"] and previous:
        first = work / "join-first.png"
        if checks.grab(clip, 0.0, first, cosmos.FRAME, qa.ENV):
            pics = [(f"Picture 1: the last frame of segment {done[-1]['index']}", HOME / previous), ("Picture 2: the first frame of this segment, at 0.00 s", first)]
            asked = checks.join_questions(categories)
            if asked:
                bad, found = yes_no(run, asker, pics, f"The join between segment {done[-1]['index']} and segment {index}: one continuous shot.", asked, opts, log,
                                    {"kind": "join", "window": 0, "at": [0.0]})
                asked_about |= {q["category"] for q in asked if found.get(q["key"])}
                for b in bad:
                    failed_about.add(b["question"]["category"])
                    failed.append(finding_of(b["question"], b["answer"], "the join with the segment before", 2, pics))
    paced = {}
    if fast:
        paced = checks.pace_verdict(pace, checks.motion(clip, qa.ENV), slow_answers, length)
        asked_about.add("pace")
        if paced["status"] in ("major", "minor"):
            failed_about.add("pace")
            failed.append({"category": "pace", "status": paced["status"], "verdict": paced["status"], "frame": 0, "subject": "", "source": "measured",
                           "evidence": paced["evidence"] + ("; the film cuts that lead" if paced["status"] == "minor" else "")})
    # one finding a question (the first window it failed in, with how many windows saw it)
    findings, seen = [], {}
    for f in failed:
        key = (f["category"], f["evidence"].split(": asked", 1)[-1])
        if key in seen:
            seen[key]["windows"] = seen[key].get("windows", 1) + 1
            continue
        seen[key] = f
        findings.append(f)
    for f in findings:
        pictures = f.pop("_pictures", None)
        if f["status"] == "major" and f["category"] != "pace" and pictures:
            continuity.settle(f, second_look(run, f, pictures, clip, work, take, bible, index), copts)
        f["action"] = continuity.act(f, copts)
    log_file = folder / f"checks-take{take}.json"
    write_json(log_file, log)
    return {"findings": [f for f in findings if f["action"] != "none"], "filtered": [f for f in findings if f["action"] == "none"],
            "passed": sorted(asked_about - failed_about), "log": log, "stats": checks.summary(log), "pace": paced,
            "windows": len(spans), "by": who[-1] if who else "", "file": relative(log_file)}


def dense_keyframe(run: dict, seg: dict, picture: Path, folder: Path) -> dict:
    """The keyframe's yes/no questions (one still: who, what they wear, the place, the era), failed majors voted on."""

    s = run["settings"]
    opts, copts = checks.options(s), continuity.options(s)
    categories = [c for c in continuity.watched(copts) if c in KEYFRAME_CATEGORIES]
    if not (opts["dense"] and opts["keyframe"] and categories):
        return {"findings": [], "log": [], "stats": checks.summary([])}
    bible, index = bible_of(run), seg["index"]
    work = folder / f"keyframe-checks-{len(seg.get('keyframes') or [])}"
    work.mkdir(parents=True, exist_ok=True)
    asked = [q for q in checks.questions(bible, index, {**seg, "mode": "keyframe"}, cast_at(run, index), categories, continuity)
             # one still: "the same person in every frame" means something only against a reference picture
             if "between these frames" not in q["text"] and ("in every frame" not in q["text"] or "reference picture" in q["text"])]
    if "pace" in continuity.watched(copts) and checks.pace_of(seg.get("pace"))["kind"] in checks.FAST_KINDS:
        asked.append(checks.MID_MOTION)  # a fight opens on a blow in motion, not a stand-off
    if not asked:
        return {"findings": [], "log": [], "stats": checks.summary([])}
    refs = references(run, bible, index, seg, work)
    log = []
    bad, _ = yes_no(run, window_asker(run, work, []), refs + [("The keyframe, the still this segment opens on", picture)],
                    f"The keyframe of segment {index}" + (", after the reference picture(s)." if refs else "."), asked, opts, log, {"kind": "keyframe", "window": 0, "at": [0.0]})
    findings = []
    for b in bad:
        f = finding_of(b["question"], b["answer"], "the keyframe", 1, None)
        f.pop("_pictures", None)
        f["action"] = continuity.act(f, copts)
        findings.append(f)
    return {"findings": findings, "log": log, "stats": checks.summary(log)}


def add_stats(run: dict, seg: dict, stats: dict) -> None:
    """The checks' counts added to the segment's and the run's (what the page and the report show)."""

    for holder in (seg.setdefault("check_stats", {}), run.setdefault("check_stats", {})):
        for key in ("requests", "questions", "failed_answers", "votes", "seconds"):
            holder[key] = round(holder.get(key, 0) + (stats.get(key) or 0), 1)


def frame_time(caption: str) -> float | None:
    found = re.search(r"at ([0-9.]+) s", caption or "")
    return float(found[1]) if found else None


def still(source: Path, target: Path, at: float | None = None, box=None, fit=(896, 512)) -> Path:
    """One picture for a closer look: the clip's frame at ``at`` at full size (or a picture), cropped to ``box``
    (fractions of the picture) and scaled to fit ``fit``; the source itself when ffmpeg cannot make it."""

    chain = []
    if box:
        x0, y0, x1, y1 = box
        chain.append(f"crop=iw*{x1 - x0:.4f}:ih*{y1 - y0:.4f}:iw*{x0:.4f}:ih*{y0:.4f}")
        fit = (448, 448)
    chain.append(f"scale={fit[0]}:{fit[1]}:force_original_aspect_ratio=decrease")
    command = ["ffmpeg", "-v", "error", "-y"] + (["-ss", f"{at:.3f}"] if at is not None else []) + ["-i", str(source), "-frames:v", "1", "-vf", ",".join(chain), str(target)]
    try:
        subprocess.run(command, check=True, env=qa.ENV, timeout=60, capture_output=True)
        return target if target.is_file() else source
    except (subprocess.SubprocessError, OSError):
        return source


def reference_for(run: dict, entry: dict, bible: dict, index: int, folder: Path, take: int) -> tuple[str, Path] | None:
    """What a second look compares with: the person's reference (their cast picture, else the clearest frame of the
    segment they were first seen in) for identity and wardrobe; the last frame of the segment before otherwise."""

    if entry["category"] in ("identity", "wardrobe"):
        who = entry.get("subject") or next(iter(cast_at(run, index)), "")
        sid = continuity.subject_of(bible, who, "character", create=False) if who else None
        ref = (bible["subjects"].get(sid) or {}).get("ref") if sid else None
        if ref and on_disk(ref):
            return (f"{continuity.name_of(bible, sid)} as established", still(HOME / ref, folder / f"ref-take{take}-{sid}.png", fit=(448, 448)))
    done = committed(run)
    if done and seg_last_frame(done[-1]):
        return (f"the last frame of segment {done[-1]['index']}, the segment before", HOME / seg_last_frame(done[-1]))
    return None


def seg_last_frame(seg: dict) -> str | None:
    frames = (seg.get("watch") or {}).get("frames") or []
    return frames[-1] if frames and on_disk(frames[-1]) else None


SECOND_SYSTEM = continuity.SECOND_SYSTEM


def second_look(run: dict, entry: dict, frames: list, clip: Path, folder: Path, take: int, bible: dict, index: int) -> dict:
    """A major finding looked at again before it counts: the frame it names, closer (cropped to where it is when the
    first look said), beside what it should match, and one question. Measured on the first real run: Cosmos3-Edge
    called a courier who got on her vehicle "identity drift"; asked only about the face it answered the question asked."""

    n = entry["frame"]
    caption, picture = frames[n - 1]
    closer = still(clip, folder / f"look-take{take}-{entry['category']}.png", frame_time(caption), entry.get("box"))
    pictures = [ref for ref in [reference_for(run, entry, bible, index, folder, take)] if ref]
    pictures.append((f"the new segment, frame {n}" + (", the part named, closer" if entry.get("box") else ""), closer))
    try:
        sheet = contact_sheet([p for _, p in pictures], folder / f"look-take{take}-{entry['category']}-sheet.png", across=len(pictures))
    except (subprocess.SubprocessError, OSError):
        sheet = closer
    prompt = continuity.second_prompt(bible, index, entry, pictures)
    try:
        data, _ = ask_json(lambda text: ask_vision(SECOND_SYSTEM, pictures, sheet, text, run["settings"].get("brain") or {})[0],
                           lambda d: "" if isinstance(d.get("confirmed"), bool) else '"confirmed" is true or false', prompt, "second look")
    except Failed as error:
        return {"error": str(error)[:200]}
    return {"confirmed": data["confirmed"], "why": clip_words(data.get("why"), 20), "pictures": [relative(p) if HOME.resolve() in p.resolve().parents else str(p.name) for _, p in pictures]}


def commit(run: dict, seg: dict, watched: dict, transcript: str, reasons: list, checked: dict | None = None) -> None:
    memory = memory_of(run)
    graph = merge(memory["graph"], watched["delta"], seg["index"])
    for event in graph["events"]:
        if event.get("segment") == seg["index"] and not re.fullmatch(r"\d{1,3}:\d{2}", str(event.get("t") or "")):
            event["t"] = mmss(float(seg["start"]))
    events = [copy.deepcopy(e) for e in graph["events"] if e.get("segment") == seg["index"]]
    archive = memory.get("archive") or []
    if len(graph["events"]) > COMPACT_AT:
        folded = copy.deepcopy(graph["events"][:FOLD])
        chapter, _ = ask_json(lambda text: ask_text(run, COMPACT_SYSTEM, text),
                              lambda d: "" if str(d.get("title") or "").strip() and str(d.get("summary") or "").strip() else '"title" and "summary" are text',
                              compact_prompt(folded, graph.get("chapters") or []), "memory's chapter fold")
        archive = archive + fold(graph, chapter, folded)  # only event detail folds: the bible keeps every canonical fact
    memory = {"graph": graph, "archive": archive, "observed_through": seg["index"]}
    folder = run_dir(run["id"])
    write_json(folder / "memory" / f"{seg['id']}.json", {**memory, "delta": watched["delta"], "cast": run.get("cast") or {},
                                                          "video": seg["output"]})
    write_json(folder / "memory.json", memory)
    bible = bible_of(run)
    checked = checked or {"skipped": "not checked", "findings": []}
    seen = continuity.absorb(bible, seg["index"], checked, events)
    save_bible(run, bible, seg["id"])
    summary = {k: checked.get(k) for k in ("by", "strictness", "categories", "checks", "findings", "skipped", "dense") if checked.get(k) is not None}
    seg.update(status="done", watch=watched, transcript=transcript, passed=not reasons, reasons=reasons, committed=slate.now(),
               continuity={**summary, "present_frames": seen}, fix=[])
    minors = [f for f in checked.get("findings") or [] if f.get("action") == "correct"]
    say(run, f"Segment {seg['index']} joins the film" + (f" (it did not pass: {'; '.join(reasons)[:160]})" if reasons else "")
        + (f"; {len(minors)} continuity correction{'s' if len(minors) != 1 else ''} for the next segment" if minors else "")
        + (f"; {seg['check_stats'].get('requests', 0):.0f} check requests, {seg['check_stats'].get('questions', 0):.0f} questions, "
           f"{seg['check_stats'].get('seconds', 0):.0f} s" if seg.get("check_stats") else "") + ".",
        stage="committed", segment=seg["index"])
    checkpoint(run, seg)


def checkpoint(run: dict, seg: dict | None = None, next_seg: dict | None = None) -> None:
    """What a run needs to go on after a crash, a stop by the memory guard or a restart, on disk after every piece
    (``checkpoint.json``, and ``checkpoints/S0001.json`` for each): the pieces kept with their seeds and lengths, the
    film so far, where the bible and the memory of that moment are, the story so far, and the next beat once planned.
    run.json carries the same state; resume_all picks up a run that was live, without anyone asking."""

    done = committed(run)
    point = {"at": slate.now(), "run": run["id"], "segments": len(done), "seconds": round(sum(float(x.get("seconds") or 0) for x in done), 3),
             "pieces": [{"index": x["index"], "output": x.get("output"), "seed": x.get("seed"), "frames": x.get("frames"), "seconds": x.get("seconds"),
                         "passed": x.get("passed"), "scene": x.get("scene"), "beat": x.get("beat")} for x in done],
             "film": run.get("film"), "story_so_far": (run.get("story_so_far") or {}).get("text", ""),
             "bible": f"bible/{seg['id']}.json" if seg else "bible.json", "memory": f"memory/{seg['id']}.json" if seg else "memory.json",
             "check_stats": run.get("check_stats") or {},
             "next": {k: next_seg.get(k) for k in ("index", "scene", "beat", "fields", "pace", "seed", "frames")} if next_seg else None}
    write_json(run_dir(run["id"]) / "checkpoint.json", point)
    if seg:
        write_json(run_dir(run["id"]) / "checkpoints" / f"{seg['id']}.json", point)
    run["checkpoint"] = {"at": point["at"], "segments": point["segments"], "next": (next_seg or {}).get("beat")}


# ---------- the film-level pass ----------

def film_check_safely(run: dict, last: int = 0) -> None:
    """The film-level pass (over the last ``last`` segments, or all), whose failure is reported and never stops the run."""

    try:
        film_check(run, last)
    except Exception as error:  # noqa: BLE001 - a report only
        run["film_check"] = {"state": "failed", "at": slate.now(), "error": str(error)[:300]}
        say(run, "The film's continuity pass could not be made: " + str(error)[:200])


def film_check(run: dict, last: int = 0) -> dict:
    """Cosmos samples across the whole film: each person's clearest frame of each segment they were seen in, an era
    at a time, asked whether any segment shows someone else (the man's face in segment 12 against segment 8), and one
    frame of each segment, asked for drift in places, props and era. A report only: nothing is rendered again."""

    bible, done = bible_of(run), committed(run)
    if last:
        done = done[-last:]  # an Infinity run's rolling pass: the last joins only
    brain = run["settings"].get("brain") or {}
    folder = run_dir(run["id"]) / ("film-check" + (f"-to{len(committed(run)):04d}" if last else ""))
    folder.mkdir(parents=True, exist_ok=True)
    report = {"state": "done", "at": slate.now(), "segments": len(committed(run)), "identity": [], "world": [], "errors": [],
              **({"window": [done[0]["index"], done[-1]["index"]]} if last and done else {})}

    def ask(prompt: str, pictures: list, name: str, check) -> dict | None:
        try:
            sheet = contact_sheet([p for _, p in pictures], folder / f"{name}.png")
            data, _ = ask_json(lambda text: ask_vision(continuity.FILM_SYSTEM, pictures, sheet, text, brain)[0], check, prompt, "film check")
            return data
        except (Failed, subprocess.SubprocessError, OSError) as error:
            report["errors"].append(f"{name}: {str(error)[:200]}")
            return None

    for sid, eras in continuity.film_samples(bible, done).items():  # noqa: B007
        for era, picked in eras.items():
            picked = [(i, path) for i, path in picked if on_disk(path)]
            if len(picked) < 2:
                continue
            look = "; ".join(f["value"] for f in bible["facts"] if f["subject"] == sid and f["category"] == "identity" and f["era"] == era)
            pictures = [(f"Segment {i}", HOME / path) for i, path in picked]
            data = ask(continuity.film_identity_prompt(continuity.name_of(bible, sid), look, picked), pictures, f"identity-{sid}-era{era}",
                       lambda d: "" if isinstance(d.get("same"), bool) else '"same" is true or false')
            if data is not None:
                different = [int(x) for x in data.get("different") or [] if str(x).isdigit() and int(x) in {i for i, _ in picked}]
                report["identity"].append({"subject": continuity.name_of(bible, sid), "era": era, "segments": [i for i, _ in picked],
                                           "same": data["same"] and not different, "different": different, "why": clip_words(data.get("why"), 24)})
    frames = [(seg["index"], (seg.get("watch") or {}).get("frames") or []) for seg in done]
    frames = [(i, f[len(f) // 2]) for i, f in frames if f and on_disk(f[len(f) // 2])]
    if len(frames) > 8:
        step = (len(frames) - 1) / 7
        frames = [frames[round(k * step)] for k in range(8)]
    if len(frames) >= 2:
        data = ask(continuity.film_world_prompt(bible, frames), [(f"Segment {i}", HOME / path) for i, path in frames], "world",
                   lambda d: "" if isinstance(d.get("issues"), list) else '"issues" is a list')
        for issue in (data or {}).get("issues") or []:
            if isinstance(issue, dict) and str(issue.get("evidence") or "").strip() and not continuity.BENIGN.search(str(issue["evidence"])):
                report["world"].append({"category": str(issue.get("category") or "other")[:20], "evidence": clip_words(issue["evidence"], 24),
                                        "segments": [int(x) for x in issue.get("segments") or [] if str(x).isdigit()][:8]})
    write_json(run_dir(run["id"]) / ("film-check.json" if not last else f"film-check-to{len(committed(run)):04d}.json"), report)
    if not last:
        run["film_check"] = report
    if last:
        run.setdefault("film_checks", []).append({k: report.get(k) for k in ("at", "window", "segments")}
                                                 | {"drift": [r for r in report["identity"] if not r["same"]], "issues": report["world"]})
        del run["film_checks"][:-20]
    drift = [r for r in report["identity"] if not r["same"]]
    say(run, "The film's continuity pass: " + (f"{len(drift)} person/era with drift, " if drift else "no face drift, ")
        + f"{len(report['world'])} other issue{'s' if len(report['world']) != 1 else ''}" + (f", {len(report['errors'])} question(s) not answered" if report["errors"] else "") + ".",
        stage="film-check")
    return report


# ---------- the film ----------

def video_seconds(clip: Path) -> float:
    """The picture's own length: the container's is the longest stream's, and a film whose picture stopped early
    still reports its sound's length there."""

    done = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets",
                           "-of", "csv=p=0", str(clip)], capture_output=True, text=True, env=qa.ENV, timeout=120)
    try:
        return int(done.stdout.strip().split(",")[0]) / FPS
    except ValueError:
        return qa.duration(clip)


def has_audio(clip: Path) -> bool:
    done = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(clip)],
                          capture_output=True, text=True, env=qa.ENV, timeout=30)
    return bool(done.stdout.strip())


EASE_IN_MAX_S = 2  # the most of a fight piece's opening the film leaves out


def ease_in(seg: dict) -> float:
    """Seconds of a fight piece's opening the film leaves out: MiniMax H3 eases out of the still it starts from, and on
    the pace and wardrobe run (2026-10-26, 1376x768) 7 of 8 fight takes opened at 2 to 5 a second whatever the prompt
    said, the keyframe already mid-strike, before the exchange ran at 10 to 16. The leading seconds measured slow (under
    checks.SLOW) are cut, at most EASE_IN_MAX_S, when a fast second follows them; a piece the plan opens slowly keeps them."""

    pace = ((seg.get("continuity") or {}).get("dense") or {}).get("pace") or {}
    motion, plan = pace.get("motion") or [], checks.pace_of(seg.get("pace"))
    if plan["kind"] not in checks.FAST_KINDS or not motion:
        return 0.0
    lead = 0
    while lead < min(EASE_IN_MAX_S, len(motion) - 1) and motion[lead] < checks.SLOW and checks.planned(plan, lead + 0.5, len(motion)) == "fast":
        lead += 1
    return float(lead) if lead and any(m >= checks.SLOW for m in motion[lead:]) else 0.0


def publish(run: dict) -> dict | None:
    """Every committed segment joined into ``film.mp4``: a continuation cross-fades over the one frame it repeats,
    a new scene dissolves in; picture and sound fade together, so they stay in step."""

    parts = [s for s in committed(run) if on_disk(s.get("output"))]
    if not parts:
        return None
    folder = run_dir(run["id"])
    say(run, f"Joining the film: {len(parts)} segment{'s' if len(parts) != 1 else ''}.", stage="preview")
    size = None
    inputs, graph, lengths_ = [], [], []
    for i, seg in enumerate(parts):
        clip = HOME / seg["output"]
        length = qa.duration(clip)
        cut = ease_in(seg)
        length = max(0.5, length - cut)
        lengths_.append(length)
        inputs += (["-ss", f"{cut:.3f}"] if cut else []) + ["-i", str(clip)]
        if size is None:
            probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(clip)],
                                   capture_output=True, text=True, env=qa.ENV, timeout=30).stdout.strip().split(",")
            size = (int(probe[0]) // 2 * 2, int(probe[1]) // 2 * 2) if len(probe) == 2 and all(x.isdigit() for x in probe) else (864, 480)
        graph.append(f"[{i}:v]fps={FPS},scale={size[0]}:{size[1]},setsar=1,format=yuv420p,settb=AVTB[v{i}]")
        if has_audio(clip):
            graph.append(f"[{i}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,apad,atrim=0:{length:.4f}[a{i}]")
        else:
            graph.append(f"anullsrc=r=48000:cl=stereo,atrim=0:{length:.4f},aformat=sample_fmts=fltp[a{i}]")
    vlast, alast, total = "[v0]", "[a0]", lengths_[0]
    for i in range(1, len(parts)):
        if parts[i].get("mode") == "previous" and not ease_in(parts[i]):  # with its ease-in cut it no longer repeats the frame
            # a continuation opens on the frame the one before ends on: that frame is dropped, and the sound cross-fades
            # over the same 1/24 s. (A one-frame xfade is not used: ffmpeg 6.1 drops the second clip with it.)
            fade = CONTINUE_FADE
            graph.append(f"[v{i}]trim=start_frame=1,setpts=PTS-STARTPTS[vt{i}]")
            graph.append(f"{vlast}[vt{i}]concat=n=2:v=1:a=0[vx{i}]")
        else:
            fade = min(CUT_FADE, lengths_[i] / 2, total / 2)
            graph.append(f"{vlast}[v{i}]xfade=transition=fade:duration={fade:.4f}:offset={total - fade:.4f}[vx{i}]")
        graph.append(f"{alast}[a{i}]acrossfade=d={fade:.4f}:c1=tri:c2=tri[ax{i}]")
        vlast, alast, total = f"[vx{i}]", f"[ax{i}]", total + lengths_[i] - fade
    target = folder / "film.mp4"
    scratch = folder / "film.part.mp4"
    command = ["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(graph), "-map", vlast, "-map", alast,
               "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
               "-movflags", "+faststart", str(scratch)]
    done = subprocess.run(command, capture_output=True, text=True, env=qa.ENV, timeout=3600)
    if done.returncode or not scratch.is_file():
        say(run, "The film could not be joined: " + (done.stderr or "").strip()[-300:])
        return None
    scratch.replace(target)
    run["film"] = {"path": relative(target), "segments": len(parts), "seconds": round(video_seconds(target), 2), "at": slate.now()}
    save(run)
    return run["film"]
