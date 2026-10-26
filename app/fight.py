"""A fight in AutoPilot as a choreographed sequence of short shots (autopilot.py hands a fight segment to ``step``).

Why. One 10 s take of a fight came out slow (MiniMax H3 eases out of the still it starts from: 7 of 8 fight takes opened
at 2 to 5 a second before the exchange ran at 10 to 16), ended in a stand-off, gave the opponent another face from one
piece to the next and kept a payroll chest in his arms through the blows. A fight is now cut like a fight:

  choreography  the Director (thinking on) writes the fight first: an ordered list of short shots (1.5 to 4 s), one
                move each, which opponent, the hero's technique by name and in plain physical words, the cause and its
                effect ("palm strike to the chest -> he is thrown backward and lands flat on his back"), the camera, the
                screen direction, who is down after it. The hero flows from one opponent to the next; a beaten man stays
                down; the count of men standing only goes down. Checked in code (``problem``) and asked again
  keyframes     each shot's keyframe is an edit of the hero's reference picture with the opponent's beside it (fixed
                references for every recurring character, their exact look and wardrobe lines word for word), made for
                all the shots first so the picture model and the video model do not swap in and out at every shot
  shots         each shot renders at the shortest length that holds its move (``SHOT_FRAMES``), from its keyframe, the
                speed in the prompt's first words and the physical reaction in its action
  checks        the eyes, yes/no per shot (``questions``): the hero, the right opponent, the wardrobe, the move, contact,
                the reaction's direction, no floating or sliding, no morphing limb or weapon, no merged bodies, the
                men already down still down, the hero's form (a vertical fist, elbows in). A failed major, voted on,
                renders that shot again (cheap: a shot is 2 to 4 s)
  ramp          every exchange shot is sped up 1.5 to 2x (frames dropped, never duplicated), the planned key move (one
                or two a fight) is left at 1x or slowed with motion-interpolated frames; the motion is measured before
                and after. No ramp over a fall: a shot whose effect is a fall is ramped no faster than RAMP_FALL
  cut           the shots join with hard cuts, no dissolve, into the segment's clip

Wing Chun is the hero's style when the seed says so (``WING_CHUN``): every move of his is one of its techniques, named
in the choreography and said in plain words in the prompt, since the video model does not know the names.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

import checks
import lengths

# the shortest clip lengths the engine renders well, a frame count a move length (17n + 5)
SHOT_FRAMES = {1.5: 39, 2.0: 56, 2.5: 56, 3.0: 73, 3.5: 90, 4.0: 90}
RAMP_EXCHANGE, RAMP_FALL, RAMP_KEY = 1.75, 1.3, 0.6  # exchanges sped up, a fall sped up less, a key move slowed
KEYS_MAX = 2
SHOT_RETAKES = 2
TRIM_S = 0.25  # the first frames hold the keyframe still: left out of every shot

WING_CHUN = {
    "Yee Ji Kim Yeung Ma": "a narrow inward stance, knees in, facing the attacker square on",
    "Juen Ma": "a short turning step on the spot",
    "Yat Ji Kuen": "a straight punch from the center of the chest with a vertical fist, thumb up, elbow down and in",
    "Lin Wan Kuen": "rapid chain punches, one vertical fist after another down the centerline, too fast to count",
    "Biu Jee": "a straight finger thrust to the throat",
    "Jing Jeung": "a straight palm strike to the chest, fingers up",
    "Wang Jeung": "a sideways palm strike to the ribs, fingers pointing across",
    "Pai Jarn": "a short horizontal elbow strike at close range",
    "low kick": "a low front kick to the knee or shin with the heel",
    "Pak Sau": "slaps the attacker's arm down with one palm and punches straight with the other hand in the same instant",
    "Tan Sau": "deflects the attack outward with the palm up and strikes with the other hand at once",
    "Bong Sau": "turns the forearm into a wing to deflect the blow while stepping in",
    "Lap Sau": "grabs the attacker's arm and pulls him forward off balance, then strikes",
    "Jut Sau": "jerks the attacker's arm down sharply to open his guard",
    "Fook Sau": "rides on top of the attacker's arm with a bent wrist, controlling it",
}

SYSTEM = """You choreograph a fight for a film made of short shots by a text-to-video model. Answer with one JSON \
object and nothing else:
{
  "location": "<the one place of the whole fight, 20 words max>",
  "shots": [
    {"opponents": [<the opponent numbers in this shot, 1 to N>],
     "technique": "<the hero's technique(s) by name, from the list given>",
     "move": "<what the hero does, in plain physical words, 25 words max>",
     "effect": "<cause -> effect: where it lands and what the body does after, in plain physical words, 20 words max>",
     "camera": "<one camera: low angle, over the shoulder, wide, tracking, close; 10 words max>",
     "screen": "left-to-right | right-to-left",
     "seconds": 1.5 | 2 | 2.5 | 3 | 3.5 | 4,
     "key": true | false,
     "down_after": [<every opponent number lying beaten on the floor after this shot>]}
  ]
}

Rules that matter:
- One move a shot, 1.5 to 4 seconds. The fight escalates: one or two opponents at a time at first, then several at once.
- The hero flows from one opponent to the next: a shot's last action hands off to the next shot's opponent (he finishes \
one and turns straight into the next one's attack, sends one attacker into another, steps over a fallen man).
- Physics: every hit lands with visible contact and the struck body reacts in the direction of the hit; bodies have \
weight, falls follow momentum onto the floor, nobody floats or slides, nothing bends or morphs. Write the cause and \
its effect in "effect".
- A beaten opponent stays down: "down_after" only grows, an opponent in it never fights again, and by the last shot \
every opponent is down.
- "key": true for one or two decisive moves only (the finishing flurry); every other shot is a fast exchange.
- Props appear only when the choreography names them. Keep it suitable for a general audience: no blood, no injury \
shown in detail.
- Write in English."""


def raw_seconds(target: float) -> tuple[float, int]:
    """(seconds of shots to render, shots) for a fight of ``target`` seconds once ramped and trimmed: the shots run about
    1.5x faster on average and lose TRIM_S each. On the first dojo run the Director wrote 9 shots, 20 s, for a 50 s fight."""

    shots = max(6, round(target / 2.0))
    return round(1.5 * (target + TRIM_S * shots)), shots


def frames_for(seconds: float) -> int:
    return SHOT_FRAMES.get(min(SHOT_FRAMES, key=lambda s: abs(s - float(seconds or 2))), 56)


def opponents_of(seed: dict) -> list[dict]:
    """The seed's opponent sheets (characters with ``role: opponent``): the faces the opponents are drawn with."""

    return [c for c in seed.get("characters") or [] if c.get("role") == "opponent"]


def hero_of(seed: dict) -> dict | None:
    return next((c for c in seed.get("characters") or [] if c.get("role") != "opponent"), None)


def face_of(seed: dict, number: int) -> dict | None:
    """Opponent ``number`` (1 to N) drawn with one of the opponent sheets in turn, so N men read as different men."""

    faces = opponents_of(seed)
    return faces[(int(number) - 1) % len(faces)] if faces else None


def prompt(run: dict, seg: dict, count: int, target: float, style: str) -> str:
    seed, hero = run["seed"], hero_of(run["seed"])
    faces = opponents_of(seed)
    wing = bool(re.search(r"wing ?chun", json.dumps(seed) + (run["settings"].get("guidance") or ""), re.I))
    techniques = "\n".join(f"- {name}: {words}" for name, words in WING_CHUN.items())
    return ((f"The hero: {hero['name']}, {hero.get('look', '')}.\n" if hero else "")
            + f"The opponents: {count} men, numbered 1 to {count}, all in the same clothes"
            + (f" ({faces[0].get('wardrobe') or faces[0].get('look')})" if faces else "") + ".\n"
            + f"The segment this fight is (its beat): {seg.get('beat', '')}\nIts setting: {(seg.get('fields') or {}).get('setting', '')}\n"
            + (f"Visual style: {style}\n" if style else "")
            + (f"The user's direction: {run['settings']['guidance']}\n" if run["settings"].get("guidance") else "")
            + "\nWrite the whole fight: about {:.0f} seconds of shots in all, {} shots or more. It escalates: at first the opponents come one at a time and most take two or three shots before they go down (an exchange, a block, then the finish); later two or three attack at once; the last ones fall together.\n".format(*raw_seconds(target))
            + (f"\nThe hero fights in Wing Chun. Every move of his is one of these techniques, named in \"technique\" and said in "
               f"plain physical words in \"move\"; attack and defense at the same time, close range, short powerful strikes, elbows "
               f"in, the centerline; no big wind-ups, high spinning kicks or acrobatics for him. The opponents may fight in other "
               f"styles (wide swings, karate kicks, charges). The signature beats: a Lap Sau pull into a chain-punch flurry (Lin Wan "
               f"Kuen) that drives a man backward across the floor (a key move), and an attacker redirected into another.\n{techniques}\n"
               if wing else ""))


def problem(data: dict, count: int, wing: bool, target: float = 0) -> str:
    """Why a choreography cannot be used ("" when it can)."""

    shots = data.get("shots")
    if not isinstance(shots, list) or len(shots) < 3:
        return '"shots" is a list of short shots, one move each'
    down: set = set()
    keys = 0
    for i, shot in enumerate(shots, 1):
        if not isinstance(shot, dict):
            return f"shot {i} is not an object"
        try:
            ops = [int(x) for x in shot.get("opponents") or []]
            after = {int(x) for x in shot.get("down_after") or []}
            seconds = float(shot.get("seconds") or 0)
        except (TypeError, ValueError):
            return f"shot {i}: opponents, down_after and seconds are numbers"
        if not ops or any(not 1 <= o <= count for o in ops):
            return f"shot {i}: name its opponent(s), numbers 1 to {count}"
        fighting = [o for o in ops if o in down]
        if fighting:
            return f"shot {i}: opponent {fighting[0]} is already down after an earlier shot: a beaten man stays down"
        if not down <= after:
            return f"shot {i}: \"down_after\" drops {sorted(down - after)[0]}, who was already down: it only grows"
        if not 1.5 <= seconds <= 4:
            return f"shot {i} is {seconds:g} s: a shot is 1.5 to 4 seconds"
        for key in ("move", "effect", "camera"):
            if not str(shot.get(key) or "").strip():
                return f'shot {i}: "{key}" is needed'
        if "->" not in str(shot.get("effect")) and "→" not in str(shot.get("effect")):
            return f'shot {i}: "effect" is cause -> effect, like "palm strike to the chest -> thrown backward onto his back"'
        if wing and not any(name.lower() in str(shot.get("technique") or "").lower() for name in WING_CHUN):
            return f"shot {i}: the hero's technique is one of the Wing Chun techniques listed, by name"
        keys += bool(shot.get("key"))
        down = after
    if target:
        raw, wanted = raw_seconds(target)
        total = sum(float(x.get("seconds") or 0) for x in shots)
        if total < 0.8 * raw or len(shots) < 0.75 * wanted:
            return (f"{len(shots)} shots, {total:g} s: the fight needs about {raw:.0f} s of shots, {wanted} shots or more. Most opponents "
                    "take two or three shots before they go down (an exchange, a block, the finish); later several attack at once")
        if count >= 5 and sum(1 for x in shots if len(x.get("opponents") or []) >= 2) < 2:
            return "the fight escalates: at least two shots where two or three opponents attack at once"
    if keys > KEYS_MAX:
        return f"{keys} key moves: one or two decisive moves only, every other shot is a fast exchange"
    if down != set(range(1, count + 1)):
        return f"by the last shot every opponent is down; still standing: {sorted(set(range(1, count + 1)) - down)}"
    return ""


def clean(data: dict) -> list[dict]:
    out, down = [], []
    for i, shot in enumerate(data["shots"], 1):
        after = sorted({int(x) for x in shot.get("down_after") or []})
        out.append({"n": i, "opponents": [int(x) for x in shot["opponents"]], "technique": str(shot.get("technique") or "")[:80],
                    "move": " ".join(str(shot["move"]).split()[:30]), "effect": " ".join(str(shot["effect"]).split()[:26]),
                    "camera": " ".join(str(shot["camera"]).split()[:12]), "screen": str(shot.get("screen") or "")[:16],
                    "seconds": float(shot["seconds"]), "key": bool(shot.get("key")), "down_before": down, "down_after": after,
                    "frames": frames_for(shot["seconds"]), "status": "keyframe", "takes": [], "keyframes": []})
        down = after
    return out


def plain(technique: str) -> str:
    """The technique's names said in plain physical words for the video model."""

    words = [w for name, w in WING_CHUN.items() if name.lower() in (technique or "").lower()]
    return "; ".join(words)


def who_line(run: dict, shot: dict) -> tuple[str, list[dict]]:
    """The look and wardrobe of the hero and of this shot's opponents, word for word, and their sheets in picture order."""

    seed = run["seed"]
    hero = hero_of(seed)
    sheets = [hero] if hero else []
    bits = []
    if hero:
        bits.append(f"{hero['name']} is {hero.get('look', '').rstrip('.')}" + (f"; he wears exactly {hero['wardrobe'].rstrip('.')}" if hero.get("wardrobe") else "") + ".")
    for o in shot["opponents"][:2]:
        face = face_of(seed, o)
        if face:
            sheets.append(face)
            bits.append(f"Opponent {o} is {face.get('look', '').rstrip('.')}" + (f", wearing exactly {face['wardrobe']}" if face.get("wardrobe") else "") + ".")
    return " ".join(bits), sheets


def floor_line(shot: dict) -> str:
    down = len(shot["down_before"])
    return (f"{down} beaten opponent{'s' if down != 1 else ''} in the same white uniform lie{'s' if down == 1 else ''} still on the floor "
            "where they fell." if down else "")


def action_of(shot: dict, hero: str) -> str:
    """The shot's action: the speed first, the move in plain words, the physical reaction, the fall's weight."""

    speed = ("The decisive moment, at real-time speed, every impact crisp:" if shot["key"] else
             "From the very first frame, already in motion, at full real-time speed and with force:")
    form = plain(shot["technique"])
    return " ".join(bit for bit in (speed, f"{hero} {shot['move'].rstrip('.')}.", f"({form})." if form else "",
                                    f"{shot['effect'].replace('->', 'and then').replace('→', 'and then').rstrip('.')}: the blow visibly lands, "
                                    "the struck body reacts in the direction of the hit, falls follow their momentum with full weight "
                                    "onto the floor, feet stay planted, nothing floats or bends.") if bit)


def ramp_of(shot: dict) -> float:
    if shot["key"]:
        return RAMP_KEY
    return RAMP_FALL if re.search(r"\b(fall|falls|falling|thrown|lands|collapse|drops|crash)", shot["effect"], re.I) else RAMP_EXCHANGE


def ramp(source: Path, target: Path, speed: float, trim: float, env=None) -> bool:
    """The shot sped up (frames dropped, never repeated) or slowed with motion-interpolated frames, its first ``trim``
    seconds left out; the sound follows (atempo)."""

    video = f"trim=start={trim:.3f},setpts=(PTS-STARTPTS)/{speed:.4f}"
    video += ",fps=24" if speed >= 1 else ",minterpolate=fps=24:mi_mode=mci:mc_mode=aobmc:me_mode=bidir"
    tempo = ",".join(f"atempo={t:.4f}" for t in ([speed] if 0.5 <= speed <= 2 else [speed ** 0.5, speed ** 0.5]))
    audio = f"atrim=start={trim:.3f},asetpts=PTS-STARTPTS,{tempo}"
    command = ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-filter_complex", f"[0:v]{video},format=yuv420p[v];[0:a]{audio}[a]",
               "-map", "[v]", "-map", "[a]", "-shortest", "-c:v", "libx264", "-crf", "16", "-preset", "veryfast", "-c:a", "aac", "-b:a", "192k", str(target)]
    done = subprocess.run(command, capture_output=True, env=env, timeout=900)
    if done.returncode:  # a clip without sound
        command = ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-vf", f"{video},format=yuv420p", "-an", "-c:v", "libx264", "-crf", "16",
                   "-preset", "veryfast", str(target)]
        done = subprocess.run(command, capture_output=True, env=env, timeout=900)
    return done.returncode == 0 and target.is_file()


def cut(parts: list[Path], target: Path, env=None) -> bool:
    """The ramped shots joined with hard cuts (no dissolve inside a fight); a shot without sound gets silence."""

    command, graph = ["ffmpeg", "-v", "error", "-y"], []
    for i, p in enumerate(parts):
        command += ["-i", str(p)]
        has = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(p)],
                             capture_output=True, text=True, env=env, timeout=30).stdout.strip()
        graph.append(f"[{i}:v]fps=24,setsar=1,format=yuv420p[v{i}]")
        graph.append(f"[{i}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,apad,atrim=0:{video_seconds(p, env):.4f}[a{i}]" if has
                     else f"anullsrc=r=48000:cl=stereo,atrim=0:{video_seconds(p, env):.4f}[a{i}]")
    graph.append("".join(f"[v{i}][a{i}]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=1:a=1[v][a]")
    done = subprocess.run(command + ["-filter_complex", ";".join(graph), "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "17",
                                     "-preset", "veryfast", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(target)],
                          capture_output=True, env=env, timeout=1800)
    return done.returncode == 0 and target.is_file()


def video_seconds(path: Path, env=None) -> float:
    """The picture's own length (the sound of a ramped shot may run a frame longer): each shot's sound is cut to it."""

    said = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets",
                           "-of", "csv=p=0", str(path)], capture_output=True, text=True, env=env, timeout=60).stdout.strip()
    try:
        return int(said.split(",")[0]) / 24
    except ValueError:
        return duration(path, env)


def duration(path: Path, env=None) -> float:
    said = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                          capture_output=True, text=True, env=env, timeout=30).stdout.strip()
    try:
        return float(said)
    except ValueError:
        return 0.0


def questions(run: dict, shot: dict, wing: bool, keyframe: bool = False) -> list[dict]:
    """The eyes' yes/no questions for one shot (or its keyframe): who, what they wear, the move and its physics."""

    seed = run["seed"]
    hero = hero_of(seed) or {"name": "the hero"}
    out = []

    def ask(category, text, expect, major=True):
        out.append({"key": f"f{len(out) + 1}", "category": category, "text": text, "expect": expect, "major": major})
    name = hero["name"]
    ask("identity", f"Is {name} the same man as in the first reference picture (face, hair)?", "yes")
    if hero.get("wardrobe"):
        ask("wardrobe", f"Does {name} wear {hero['wardrobe']}?", "yes")
    if opponents_of(seed):
        ask("identity", f"Is {name} fighting a man who looks like the second reference picture (face, hair, clothes)?", "yes")
    if shot["down_before"]:
        ask("story", "Is at least one beaten man lying on the floor?", "yes")
    ask("setting", f"Is this place {(run.get('fight_location') or '')[:120] or 'the place of the fight'}?", "yes")
    if keyframe:
        ask("pace", "Is someone caught in the middle of a fast movement, a blow or a block already in motion, rather than standing still?", "yes")
        return out
    ask("action", f"Does this happen: {name} {shot['move']}?", "yes")
    if re.search(r"punch|strike|palm|elbow|kick|thrust|fist|hit", shot["move"] + shot["technique"], re.I):
        ask("action", "Does a blow visibly land on a body?", "yes")
    ask("action", f"Does the struck man react like this: {shot['effect'].split('->')[-1].split('→')[-1].strip()}?", "yes")
    ask("physics", "Does anyone float, hover, or slide across the floor without stepping?", "no")
    ask("physics", "Does any arm, leg, hand or weapon bend, stretch, melt, duplicate or vanish?", "no")
    ask("physics", "Do two bodies merge into each other or pass through each other?", "no")
    if wing and re.search(r"kuen|punch|fist", shot["technique"] + shot["move"], re.I):
        ask("form", f"Does {name} punch with a vertical fist (thumb on top) with his elbows kept close to his body?", "yes", major=False)
    if wing and re.search(r"pak sau|tan sau|bong sau|lap sau", shot["technique"], re.I):
        ask("form", f"Does {name} block or grab with one hand and strike with the other at the same moment?", "yes", major=False)
    return out
