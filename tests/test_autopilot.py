"""AutoPilot (app/autopilot.py): the self-directing loop with a scripted Director and watcher, a queue whose
clips are made at once by ffmpeg (tiny test patterns), and the QA gate stood in. No GPU, no network, no model."""
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("STUDIO_HOME", str(Path(tempfile.mkdtemp()) / "ImaginationStudio"))
sys.path.insert(0, str(ROOT / "app"))
from fastapi.testclient import TestClient  # noqa: E402

import autopilot  # noqa: E402
import cosmos  # noqa: E402
import server  # noqa: E402  (attaches AutoPilot)

HOME = server.HOME
client = TestClient(server.app)
pytestmark = pytest.mark.skipif(not (os.path.exists("/usr/bin/ffmpeg") or os.path.exists("/opt/homebrew/bin/ffmpeg")), reason="needs ffmpeg")

SEED = {"world": "A rain-soaked neon megacity of the 2080s, hover-trams over flooded streets.",
        "situation": "A courier on a mono-wheel races a storm front to deliver a glowing data core.",
        "characters": [{"name": "Vex", "look": "a wiry courier in her twenties, shaved undercut, chrome visor, orange reflective jacket"}]}
BEATS = ["Vex launches off the tram platform", "Vex threads between hover-trams", "Vex skids under a falling sign",
         "Vex reaches the rooftop relay", "Vex hands the core to a drone", "Vex watches the city light up"]


def shot(beat: str, scene: str = "Flooded avenue", change: bool = False) -> dict:
    return {"scene": scene, "scene_change": change, "beat": beat, "new_characters": [], "keyframe": f"Vex mid-move: {beat}, wide shot, neon night",
            "shot": {"setting": "A flooded neon avenue at night, magenta and cyan reflections", "camera": "Tracking shot alongside the mono-wheel",
                     "action": f"{beat}, spray fanning from the wheel.", "dialogue": [], "soundscape": "Rain, a whining electric motor", "music": "N/A"}}


class Director:
    """Writes the next beat; ``repeat_once`` first answers with the beat before (to be turned back)."""

    def __init__(self, repeat_once: bool = False):
        self.asked, self.repeat = [], repeat_once

    def __call__(self, run, system, prompt):
        self.asked.append(prompt)
        if "Fold this stretch" in prompt:
            return json.dumps({"title": "The run", "summary": "Vex crossed the city.", "carried_forward": ["the core is delivered"]})
        n = len(autopilot.committed(run))
        if self.repeat and n >= 1:
            self.repeat = False
            return "```json\n" + json.dumps(shot(BEATS[n - 1])) + "\n```"
        return json.dumps(shot(BEATS[n % len(BEATS)], change=n == 0))


class Watcher:
    """The vision brain stood in: one fixed observation per call; ``drift`` lists the calls that see identity drift.
    It answers the continuity checklist too (``checks``/``facts``: {checklist call number: answer part}, all ok by
    default), the second look (``second``: what it says in turn, confirmed by default) and the film-level pass."""

    def __init__(self, drift=(), same=False, confirms=True, checks=None, facts=None, second=(), film=None, world=None, present="Vex", keyframes=None,
                 yesno=None):
        self.calls, self.drift, self.same, self.prompts, self.confirms, self.checked = 0, set(drift), same, [], confirms, 0
        self.checks, self.facts, self.second, self.film, self.world, self.present = checks or {}, facts or {}, list(second), film, world, present
        self.checklists, self.looks, self.films, self.keyframes, self.keyframe_checks = [], [], [], keyframes or {}, []
        self.yesno, self.answer_yesno = [], yesno  # the dense checks: unanswered ("unclear") unless ``yesno`` says

    def __call__(self, system, frames, sheet, prompt, brain):
        if prompt.startswith("YES/NO CHECK"):
            self.yesno.append((prompt, [p for _, p in frames]))
            assert all(Path(p).is_file() for _, p in frames)
            return (self.answer_yesno(prompt, frames) if self.answer_yesno else "{}"), "stand-in"
        if "clearly the same person" in prompt:  # the narrower second look (confirm_drift)
            self.checked += 1
            assert len(frames) == 3
            return json.dumps({"same": not self.confirms, "why": "another face"}), "stand-in"
        if "CONTINUITY CHECKLIST" in prompt:
            self.checklists.append(prompt)
            n = len(self.checklists)
            assert len(frames) >= 4 and all(Path(p).is_file() for _, p in frames)
            checks = {c: {"verdict": "ok", "frame": 0, "evidence": ""} for c in autopilot.continuity.WATCHED}
            checks.update(self.checks.get(n) or {})
            return json.dumps({"checks": checks, "facts": self.facts.get(n) or [],
                               "present": [{"who": self.present, "frame": 2}] if self.present else [], "positions": [], "open_actions": [],
                               "light": "night, rain, neon"}), "stand-in"
        if "KEYFRAME CHECK" in prompt:
            self.keyframe_checks.append(prompt)
            assert len(frames) == 1 and Path(frames[0][1]).is_file()
            return json.dumps({"checks": self.keyframes.get(len(self.keyframe_checks)) or {}}), "stand-in"
        if "A FIRST LOOK REPORTED" in prompt:
            self.looks.append((prompt, [p for _, p in frames]))
            assert all(Path(p).is_file() for _, p in frames) and sheet.is_file()
            return json.dumps({"confirmed": self.second.pop(0) if self.second else True, "why": "looked closer"}), "stand-in"
        if "SAME CHARACTER ACROSS THE FILM" in prompt:
            self.films.append(prompt)
            return json.dumps(self.film or {"same": True, "different": [], "why": "one person"}), "stand-in"
        if "ACROSS THE WHOLE FILM" in prompt:
            self.films.append(prompt)
            return json.dumps(self.world or {"issues": []}), "stand-in"
        self.calls += 1
        self.prompts.append(prompt)
        assert len(frames) >= 4 and all(Path(p).is_file() for _, p in frames) and sheet.is_file()
        tag = "x" if self.same else str(self.calls)
        return json.dumps({"add_nodes": [{"id": "vex", "type": "character", "label": "Vex", "state": "riding hard"}] if self.calls == 1 else [],
                           "update_nodes": [{"id": "vex", "state": f"riding hard {tag}"}],
                           "add_events": [{"t": "00:01", "subject": "vex", "action": f"swerves past obstacle {tag}"}],
                           "summary": "Vex rides.", "ending": "Vex mid-turn, spray in the air.", "continuity": [],
                           "identity_drift": self.calls in self.drift}), "stand-in"


class Queue:
    def __init__(self, after=None):
        self.jobs, self.bodies, self.after = {}, [], after

    def submit(self, body):
        params = body["params"]
        if body["kind"] in ("scout", "edit"):  # a keyframe: one still (made, or edited from a portrait), where those jobs put it
            job_id = f"j{len(self.jobs) + 1}"
            picture = HOME / "scouts" / job_id / f"scout_s{params['seed']}.png"
            picture.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=teal:size=160x96", "-frames:v", "1", str(picture)], check=True)
            job = {"id": job_id, "status": "done", "outputs": [str(picture), str(picture.parent / "sheet.jpg")], "params": params}
            self.jobs[job_id] = job
            self.bodies.append(body)
            return job
        target = HOME / params["output"]
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x96:rate=24:duration=1",
                        "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", str(target)], check=True)
        job = {"id": f"j{len(self.jobs) + 1}", "status": "done", "outputs": [params["output"]], "params": params}
        self.jobs[job["id"]] = job
        self.bodies.append(body)
        if self.after:
            self.after(len(self.bodies))
        return job

    def lookup(self, job_id):
        return self.jobs.get(job_id)


@pytest.fixture
def loop(monkeypatch):
    queue = Queue()
    monkeypatch.setattr(autopilot, "submit", queue.submit)
    monkeypatch.setattr(autopilot, "job_of", queue.lookup)
    monkeypatch.setattr(autopilot, "cancel", lambda job_id: None)
    monkeypatch.setattr(autopilot, "room", lambda params: None)  # this machine's memory is not the test's
    monkeypatch.setattr(autopilot, "gate", lambda clip, prompt, sheet, brain: {"passed": True, "audio": {"result": "pass", "heard": "go go go"}, "identity": {"result": "pass"}})
    monkeypatch.setattr(autopilot.qa, "ready", lambda: {"runs": True, "transcript": True, "sheet": True, "missing": []})
    director, watcher = Director(), Watcher()
    monkeypatch.setattr(autopilot, "ask_text", director)
    monkeypatch.setattr(autopilot, "ask_vision", watcher)
    return {"queue": queue, "director": director, "watcher": watcher}


def new_run(segments=3, **settings) -> dict:
    said = client.post("/api/autopilot/runs", json={"name": "Storm Courier", "seed": SEED, "settings": {"segments": segments, "preview_every": 2, **settings}})
    assert said.status_code == 200, said.text
    return autopilot.load(said.json()["id"])


def go(run):
    autopilot.work(run, threading.Event())
    return autopilot.load(run["id"])


# ---------- the scene graph ----------

def test_merge_folds_ids_clips_states_and_keeps_two_events_a_segment():
    graph = autopilot.merge(autopilot.EMPTY["graph"], {"add_nodes": [{"id": "guard", "type": "character", "label": "The night guard", "state": "asleep"}]})
    delta = {"add_nodes": [{"id": "guard_2", "label": "Guard", "state": "awake now and very very very angry at the courier on the wheel"}],
             "update_nodes": [], "add_events": [{"subject": "guard_2", "action": "a"}, {"subject": "guard", "action": "b"}, {"subject": "guard", "action": "c"}]}
    graph = autopilot.merge(graph, delta, segment=4)
    assert [n["id"] for n in graph["nodes"]] == ["guard"]
    assert len(graph["nodes"][0]["state"].split()) == autopilot.STATE_WORDS
    assert [e["action"] for e in graph["events"]] == ["a", "b"] and graph["events"][0]["subject"] == "guard"
    assert all(e["segment"] == 4 for e in graph["events"])


def test_fold_makes_a_chapter_and_the_view_stays_bounded():
    graph = {"nodes": [{"id": "vex", "type": "character", "label": "Vex", "state": "soaked"}, {"id": "core", "type": "object", "label": "Data core", "state": "glowing"}],
             "events": [{"t": f"00:{i:02d}", "subject": "vex", "action": f"step {i}"} for i in range(30)], "chapters": []}
    folded = autopilot.fold(graph, {"title": "Across the flooded city at night", "summary": "Vex crossed.", "carried_forward": ["a", "b", "c", "d"]}, graph["events"][:10])
    assert len(folded) == 10 and len(graph["events"]) == 20
    assert graph["chapters"][0]["title"] == "Across the flooded city at" and len(graph["chapters"][0]["carried_forward"]) == 3
    text = autopilot.as_text(graph)
    assert "Across the flooded city" in text and "step 29" in text and "step 10" not in text and "Data core (glowing)" in text


def test_stagnation_needs_three_same_observations_in_one_scene():
    seen = {"add_nodes": [], "update_nodes": [{"id": "vex", "state": "waiting"}], "add_events": [{"subject": "vex", "action": "waits"}]}
    segs = [{"index": i, "scene": "Alley", "watch": {"delta": seen}} for i in range(1, 4)]
    assert autopilot.stagnation(segs)["segments"] == [1, 2, 3]
    segs[1]["scene"] = "Roof"
    assert autopilot.stagnation(segs) is None


# ---------- the loop ----------

def test_a_run_renders_watches_commits_and_joins_the_film(loop):
    run = go(new_run(3))
    assert run["state"] == "done" and [s["status"] for s in run["segments"]] == ["done"] * 3
    bodies = loop["queue"].bodies
    assert all(b["kind"] == "video" and b["params"]["qa"] is False for b in bodies)
    assert "first_frame_from" not in bodies[0]["params"]
    assert bodies[1]["params"]["first_frame_from"] == run["segments"][0]["output"]  # last-frame continuation
    assert [s["mode"] for s in run["segments"]] == ["text", "previous", "previous"]
    assert [b["params"]["seed"] for b in bodies] == [1, 2, 3]
    assert "Vex is a wiry courier" in bodies[1]["params"]["prompt"]  # the pinned look, said every time
    folder = autopilot.run_dir(run["id"])
    memory = json.loads((folder / "memory.json").read_text())
    assert memory["observed_through"] == 3 and memory["graph"]["nodes"][0]["id"] == "vex" and len(memory["graph"]["events"]) == 3
    assert all((folder / "memory" / f"S{i:04d}.json").is_file() for i in (1, 2, 3))
    assert json.loads((folder / "scene-graph.schema.json").read_text())["title"] == "AutoPilot scene graph"
    film = run["film"]
    assert film["segments"] == 3 and (HOME / film["path"]).is_file()
    assert abs(film["seconds"] - (3 * 24 - 2) / 24) < 0.01  # three 1 s clips, the repeated frame dropped twice
    assert abs(autopilot.video_seconds(HOME / film["path"]) - autopilot.qa.duration(HOME / film["path"])) < 0.1  # picture and sound together
    # the Director saw what the watcher saw, and the transcript
    assert "Vex mid-turn, spray in the air." in loop["director"].asked[1] and "go go go" in loop["director"].asked[1]
    assert "go go go" in loop["watcher"].prompts[0]


def test_the_director_is_turned_back_when_it_repeats_the_beat_before(loop, monkeypatch):
    director = Director(repeat_once=True)
    monkeypatch.setattr(autopilot, "ask_text", director)
    run = go(new_run(2))
    second = run["segments"][1]
    assert second["beat"] == BEATS[1] and len(second["rejected"]) == 1
    assert "repeats the segment before" in second["rejected"][0] or "restages segment 1" in second["rejected"][0]
    assert "YOUR LAST ANSWER COULD NOT BE USED" in director.asked[2]


def test_a_spent_scene_and_a_negation_are_turned_back():
    run = {"settings": dict(autopilot.DEFAULTS, max_scene_segments=2), "segments": [
        {"index": 1, "status": "done", "scene": "Avenue", "scene_change": True, "beat": "launch", "fields": {"action": "Vex launches."}},
        {"index": 2, "status": "done", "scene": "Avenue", "scene_change": False, "beat": "weave", "fields": {"action": "Vex weaves."}}]}
    answer = shot("Vex climbs the relay mast")
    assert "budget" in autopilot.director_problem(run, answer, 3, None)
    assert autopilot.director_problem(run, {**answer, "scene_change": True, "scene": "Relay mast"}, 3, None) == ""
    bad = shot("Vex climbs")
    bad["shot"]["setting"] = "A rooftop with no people in sight"
    assert "names what must not be seen" in autopilot.director_problem(run, {**bad, "scene_change": True}, 3, None)
    assert "circling" in autopilot.director_problem(run, {**answer, "scene_change": True, "scene": "Avenue"}, 3, {"reason": "x"})


def test_identity_drift_renders_the_segment_again_at_the_next_seed(loop, monkeypatch):
    watcher = Watcher(drift={2})
    monkeypatch.setattr(autopilot, "ask_vision", watcher)
    run = go(new_run(2, retakes=1))
    second = run["segments"][1]
    assert [t["seed"] for t in second["takes"]] == [2, 3] and second["takes"][0]["passed"] is False
    assert "identity drift" in second["takes"][0]["reasons"][0] and second["passed"] is True
    assert [b["params"]["seed"] for b in loop["queue"].bodies] == [1, 2, 3]
    assert watcher.checked == 1 and second["takes"][0]["watch"]["drift_check"]["same"] is False


def test_drift_the_narrower_look_does_not_confirm_is_not_a_failure(loop, monkeypatch):
    watcher = Watcher(drift={2}, confirms=False)
    monkeypatch.setattr(autopilot, "ask_vision", watcher)
    run = go(new_run(2, retakes=1))
    second = run["segments"][1]
    assert len(second["takes"]) == 1 and second["passed"] is True and second["watch"]["identity_drift"] is False
    assert second["watch"]["drift_check"] == {"same": True, "why": "another face"}


def test_pause_then_resume_carries_on(loop):
    run = new_run(3)
    loop["queue"].after = lambda n: run.__setitem__("pause", True) if n == 1 else None
    run = go(run)
    assert run["state"] == "paused" and len(autopilot.committed(run)) == 1 and run["film"]["segments"] == 1
    loop["queue"].after = None
    run = go(run)
    assert run["state"] == "done" and len(autopilot.committed(run)) == 3 and len(loop["queue"].bodies) == 3


def test_a_restart_with_the_clip_on_disk_picks_the_segment_up(loop):
    run = new_run(2)
    run["segments"].append({"index": 1, "id": "S0001", "status": "queued", "job": "gone", "scene": "Avenue", "scene_change": True,
                            "beat": "launch", "fields": shot("launch")["shot"], "rejected": [], "seed": 1, "frames": 124, "seconds": 5.17,
                            "start": 0, "takes": [], "mode": "text", "prompt": "p", "output": f"autopilot/{run['id']}/segments/S0001/take1.mp4"})
    loop["queue"].submit({"kind": "video", "params": {"output": run["segments"][0]["output"]}})  # the clip a lost job made
    loop["queue"].jobs.clear()
    loop["queue"].bodies.clear()
    run = go(run)
    assert run["state"] == "done" and len(loop["queue"].bodies) == 1  # only segment 2 was rendered


def test_branch_shares_the_past_and_directs_the_new_part(loop):
    source = go(new_run(3))
    said = client.post(f"/api/autopilot/runs/{source['id']}/branch", json={"after": 2, "guidance": "A rival courier cuts in and steals the core."})
    assert said.status_code == 200, said.text
    branch = autopilot.load(said.json()["id"])
    assert branch["branch"]["after"] == 2 and [s["shared"] for s in branch["segments"]] == [source["id"]] * 2
    memory = json.loads((autopilot.run_dir(branch["id"]) / "memory.json").read_text())
    assert memory["observed_through"] == 2
    before = len(loop["director"].asked)
    branch = go(branch)
    assert branch["state"] == "done" and len(autopilot.committed(branch)) == 3
    asked = loop["director"].asked[before]
    assert "rival courier" in asked and "NOT something that already happened" in asked
    assert loop["queue"].bodies[-1]["params"]["first_frame_from"] == source["segments"][1]["output"]
    assert client.post(f"/api/autopilot/runs/{source['id']}/branch", json={"after": 9, "guidance": "x"}).status_code == 400
    assert client.post(f"/api/autopilot/runs/{source['id']}/branch", json={"after": 1, "guidance": " "}).status_code == 400


def test_a_new_scene_opens_on_the_cast_pictures_where_ref2va_is_installed(loop, monkeypatch):
    picture = HOME / "uploads" / "vex.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=orange:size=64x64", "-frames:v", "1", str(picture)], check=True)
    monkeypatch.setattr(autopilot, "ref2va", lambda: True)
    seed = {**SEED, "characters": [{**SEED["characters"][0], "image": "uploads/vex.png"}]}
    said = client.post("/api/autopilot/runs", json={"name": "Refs", "seed": seed, "settings": {"segments": 2}})
    run = go(autopilot.load(said.json()["id"]))
    first, second = (b["params"] for b in loop["queue"].bodies)
    assert first["references"] == ["uploads/vex.png"] and "quality" not in first and "first_frame_from" not in first
    assert second["first_frame_from"] and "references" not in second  # a reference is never combined with a first frame


# ---------- the routes ----------

def test_routes_refuse_bad_seeds_and_far_vision_servers():
    assert client.post("/api/autopilot/runs", json={"seed": {}}).status_code == 400
    assert client.post("/api/autopilot/runs", json={"seed": {**SEED, "characters": [{"name": "X", "image": "../etc/passwd"}]}}).status_code == 400
    assert client.post("/api/autopilot/vision", json={"endpoint": "http://203.0.113.9:8000"}).status_code == 400
    said = client.post("/api/autopilot/vision", json={"endpoint": "http://127.0.0.1:8341", "model": "cosmos3", "frames": 6}).json()
    assert said["endpoint"] == "http://127.0.0.1:8341" and said["frames"] == 6 and not said["managed"]
    assert client.post("/api/autopilot/vision", json={"endpoint": "studio:cosmos"}).json()["managed"] is True
    assert client.get("/api/autopilot/runs").status_code == 200


def test_the_vision_server_gets_every_frame_in_order_and_falls_back_to_the_sheet(monkeypatch, tmp_path):
    frames = []
    for i in range(3):
        p = tmp_path / f"f{i}.png"
        p.write_bytes(b"\x89PNG fake")
        frames.append((f"Frame {i + 1}, at {i:.2f} s", p))
    sheet = tmp_path / "sheet.png"
    sheet.write_bytes(b"\x89PNG sheet")
    sent = []

    def chat(endpoint, model, system, given, text, max_tokens=1500):
        sent.append([c for c, _ in given])
        if len(given) > 1:
            raise cosmos.TooMany("too many images")
        return "{}"
    monkeypatch.setattr(cosmos, "chat", chat)
    monkeypatch.setattr(cosmos, "settings", lambda: {"endpoint": "http://127.0.0.1:8341", "model": "cosmos3", "frames": 7})
    answer, who = cosmos.look("sys", frames, sheet, "what changed?")
    assert answer == "{}" and sent[0] == [c for c, _ in frames] and sent[1] == ["Frame sheet"] and "frame sheet" in who


def test_a_new_scene_dissolves_in_and_picture_and_sound_stay_together(loop, monkeypatch):
    class Cutter(Director):
        def __call__(self, run, system, prompt):
            n = len(autopilot.committed(run))
            return json.dumps(shot(BEATS[n], scene=f"Scene {n}", change=True))
    monkeypatch.setattr(autopilot, "ask_text", Cutter())
    run = go(new_run(3))
    assert [s["mode"] for s in run["segments"]] == ["text", "text", "text"]
    film = HOME / run["film"]["path"]
    assert abs(run["film"]["seconds"] - (3 - 2 * autopilot.CUT_FADE)) < 0.06
    assert abs(autopilot.video_seconds(film) - autopilot.qa.duration(film)) < 0.1


def test_an_outline_is_followed_beat_by_beat_with_each_era_its_own_cast(loop):
    still = HOME / "uploads" / "grown.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=red:size=64x64", "-frames:v", "1", str(still)], check=True)
    seed = {**SEED, "characters": [{"name": "Eirik the boy", "look": "a ten-year-old with copper braids", "segments": "1-2"},
                                   {"name": "Eirik the man", "look": "a man of twenty-five with copper braids", "segments": "3-"}],
            "outline": ["The storm", {"text": "Overboard"}, {"text": "Years later, in Boston", "image": "uploads/grown.png"}]}
    said = client.post("/api/autopilot/runs", json={"name": "Saga", "seed": seed, "settings": {"segments": 4, "keyframes": "off"}})
    assert said.status_code == 200, said.text
    run = go(autopilot.load(said.json()["id"]))
    asked = loop["director"].asked
    assert "This is segment 1: it carries beat 1 (\"The storm\")" in asked[0]
    assert "it carries beat 3 (\"Years later, in Boston\")" in asked[2] and "opens on a picture of its own" in asked[2]
    assert "outline (3 beats) is done" in asked[3]
    bodies = [b["params"] for b in loop["queue"].bodies]
    assert bodies[2]["image"] == "uploads/grown.png" and "first_frame_from" not in bodies[2]
    assert run["segments"][2]["mode"] == "image" and run["segments"][2]["scene_change"] is True
    # each era's look in its own segments only
    assert "a ten-year-old" in asked[1] and "a man of twenty-five" not in asked[1].split("Cast -- repeat")[1].split("Beats asked")[0]
    assert "a man of twenty-five" in asked[3].split("Cast -- repeat")[1].split("Beats asked")[0]
    assert autopilot.span("8-") == (8, None) and autopilot.span("1-7") == (1, 7) and autopilot.span("3") == (3, 3) and autopilot.span("") == (None, None)


def test_old_runs_move_to_the_new_folder(tmp_path):
    old = tmp_path / "autorl" / "r1"
    (old / "memory").mkdir(parents=True)
    (old / "run.json").write_text(json.dumps({"id": "r1", "segments": [{"output": "autorl/r1/segments/S0001/take1.mp4"}]}))
    autopilot.migrate(tmp_path)
    assert not (tmp_path / "autorl").exists()
    assert json.loads((tmp_path / "autopilot" / "r1" / "run.json").read_text())["segments"][0]["output"] == "autopilot/r1/segments/S0001/take1.mp4"
