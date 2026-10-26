"""AutoPilot's panel (app/static/autopilot.js), run in node after slate.js with the page's helpers stood in for
(tests/test_slate_page.py's harness). Skipped where node is not installed; no browser, no GPU, no network."""
import json
import subprocess
import tempfile
from pathlib import Path

from test_shell_page import need_node
from test_slate_page import HARNESS

ROOT = Path(__file__).resolve().parent.parent
SETUP = """
S.presets = { small: { label: "Small 864x480" }, sharp: { label: "Sharp Spec" } }; S.qualities = { fh8: "FastH3 8", fh4: "FastH3 4" };
SL.open = new Proxy({}, { get: () => true });
globalThis.keep = node => node;
"""


def run(code: str) -> dict:
    need_node()
    with tempfile.TemporaryDirectory() as scratch:
        page = Path(scratch) / "page.js"
        page.write_text((ROOT / "app/static/slate.js").read_text() + "\n" + (ROOT / "app/static/autopilot.js").read_text())
        (Path(scratch) / "harness.js").write_text(HARNESS)
        (Path(scratch) / "test.js").write_text(SETUP + code)
        done = subprocess.run(["node", str(Path(scratch) / "harness.js"), str(page), str(Path(scratch) / "test.js")],
                              capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


RUN = {"id": "r1", "name": "Storm Courier", "state": "paused", "running": False, "rev": 3, "settings": {"segments": 6, "preset": "small", "quality": "fh8", "retakes": 1, "qa": True, "loras": []},
       "segments": [{"index": 1, "status": "done", "scene": "Flooded avenue", "scene_change": True, "beat": "Vex launches off the tram platform", "seconds": 5.17, "seed": 1,
                     "mode": "text", "output": "autopilot/r1/segments/S0001/take1.mp4", "fields": {"action": "Vex launches.", "dialogue": []},
                     "watch": {"summary": "Vex jumps the gap.", "ending": "Vex mid-air over the tram.", "continuity": [{"kind": "costume", "who": "vex", "note": "jacket turns red"}], "by": "Cosmos3-Edge"},
                     "takes": [{"take": 1, "seed": 1, "passed": True, "reasons": []}], "passed": True, "rejected": ["the beat restages segment 0"],
                     "continuity": {"by": "Cosmos3-Edge", "strictness": "standard", "checks": {"identity": {"status": "filtered", "evidence": "her face changes as she mounts", "second_look": {"confirmed": False, "why": "same face"}},
                                                                                             "props": {"status": "minor", "evidence": "core in the left hand"}},
                                    "findings": [{"category": "props", "status": "minor", "frame": 4, "action": "correct", "evidence": "core in the left hand"}]}}],
       "film": {"path": "autopilot/r1/film.mp4", "segments": 1, "seconds": 5.2, "at": "t"}, "log": [{"at": "2026-10-25T19:00:00", "text": "Paused."}],
       "bible": {"eras": [{"id": 1, "from": 1, "label": "Neon 2080s", "rules": ["hover-trams, chrome"]}], "subjects": {"vex": {"name": "Vex"}},
                 "facts": [{"subject": "vex", "key": "marks", "value": "thin scar, left eyebrow", "from": 1, "to": None}, {"subject": "vex", "key": "wardrobe", "value": "dry jacket", "from": 1, "to": 1}],
                 "story": [{"t": "00:02", "text": "vex jumps the gap"}], "corrections": [{"note": "props: keep the core in her right hand"}]},
       "film_check": {"state": "done", "identity": [{"subject": "Vex", "era": 1, "segments": [1, 2], "same": False, "different": [2], "why": "another face"}], "world": []},
       "memory": {"graph": {"nodes": [{"id": "vex", "type": "character", "label": "Vex", "state": "airborne"}], "events": [{"t": "00:02", "subject": "vex", "action": "jumps the gap"}], "chapters": []}}}


def test_a_first_visit_shows_the_seed_form_and_creates_a_run():
    got = run("""
      answers["/api/autopilot/runs"] = []; AR.runs = [];
      const view = viewAutoPilot(el("div", {}, "switch"));
      out.view = text(view);
      AR.form.name = "Storm Courier"; AR.form.seed.world = "A neon megacity"; AR.form.seed.characters[0].name = "Vex"; AR.form.seed.characters[0].look = "orange jacket";
      answers["/api/autopilot/runs"] = { id: "r9" };
      await arCreate(true);
      out.sent = calls.find(c => c[0] === "/api/autopilot/runs" && c[1]).at(1);
    """)
    assert "AutoPilot" in got["view"] and "Create and start" in got["view"] and "Vision brain" in got["view"]
    sent = got["sent"]
    assert sent["start"] is True and sent["seed"]["characters"] == [{"name": "Vex", "look": "orange jacket", "image": None}]
    assert sent["settings"]["preset"] == "sharp" and sent["settings"]["quality"] == "fh8" and sent["settings"]["seconds"] == 10  # the lessons of the first long run
    assert sent["settings"]["retakes"] == 2 and sent["settings"]["keyframes"] == "auto" and sent["settings"]["continuity"]["strictness"] == "standard"
    assert "setting" in sent["settings"]["continuity"]["categories"] and sent["seed"]["rules"] == []
    for said in ("Continuity strictness", "Checked against the continuity bible", "Each piece opens on", "Era and setting rules", "+ Style for an era or a setting"):
        assert said in got["view"], said


def test_a_run_shows_its_film_segments_watcher_notes_and_branch():
    got = run(f"""
      AR.runs = [{{ id: "r1", name: "Storm Courier", segments: 1 }}]; AR.id = "r1"; AR.run = {json.dumps(RUN)};
      AR.branch[1] = {{ open: true, text: "A rival steals the core" }};
      out.view = text(viewAutoPilot(el("div", {{}}, "switch")));
      answers["/api/autopilot/runs/r1/branch"] = {{ id: "r2" }}; answers["/api/autopilot/runs"] = [];
      await arDo("/branch", {{ after: 1, guidance: "A rival steals the core", start: true }});
      out.calls = calls.map(c => c[0]);
    """)
    view = got["view"]
    for said in ("Resume", "The film so far", "Vex launches off the tram platform", "Vex mid-air over the tram.", "jacket turns red",
                 "Cosmos3-Edge", "Branch and start", "Scene graph", "airborne", "the beat restages segment 0",
                 "Continuity bible", "Neon 2080s", "thin scar, left eyebrow", "keep the core in her right hand", "vex jumps the gap",
                 "identity · filtered by the second look", "the Director's next correction", "drift in segment 2", "Check the whole film", "runs on its own"):
        assert said in view, said
    assert "/api/autopilot/runs/r1/branch" in got["calls"]
