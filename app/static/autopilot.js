"use strict";
/* AutoPilot, the third way to plan a film in Production (app/autopilot.py): a story seed becomes an open-ended
   film that watches each segment it renders, keeps a scene graph of what changed, and directs the next segment, until
   it is stopped. The run lives on the server (autopilot/<run>/run.json), so it carries on with the page closed; this
   panel starts it, follows it, pauses, resumes and branches it, and plays the film as it grows.
   The idea is Echo-Infinity's; the loop's design is adapted from Video Rebirth's HyperFlow-Infinity (Apache-2.0). */
const AR = { runs: null, id: null, run: null, creating: false, poll: null, vision: null, films: null, busy: false, branch: {}, form: null };
try { AR.id = localStorage.getItem("autopilot-run") || null; } catch {}
const AR_STATES = { idle: "ready", running: "running", paused: "paused", done: "done", failed: "failed" };
// the continuity bible's categories (app/continuity.py CATEGORIES); audio is held in the prompts, the others are watched
const AR_CATEGORIES = ["identity", "wardrobe", "props", "location", "setting", "lighting", "era", "spatial", "action", "audio", "story"];
const AR_VERDICT = { ok: "ok", not_visible: "not visible", minor: "minor", major: "major", filtered: "filtered by the second look", no_evidence: "no evidence", benign: "said nothing wrong" };
const arBlank = () => ({ name: "", from: "fresh", slate: "",
  seed: { inspiration: "", world: "", situation: "", characters: [{ name: "", look: "", image: "", segments: "" }], opening_image: "", rules: "" }, outline: "",
  // the defaults of the first long run's lessons: the Sharp Spec, fewer and longer pieces, a keyframe a piece with an outline
  settings: { style: "", styles: [], guidance: "", preset: S.presets.sharp ? "sharp" : S.presets.small ? "small" : Object.keys(S.presets)[0], quality: S.qualities.fh8 ? "fh8" : Object.keys(S.qualities)[0],
    seconds: S.durations && S.durations[10] ? 10 : 5, segments: 6, seed: 1, max_scene_segments: 8, qa: true, retakes: 2, preview_every: 3, keyframes: "auto", identity: "auto", ref_turbo: true, music: "", loras: [],
    continuity: { strictness: "standard", categories: AR_CATEGORIES.slice(), film_pass: true },
    brain: { backend: "auto", endpoint: "", model: "" } } });
async function arList() { try { const l = await api("/api/autopilot/runs"); AR.runs = Array.isArray(l) ? l : []; } catch (e) { AR.runs = []; toast(e.message); }
  if (!AR.id && AR.runs.length) AR.id = AR.runs[0].id; if (AR.id && !AR.runs.some(r => r.id === AR.id)) AR.id = AR.runs.length ? AR.runs[0].id : null;
  if (AR.id) await arLoad(AR.id); else render(); }
async function arLoad(id) { try { AR.run = await api("/api/autopilot/runs/" + id); AR.id = id; try { localStorage.setItem("autopilot-run", id); } catch {} } catch (e) { AR.run = null; toast(e.message); }
  render(); arWatch(); }
function arWatch() { clearTimeout(AR.poll); if (AR.run && AR.run.running) AR.poll = setTimeout(async () => { const was = AR.run && AR.run.rev; await arLoad(AR.id); if (AR.run && AR.run.rev !== was && typeof poll === "function") poll().catch(() => {}); }, 3000); }
async function arDo(path, body = {}, what = "") { AR.busy = true; render();
  try { const r = await api("/api/autopilot/runs/" + AR.id + path, body); if (what) toast(what); if (r && r.id && r.id !== AR.id) { AR.id = r.id; } } catch (e) { toast(e.message); }
  AR.busy = false; await arList(); }
async function arCreate(start) { const f = AR.form; AR.busy = true; render();
  // the outline: one beat a line, in order; "beat text | picture path" opens that beat on a picture of its own
  const outline = (f.outline || "").split("\n").map(l => l.trim()).filter(Boolean).map(l => { const [text, image] = l.split("|").map(x => x.trim()); return image ? { text, image } : { text }; });
  const rules = (f.seed.rules || "").split("\n").map(l => l.trim()).filter(Boolean);
  const seed = { ...f.seed, rules, characters: f.seed.characters.filter(c => (c.name || "").trim()).map(c => ({ name: c.name.trim(), look: (c.look || "").trim(), image: (c.image || "").trim() || null, ...((c.segments || "").trim() ? { segments: c.segments.trim() } : {}) })), ...(outline.length ? { outline } : {}) };
  try { const r = await api("/api/autopilot/runs", { name: f.name || seed.inspiration || "AutoPilot film", seed, settings: arOut(f.settings), start });
    AR.id = r.id; AR.creating = false; AR.form = null; toast(start ? "AutoPilot is at work" : "Created: press Start when ready"); }
  catch (e) { toast(e.message); }
  AR.busy = false; await arList(); }
async function arFromSlate(id) { if (!id) return; try { const r = await api("/api/autopilot/seed/slate/" + id); const f = AR.form;
  f.name = f.name || r.name; f.seed = { ...f.seed, ...r.seed, characters: r.seed.characters.length ? r.seed.characters.map(c => ({ name: c.name, look: c.look || "", image: c.image || "" })) : f.seed.characters };
  if (r.style) f.settings.style = r.style; toast("Seeded from the Slate film's bible"); } catch (e) { toast(e.message); } render(); }
async function arVision() { try { AR.vision = await api("/api/autopilot/vision"); } catch { AR.vision = { endpoint: "", model: "", frames: 7 }; } render(); }
function arVisionCard() { if (AR.vision === null) { AR.vision = false; arVision(); }
  const v = AR.vision || { endpoint: "", model: "", frames: 7 };
  return slFold("autopilot-vision", "Vision brain · " + (v.managed ? "NVIDIA Cosmos3-Edge (the Studio's own)" : v.endpoint ? v.endpoint : "the QA gate's vision path"), () => el("div", {},
    el("div", { class: "note" }, "The watcher looks at each segment: eight frames in time order and what is heard in it, and reports what changed, how the segment ends and any continuity problem. "
      + "“studio:cosmos” is NVIDIA Cosmos3-Edge on TensorFold, started by the Studio on its own port while it watches and unloaded before any render. Any OpenAI-compatible vision server on this computer or network works too; empty uses the QA gate's eyes (Claude Code, Codex or a local model) on one frame sheet."),
    v.why ? el("div", { class: "tip" }, v.why) : null,
    el("div", { class: "row" }, field("Vision server", bind(v, "endpoint", el("input", { placeholder: "studio:cosmos" }))), field("Model", bind(v, "model", el("input", { placeholder: "cosmos3" }))),
      el("div", { class: "fit", style: "width:110px" }, el("label", {}, "Frames"), bind(v, "frames", el("input", { type: "number", min: 3, max: 8 }))),
      el("button", { class: "fit", onclick: async () => { try { AR.vision = await api("/api/autopilot/vision", { endpoint: v.endpoint, model: v.model, frames: v.frames }); toast("Saved"); } catch (e) { toast(e.message); } render(); } }, "Save")))); }
function arCharacter(c, i, list) { return el("div", { class: "row" }, keep(field("Name", bind(c, "name", el("input", { placeholder: "Vex" })))),
  field("Look (face, age, hair, clothing: repeated in every segment)", bind(c, "look", el("input", { placeholder: "a wiry courier in her twenties, shaved undercut, chrome visor, orange reflective jacket" }))),
  keep(field("Reference picture (a path in the Library)", bind(c, "image", el("input", { placeholder: "scouts/…/scout_s1.png" })))),
  el("div", { class: "fit", style: "width:110px" }, el("label", {}, "Segments"), bind(c, "segments", el("input", { placeholder: "all", title: "Which segments this sheet is for: 1-7, 8- (the same person older is a second sheet)" }))),
  el("button", { class: "small fit", onclick: () => { list.splice(i, 1); render(); } }, "Remove")); }
function arOut(s) { return { ...s, loras: lorasOf(s), styles: (s.styles || []).filter(x => (x.style || "").trim()).map(x => ({ style: x.style.trim(), segments: (x.segments || "").trim() || (x.from ? x.from + "-" + (x.to || "") : "") })) }; }
function arStyles(s) { s.styles = s.styles || [];
  return el("div", {}, s.styles.map((x, i) => el("div", { class: "row" }, field("Style for some segments only (an era or a setting)", bind(x, "style", el("input", { placeholder: "1770s Boston: red-brick streets, lanterns, grey winter light" }))),
      el("div", { class: "fit", style: "width:110px" }, el("label", {}, "Segments"), bind(x, "segments", el("input", { placeholder: x.from ? x.from + "-" + (x.to || "") : "8-", title: "Which segments this style is for: 1-7, 8-" }))),
      el("button", { class: "small fit", onclick: () => { s.styles.splice(i, 1); render(); } }, "Remove"))),
    el("div", { class: "row" }, el("button", { class: "small fit", onclick: () => { s.styles.push({ style: "", segments: "" }); render(); } }, "+ Style for an era or a setting"), el("span"))); }
function arContinuitySettings(s) { const c = s.continuity || (s.continuity = { strictness: "standard", categories: AR_CATEGORIES.slice(), film_pass: true });
  return el("div", {}, el("div", { class: "row" }, field("Continuity strictness", select(c, "strictness", { lenient: "lenient: only a face or an era object renders again", standard: "standard: also the setting, the story and broken actions", strict: "strict: every confirmed major finding" })),
      el("div", { class: "fit" }, el("label", {}, "Film-level pass at the end"), el("button", { type: "button", class: "toggle" + (c.film_pass ? " on" : ""), "aria-pressed": c.film_pass ? "true" : "false", onclick: () => { c.film_pass = !c.film_pass; render(); } }, el("span", { class: "dotl" + (c.film_pass ? "" : " o") }), c.film_pass ? "On" : "Off"))),
    el("div", { class: "fit" }, el("label", {}, "Checked against the continuity bible"), el("div", { class: "seg-ctl" }, AR_CATEGORIES.map(k => el("button", { type: "button", class: c.categories.includes(k) ? "on" : "",
      onclick: () => { c.categories = c.categories.includes(k) ? c.categories.filter(x => x !== k) : AR_CATEGORIES.filter(x => x === k || c.categories.includes(x)); render(); } }, k))))); }
function arSettings(s, locked) { const num = (key, label, min, max, w = 120) => el("div", { class: "fit", style: "width:" + w + "px" }, el("label", {}, label), bind(s, key, el("input", { type: "number", min, max, disabled: locked })));
  return [field("Visual style (sent with every segment)", bind(s, "style", el("textarea", { style: "min-height:48px", placeholder: "Photoreal night-time cyberpunk, anamorphic lens flares, wet reflective streets, teal and magenta neon, film grain." }))),
    field("Director guidance (your creative direction, read before every segment)", bind(s, "guidance", el("textarea", { style: "min-height:48px", placeholder: "Keep the chase kinetic; vary the obstacles; let small kindnesses happen between strangers." }))),
    el("div", { class: "row" }, field("Output", select(s, "preset", Object.fromEntries(Object.entries(S.presets).map(([k, v]) => [k, v.label])))), field("Quality", select(s, "quality", S.qualities)),
      el("div", { class: "fit", style: "width:150px" }, el("label", {}, "Seconds a segment"), select(s, "seconds", Object.fromEntries(Object.keys(S.durations).map(k => [k, k + " s"]))))),
    el("div", { class: "row" }, num("segments", "Segments (0 = until stopped)", 0, 500, 190), num("seed", "Seed", 0, 2147483647), num("max_scene_segments", "Scene budget", 2, 50), num("preview_every", "Join every", 1, 50),
      el("div", { class: "fit" }, el("label", {}, "A segment that fails the check"), el("div", { class: "seg-ctl" }, [[0, "Keep it"], [1, "One retake"], [2, "Two retakes"], [3, "Three"], [4, "Four"]].map(([n, label]) => el("button", { type: "button", class: Number(s.retakes) === n ? "on" : "", onclick: () => { s.retakes = n; render(); } }, label))))),
    arStyles(s),
    el("div", { class: "row" }, field("Each piece opens on", select(s, "keyframes", { auto: "a keyframe of its own when the run has an outline, else the piece before", on: "a keyframe of its own (written by the Director, checked, drawn again if wrong)", off: "the last frame of the piece before" })), el("span")),
    el("div", { class: "row" }, field("A new scene keeps who is who with", select(s, "identity", { auto: "the cast's pictures where Ref2VA is installed, else words", references: "the cast's pictures (Ref2VA)", frame: "words only" })),
      field("Score under every segment (optional)", bind(s, "music", el("input", { placeholder: "a low pulsing synth bed, 100 bpm" }))),
      el("div", { class: "fit" }, el("label", {}, "QA gate"), el("button", { type: "button", class: "toggle" + (s.qa ? " on" : ""), "aria-pressed": s.qa ? "true" : "false", onclick: () => { s.qa = !s.qa; render(); } }, el("span", { class: "dotl" + (s.qa ? "" : " o") }), s.qa ? "On" : "Off"))),
    arContinuitySettings(s), loraHub(s, () => {})]; }
function arForm() { const f = AR.form || (AR.form = arBlank());
  if (AR.films === null) { AR.films = []; api("/api/slate/projects").then(l => { AR.films = Array.isArray(l) ? l : []; render(); }).catch(() => {}); }
  return el("div", { "data-clear": "autopilot-seed" }, el("h3", {}, "A new run"),
    el("div", { class: "row" }, keep(field("Name", bind(f, "name", el("input", { placeholder: "Storm Courier" })))),
      field("Seed from", select(f, "from", { fresh: "this form", slate: "a Slate film's bible" }, render)),
      f.from === "slate" ? field("Slate film", select(f, "slate", { "": "Choose a film", ...Object.fromEntries((AR.films || []).map(p => [p.id, p.name || p.id])) }, () => arFromSlate(f.slate))) : null),
    field("Inspiration (a starting point, not a plot to retell)", bind(f.seed, "inspiration", el("input", { placeholder: "a courier's last delivery before the storm" }))),
    field("World", bind(f.seed, "world", el("textarea", { style: "min-height:54px", placeholder: "A rain-soaked neon megacity of the 2080s, hover-trams over flooded streets, rooftop gardens and relay masts." }))),
    field("Where it starts", bind(f.seed, "situation", el("textarea", { style: "min-height:54px", placeholder: "A courier on a mono-wheel races a storm front to deliver a glowing data core across the city." }))),
    field("Era and setting rules (optional, one a line: what belongs and what would be out of place; the eyes check every piece against them)",
      bind(f.seed, "rules", el("textarea", { style: "min-height:48px", placeholder: "Norse seas, 1750s: wool, furs, iron, longships\nBoston, 1770s: muskets, tricorns, brick; nothing electric" }))),
    field("Outline (optional): one beat a line, in order, one segment each; “beat | picture path” opens that beat on a picture of its own. Empty: open-ended",
      bind(f, "outline", el("textarea", { style: "min-height:64px", placeholder: "The longship pitches in the storm; the boy grips the rail beside his parents\nA wave throws him overboard; his parents reach for him\n…" }))),
    el("h3", {}, "Characters"), f.seed.characters.map((c, i, list) => arCharacter(c, i, list)),
    el("div", { class: "row" }, el("button", { class: "small fit", onclick: () => { f.seed.characters.push({ name: "", look: "", image: "", segments: "" }); render(); } }, "+ Character"),
      keep(field("Opening picture (optional: segment 1 opens on it)", bind(f.seed, "opening_image", el("input", { placeholder: "scouts/…/scout_s3.png" }))))),
    el("h3", {}, "Settings"), arSettings(f.settings, false), arVisionCard(),
    el("div", { class: "row", style: "margin-top:10px" }, el("button", { class: "primary fit", disabled: AR.busy, onclick: () => arCreate(true) }, "Create and start"),
      el("button", { class: "fit", disabled: AR.busy, onclick: () => arCreate(false) }, "Create"), clearButton("autopilot-seed"),
      AR.runs && AR.runs.length ? el("button", { class: "fit", onclick: () => { AR.creating = false; render(); } }, "Cancel") : null, el("span"))); }
function arGraph(memory) { const g = (memory || {}).graph || {}, cast = (g.nodes || []).filter(n => (n.type || "character") === "character"), world = (g.nodes || []).filter(n => (n.type || "character") !== "character");
  return el("div", { class: "receipt" }, (g.chapters || []).map(c => el("div", {}, el("b", {}, "[" + c.t_start + "–" + c.t_end + "] " + c.title + ": "), c.summary + ((c.carried_forward || []).length ? " (still matters: " + c.carried_forward.join("; ") + ")" : ""))),
    cast.length ? el("div", {}, el("b", {}, "Cast: "), cast.map(n => n.label + " — " + n.state).join("; ")) : null,
    world.length ? el("div", {}, el("b", {}, "Places and things: "), world.map(n => n.label + (n.state ? " (" + n.state + ")" : "")).join(", ")) : null,
    (g.events || []).slice(-12).map(e => el("div", {}, (e.t || "") + "  " + e.subject + " " + e.action + (e.target ? " → " + e.target : ""))),
    !(g.nodes || []).length ? el("div", {}, "Empty until the first segment is watched.") : null); }
function arSegment(run, seg) { const w = seg.watch || {}, last = (seg.takes || []).at(-1) || {}, job = seg.job && jobs.find(j => j.id === seg.job), b = AR.branch[seg.index] || (AR.branch[seg.index] = { open: false, text: "" });
  const pill = seg.status === "done" ? (seg.passed === false ? el("span", { class: "pill failed" }, "kept, not passed") : el("span", { class: "pill done" }, "in the film")) : el("span", { class: "pill " + (["queued", "rendered"].includes(seg.status) ? "running" : seg.status === "failed" ? "failed" : "queued") }, seg.status);
  return el("div", { class: "seg" }, el("div", { class: "seghead" }, el("b", {}, seg.index + " · " + (seg.scene || "") + (seg.scene_change ? " (new scene)" : "")), pill, seg.shared ? el("span", { class: "pill" }, "shared") : null,
      el("span", { style: "flex:1" }), seg.status === "done" ? el("button", { class: "small", onclick: () => { b.open = !b.open; render(); } }, "Branch from here") : null),
    el("div", { class: "note" }, seg.beat || ""),
    run.seed && (run.seed.outline || [])[seg.index - 1] ? el("div", { class: "note" }, el("b", {}, "Outline beat " + seg.index + ": "), run.seed.outline[seg.index - 1].text) : null,
    b.open ? el("div", { class: "row" }, field("What should happen from here, in this branch", bind(b, "text", el("input", { placeholder: "A rival courier cuts in and snatches the core." }))),
      el("button", { class: "primary fit", disabled: AR.busy || !b.text.trim(), onclick: () => { b.open = false; arDo("/branch", { after: seg.index, guidance: b.text, start: true }, "Branched after segment " + seg.index); } }, "Branch and start")) : null,
    job && ["queued", "running"].includes(job.status) ? jobCard(job) : null,
    seg.output && ["done", "rendered"].includes(seg.status) ? el("video", { src: file(seg.output), controls: true, preload: "metadata", style: "width:100%;max-width:520px;border-radius:8px;margin-top:6px" }) : null,
    w.summary || w.ending ? el("div", { class: "note" }, el("b", {}, "Watcher" + (w.by ? " (" + w.by + ")" : "") + ": "), (w.summary || "") + (w.ending ? " Ends on: " + w.ending : "")) : null,
    (w.continuity || []).length ? el("div", { class: "tip" }, el("b", {}, "Continuity"), w.continuity.map(c => c.kind + (c.who ? " · " + c.who : "") + ": " + c.note).join(" · ")) : null,
    (seg.reasons || []).length ? el("div", { class: "tip" }, el("b", {}, "Did not pass"), seg.reasons.join(" · ")) : null,
    arCheck(seg.continuity || (last && last.continuity), "Continuity"),
    slFold("autopilot-seg-" + run.id + "-" + seg.index, "The Director's segment, the memory notes and the checks", () => el("div", { class: "receipt" },
      el("div", {}, el("b", {}, "Action: "), (seg.fields || {}).action || ""), el("div", {}, el("b", {}, "Setting: "), (seg.fields || {}).setting || ""), el("div", {}, el("b", {}, "Camera: "), (seg.fields || {}).camera || ""),
      ((seg.fields || {}).dialogue || []).map(d => el("div", {}, el("b", {}, (d.speaker || "Someone") + ": "), d.text)),
      (seg.rejected || []).length ? el("div", {}, el("b", {}, "Turned back first: "), seg.rejected.join(" · ")) : null,
      seg.mode ? el("div", {}, el("b", {}, "Opened: "), { previous: "on the last frame of the segment before", references: "on the cast's pictures", image: "on the opening picture", keyframe: "on its keyframe", text: "from text" }[seg.mode] + " · seed " + seg.seed) : null,
      seg.transcript ? el("div", {}, el("b", {}, "Heard: "), seg.transcript) : null,
      w.delta ? el("div", {}, el("b", {}, "Memory delta: "), JSON.stringify(w.delta)) : null,
      (seg.keyframes || []).map(k => el("div", {}, k.path ? el("img", { src: file(k.path), alt: "keyframe " + k.take, style: "width:160px;border-radius:6px;vertical-align:middle;margin-right:8px" }) : null,
        "Keyframe " + k.take + ((k.continuity || {}).findings || []).map(f => " · " + f.category + " " + f.status + ": " + f.evidence).join("") + (seg.keyframe && seg.keyframe === k.path ? " · opens the piece" : ""))),
      (seg.fixed || []).map(x => el("div", {}, el("b", {}, "Rewritten after take " + x.take + ": "), (x.findings || []).map(f => f.category + ": " + f.evidence).join(" · "))),
      (seg.takes || []).map(t => el("div", {}, "Take " + t.take + " at seed " + t.seed + ": " + (t.passed ? "passed" : "failed · " + (t.reasons || []).join("; ")) + (t.qa && t.qa.audio ? " · line check " + t.qa.audio.result : "") + (t.qa && t.qa.identity ? " · identity " + t.qa.identity.result : ""))),
      seg.prompt ? el("pre", { style: "white-space:pre-wrap;margin-top:6px" }, seg.prompt) : null))); }
function arCheck(c, title) { if (!c) return null;
  if (c.skipped) return el("div", { class: "note" }, el("b", {}, title + ": "), "not checked (" + c.skipped + ")");
  const checks = Object.entries(c.checks || {}), found = c.findings || [];
  return el("div", { class: "receipt" }, el("b", {}, title + (c.by ? " (" + c.by + ", " + (c.strictness || "standard") + ")" : "") + ": "),
    el("div", {}, checks.map(([k, v]) => el("span", { class: "pill " + ({ major: "failed", minor: "queued", ok: "done" }[v.status] || ""), title: v.evidence || "" }, k + " · " + (AR_VERDICT[v.status] || v.status)))),
    found.map(f => el("div", {}, el("b", {}, f.category + " · " + f.status + (f.frame ? " · frame " + f.frame : "") + " → " + ({ retake: "retake", correct: "the Director's next correction", note: "noted" }[f.action] || f.action) + ": "), f.evidence,
      f.second_look && f.second_look.why ? " (second look: " + (f.second_look.confirmed ? "confirmed" : "not seen") + ", " + f.second_look.why + ")" : "")),
    checks.filter(([, v]) => v.status === "filtered").map(([k, v]) => el("div", { class: "note" }, "Filtered: " + k + " · " + v.evidence + (v.second_look && v.second_look.why ? " (second look: " + v.second_look.why + ")" : "")))); }
function arBible(b, run) { if (!b) return el("div", { class: "receipt" }, "Empty until the run starts.");
  const now = (run.segments || []).filter(s => s.status === "done").length + 1, live = f => f.from <= now && (f.to == null || f.to >= now) && (f.to == null || f.to >= f.from);
  const facts = b.facts || [], subjects = b.subjects || {}, name = id => (subjects[id] || {}).name || id;
  const by = {}; facts.filter(live).forEach(f => (by[f.subject] = by[f.subject] || []).push(f));
  return el("div", { class: "receipt" },
    (b.eras || []).map(e => el("div", {}, el("b", {}, "Era " + e.id + (e.label ? " · " + e.label : "") + " (from segment " + e.from + "): "), (e.rules || []).join("; ") || e.world || "")),
    Object.entries(by).map(([id, list]) => el("div", {}, el("b", {}, (id === "_scene" ? "This scene" : name(id)) + ": "), list.map(f => f.key + ": " + f.value).join("; "))),
    (b.corrections || []).length ? el("div", {}, el("b", {}, "Corrections for the next segment: "), b.corrections.map(c => c.note).join(" · ")) : null,
    (b.story || []).length ? el("div", {}, el("b", {}, "What has happened (" + b.story.length + ", never folded): "), b.story.slice(-12).map(s => (s.t ? s.t + " " : "") + s.text).join("; ")) : null,
    el("div", { class: "note" }, facts.length + " facts in all, " + facts.filter(f => !live(f)).length + " of them closed by an era or a scene and kept as history.")); }
function arFilmCheck(run) { const r = run.film_check, done = (run.segments || []).filter(s => s.status === "done").length;
  return el("div", {}, r && r.state === "running" ? el("div", { class: "note" }, "The film is being checked…") : null,
    r && r.state === "failed" ? el("div", { class: "tip" }, "The film's check could not be made: " + (r.error || "")) : null,
    r && r.state === "done" ? el("div", { class: "receipt" }, (r.identity || []).map(x => el("div", {}, el("b", {}, x.subject + ", era " + x.era + ": "), x.same ? "the same person in segments " + x.segments.join(", ") : "drift in segment " + x.different.join(", ") + " (" + x.why + ")")),
      (r.world || []).map(x => el("div", {}, el("b", {}, x.category + " · segments " + x.segments.join(", ") + ": "), x.evidence)),
      !(r.identity || []).length && !(r.world || []).length ? el("div", {}, "Nothing to compare yet.") : null, (r.errors || []).map(e => el("div", { class: "note" }, e))) : null,
    el("div", { class: "row" }, el("button", { class: "small fit", disabled: AR.busy || run.running || done < 2, onclick: () => arDo("/film-check", {}, "Checking the whole film") }, "Check the whole film"),
      el("span", { class: "note" }, "A report only: nothing is rendered again."))); }
function arRun(run) { const done = (run.segments || []).filter(s => s.status === "done"), secs = done.reduce((a, s) => a + Number(s.seconds || 0), 0), limit = Number(run.settings.segments || 0);
  const film = run.film, live = run.running, s = run.settings;
  return el("div", {}, el("div", { class: "seg" }, el("div", { class: "seghead" }, el("b", {}, run.name), el("span", { class: "pill " + (live ? "running" : run.state === "done" ? "done" : run.state === "failed" ? "failed" : "queued") }, live ? "running" : AR_STATES[run.state] || run.state),
        run.branch ? el("span", { class: "pill" }, "branch of " + run.branch.from + " at " + run.branch.after) : null, el("span", { style: "flex:1" }),
        live ? [el("button", { class: "small", disabled: AR.busy || run.pause, onclick: () => arDo("/pause", {}, "Pausing after this segment") }, run.pause ? "Pausing…" : "Pause"),
          el("button", { class: "small", disabled: AR.busy, onclick: () => arDo("/stop", {}, "Stopping") }, "Stop")]
          : el("button", { class: "primary small", disabled: AR.busy || (limit && done.length >= limit), onclick: () => arDo("/start", {}, done.length ? "Resumed" : "Started") }, done.length ? "Resume" : "Start"),
        el("button", { class: "small", disabled: AR.busy || !done.length, onclick: () => arDo("/assemble", {}, "Joining the film") }, "Join the film now")),
      el("div", { class: "note" }, done.length + (limit ? " of " + limit : "") + " segments · " + secs.toFixed(1) + " s" + (live && run.current && run.current.text ? " · " + run.current.text : "")),
      run.state === "failed" ? el("div", { class: "tip" }, el("b", {}, "Stopped on a problem"), (run.error || "") + " Resume tries again from here.") : null,
      film ? el("div", {}, el("h3", {}, "The film so far · " + film.segments + " segments, " + Number(film.seconds).toFixed(1) + " s"),
        el("video", { src: file(film.path) + "?v=" + encodeURIComponent(film.at || ""), controls: true, preload: "metadata", style: "width:100%;max-width:860px;border-radius:8px" }),
        el("div", { class: "row", style: "margin-top:6px" }, el("a", { class: "fit", href: file(film.path), download: "" }, "Download MP4"),
          el("button", { class: "small fit", onclick: async () => { await addToTimeline(film.path); } }, "Add to the editor timeline"), el("span"))) : null),
    el("div", { class: "note" }, "AutoPilot runs on its own: the Director chooses each step and its camera, every piece is checked against the continuity bible and rendered again when it breaks it. Watch, pause or branch if you like; it never waits for you."),
    slFold("autopilot-graph-" + run.id, "Scene graph · what the film remembers", () => arGraph(run.memory)),
    slFold("autopilot-bible-" + run.id, "Continuity bible · what every piece is checked against", () => arBible(run.bible, run)),
    slFold("autopilot-filmcheck-" + run.id, "The whole film's continuity" + (run.film_check && run.film_check.state === "done" ? " · " + ((run.film_check.identity || []).filter(x => !x.same).length + (run.film_check.world || []).length) + " issues" : ""), () => arFilmCheck(run)),
    slFold("autopilot-settings-" + run.id, "Settings (they apply from the next segment)", () => el("div", {}, arSettings(s, false),
      el("div", { class: "row" }, el("button", { class: "fit", onclick: async () => { try { await api("/api/autopilot/runs/" + run.id, { settings: arOut(s) }, "PUT"); toast("Saved"); } catch (e) { toast(e.message); } arLoad(run.id); } }, "Save settings"), el("span")), arVisionCard())),
    el("h3", {}, "Segments"), (run.segments || []).slice().reverse().map(seg => arSegment(run, seg)),
    (run.log || []).length ? slFold("autopilot-log-" + run.id, "The log · " + run.log.length + " lines", () => el("div", { class: "receipt" }, run.log.slice().reverse().map(l => el("div", {}, (l.at || "").slice(11, 19) + "  " + l.text)))) : null); }
function viewAutoPilot(head) { const box = el("section", { class: "card" }, head, el("h2", {}, "AutoPilot"),
    el("div", { class: "note" }, "Give it a story seed and it keeps going: it renders a segment on this Studio's engines, watches it, remembers what changed in a scene graph, writes the next segment from that, and continues from the last frame, until you pause or stop it. Branch from any segment with new direction. The idea is Echo-Infinity's; the loop is adapted from Video Rebirth's HyperFlow-Infinity."));
  if (AR.runs === null) { AR.runs = []; arList(); }
  const list = AR.runs || [];
  const pick = { v: AR.id || "" };
  put(box, el("div", { class: "row" }, list.length ? field("Run", select(pick, "v", Object.fromEntries(list.map(r => [r.id, r.name + " · " + r.segments + " segments" + (r.running ? " · running" : "")])), () => { AR.creating = false; arLoad(pick.v); })) : null,
    el("button", { class: "fit", onclick: () => { AR.creating = true; render(); } }, "New run"), el("span")));
  if (AR.creating || !list.length) put(box, arForm());
  else if (AR.run && AR.run.id === AR.id) put(box, arRun(AR.run));
  return box; }
