"use strict";
/* The Director, in Slate's First AD tab (app/director.py): a logline becomes a written, cast, rendered, checked and cut
   film while you watch. The run lives on the server (director.json in the film's folder), so it carries on with the
   page closed; this panel starts it, follows it and stops it. While it runs the film file is the server's: the page
   reads it again whenever the run changes it. */
const DR = { st: {}, ask: {}, poll: null, songs: null, busy: false };
const DR_STAGES = [["writing", "Writing"], ["casting", "Casting"], ["rendering", "Rendering"], ["checking", "Checking"], ["cutting", "Cutting"]];
const DR_ORDER = ["writing", "casting", "rendering", "checking", "retaking", "cutting", "done"];
const drAsk = film => DR.ask[film.id] || (DR.ask[film.id] = { logline: film.logline || "", scenes: 3, shotsPerScene: 3, seconds: Number((film.defaults || {}).durationSec) || 8,
  quality: (film.tf || {}).quality || Object.keys(S.qualities || {})[0] || "", retakes: 1, cut: true, song: "" });
async function drLoad(id) { let s = null; try { s = await api("/api/slate/direct/" + id); } catch { return; }
  const was = DR.st[id]; DR.st[id] = s;
  if (was && (s.running || was.running) && s.filmUpdatedAt !== was.filmUpdatedAt && SL.p && SL.p.id === id) { try { clearTimeout(slTimer); SL.p = await api("/api/slate/projects/" + id); SL.plan = {}; } catch {} }
  if (s.running && typeof poll === "function") poll().catch(() => {});
  render(); drWatch(id); }
function drWatch(id) { clearTimeout(DR.poll); const s = DR.st[id]; if (s && s.running) DR.poll = setTimeout(() => drLoad(id), 2000); }
async function drStart(film) { const a = drAsk(film); if (!a.logline.trim() && !film.scenes.length) return toast("Give the film a logline first");
  DR.busy = true; render();
  try { clearTimeout(slTimer); await slPut(film);  // the film as it is now: the run reads it from the file
    DR.st[film.id] = await api("/api/slate/direct", { project: film.id, ...a, logline: a.logline.trim() }); toast("The Director is at work on " + film.name); }
  catch (e) { toast(e.message); }
  DR.busy = false; drLoad(film.id); }
async function drStop(film) { try { await api("/api/slate/direct/" + film.id + "/stop", {}); toast("Stopping: jobs still waiting are taken out of the queue"); } catch (e) { toast(e.message); } drLoad(film.id); }
// where a stage stands: done, the one at work, or still to come
const drPill = (s, key) => { const at = DR_ORDER.indexOf(s.state === "retaking" ? "rendering" : s.state), me = DR_ORDER.indexOf(key), skip = key === "cutting" && !(s.asked || {}).cut;
  const live = s.state === key || (key === "rendering" && s.state === "retaking");
  return el("span", { class: "pill " + (skip ? "" : live && s.running ? "running" : (s.done || []).includes(key) || s.state === "done" || (key === "checking" && (s.done || []).includes("rendering")) ? "done" : live ? "failed" : at > me ? "done" : "queued") }, DR_STAGES.find(x => x[0] === key)[1]); };
function drCounts(s) { const c = s.counts || {}, bits = [];
  if (c.cast) bits.push(c.castDone + " of " + c.cast + " pictures made");
  if (c.shots) bits.push(c.rendered + " of " + c.shots + " shots rendered", (c.passed || 0) + " passed the check");
  if (c.kept) bits.push(c.kept + " kept as rendered before");
  // what did not make it, by why: a refusal or a failed render made no clip; a failed take is in the Library
  for (const [k, words] of [["failedCheck", "failed the check"], ["refused", "refused by the queue"], ["broken", "failed to render"], ["stopped", "stopped by the workload limits"]])
    if (c[k]) bits.push(c[k] + " " + words);
  if (c.retakes) bits.push(c.retakes + " retake" + (c.retakes === 1 ? "" : "s"));
  return bits.join(", ") || (c.scenes ? c.scenes + " scenes written" : "Starting…"); }
const DR_WHY = { failed: "failed the check: its take is in the Library and the shot's takes, rated no-good; circle it to cut it in", refused: "refused by the queue: no clip was made",
  broken: "the render failed: no clip was made", stopped: "stopped twice by the workload limits (heat or memory): no clip was made" };
// the shots a finished run left out of the cut, each with its reason
function drLeftOut(s) { const out = (s.order || []).map(id => (s.shots || {})[id]).filter(r => r && r.settled && DR_WHY[r.outcome]);
  return out.length ? el("div", { class: "tip" }, el("b", {}, "Left out of the cut · " + out.length), out.map(r => el("div", {}, r.name + ": " + DR_WHY[r.outcome] + (r.why && r.outcome !== "failed" ? " (" + r.why.slice(0, 240) + ")" : "")))) : null; }
function drProgress(film, s) { const job = s.current && s.current.job && jobs.find(j => j.id === s.current.job), cut = s.cut && s.cut.files || {}, log = s.log || [];
  return el("div", { class: "seg" }, el("div", { class: "seghead" }, DR_STAGES.map(([k]) => drPill(s, k)), el("span", { style: "flex:1" }),
      s.running ? el("button", { class: "small", onclick: () => drStop(film) }, "Stop") : null),
    el("div", { class: "note" }, drCounts(s) + (s.running && s.current && s.current.text ? " · " + s.current.text : "")),
    s.state === "failed" ? el("div", { class: "tip" }, el("b", {}, "Stopped on a problem"), s.error || "no reason given") : null,
    s.state === "stopped" ? el("div", { class: "note" }, "Stopped. Direct it again to carry on: what is written, cast and rendered is kept.") : null,
    s.running && job && ["queued", "running"].includes(job.status) ? jobCard(job) : null,
    s.running ? null : drLeftOut(s),
    s.state === "done" ? el("div", { class: "row", style: "margin-top:8px" }, cut.mp4 ? el("a", { class: "fit", href: file(cut.mp4), target: "_blank" }, el("button", { class: "primary small", type: "button" }, "Watch the cut")) : null,
      cut.fcpxml ? el("a", { class: "fit", href: cut.fcpxml, download: "" }, "FCPXML") : null, cut.edl ? el("a", { class: "fit", href: cut.edl, download: "" }, "EDL") : null,
      cut.zip ? el("a", { class: "fit", href: cut.zip, download: "" }, "Package (ZIP)") : null,
      el("button", { class: "small fit", onclick: () => { if (tab !== "Production" && typeof slateMode === "function") return slateMode("slate", "shots"); SL.view = "shots"; render(); } }, "Review takes"), el("span")) : null,
    log.length ? slFold("director-log", "The log · " + log.length + " lines", () => el("div", { class: "receipt" }, log.slice().reverse().map(l => el("div", {}, (l.at || "").slice(11, 19) + "  " + l.text)))) : null); }
function directorPanel(film) { if (!(film.id in DR.st)) { DR.st[film.id] = null; drLoad(film.id); }
  if (DR.songs === null) { DR.songs = []; api("/api/music/takes").then(list => { DR.songs = Array.isArray(list) ? list : []; render(); }).catch(() => {}); }
  const s = DR.st[film.id], a = drAsk(film), running = !!(s && s.running);
  const num = (key, label, min, max) => el("div", { class: "fit", style: "width:110px" }, el("label", {}, label), bind(a, key, el("input", { type: "number", min, max, disabled: running }), () => { a[key] = Math.min(max, Math.max(min, Math.round(Number(a[key]) || min))); }));
  return el("div", {}, el("h3", {}, "Direct the whole film"),
    el("div", { class: "note" }, "From the logline the Director writes what the film still needs (cast, places, scenes, shots with their lines), makes a reference picture and a turnaround of each character and a plate of each place, renders every shot in order, checks each take and takes a failed one again, and cuts the takes that passed together. A take that fails stays in the Library, out of the cut unless you circle it. It keeps what is already there."),
    s && s.state !== "none" ? drProgress(film, s) : null,
    running ? null : [field("Logline", bind(a, "logline", el("input", { placeholder: "A night-shift nurse finds a letter that was never sent, and has until dawn to deliver it." }))),
      el("div", { class: "row" }, num("scenes", "Scenes", 1, 12), num("shotsPerScene", "Shots a scene", 1, 12), num("seconds", "Seconds a shot", 1, 15),
        el("div", { class: "fit", style: "width:230px" }, el("label", {}, "Quality"), select(a, "quality", S.qualities || {})),
        el("div", { class: "fit" }, el("label", {}, "A take that fails the check"), el("div", { class: "seg-ctl" }, [[0, "No retake"], [1, "One retake"], [2, "Two retakes"]].map(([n, label]) => el("button", { type: "button", class: a.retakes === n ? "on" : "", onclick: () => { a.retakes = n; render(); } }, label))))),
      el("div", { class: "row", style: "align-items:flex-end" }, el("div", { class: "fit" }, el("button", { type: "button", class: "toggle" + (a.cut ? " on" : ""), "aria-pressed": a.cut ? "true" : "false", onclick: () => { a.cut = !a.cut; render(); } }, el("span", { class: "dotl" + (a.cut ? "" : " o") }), "Cut it when done")),
        a.cut ? el("div", { style: "max-width:320px" }, el("label", {}, "Song under the cut"), select(a, "song", { "": "No song", ...Object.fromEntries((DR.songs || []).map(t => [t.id, t.title || t.id])) })) : null, el("span")),
      el("div", { class: "row", style: "margin-top:8px" }, el("button", { class: "primary fit", disabled: DR.busy || !!SL.busy, onclick: () => drStart(film) }, s && ["stopped", "failed"].includes(s.state) ? "Direct it again" : "Direct it"), el("span"))]); }
