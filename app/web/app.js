// GRACE web page. Talks to app/server.py: upload (with progress) -> job id -> poll until done -> draw the result.
// All text from the server or the user (file names) is set with textContent, never as HTML.
"use strict";

const $ = (id) => document.getElementById(id);
const ICONS = {
  warn: '<path d="M12 3l9 16H3z"/><path d="M12 10v4"/><path d="M12 17h.01"/>',
  question: '<circle cx="12" cy="12" r="9"/><path d="M9.5 9a2.5 2.5 0 1 1 3.5 2.3c-.7.3-1 .9-1 1.7"/><path d="M12 17h.01"/>',
  check: '<circle cx="12" cy="12" r="9"/><path d="M8 12.5l2.7 2.7L16 10"/>',
};
const STAGES = {
  waiting: "Waiting for its turn", starting: "Starting", frames: "Reading the video",
  faces: "Finding faces", scoring: "Looking for signs of editing", explaining: "Preparing the explanation",
};

let cfg = null;      // /api/config
let jobId = null;    // current job
let pollTimer = null;
let xhr = null;

function el(tag, attrs = {}, text) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
}

function show(which) {  // one of: none, status, message, result
  $("status").hidden = which !== "status";
  $("message").hidden = which !== "message";
  $("result").hidden = which !== "result";
}

function message(title, body) {
  $("message-title").textContent = title;
  $("message-body").textContent = body;
  show("message");
  $("message").scrollIntoView({ block: "nearest" });
}

function errorFrom(payload, fallbackTitle) {
  const d = payload && payload.detail;
  if (d && typeof d === "object" && d.title) return d;
  return { title: fallbackTitle || "Something went wrong", message: "Please try again in a moment." };
}

// ---------- setup ----------
async function init() {
  try {
    const r = await fetch("/api/config");
    cfg = await r.json();
  } catch (e) {
    message("We can't reach the server", "Check your internet connection and reload the page.");
    return;
  }
  $("file").accept = cfg.accept.join(",");
  $("drop-limit").textContent = `Up to ${cfg.max_mb} MB · JPG, PNG, WEBP, MP4, MOV, AVI, MKV, WEBM`;
  const box = $("modes");
  const desc = { Deep: "Our full model. Most accurate; takes a few seconds.", Quick: "A smaller model. Faster, a little less accurate." };
  let first = true;
  for (const m of cfg.modes) {
    const lab = el("label", { class: "g-mode" + (m.available ? "" : " is-off") });
    const inp = el("input", { type: "radio", name: "mode", value: m.id });
    if (!m.available) inp.disabled = true;
    else if (first) { inp.checked = true; first = false; }
    const txt = el("span");
    txt.append(el("b", {}, m.label + (m.id === "Deep" ? " (recommended)" : "")), el("span", { class: "g-small" }, m.available ? (desc[m.id] || "") : "Not available on this server."));
    lab.append(inp, txt);
    box.append(lab);
  }
  const sel = $("t-concept");
  cfg.concepts.forEach((c, i) => sel.append(el("option", { value: String(i) }, c)));
  const offs = $("t-off");
  cfg.concepts.forEach((c, i) => {
    const lab = el("label");
    lab.append(el("input", { type: "checkbox", value: String(i) }), document.createTextNode(c));
    offs.append(lab);
  });
}

function pickedFile() { return $("file").files[0] || null; }

function onFileChosen() {
  const f = pickedFile();
  $("drop").classList.toggle("has-file", !!f);
  $("go").disabled = !f;
  const t = $("drop-text");
  t.textContent = "";
  if (f) {
    t.append(el("b", {}, f.name), document.createTextNode(` · ${(f.size / 1e6).toFixed(1)} MB`));
    if (cfg && f.size > cfg.max_mb * 1e6) {
      message("This file is too large", `Please upload a file smaller than ${cfg.max_mb} MB, or a shorter clip.`);
      $("go").disabled = true;
    } else if (!$("message").hidden) show("none");
  } else {
    t.append(el("b", {}, "Choose a photo or video"), document.createTextNode(" or drop it here"));
  }
}

// ---------- upload + polling ----------
function status(title, sub, pct) {
  $("status-title").textContent = title;
  $("status-sub").textContent = sub || "";
  $("bar").style.width = `${Math.max(0, Math.min(100, pct))}%`;
}

function submit(ev) {
  ev.preventDefault();
  const f = pickedFile();
  if (!f) { message("Choose a file first", "Upload a photo or a short video that shows a face, then press Check."); return; }
  const mode = (document.querySelector('input[name="mode"]:checked') || {}).value || "Deep";
  const fd = new FormData();
  fd.append("file", f);
  fd.append("mode", mode);
  show("status");
  status("Uploading…", f.name, 0);
  $("go").disabled = true;
  xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/check");
  xhr.responseType = "json";
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) status(`Uploading… ${Math.round((e.loaded / e.total) * 100)}%`, f.name, (e.loaded / e.total) * 30);
  };
  xhr.onload = () => {
    const body = xhr.response || {};
    const ok = xhr.status === 202 && body.job_id;
    xhr = null;
    $("go").disabled = !pickedFile();
    if (ok) { jobId = body.job_id; poll(); return; }
    const err = errorFrom(body, "The upload didn't work");
    message(err.title, err.message);
  };
  xhr.onerror = () => { xhr = null; $("go").disabled = !pickedFile(); message("The upload didn't work", "Check your internet connection and try again."); };
  xhr.onabort = () => { xhr = null; $("go").disabled = !pickedFile(); show("none"); };
  xhr.send(fd);
}

function ordinal(n) { return n === 1 ? "next" : n === 2 ? "2nd" : n === 3 ? "3rd" : `${n}th`; }
function secs(s) { return s == null ? "" : s < 60 ? `about ${Math.max(1, Math.round(s / 5) * 5)} s` : `about ${Math.round(s / 60)} min`; }

async function poll() {
  clearTimeout(pollTimer);
  if (!jobId) return;
  let j;
  try {
    const r = await fetch(`/api/jobs/${jobId}`);
    j = await r.json();
    if (!r.ok) { const e = errorFrom(j); jobId = null; message(e.title, e.message); return; }
  } catch (e) {
    pollTimer = setTimeout(poll, 2000);  // brief network blip: keep trying
    return;
  }
  if (j.state === "queued") {
    status(j.position === 1 ? "You're next in line" : `You're ${ordinal(j.position)} in line`,
      `Other people are checking files. ${secs(j.eta_s)} to go.`, 32);
  } else if (j.state === "running") {
    const frac = j.total ? j.done / j.total : 0;
    const base = { starting: 32, frames: 35, faces: 40, scoring: 85, explaining: 93 }[j.stage] || 35;
    const pct = j.stage === "faces" ? 40 + frac * 45 : base;
    const count = j.stage === "faces" && j.total > 1 ? ` · ${j.done} of ${j.total}` : "";
    status(`${STAGES[j.stage] || "Working"}…${count}`, secs(j.eta_s) ? `${secs(j.eta_s)} to go.` : "", pct);
  } else if (j.state === "done") {
    status("Done", "", 100);
    render(j.result);
    return;
  } else if (j.state === "error") {
    jobId = null;
    message(j.error.title, j.error.message);
    return;
  } else {  // cancelled
    jobId = null;
    show("none");
    return;
  }
  pollTimer = setTimeout(poll, 1000);
}

async function cancel() {
  if (xhr) { xhr.abort(); return; }
  clearTimeout(pollTimer);
  if (jobId) {
    try { await fetch(`/api/jobs/${jobId}/cancel`, { method: "POST" }); } catch (e) { /* the server drops it anyway */ }
  }
  jobId = null;
  show("none");
}

function again() {
  if (jobId && !$("status").hidden) cancel();
  jobId = null;
  $("file").value = "";
  onFileChosen();
  show("none");
  $("tech").open = false;
  $("upload").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---------- result ----------
function svgIcon(name) {
  const ns = "http://www.w3.org/2000/svg";
  const s = document.createElementNS(ns, "svg");
  for (const [k, v] of Object.entries({ width: 18, height: 18, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor",
    "stroke-width": 2.2, "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true" })) s.setAttribute(k, v);
  s.innerHTML = ICONS[name] || ICONS.question;  // fixed strings from this file, not server text
  return s;
}

function render(r) {
  const v = r.verdict;
  $("r-media").src = v.media;
  $("r-media").alt = `Your ${v.kind}`;
  $("r-pill").textContent = v.pill;
  $("r-caption").textContent = `${v.file} · checked ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
  const badge = $("r-badge");
  badge.textContent = "";
  badge.style.background = v.tint;
  badge.style.color = v.colour;
  badge.append(svgIcon(v.icon), el("span", {}, v.label));
  $("r-headline").textContent = v.headline;
  $("r-expl").textContent = v.explanation;
  $("r-knob").style.left = `${v.knob}%`;
  $("r-meter").setAttribute("aria-label", `Position on the scale from looks real to likely fake: ${v.label}`);
  $("r-basis").textContent = v.basis;
  $("r-note").hidden = !v.note;
  $("r-note").textContent = v.note || "";

  $("r-clues-title").textContent = r.clues.title;
  const grid = $("r-clues");
  grid.textContent = "";
  for (const c of r.clues.cards) {
    const art = el("article", { class: "g-card g-reason" });
    art.append(el("img", { src: c.img, alt: "" }), el("span", { class: `g-tag g-tag-${c.level}` }, c.tag),
      el("h3", {}, c.title), el("p", {}, c.body));
    grid.append(art);
  }
  $("r-hint").hidden = !r.clues.hint;
  $("r-save").href = `/api/jobs/${jobId}/report`;

  const t = r.technical;
  $("t-summary").textContent = `${t.verdict} · calibrated fake probability ${t.prob.toFixed(2)} · ${t.model} · ${t.used}` +
    (t.flicker != null ? ` · concept flicker ${t.flicker.toFixed(3)}` : "") + (t.drift != null ? ` · identity drift ${t.drift.toFixed(3)}` : "");
  $("t-timeline-fig").hidden = !t.timeline;
  if (t.timeline) $("t-timeline").src = t.timeline;
  $("t-concepts").src = t.concepts;
  $("t-concept").value = "0";
  $("t-opacity").value = "0.45";
  gallery($("t-gallery"), t.gallery);
  $("t-evidence-wrap").hidden = !t.evidence.length;
  gallery($("t-evidence"), t.evidence);
  document.querySelectorAll("#t-off input").forEach((i) => { i.checked = false; });
  $("t-rescore-out").textContent = "";

  show("result");
  $("result").scrollIntoView({ behavior: "smooth", block: "start" });
}

function gallery(box, items) {
  box.textContent = "";
  for (const g of items) {
    const fig = el("figure");
    fig.append(el("img", { src: g.img, alt: g.caption }), el("figcaption", {}, g.caption));
    box.append(fig);
  }
}

let overlayTimer = null;
function overlayChanged() {
  clearTimeout(overlayTimer);
  overlayTimer = setTimeout(async () => {
    if (!jobId) return;
    const r = await fetch(`/api/jobs/${jobId}/overlay`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ concept: Number($("t-concept").value), opacity: Number($("t-opacity").value) }),
    });
    const j = await r.json();
    if (r.ok) gallery($("t-gallery"), j.gallery);
    else $("t-rescore-out").textContent = errorFrom(j).message;
  }, 200);
}

async function rescore() {
  if (!jobId) return;
  const off = [...document.querySelectorAll("#t-off input:checked")].map((i) => Number(i.value));
  $("t-rescore-out").textContent = "Re-scoring…";
  const r = await fetch(`/api/jobs/${jobId}/rescore`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ off }),
  });
  const j = await r.json();
  if (!r.ok) { $("t-rescore-out").textContent = errorFrom(j).message; return; }
  $("t-rescore-out").textContent = `With ${j.off.length ? j.off.join(", ") : "no concepts"} switched off: probability ` +
    `${j.prob.toFixed(2)} → ${j.verdict} (was ${j.was_prob.toFixed(2)}, ${j.was_verdict}).`;
}

// ---------- wiring ----------
document.addEventListener("DOMContentLoaded", () => {
  init();
  $("file").addEventListener("change", onFileChosen);
  const drop = $("drop");
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("is-over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("is-over"); }));
  drop.addEventListener("drop", (e) => {
    if (e.dataTransfer.files.length) { $("file").files = e.dataTransfer.files; onFileChosen(); }
  });
  $("upload").addEventListener("submit", submit);
  $("cancel").addEventListener("click", cancel);
  document.querySelectorAll(".js-again").forEach((b) => b.addEventListener("click", again));
  $("t-concept").addEventListener("change", overlayChanged);
  $("t-opacity").addEventListener("input", overlayChanged);
  $("t-rescore").addEventListener("click", rescore);
  window.addEventListener("beforeunload", () => { if (jobId && !$("status").hidden) navigator.sendBeacon(`/api/jobs/${jobId}/cancel`); });
});
