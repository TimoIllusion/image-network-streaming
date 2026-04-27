const POLL_MS = 1000;
const PENDING_TTL_MS = 5000;

const backendSelect = document.getElementById("backend");
const switchAllBtn = document.getElementById("switchAll");
const clearAllBtn = document.getElementById("clearAll");
const copyMdBtn = document.getElementById("copyMd");
const downloadCsvBtn = document.getElementById("downloadCsv");
const switchStatus = document.getElementById("switchStatus");
const clientsDiv = document.getElementById("clients");
const clientCount = document.getElementById("clientCount");
const serverHost = document.getElementById("serverHost");
const batchEnabledEl = document.getElementById("batchEnabled");
const batchSizeEl = document.getElementById("batchSize");
const batchWaitEl = document.getElementById("batchWait");
const batchApplyBtn = document.getElementById("batchApply");
const batchStatus = document.getElementById("batchStatus");

serverHost.textContent = window.location.host;

let activeTransport = null;
let lastClientsPayload = null;
// Optimistic toggles: keep user-set values visible until the heartbeat catches up
// or PENDING_TTL_MS elapses. Keyed by `${clientName}:${action}`.
const pendingChanges = {};

function pendingKey(name, action) {
  return `${name}:${action}`;
}

function effectiveValue(name, action, serverValue) {
  const key = pendingKey(name, action);
  const p = pendingChanges[key];
  if (!p) return serverValue;
  if (Date.now() > p.expiresAt || p.value === serverValue) {
    delete pendingChanges[key];
    return serverValue;
  }
  return p.value;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[c]);
}

function ageLabel(age_s) {
  if (age_s < 1) return `${Math.round(age_s * 1000)}ms ago`;
  if (age_s < 60) return `${age_s.toFixed(1)}s ago`;
  return `${Math.round(age_s / 60)}m ago`;
}

// Track which fields the user has touched, so periodic refresh doesn't clobber in-progress edits.
const batchFieldDirty = { enabled: false, max_batch_size: false, max_wait_ms: false };

function markDirty(field) {
  batchFieldDirty[field] = true;
}

async function refreshBatching() {
  try {
    const r = await fetch("/batching");
    const s = await r.json();
    if (!batchFieldDirty.enabled) batchEnabledEl.checked = !!s.enabled;
    if (!batchFieldDirty.max_batch_size && document.activeElement !== batchSizeEl) {
      batchSizeEl.value = s.max_batch_size;
    }
    if (!batchFieldDirty.max_wait_ms && document.activeElement !== batchWaitEl) {
      batchWaitEl.value = s.max_wait_ms;
    }
  } catch {
    /* transient */
  }
}

async function applyBatching() {
  const body = {
    enabled: batchEnabledEl.checked,
    max_batch_size: parseInt(batchSizeEl.value, 10),
    max_wait_ms: parseFloat(batchWaitEl.value),
  };
  batchApplyBtn.disabled = true;
  batchStatus.textContent = "applying…";
  try {
    const r = await fetch("/batching", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const txt = await r.text();
      batchStatus.textContent = `failed: ${txt}`;
    } else {
      const s = await r.json();
      batchEnabledEl.checked = !!s.enabled;
      batchSizeEl.value = s.max_batch_size;
      batchWaitEl.value = s.max_wait_ms;
      batchFieldDirty.enabled = false;
      batchFieldDirty.max_batch_size = false;
      batchFieldDirty.max_wait_ms = false;
      batchStatus.textContent = `applied (${s.enabled ? "on" : "off"}, size ${s.max_batch_size}, wait ${s.max_wait_ms}ms)`;
      setTimeout(() => { batchStatus.textContent = ""; }, 3000);
    }
  } finally {
    batchApplyBtn.disabled = false;
  }
}

batchEnabledEl.addEventListener("change", () => markDirty("enabled"));
batchSizeEl.addEventListener("input", () => markDirty("max_batch_size"));
batchWaitEl.addEventListener("input", () => markDirty("max_wait_ms"));
batchApplyBtn.addEventListener("click", applyBatching);

async function refreshTransports() {
  try {
    const r = await fetch("/transports");
    const items = await r.json();
    const previous = backendSelect.value;
    backendSelect.innerHTML = "";
    activeTransport = null;
    for (const item of items) {
      const opt = document.createElement("option");
      opt.value = item.name;
      opt.textContent = item.active ? `${item.name} · active` : item.name;
      backendSelect.appendChild(opt);
      if (item.active) activeTransport = item.name;
    }
    backendSelect.value = previous || activeTransport || items[0]?.name || "";
  } catch {
    /* server might be reloading; the next tick will retry */
  }
}

function renderBenchTable(rows) {
  if (!rows || !rows.length) {
    return '<p class="hint small">No data yet — enable inference to start collecting. Switching transports preserves prior data.</p>';
  }
  const cols = Object.keys(rows[0]);
  const head = cols.map((c) => `<th>${escapeHtml(c)}</th>`).join("");
  const body = rows
    .map((r) => `<tr>${cols.map((c) => `<td>${escapeHtml(r[c])}</td>`).join("")}</tr>`)
    .join("");
  return `<table class="bench-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function renderClientCard(c) {
  const s = c.stats || {};
  const ageClass = c.age_s < 5 ? "age-fresh" : "age-stale";
  const mockChecked = effectiveValue(c.name, "mock_camera", !!s.mock_camera) ? "checked" : "";
  const inferChecked = effectiveValue(c.name, "inference", !!s.inference) ? "checked" : "";
  const benchRows = s.bench_rows || [];

  return `
    <div class="client-card">
      <div class="client-header">
        <div class="client-meta">
          <a class="client-link" href="${escapeHtml(c.ui_url)}" target="_blank" rel="noopener">${escapeHtml(c.name)}</a>
          <span class="${ageClass} small">${ageLabel(c.age_s || 0)}</span>
          <span class="muted small">backend: <strong>${escapeHtml(s.backend || "—")}</strong></span>
        </div>
        <div class="client-controls">
          <label class="control-pill">
            <input type="checkbox" data-action="mock_camera" data-name="${escapeHtml(c.name)}" ${mockChecked} />
            <span>mock cam</span>
          </label>
          <label class="control-pill">
            <input type="checkbox" data-action="inference" data-name="${escapeHtml(c.name)}" ${inferChecked} />
            <span>inference</span>
          </label>
          <button class="ghost-btn" data-clear="${escapeHtml(c.name)}">Clear</button>
        </div>
      </div>
      ${renderBenchTable(benchRows)}
    </div>`;
}

function renderClients(payload) {
  const rows = payload.clients || [];
  clientCount.textContent = String(rows.length);
  if (!rows.length) {
    clientsDiv.innerHTML = '<p class="hint">No clients connected. Start a client with <code>INFSB_CONTROL_HOST</code> pointing here.</p>';
    return;
  }
  clientsDiv.innerHTML = rows.map(renderClientCard).join("");
}

async function refreshClients() {
  try {
    const r = await fetch("/clients");
    const payload = await r.json();
    lastClientsPayload = payload;
    renderClients(payload);
  } catch {
    /* transient — try again next tick */
  }
}

function flattenForExport(payload) {
  const rows = [];
  for (const c of payload?.clients || []) {
    const benchRows = (c.stats || {}).bench_rows || [];
    for (const r of benchRows) {
      rows.push({ Client: c.name, ...r });
    }
  }
  return rows;
}

function toMarkdown(rows) {
  if (!rows.length) return "";
  const cols = Object.keys(rows[0]);
  const header = `| ${cols.join(" | ")} |`;
  const sep = `| ${cols.map(() => "---").join(" | ")} |`;
  const body = rows.map((r) => `| ${cols.map((c) => r[c] ?? "").join(" | ")} |`).join("\n");
  return [header, sep, body].join("\n");
}

function toCSV(rows) {
  if (!rows.length) return "";
  const cols = Object.keys(rows[0]);
  const esc = (v) => {
    const s = v == null ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const header = cols.join(",");
  const body = rows.map((r) => cols.map((c) => esc(r[c])).join(",")).join("\n");
  return `${header}\n${body}\n`;
}

async function copyTextToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const ta = Object.assign(document.createElement("textarea"), { value: text });
    Object.assign(ta.style, { position: "fixed", opacity: "0" });
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand("copy");
    ta.remove();
    return ok;
  }
}

function downloadFile(filename, content, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function flashButton(btn, ok, okText, failText) {
  const original = btn.textContent;
  btn.textContent = ok ? okText : failText;
  btn.classList.toggle("copied", ok);
  setTimeout(() => {
    btn.textContent = original;
    btn.classList.remove("copied");
  }, 2000);
}

async function postClientControl(name, body) {
  const r = await fetch(`/clients/${encodeURIComponent(name)}/control`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) console.warn(`control failed for ${name}:`, await r.text());
}

async function postClientClear(name) {
  const r = await fetch(`/clients/${encodeURIComponent(name)}/clear`, { method: "POST" });
  if (!r.ok) console.warn(`clear failed for ${name}:`, await r.text());
  refreshClients();
}

clientsDiv.addEventListener("change", (ev) => {
  const el = ev.target;
  if (!(el instanceof HTMLInputElement)) return;
  const action = el.dataset.action;
  const name = el.dataset.name;
  if (!action || !name) return;
  pendingChanges[pendingKey(name, action)] = { value: el.checked, expiresAt: Date.now() + PENDING_TTL_MS };
  const body = { [action]: el.checked };
  if (action === "inference" && el.checked && activeTransport) {
    body.backend = activeTransport;
  }
  postClientControl(name, body);
});

clientsDiv.addEventListener("click", (ev) => {
  const el = ev.target;
  if (!(el instanceof HTMLButtonElement)) return;
  const clearName = el.dataset.clear;
  if (clearName) postClientClear(clearName);
});

switchAllBtn.addEventListener("click", async () => {
  const name = backendSelect.value;
  if (!name) return;
  switchStatus.textContent = `switching to ${name}…`;
  switchAllBtn.disabled = true;
  try {
    const r = await fetch("/switch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, cascade: true }),
    });
    if (!r.ok) {
      switchStatus.textContent = `failed: ${await r.text()}`;
    } else {
      switchStatus.textContent = `switched to ${name}`;
      setTimeout(() => { switchStatus.textContent = ""; }, 3000);
    }
  } catch (e) {
    switchStatus.textContent = `error: ${e}`;
  } finally {
    switchAllBtn.disabled = false;
    refreshTransports();
  }
});

clearAllBtn.addEventListener("click", async () => {
  if (!confirm("Clear stats on all connected clients?")) return;
  const r = await fetch("/clients/clear-all", { method: "POST" });
  if (!r.ok) console.warn("clear-all failed:", await r.text());
  refreshClients();
});

copyMdBtn.addEventListener("click", async () => {
  const rows = flattenForExport(lastClientsPayload);
  if (!rows.length) {
    flashButton(copyMdBtn, false, "", "No data");
    return;
  }
  const ok = await copyTextToClipboard(toMarkdown(rows));
  flashButton(copyMdBtn, ok, "Copied!", "Copy failed");
});

downloadCsvBtn.addEventListener("click", () => {
  const rows = flattenForExport(lastClientsPayload);
  if (!rows.length) {
    flashButton(downloadCsvBtn, false, "", "No data");
    return;
  }
  const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  downloadFile(`inference-benchmark-${stamp}.csv`, toCSV(rows), "text/csv;charset=utf-8");
  flashButton(downloadCsvBtn, true, "Downloaded", "");
});

refreshTransports();
refreshClients();
refreshBatching();
setInterval(refreshTransports, POLL_MS * 3);
setInterval(refreshClients, POLL_MS);
setInterval(refreshBatching, POLL_MS * 3);
