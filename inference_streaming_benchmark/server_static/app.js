const POLL_MS = 1000;
const PENDING_TTL_MS = 5000;

const backendSelect = document.getElementById("backend");
const switchAllBtn = document.getElementById("switchAll");
const clearAllBtn = document.getElementById("clearAll");
const switchStatus = document.getElementById("switchStatus");
const clientsDiv = document.getElementById("clients");
const clientCount = document.getElementById("clientCount");
const serverHost = document.getElementById("serverHost");

serverHost.textContent = window.location.host;

let activeTransport = null;
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
    renderClients(payload);
  } catch {
    /* transient — try again next tick */
  }
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

refreshTransports();
refreshClients();
setInterval(refreshTransports, POLL_MS * 3);
setInterval(refreshClients, POLL_MS);
