const POLL_MS = 1000;
const PENDING_TTL_MS = 5000;

const backendSelect = document.getElementById("backend");
const switchAllBtn = document.getElementById("switchAll");
const switchStatus = document.getElementById("switchStatus");
const clientsDiv = document.getElementById("clients");
const clientCount = document.getElementById("clientCount");
const serverHost = document.getElementById("serverHost");

serverHost.textContent = window.location.host;

let activeTransport = null;
// Optimistic toggles: keep user-set values visible until the heartbeat catches up
// or PENDING_TTL_MS elapses. Keyed by `${clientName}:${action}` (action is mock_camera | inference).
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
  } catch (e) {
    /* server might be reloading; the next tick will retry */
  }
}

function ageLabel(age_s) {
  if (age_s < 1) return `${Math.round(age_s * 1000)}ms ago`;
  if (age_s < 60) return `${age_s.toFixed(1)}s ago`;
  return `${Math.round(age_s / 60)}m ago`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[c]);
}

function renderClients(payload) {
  const rows = payload.clients || [];
  clientCount.textContent = String(rows.length);

  if (!rows.length) {
    clientsDiv.innerHTML = '<p class="hint">No clients connected. Start a client with <code>INFSB_CONTROL_HOST</code> pointing here.</p>';
    return;
  }

  const head = `
    <thead><tr>
      <th>Name</th>
      <th>Last seen</th>
      <th>Backend</th>
      <th>FPS</th>
      <th>total (ms)</th>
      <th>infer (ms)</th>
      <th>w/o infer (ms)</th>
      <th>Frames</th>
      <th>Mock cam</th>
      <th>Inference</th>
    </tr></thead>`;

  const body = rows.map((c) => {
    const s = c.stats || {};
    const ageClass = c.age_s < 5 ? "age-fresh" : "age-stale";
    const fmt = (v, d = 1) => (typeof v === "number" ? v.toFixed(d) : "—");
    const mockChecked = effectiveValue(c.name, "mock_camera", !!s.mock_camera) ? "checked" : "";
    const inferChecked = effectiveValue(c.name, "inference", !!s.inference) ? "checked" : "";
    const backend = s.backend || "—";
    return `
      <tr>
        <td><a class="client-link" href="${escapeHtml(c.ui_url)}" target="_blank" rel="noopener">${escapeHtml(c.name)}</a></td>
        <td class="${ageClass}">${ageLabel(c.age_s || 0)}</td>
        <td>${escapeHtml(backend)}</td>
        <td>${fmt(s.fps, 1)}</td>
        <td>${fmt(s.total_ms, 1)}</td>
        <td>${fmt(s.infer_ms, 1)}</td>
        <td>${fmt(s.transmission_ms, 1)}</td>
        <td>${s.frames ?? "—"}</td>
        <td>
          <label class="toggle">
            <input type="checkbox" data-action="mock_camera" data-name="${escapeHtml(c.name)}" ${mockChecked} />
            <span class="toggle-slider"></span>
          </label>
        </td>
        <td>
          <label class="toggle">
            <input type="checkbox" data-action="inference" data-name="${escapeHtml(c.name)}" ${inferChecked} />
            <span class="toggle-slider"></span>
          </label>
        </td>
      </tr>`;
  }).join("");

  clientsDiv.innerHTML = `<table>${head}<tbody>${body}</tbody></table>`;
}

async function refreshClients() {
  try {
    const r = await fetch("/clients");
    const payload = await r.json();
    renderClients(payload);
  } catch (e) {
    /* transient — try again next tick */
  }
}

async function postClientControl(name, body) {
  const r = await fetch(`/clients/${encodeURIComponent(name)}/control`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const txt = await r.text();
    console.warn(`control failed for ${name}:`, txt);
  }
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
    // Without a backend hint the client would have to query the server; pass it explicitly to skip a round-trip.
    body.backend = activeTransport;
  }
  postClientControl(name, body);
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
      const txt = await r.text();
      switchStatus.textContent = `failed: ${txt}`;
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

refreshTransports();
refreshClients();
setInterval(refreshTransports, POLL_MS * 3);
setInterval(refreshClients, POLL_MS);
