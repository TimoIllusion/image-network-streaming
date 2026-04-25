const STATUS_POLL_MS = 3000;
const STATS_POLL_MS = 1000;

const backendSelect = document.getElementById("backend");
const inferCheckbox = document.getElementById("infer");
const clearButton = document.getElementById("clear");
const statsDiv = document.getElementById("stats");
const statusDot = document.getElementById("statusDot");
const statusLabel = document.getElementById("statusLabel");

function updateStatusOverlay() {
  const backend = backendSelect.value;
  const live = inferCheckbox.checked && !!backend;
  statusDot.className = "status-dot" + (live ? " live" : "");
  statusLabel.textContent = backend
    ? `${backend} · ${live ? "detecting" : "idle"}`
    : "no backend selected";
}

async function postControl() {
  await fetch("/api/control", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      backend: backendSelect.value,
      infer: inferCheckbox.checked,
    }),
  });
  updateStatusOverlay();
}

async function refreshStatus() {
  const r = await fetch("/api/status");
  const statuses = await r.json();
  const previous = backendSelect.value;
  backendSelect.innerHTML = "";
  for (const [name, online] of Object.entries(statuses)) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = online ? name : `${name} · offline`;
    opt.disabled = !online;
    backendSelect.appendChild(opt);
  }
  // Preserve previous selection if it is still online; otherwise pick the first online one.
  if (previous && statuses[previous]) {
    backendSelect.value = previous;
  } else {
    const firstOnline = Object.entries(statuses).find(([, ok]) => ok);
    if (firstOnline) backendSelect.value = firstOnline[0];
  }
  updateStatusOverlay();
}

async function refreshStats() {
  const r = await fetch("/api/stats");
  const rows = await r.json();
  if (!rows.length) {
    statsDiv.innerHTML = '<p class="hint">No measurements yet — enable object detection to start collecting.</p>';
    return;
  }
  const cols = Object.keys(rows[0]);
  const thead = `<tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr>`;
  const tbody = rows
    .map((row) => `<tr>${cols.map((c) => `<td>${row[c]}</td>`).join("")}</tr>`)
    .join("");
  statsDiv.innerHTML = `<table><thead>${thead}</thead><tbody>${tbody}</tbody></table>`;
}

backendSelect.addEventListener("change", postControl);
inferCheckbox.addEventListener("change", postControl);
clearButton.addEventListener("click", async () => {
  await fetch("/api/clear", { method: "POST" });
  refreshStats();
});

refreshStatus();
refreshStats();
setInterval(refreshStatus, STATUS_POLL_MS);
setInterval(refreshStats, STATS_POLL_MS);
