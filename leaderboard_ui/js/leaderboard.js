const CONFIG = {
  REPO_BASE:
    "https://raw.githubusercontent.com/hisandan/agentbeats-werewolves-leaderboard/main/",
  LEADERBOARD_URL: "indexes/leaderboard.json",
  GAMES_URL: "indexes/games.json",
  AGENTS_DIR: "indexes/agents/",
};

document.addEventListener("DOMContentLoaded", () => {
  initApp();
});

async function initApp() {
  const params = new URLSearchParams(window.location.search);
  const agentId = params.get("agentId");
  const view = params.get("view");
  const run = params.get("run");

  if (run) {
    renderIArena(run);
  } else if (agentId) {
    renderAgentDetail(agentId);
  } else if (view === "games") {
    renderGames();
  } else {
    renderLeaderboard();
  }
}

async function fetchData(endpoint) {
  try {
    const response = await fetch(CONFIG.REPO_BASE + endpoint);
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    return await response.json();
  } catch (e) {
    console.error("Could not fetch data: ", e);
    return null;
  }
}

async function renderLeaderboard() {
  const content = document.getElementById("content-area");
  const tpl = document.getElementById("tpl-leaderboard");
  content.innerHTML = "";
  content.appendChild(tpl.content.cloneNode(true));

  const data = await fetchData(CONFIG.LEADERBOARD_URL);
  if (!data) {
    content.innerHTML =
      '<p class="error">Error cargando el leaderboard. Por favor intente más tarde.</p>';
    return;
  }

  document.getElementById("total-agents").textContent =
    data.total_agents || data.rankings.length;

  // For total games, we might need games.json or just a placeholder if not in leaderboard.json
  const gamesData = await fetchData(CONFIG.GAMES_URL);
  document.getElementById("total-games").textContent = gamesData
    ? gamesData.total_games
    : "-";

  const tbody = document.querySelector("#leaderboard-table tbody");
  data.rankings.forEach((agent) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
            <td><span class="rank-badge">${agent.rank}</span></td>
            <td><a href="?agentId=${
              agent.agent_id
            }" class="btn-link">${agent.agent_id.substring(0, 8)}...</a></td>
            <td><span class="elo-val">${agent.general_elo.toFixed(
              1
            )}</span></td>
            <td style="color: var(--accent-purple)">${agent.werewolf_elo.toFixed(
              1
            )}</td>
            <td style="color: var(--accent-green)">${agent.villager_elo.toFixed(
              1
            )}</td>
            <td>${agent.games_played}</td>
            <td><span class="winrate-tag">${agent.win_rate.toFixed(
              1
            )}%</span></td>
        `;
    tbody.appendChild(tr);
  });
}

async function renderGames(highlightRun) {
  const content = document.getElementById("content-area");
  const tpl = document.getElementById("tpl-games");
  content.innerHTML = "";
  content.appendChild(tpl.content.cloneNode(true));

  document.getElementById("view-games-link").classList.add("active");
  document
    .querySelector("nav .nav-link:not(#view-games-link)")
    .classList.remove("active");

  const data = await fetchData(CONFIG.GAMES_URL);
  if (!data) {
    content.innerHTML =
      '<p class="error">Error cargando el historial de partidas.</p>';
    return;
  }

  const tbody = document.querySelector("#games-table tbody");
  data.games.forEach((game) => {
    const isTarget = highlightRun && game.filename.includes(highlightRun);
    const tr = document.createElement("tr");
    if (isTarget) tr.style.background = "rgba(0, 255, 136, 0.05)";

    const date = new Date(game.start_time).toLocaleDateString();
    tr.innerHTML = `
            <td title="${
              game.filename
            }">${date} <small>${game.filename.substring(0, 15)}...</small></td>
            <td><span class="winrate-tag" style="background: ${
              game.winner === "werewolves"
                ? "rgba(157, 0, 255, 0.1)"
                : "rgba(0, 255, 136, 0.1)"
            }; color: ${
      game.winner === "werewolves"
        ? "var(--accent-purple)"
        : "var(--accent-green)"
    }">
                ${game.winner.toUpperCase()}
            </span></td>
            <td>${game.participant_count} Agentes</td>
            <td>
              <a href="?run=${game.filename.replace(
                ".json",
                ""
              )}" class="btn-link">IArena Replay 🕹️</a>
              <a href="${
                game.traceability_url
              }" target="_blank" class="btn-link" style="margin-left: 10px; opacity: 0.6">Raw ↗</a>
            </td>
        `;
    tbody.appendChild(tr);
  });
}

async function renderAgentDetail(agentId) {
  const content = document.getElementById("content-area");
  const tpl = document.getElementById("tpl-agent-detail");
  content.innerHTML = "";
  content.appendChild(tpl.content.cloneNode(true));

  // Try to find the agent in the leaderboard index first
  const lbData = await fetchData(CONFIG.LEADERBOARD_URL);
  const agent = lbData.rankings.find((a) => a.agent_id === agentId);

  if (!agent) {
    content.innerHTML = '<p class="error">Agente no encontrado.</p>';
    return;
  }

  document.getElementById("det-agent-id").textContent = agentId;
  document.getElementById("det-elo-general").textContent =
    agent.general_elo.toFixed(1);
  document.getElementById("det-winrate").textContent =
    agent.win_rate.toFixed(1) + "%";
  document.getElementById("det-games").textContent = agent.games_played;

  document.getElementById("det-elo-wolf").textContent =
    agent.werewolf_elo.toFixed(1);
  document.getElementById(
    "det-stats-wolf"
  ).textContent = `${agent.wins_as_werewolf} victorias en ${agent.games_as_werewolf} partidas`;

  document.getElementById("det-elo-villager").textContent =
    agent.villager_elo.toFixed(1);
  document.getElementById(
    "det-stats-villager"
  ).textContent = `${agent.wins_as_villager} victorias en ${agent.games_as_villager} partidas`;
}

// --- IArena REPLAY ENGINE ---
let replayState = {
  events: [],
  currentIndex: 0,
  autoPlay: false,
  typingInterval: null,
  suspicionScores: {},
};

async function renderIArena(runId) {
  const content = document.getElementById("content-area");
  const tpl = document.getElementById("tpl-iarena");
  content.innerHTML = "";
  content.appendChild(tpl.content.cloneNode(true));

  document.getElementById("ia-run-id").textContent = runId;

  // Hide hero section to focus on forensic machine
  const hero = document.getElementById("hero");
  if (hero) hero.style.display = "none";

  const rawData = await fetchData(`results/${runId}.json`);
  if (!rawData) {
    content.innerHTML = '<div class="iarena-container"><h1 style="color: var(--blood-alert)">CORRUPTED ARCHIVE</h1><p>FALLO EN LA RECUPERACIÓN DE DATOS</p></div>';
    return;
  }

  // Parse events
  const game = rawData.results[0];
  const participants = rawData.participants;
  const events = parseEvents(game, participants);

  replayState.events = events;
  replayState.currentIndex = 0;
  replayState.suspicionScores = {};
  Object.keys(participants).forEach((p) => (replayState.suspicionScores[p] = 0));

  // Initialize UI
  initIarenaUI(participants);

  // Bind controls
  document.getElementById("ia-btn-next").onclick = () => stepReplay(1);
  document.getElementById("ia-btn-prev").onclick = () => stepReplay(-1);
  document.getElementById("ia-btn-auto").onclick = toggleAutoPlay;

  // Show first event
  updateReplayView();
}

function parseEvents(game, participants) {
  let events = [];
  let roles = {};

  // 1. Role Assignment
  const roleLog = game.game_log.find((l) => l.event === "role_assignment");
  if (roleLog) {
    roles = roleLog.roles;
    events.push({
      type: "SYSTEM",
      text: "ARCHIVO RECUPERADO. PROTOCOLO DE VIGILANCIA ACTIVADO. IDENTIDADES ENCRIPTRADAS.",
      meta: "ESTADO: INICIAL",
    });
  }

  // 2. Aggregate actions
  const actionLogs = game.action_log || [];
  actionLogs.forEach((log) => {
    events.push({
      type: log.action === "debate" ? "SPEAK" : "ACTION",
      actor: log.player,
      text: log.reasoning || `REGISTRO: ${log.player} ejecutó ${log.action} sobre ${log.decision || 'N/A'}.`,
      meta: `TICK: R${log.round} - ${log.phase?.toUpperCase()}`,
      decision: log.decision,
      phase: log.phase,
    });
  });

  // Fallback for logic
  if (events.length <= 1 && game.game_log) {
    game.game_log.forEach((log) => {
      if (log.event === "night_phase") {
        events.push({
          type: "SYSTEM",
          text: "ACTIVIDAD DETECTADA EN SOMBRAS. ELIMINACIÓN EN PROGRESO.",
          meta: `TICK: R${log.round} - NOCHE`,
        });
      } else if (log.event === "vote_exile") {
        events.push({
          type: "ACTION",
          text: `VOTACIÓN FINALIZADA. SUJETO ${log.exiled} HA SIDO EXPULSADO DEL PERÍMETRO.`,
          meta: `TICK: R${log.round} - DÍA`,
          actor: log.exiled,
          isElimination: true,
        });
      }
    });
  }

  events.push({
    type: "REVEAL",
    text: `RECONSTRUCCIÓN FINALIZADA. FACCION VICTORIOSA: ${game.winner?.toUpperCase()}. CARGANDO IDENTIDADES REALES...`,
    meta: "ESTADO: CIERRE",
    roles: roles,
  });

  return events;
}

function initIarenaUI(participants) {
  const map = document.getElementById("ia-map");
  const suspList = document.getElementById("ia-suspicion-list");
  if (!map || !suspList) return;

  map.innerHTML = "";
  suspList.innerHTML = "";

  const pKeys = participants ? Object.keys(participants) : [];
  pKeys.forEach((pName) => {
    // Stage Node
    const node = document.createElement("div");
    node.className = "ia-actor-node";
    node.id = `node-${pName}`;
    node.innerHTML = `<span>${pName}</span>`;
    map.appendChild(node);

    // Suspicion Bar
    const sItem = document.createElement("div");
    sItem.className = "ia-suspicion-item";
    sItem.innerHTML = `
      <div style="font-size: 0.5rem">${pName}</div>
      <div class="suspicion-bar-container">
        <div class="suspicion-bar" id="susp-bar-${pName}"></div>
      </div>
    `;
    suspList.appendChild(sItem);
  });
}

function stepReplay(delta) {
  if (!replayState.events.length) return;
  replayState.currentIndex = Math.max(
    0,
    Math.min(replayState.events.length - 1, replayState.currentIndex + delta)
  );
  updateReplayView();
}

function toggleAutoPlay() {
  replayState.autoPlay = !replayState.autoPlay;
  const btn = document.getElementById("ia-btn-auto");
  if (btn) btn.textContent = replayState.autoPlay ? "PAUSA" : "AUTO";
  if (replayState.autoPlay) autoPlayLoop();
}

async function autoPlayLoop() {
  while (replayState.autoPlay && replayState.currentIndex < replayState.events.length - 1) {
    stepReplay(1);
    await new Promise((r) => setTimeout(r, 3000));
  }
}

function updateReplayView() {
  if (replayState.typingInterval) clearInterval(replayState.typingInterval);

  const event = replayState.events[replayState.currentIndex];
  if (!event) return;

  const progress =
    replayState.events.length > 1
      ? (replayState.currentIndex / (replayState.events.length - 1)) * 100
      : 100;

  document.getElementById("ia-progress-fill").style.width = `${progress}%`;
  document.getElementById("ia-meta").textContent = event.meta;
  document.getElementById("ia-actor-name").textContent = event.actor || "SYSTEM";

  // Typing effect
  const dialogText = document.getElementById("ia-dialog-text");
  dialogText.textContent = "";
  let i = 0;
  replayState.typingInterval = setInterval(() => {
    if (i < event.text.length) {
      dialogText.textContent += event.text.charAt(i);
      i++;
    } else {
      clearInterval(replayState.typingInterval);
    }
  }, 20);

  // Update nodes and suspicion
  const container = document.querySelector(".iarena-container");
  if (container) {
    container.classList.remove("day", "night");
    if (event.phase) container.classList.add(event.phase);
  }

  document.querySelectorAll(".ia-actor-node").forEach((n) => n.classList.remove("active", "glitch"));
  
  if (event.actor) {
    const node = document.getElementById(`node-${event.actor}`);
    if (node) {
      node.classList.add("active");
      if (event.isElimination) {
        node.classList.add("eliminated", "glitch");
        if (container) {
          container.classList.add("glitch");
          setTimeout(() => container.classList.remove("glitch"), 500);
        }
      }
    }

    // Heuristic suspicion: if someone accuses or votes, increase score slightly
    if (event.type === "ACTION" || event.text.toLowerCase().includes("acus") || event.text.toLowerCase().includes("sospech")) {
      replayState.suspicionScores[event.actor] = Math.min(100, (replayState.suspicionScores[event.actor] || 0) + 15);
    }
  }

  // Update Suspicion Bars
  Object.keys(replayState.suspicionScores).forEach((p) => {
    const bar = document.getElementById(`susp-bar-${p}`);
    if (bar) {
      const score = replayState.suspicionScores[p];
      bar.style.width = `${score}%`;
      bar.classList.toggle("medium", score > 40);
      bar.classList.toggle("high", score > 75);
    }
  });

  // Final Reveal
  if (event.type === "REVEAL" && event.roles) {
    Object.keys(event.roles).forEach((pName) => {
      const role = event.roles[pName];
      const node = document.getElementById(`node-${pName}`);
      if (node) {
        node.style.borderColor = role === "werewolf" ? "var(--blood-alert)" : "var(--gb-phosphor)";
        node.style.color = role === "werewolf" ? "var(--blood-alert)" : "var(--gb-phosphor)";
        node.innerHTML = `<small style="font-size: 0.4rem">${role.toUpperCase()}</small><span>${pName}</span>`;
      }
    });
  }
}
