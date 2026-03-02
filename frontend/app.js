/* ── app.js: RAG Coding Agent Demo UI ──────────────────────────────────── */

const WS_URL = `ws://${location.host}/ws/chat`;

// DOM refs
const chatMessages  = document.getElementById('chat-messages');
const chatForm      = document.getElementById('chat-form');
const chatInput     = document.getElementById('chat-input');
const sendBtn       = document.getElementById('send-btn');
const fileSelector  = document.getElementById('file-selector');
const fileCode      = document.getElementById('file-code');
const fileContent   = document.getElementById('file-content');
const logEntries    = document.getElementById('log-entries');
const clearLogBtn   = document.getElementById('clear-log-btn');
const statusDot     = document.getElementById('status-dot');

let ws = null;
let currentAgentBubble = null;  // streaming text bubble
let agentRunning = false;

// ── WebSocket ──────────────────────────────────────────────────────────────
function connect() {
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    setStatus('connected');
    loadFileList();
  };

  ws.onclose = () => {
    setStatus('disconnected');
    setTimeout(connect, 2000);
  };

  ws.onerror = () => setStatus('disconnected');

  ws.onmessage = (ev) => {
    const event = JSON.parse(ev.data);
    handleEvent(event);
  };
}

function setStatus(state) {
  statusDot.className = 'dot ' + state;
}

// ── Event dispatcher ───────────────────────────────────────────────────────
function handleEvent(ev) {
  switch (ev.type) {
    case 'llm_token':
      appendToken(ev.token);
      break;
    case 'llm_done':
      finaliseAgentBubble(ev.content);
      setAgentRunning(false);
      break;
    case 'rag_query':
      startRagCard(ev.query);
      break;
    case 'rag_results':
      finishRagCard(ev.query, ev.chunks);
      break;
    case 'tool_call':
      addLogEntry('tool', `🔧 ${ev.tool}`, ev);
      break;
    case 'tool_result':
      addLogEntry('result', `✅ ${ev.tool} result`, ev);
      // If a file was written, refresh the editor
      if (ev.tool === 'write_file' && ev.result && ev.result.path) {
        loadFile(ev.result.path);
      }
      break;
    case 'error':
      appendError(ev.message);
      setAgentRunning(false);
      break;
  }
}

// ── Chat bubbles ───────────────────────────────────────────────────────────
function appendUserMessage(text) {
  const div = document.createElement('div');
  div.className = 'msg user';
  div.innerHTML = `<div class="msg-role">You</div><div class="msg-body">${escHtml(text)}</div>`;
  chatMessages.appendChild(div);
  scrollChat();
}

function getOrCreateAgentBubble() {
  if (!currentAgentBubble) {
    const div = document.createElement('div');
    div.className = 'msg agent';
    div.innerHTML = `<div class="msg-role">Agent</div><div class="msg-body"></div>`;
    chatMessages.appendChild(div);
    currentAgentBubble = div.querySelector('.msg-body');
  }
  return currentAgentBubble;
}

function appendToken(token) {
  const body = getOrCreateAgentBubble();
  body.textContent += token;
  scrollChat();
}

function finaliseAgentBubble() {
  currentAgentBubble = null;
}

function appendError(msg) {
  const div = document.createElement('div');
  div.className = 'msg agent';
  div.innerHTML = `<div class="msg-role">Error</div><div class="msg-body" style="color:var(--red)">${escHtml(msg)}</div>`;
  chatMessages.appendChild(div);
  scrollChat();
}

// ── RAG visualisation cards ────────────────────────────────────────────────
const pendingRagCards = {};  // query -> card element

function startRagCard(query) {
  addLogEntry('rag', `🔍 RAG Query`, { query });

  const card = document.createElement('div');
  card.className = 'rag-card';
  card.innerHTML = `
    <div class="rag-card-header">
      <span>🔮 RAG Retrieval</span>
      <span class="toggle">▼</span>
    </div>
    <div class="rag-card-body">
      <div class="rag-step">
        <div class="rag-step-num">1</div>
        <div class="rag-step-content">
          <div class="rag-step-label">Query</div>
          <div class="rag-step-value">${escHtml(query)}</div>
        </div>
      </div>
      <div class="rag-step">
        <div class="rag-step-num">2</div>
        <div class="rag-step-content">
          <div class="rag-step-label">Embedding → ChromaDB search</div>
          <div class="rag-step-value" style="color:var(--text-muted)">Searching…</div>
        </div>
      </div>
    </div>`;

  card.querySelector('.rag-card-header').addEventListener('click', () => {
    const header = card.querySelector('.rag-card-header');
    const body   = card.querySelector('.rag-card-body');
    header.classList.toggle('collapsed');
    body.style.display = header.classList.contains('collapsed') ? 'none' : '';
  });

  chatMessages.appendChild(card);
  pendingRagCards[query] = card;
  scrollChat();
}

function finishRagCard(query, chunks) {
  addLogEntry('rag', `📦 RAG Results (${chunks.length} chunks)`, { query, chunks });

  const card = pendingRagCards[query];
  if (!card) return;
  delete pendingRagCards[query];

  const body = card.querySelector('.rag-card-body');

  // Update step 2
  const steps = body.querySelectorAll('.rag-step');
  if (steps[1]) {
    steps[1].querySelector('.rag-step-value').innerHTML =
      `Found <strong>${chunks.length}</strong> relevant chunk(s)`;
  }

  // Step 3: injected context
  const step3 = document.createElement('div');
  step3.className = 'rag-step';
  step3.innerHTML = `
    <div class="rag-step-num">3</div>
    <div class="rag-step-content">
      <div class="rag-step-label">Retrieved Chunks (injected into prompt)</div>
      <div class="chunk-list">${chunks.map(chunkHtml).join('')}</div>
    </div>`;
  body.appendChild(step3);
  scrollChat();
}

function chunkHtml(c) {
  return `<div class="chunk-item">
    <div class="chunk-meta">
      <span>📄 ${escHtml(c.file)}</span>
      <span>lines ${c.start_line}–${c.end_line}</span>
      <span class="score">score ${c.score}</span>
    </div>
    <div class="chunk-text">${escHtml(c.text.slice(0, 300))}${c.text.length > 300 ? '…' : ''}</div>
  </div>`;
}

// ── Request log ────────────────────────────────────────────────────────────
function addLogEntry(tagType, label, data) {
  const entry = document.createElement('div');
  entry.className = 'log-entry';

  const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const bodyJson = JSON.stringify(data, null, 2);

  entry.innerHTML = `
    <div class="log-entry-header" style="background:${headerBg(tagType)}">
      <span class="tag ${tagType}">${tagType}</span>
      <span class="log-label">${escHtml(label)}</span>
      <span class="log-time">${now}</span>
    </div>
    <div class="log-entry-body hidden">${escHtml(bodyJson)}</div>`;

  entry.querySelector('.log-entry-header').addEventListener('click', () => {
    const body = entry.querySelector('.log-entry-body');
    body.classList.toggle('hidden');
  });

  logEntries.prepend(entry);  // newest on top
}

function headerBg(tagType) {
  return { rag: '#1a1228', tool: '#0f1f0f', result: '#0f1a0f', error: '#1f0f0f' }[tagType] || 'var(--surface2)';
}

clearLogBtn.addEventListener('click', () => { logEntries.innerHTML = ''; });

// ── File editor ────────────────────────────────────────────────────────────
async function loadFileList() {
  try {
    const res = await fetch('/files');
    const data = await res.json();
    fileSelector.innerHTML = '<option value="">— select file —</option>' +
      data.files.map(f => `<option value="${escAttr(f)}">${escHtml(f)}</option>`).join('');
    if (data.files.length > 0) loadFile(data.files[0]);
  } catch {}
}

async function loadFile(filename) {
  try {
    const res = await fetch(`/files/${encodeURIComponent(filename)}`);
    const data = await res.json();
    if (data.error) return;

    // Update selector
    for (const opt of fileSelector.options) {
      if (opt.value === filename) { opt.selected = true; break; }
    }

    fileCode.innerHTML = highlight(data.content);
    fileContent.classList.add('file-updated');
    fileContent.addEventListener('animationend', () => fileContent.classList.remove('file-updated'), { once: true });
  } catch {}
}

fileSelector.addEventListener('change', () => {
  if (fileSelector.value) loadFile(fileSelector.value);
});

// ── Minimal syntax highlighter ─────────────────────────────────────────────
function highlight(code) {
  const escaped = escHtml(code);
  return escaped
    // Comments
    .replace(/(#[^\n]*)/g, '<span class="cmt">$1</span>')
    // Strings (simple — triple-quoted first, then single/double)
    .replace(/("""[\s\S]*?"""|'''[\s\S]*?''')/g, '<span class="str">$1</span>')
    .replace(/("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/g, '<span class="str">$1</span>')
    // Decorators
    .replace(/(@\w+)/g, '<span class="dec">$1</span>')
    // Keywords
    .replace(/\b(def|class|return|if|elif|else|for|while|in|not|and|or|import|from|as|with|try|except|raise|pass|True|False|None|self|yield|async|await|lambda)\b/g, '<span class="kw">$1</span>')
    // Numbers
    .replace(/\b(\d+\.?\d*)\b/g, '<span class="num">$1</span>')
    // Function/class definitions
    .replace(/\b(def|class)\s+(\w+)/g, (m, kw, name) => `${m.slice(0,m.indexOf(name))}<span class="fn">${name}</span>`);
}

// ── Chat form ──────────────────────────────────────────────────────────────
chatForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const msg = chatInput.value.trim();
  if (!msg || agentRunning || !ws || ws.readyState !== WebSocket.OPEN) return;

  appendUserMessage(msg);
  chatInput.value = '';
  setAgentRunning(true);
  setStatus('thinking');
  ws.send(JSON.stringify({ message: msg }));
});

function setAgentRunning(running) {
  agentRunning = running;
  sendBtn.disabled = running;
  chatInput.disabled = running;
  if (!running) setStatus('connected');
}

// ── Utilities ──────────────────────────────────────────────────────────────
function scrollChat() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function escAttr(str) {
  return String(str).replace(/"/g, '&quot;');
}

// ── Boot ───────────────────────────────────────────────────────────────────
connect();
