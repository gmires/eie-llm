/* ═════════════════════════════════════════════════════════════════════════════
   EIE-LLM Web UI — Applicazione client-side
   ═════════════════════════════════════════════════════════════════════════════
   Funzionalità:
   • Chat con streaming SSE (fetch + ReadableStream)
   • Cronologia chat in localStorage (nuova, seleziona, elimina, importa/esporta)
   • Impostazioni sampling (temperatura, top-k, top-p, ecc.)
   • Tema chiaro/scuro
   • Visualizzazione attention heatmap su canvas
   • Markdown rendering (marked.js) + syntax highlighting (highlight.js)
   ═════════════════════════════════════════════════════════════════════════════ */

(function() {
'use strict';

/* ── Default settings ─────────────────────────────────────────────────────── */
const DEFAULTS = {
  temperature: 1.0,
  max_tokens: 200,
  top_k: 40,
  top_p: 0.9,
  repetition_penalty: 1.1,
  stream: true,
  greedy: false,
  chatMode: true,
  enableThinking: true,  // thinking mode Qwen3: on/off
};

/* ── State ────────────────────────────────────────────────────────────────── */
const state = {
  chats: [],
  currentChatId: null,
  settings: { ...DEFAULTS },
  abortCtrl: null,
  generating: false,
  attentionData: null,
  sidebarOpen: window.innerWidth > 768,
};

/* ── DOM refs ─────────────────────────────────────────────────────────────── */
const $ = id => document.getElementById(id);
const el = {
  sidebar: $('sidebar-left'),
  chatList: $('chat-list'),
  btnNewChat: $('btn-new-chat'),
  btnToggleSidebar: $('btn-toggle-sidebar'),
  btnTheme: $('btn-theme'),
  btnSettings: $('btn-settings'),
  btnAttention: $('btn-attention'),
  btnSend: $('btn-send'),
  btnStop: $('btn-stop'),
  msgInput: $('msg-input'),
  chatMessages: $('chat-messages'),
  modelStatus: $('model-status'),
  settingsPanel: $('settings-panel'),
  attentionPanel: $('attention-panel'),
  btnImport: $('btn-import'),
  btnExport: $('btn-export'),
  fileImport: $('file-import'),
  // Settings inputs
  sModel: $('setting-model'),
  sTemp: $('setting-temperature'),
  sMaxTok: $('setting-max-tokens'),
  sTopK: $('setting-top-k'),
  sTopP: $('setting-top-p'),
  sRep: $('setting-rep-penalty'),
  sStream: $('setting-stream'),
  sGreedy: $('setting-greedy'),
  sChatMode: $('setting-chat-mode'),
  sEnableThinking: $('setting-enable-thinking'),
  // Value labels
  vTemp: $('val-temperature'),
  vMaxTok: $('val-max-tokens'),
  vTopK: $('val-top-k'),
  vTopP: $('val-top-p'),
  vRep: $('val-rep-penalty'),
  // Attention
  attInput: $('attention-input'),
  btnAnalyze: $('btn-analyze'),
  btnAttClear: $('btn-attention-clear'),
  attLayer: $('attention-layer'),
  attHead: $('attention-head'),
  attAvg: $('attention-average'),
  attCanvas: $('attention-canvas'),
  attContainer: $('attention-canvas-container'),
  attControls: $('attention-controls'),
  attInfo: $('attention-info'),
};

/* ── Helpers ───────────────────────────────────────────────────────────────── */
function uuid() { return Date.now().toString(36) + Math.random().toString(36).slice(2); }
function nowISO() { return new Date().toISOString(); }
function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

/* ── Theme ─────────────────────────────────────────────────────────────────── */
function loadTheme() {
  const saved = localStorage.getItem('eie-theme');
  const isDark = saved ? saved === 'dark' : true;
  document.body.classList.toggle('dark-theme', isDark);
  document.body.classList.toggle('light-theme', !isDark);
  updateThemeIcon(isDark);
  updateHljsTheme(isDark);
}
function toggleTheme() {
  const isDark = document.body.classList.contains('dark-theme');
  document.body.classList.toggle('dark-theme', !isDark);
  document.body.classList.toggle('light-theme', isDark);
  localStorage.setItem('eie-theme', !isDark ? 'dark' : 'light');
  updateThemeIcon(!isDark);
  updateHljsTheme(!isDark);
}
function updateThemeIcon(isDark) { el.btnTheme.textContent = isDark ? '🌙' : '☀️'; }
function updateHljsTheme(isDark) {
  const link = document.getElementById('hljs-theme');
  if (!link) return;
  link.href = isDark
    ? 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css'
    : 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css';
}

/* ── Settings ──────────────────────────────────────────────────────────────── */
function loadSettings() {
  try {
    const raw = localStorage.getItem('eie-settings');
    if (raw) Object.assign(state.settings, JSON.parse(raw));
  } catch (e) { console.warn('loadSettings error', e); }
  applySettingsToUI();
}
function saveSettings() {
  localStorage.setItem('eie-settings', JSON.stringify(state.settings));
}
function applySettingsToUI() {
  const s = state.settings;
  el.sTemp.value = s.temperature; el.vTemp.textContent = s.temperature.toFixed(1);
  el.sMaxTok.value = s.max_tokens; el.vMaxTok.textContent = s.max_tokens;
  el.sTopK.value = s.top_k; el.vTopK.textContent = s.top_k;
  el.sTopP.value = s.top_p; el.vTopP.textContent = s.top_p.toFixed(2);
  el.sRep.value = s.repetition_penalty; el.vRep.textContent = s.repetition_penalty.toFixed(2);
  el.sStream.checked = s.stream;
  el.sGreedy.checked = s.greedy;
  el.sChatMode.checked = s.chatMode;
  el.sEnableThinking.checked = s.enableThinking;
}
function readSettingsFromUI() {
  state.settings.temperature = parseFloat(el.sTemp.value);
  state.settings.max_tokens = parseInt(el.sMaxTok.value, 10);
  state.settings.top_k = parseInt(el.sTopK.value, 10);
  state.settings.top_p = parseFloat(el.sTopP.value);
  state.settings.repetition_penalty = parseFloat(el.sRep.value);
  state.settings.stream = el.sStream.checked;
  state.settings.greedy = el.sGreedy.checked;
  state.settings.chatMode = el.sChatMode.checked;
  state.settings.enableThinking = el.sEnableThinking.checked;
  saveSettings();
}

/* ── Chat storage ──────────────────────────────────────────────────────────── */
function loadChats() {
  try {
    const raw = localStorage.getItem('eie-chats');
    if (raw) state.chats = JSON.parse(raw);
  } catch (e) { state.chats = []; }
  if (!Array.isArray(state.chats)) state.chats = [];
}
function saveChats() {
  localStorage.setItem('eie-chats', JSON.stringify(state.chats));
}
function createNewChat() {
  const chat = {
    id: uuid(),
    title: 'Nuova chat',
    messages: [],
    createdAt: Date.now(),
  };
  state.chats.unshift(chat);
  saveChats();
  renderChatList();
  selectChat(chat.id);
  return chat.id;
}
function selectChat(id) {
  state.currentChatId = id;
  renderChatList();
  renderMessages();
}
function deleteChat(id, ev) {
  if (ev) ev.stopPropagation();
  if (!confirm('Eliminare questa chat?')) return;
  state.chats = state.chats.filter(c => c.id !== id);
  saveChats();
  if (state.currentChatId === id) {
    state.currentChatId = state.chats[0]?.id || null;
  }
  renderChatList();
  renderMessages();
}
function updateChatTitleFromFirstMessage(chat) {
  const firstUser = chat.messages.find(m => m.role === 'user');
  if (firstUser) {
    let t = firstUser.content.replace(/\n/g, ' ').trim();
    if (t.length > 30) t = t.slice(0, 30) + '…';
    chat.title = t || 'Chat';
  }
}
function exportChats() {
  const blob = new Blob([JSON.stringify(state.chats, null, 2)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `eie-chats-${nowISO().slice(0,10)}.json`;
  a.click();
}
function importChats(file) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const arr = JSON.parse(reader.result);
      if (!Array.isArray(arr)) throw new Error('Formato non valido');
      state.chats = arr;
      saveChats();
      renderChatList();
      selectChat(state.chats[0]?.id || createNewChat());
    } catch (e) { alert('Errore importazione: ' + e.message); }
  };
  reader.readAsText(file);
}

/* ── UI rendering ──────────────────────────────────────────────────────────── */
function renderChatList() {
  el.chatList.innerHTML = '';
  state.chats.forEach(chat => {
    const div = document.createElement('div');
    div.className = 'chat-item' + (chat.id === state.currentChatId ? ' active' : '');
    div.innerHTML = `<span class="title">${escapeHtml(chat.title)}</span><button class="del" title="Elimina">×</button>`;
    div.onclick = () => selectChat(chat.id);
    div.querySelector('.del').onclick = (ev) => deleteChat(chat.id, ev);
    el.chatList.appendChild(div);
  });
}

function renderMessages() {
  el.chatMessages.innerHTML = '';
  const chat = state.chats.find(c => c.id === state.currentChatId);
  if (!chat) return;
  chat.messages.forEach((m, idx) => appendMessageBubble(m.role, m.content, false));
  scrollToBottom();
}

// Formatta il contenuto con eventuali tag <think>...</think>.
// Il thinking viene reso come blocco collassabile <details>.
function formatContentWithThinking(content) {
  const thinkStart = content.indexOf('<think>');
  if (thinkStart === -1) return content;

  const thinkEnd = content.indexOf('</think>', thinkStart);
  if (thinkEnd === -1) return content;

  const before = content.substring(0, thinkStart);
  const thinking = content.substring(thinkStart + 7, thinkEnd).trim();
  const after = content.substring(thinkEnd + 8).trim();

  let html = '';
  if (before.trim()) html += before;
  if (thinking && thinking.trim()) {
    html += '<details class="think-block" open>';
    html += '<summary>🧠 Pensiero</summary>';
    html += '<div class="think-content">' + thinking.replace(/\n/g, '<br>') + '</div>';
    html += '</details>';
  }
  if (after) html += '\n\n' + after;
  return html;
}

function appendMessageBubble(role, content, animate = true) {
  const div = document.createElement('div');
  div.className = 'message ' + role + (animate ? ' generating' : '');
  const avatar = role === 'user' ? 'U' : 'AI';
  const formatted = role === 'assistant' ? formatContentWithThinking(content) : content;
  const html = marked.parse(formatted, {breaks: true, gfm: true});
  div.innerHTML = `<div class="avatar">${avatar}</div><div class="bubble">${html}</div>`;
  el.chatMessages.appendChild(div);
  div.querySelectorAll('pre code').forEach(block => {
    if (window.hljs) hljs.highlightElement(block);
  });
  if (!animate) scrollToBottom();
  return div;
}

function updateLastBubble(content) {
  const chat = state.chats.find(c => c.id === state.currentChatId);
  if (!chat) return;
  const last = el.chatMessages.lastElementChild;
  if (!last) return;
  const bubble = last.querySelector('.bubble');
  const formatted = formatContentWithThinking(content);
  const html = marked.parse(formatted, {breaks: true, gfm: true});
  bubble.innerHTML = html;
  bubble.querySelectorAll('pre code').forEach(block => {
    if (window.hljs) hljs.highlightElement(block);
  });
  scrollToBottom();
}

function scrollToBottom() {
  el.chatMessages.scrollTop = el.chatMessages.scrollHeight;
}

function setGenerating(gen) {
  state.generating = gen;
  el.btnSend.style.display = gen ? 'none' : 'flex';
  el.btnStop.style.display = gen ? 'flex' : 'none';
  const last = el.chatMessages.lastElementChild;
  if (last) {
    if (gen) last.classList.add('generating');
    else last.classList.remove('generating');
  }
}

/* ── Health check ──────────────────────────────────────────────────────────── */
async function checkHealth() {
  try {
    const res = await fetch('/health', {cache: 'no-store'});
    if (res.ok) {
      const data = await res.json();
      el.modelStatus.textContent = '🟢 ' + (data.model || 'Connesso');
      el.modelStatus.classList.add('connected');
      const modelId = data.arch || 'llama';
      el.sModel.value = modelId;

      // Applica parametri consigliati dal server se l'utente
      // non li ha personalizzati (primo avvio o reset)
      if (data.recommended && !localStorage.getItem('eie-settings')) {
        const r = data.recommended;
        state.settings.temperature = r.temperature ?? 1.0;
        state.settings.top_k = r.top_k ?? 40;
        state.settings.top_p = r.top_p ?? 0.9;
        state.settings.repetition_penalty = r.repetition_penalty ?? 1.1;
        state.settings.enableThinking = r.enable_thinking ?? true;
        applySettingsToUI();
      }
    } else throw new Error('status ' + res.status);
  } catch (e) {
    el.modelStatus.textContent = '🔴 Disconnesso';
    el.modelStatus.classList.remove('connected');
  }
}

/* ── Send / Stream ─────────────────────────────────────────────────────────── */
async function sendMessage() {
  const text = el.msgInput.value.trim();
  if (!text || state.generating) return;
  el.msgInput.value = '';
  el.msgInput.style.height = 'auto';

  const chat = state.chats.find(c => c.id === state.currentChatId);
  if (!chat) return;

  // Aggiungi messaggio utente
  chat.messages.push({role: 'user', content: text});
  if (chat.messages.length === 1) updateChatTitleFromFirstMessage(chat);
  saveChats();
  renderChatList();
  appendMessageBubble('user', text, false);

  // Placeholder assistant
  const s = state.settings;
  const assistantMsg = {role: 'assistant', content: ''};
  chat.messages.push(assistantMsg);
  appendMessageBubble('assistant', '', true);
  setGenerating(true);

  const abortCtrl = new AbortController();
  state.abortCtrl = abortCtrl;

  try {
    if (s.stream) {
      console.log('[CLIENT] inizio streamResponse');
      await streamResponse(chat, assistantMsg, abortCtrl);
      console.log('[CLIENT] streamResponse terminata');
    } else {
      console.log('[CLIENT] inizio blockingResponse');
      await blockingResponse(chat, assistantMsg, abortCtrl);
      console.log('[CLIENT] blockingResponse terminata');
    }
  } catch (e) {
    console.error('[CLIENT] ERRORE:', e.name, e.message);
    if (e.name !== 'AbortError') {
      assistantMsg.content += '\n\n*[Errore: ' + e.message + ']*';
      updateLastBubble(assistantMsg.content);
    }
  } finally {
    setGenerating(false);
    state.abortCtrl = null;
    saveChats();
  }
}

async function blockingResponse(chat, assistantMsg, abortCtrl) {
  const s = state.settings;
  const endpoint = s.chatMode ? '/v1/chat/completions' : '/v1/completions';
  const body = s.chatMode
    ? JSON.stringify({
        messages: chat.messages.filter(m => m.role === 'user' || m.role === 'assistant').map(m => ({role: m.role, content: m.content})),
        max_tokens: s.max_tokens,
        temperature: s.greedy ? 0 : s.temperature,
        top_k: s.top_k,
        top_p: s.top_p,
        repetition_penalty: s.repetition_penalty,
        enable_thinking: s.enableThinking,
        stream: false,
      })
    : JSON.stringify({
        prompt: buildPromptFromMessages(chat.messages),
        max_tokens: s.max_tokens,
        temperature: s.greedy ? 0 : s.temperature,
        top_k: s.top_k,
        top_p: s.top_p,
        repetition_penalty: s.repetition_penalty,
        stream: false,
        chat: s.chatMode,
      });

  console.log('[CLIENT] fetch POST (blocking)', endpoint);
  const res = await fetch(endpoint, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body,
    signal: abortCtrl.signal,
  });
  console.log('[CLIENT] blocking risposta:', res.status);
  if (!res.ok) throw new Error('HTTP ' + res.status);
  const data = await res.json();
  console.log('[CLIENT] blocking JSON:', data);
  const text = data.choices?.[0]?.text ?? data.choices?.[0]?.message?.content ?? '';
  assistantMsg.content = text;
  updateLastBubble(text);
}

async function streamResponse(chat, assistantMsg, abortCtrl) {
  const s = state.settings;
  const endpoint = s.chatMode ? '/v1/chat/completions' : '/v1/completions';
  const body = s.chatMode
    ? JSON.stringify({
        messages: chat.messages.filter(m => m.role === 'user' || m.role === 'assistant').map(m => ({role: m.role, content: m.content})),
        max_tokens: s.max_tokens,
        temperature: s.greedy ? 0 : s.temperature,
        top_k: s.top_k,
        top_p: s.top_p,
        repetition_penalty: s.repetition_penalty,
        enable_thinking: s.enableThinking,
        stream: true,
      })
    : JSON.stringify({
        prompt: buildPromptFromMessages(chat.messages),
        max_tokens: s.max_tokens,
        temperature: s.greedy ? 0 : s.temperature,
        top_k: s.top_k,
        top_p: s.top_p,
        repetition_penalty: s.repetition_penalty,
        stream: true,
        chat: s.chatMode,
      });

  console.log('[CLIENT] fetch POST', endpoint);
  const res = await fetch(endpoint, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body,
    signal: abortCtrl.signal,
  });
  console.log('[CLIENT] fetch risposta:', res.status, res.statusText);
  console.log('[CLIENT] headers:', [...res.headers.entries()]);
  if (!res.ok) throw new Error('HTTP ' + res.status);

  const reader = res.body.getReader();
  console.log('[CLIENT] reader ottenuto');
  const decoder = new TextDecoder();
  let buffer = '';
  let chunkCount = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) { console.log('[CLIENT] reader done'); break; }
    if (abortCtrl.signal.aborted) { console.log('[CLIENT] abort'); break; }

    chunkCount++;
    const text = decoder.decode(value, {stream: true});
    console.log('[CLIENT] chunk #' + chunkCount, 'bytes=', value?.length, 'text=', text.substring(0, 80));
    buffer += text;
    const parts = buffer.split('\n\n');
    buffer = parts.pop();

    for (const part of parts) {
      const lines = part.split('\n');
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6).trim();
        if (data === '[DONE]') { console.log('[CLIENT] [DONE]'); continue; }
        try {
          const json = JSON.parse(data);
          const delta = json.choices?.[0]?.delta?.content ?? json.choices?.[0]?.text ?? '';
          if (delta) {
            assistantMsg.content += delta;
            updateLastBubble(assistantMsg.content);
          }
        } catch (e) { console.warn('[CLIENT] JSON parse error:', e.message, 'data=', data); }
      }
    }
  }
  console.log('[CLIENT] stream finito, chunk totali:', chunkCount);
}

function buildPromptFromMessages(messages) {
  // Per /v1/completions in modalità non-chat, concateniamo i messaggi
  return messages.map(m => (m.role === 'user' ? 'User: ' : 'Assistant: ') + m.content).join('\n') + '\nAssistant: ';
}

function stopGeneration() {
  if (state.abortCtrl) {
    state.abortCtrl.abort();
    state.abortCtrl = null;
  }
  setGenerating(false);
}

/* ── Attention heatmap — ristrutturata ────────────────────────────────────── */
//
//  Miglioramenti rispetto alla versione precedente:
//  • ImageData pixel buffer per rendering ~100x più veloce
//  • Colormap uniforme (blu → ciano → giallo → rosso) invece di HSL singolo
//  • Zoom con rotellina mouse + pan con trascinamento
//  • Etichette token troncate senza sovrapposizioni
//  • Media tra layer/head calcolata in batch e cache
// ─────────────────────────────────────────────────────────────────────────────

// Colormap a 256 colori: blue → cyan → yellow → red
function buildHeatmapColormap() {
  const map = new Uint8Array(256 * 4); // RGBA
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let r, g, b;
    if (t < 0.25) {
      const u = t / 0.25;
      r = 0; g = u * 255; b = 255;
    } else if (t < 0.5) {
      const u = (t - 0.25) / 0.25;
      r = 0; g = 255; b = (1 - u) * 255;
    } else if (t < 0.75) {
      const u = (t - 0.5) / 0.25;
      r = u * 255; g = 255; b = 0;
    } else {
      const u = (t - 0.75) / 0.25;
      r = 255; g = (1 - u) * 255; b = 0;
    }
    const idx = i * 4;
    map[idx] = r; map[idx+1] = g; map[idx+2] = b; map[idx+3] = 255;
  }
  return map;
}
const COLORMAP = buildHeatmapColormap();

// Tronca etichetta a maxLen caratteri
function truncLabel(s, maxLen) {
  if (!s) return '';
  if (s.length <= maxLen) return s;
  return s.slice(0, maxLen - 1) + '…';
}

async function analyzeAttention() {
  const text = el.attInput.value.trim();
  if (!text) return;
  el.btnAnalyze.textContent = 'Analisi…';
  el.btnAnalyze.disabled = true;
  try {
    const res = await fetch('/v1/inspect/attention', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt: text, max_len: 100}),
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    state.attentionData = await res.json();
    state._cachedAvg = null; // invalida cache media
    populateAttentionControls();
    drawAttention();
  } catch (e) {
    alert('Errore: ' + e.message);
  } finally {
    el.btnAnalyze.textContent = 'Analizza';
    el.btnAnalyze.disabled = false;
  }
}

function populateAttentionControls() {
  const d = state.attentionData;
  if (!d || !d.layers) return;
  el.attControls.style.display = 'flex';
  el.attControls.style.flexWrap = 'wrap';
  el.attControls.style.gap = '8px';

  el.attLayer.innerHTML = '<option value="-1">Media tutti</option>';
  d.layers.forEach((L, i) => {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = 'Layer ' + L.layer;
    el.attLayer.appendChild(opt);
  });
  if (d.layers.length) {
    const nHeads = d.layers[0].heads.length;
    el.attHead.innerHTML = '<option value="-1">Media tutti</option>';
    for (let h = 0; h < nHeads; h++) {
      const opt = document.createElement('option');
      opt.value = h;
      opt.textContent = 'Head ' + h;
      el.attHead.appendChild(opt);
    }
  }
  el.attLayer.onchange = drawAttention;
  el.attHead.onchange = drawAttention;
}

// Calcola la matrice da visualizzare con caching della media
function computeAttentionMatrix() {
  const d = state.attentionData;
  if (!d || !d.tokens) return null;
  const seqLen = d.tokens.length;
  const li = parseInt(el.attLayer.value, 10);
  const hi = parseInt(el.attHead.value, 10);

  // Layer e head specifici
  if (li >= 0 && hi >= 0 && li < d.layers.length && hi < d.layers[li].heads.length) {
    return d.layers[li].heads[hi].weights;
  }

  // Media su layer/head — calcolata una volta e cache
  if (li < 0 && hi < 0) {
    if (state._cachedAvg && state._cachedAvg.length === seqLen) return state._cachedAvg;
  }
  if (state._cachedAvg && state._cachedAvg.length === seqLen) return state._cachedAvg;

  const nL = d.layers.length;
  const nH = d.layers[0]?.heads.length || 1;
  const layerSet = li >= 0 ? [li] : [...Array(nL).keys()];
  const headSet  = hi >= 0 ? [hi] : [...Array(nH).keys()];

  // Pre-allocated array di float64 per performance
  const matrix = Array.from({length: seqLen}, () => new Float64Array(seqLen));
  for (let q = 0; q < seqLen; q++)
    for (let k = 0; k < seqLen; k++) {
      let sum = 0;
      for (const l of layerSet)
        for (const h of headSet)
          sum += d.layers[l].heads[h].weights[q][k];
      matrix[q][k] = sum / (layerSet.length * headSet.length);
    }
  state._cachedAvg = matrix;
  return matrix;
}

// Stato zoom/pan
let _attZoom = {scale:1, offsetX:0, offsetY:0, seqLen:0};
function resetAttZoom(seqLen) {
  _attZoom = {scale:1, offsetX:0, offsetY:0, seqLen};
}

function drawAttention() {
  const d = state.attentionData;
  if (!d || !d.tokens) return;
  const tokens = d.tokens;
  const seqLen = tokens.length;
  const canvas = el.attCanvas;
  const ctx = canvas.getContext('2d');
  const isDark = document.body.classList.contains('dark-theme');

  const matrix = computeAttentionMatrix();
  if (!matrix) return;

  const margin = {top: 50, right: 10, bottom: 10, left: 80};
  const zoom = _attZoom.seqLen !== seqLen ? 1 : _attZoom.scale;
  if (_attZoom.seqLen !== seqLen) resetAttZoom(seqLen);

  const baseCell = Math.max(4, Math.min(48, Math.floor(600 / seqLen)));
  const cellSize = Math.max(4, Math.floor(baseCell * zoom));
  const ox = _attZoom.offsetX, oy = _attZoom.offsetY;

  const w = margin.left + seqLen * cellSize + margin.right;
  const h = margin.top + seqLen * cellSize + margin.bottom;
  canvas.width = Math.max(w + ox, 200);
  canvas.height = Math.max(h + oy, 200);

  // Sfondo
  ctx.fillStyle = isDark ? '#1a1a2e' : '#f8f9fa';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Disegna la matrice con ImageData (~100x più veloce di fillRect)
  const pixelW = seqLen * cellSize;
  const pixelH = seqLen * cellSize;
  const imageData = ctx.createImageData(pixelW, pixelH);
  const px = imageData.data;

  for (let q = 0; q < seqLen; q++) {
    for (let k = 0; k < seqLen; k++) {
      const val = matrix[q][k];
      const ci = Math.min(255, Math.max(0, Math.round(val * 255)));
      const r = COLORMAP[ci * 4]; const g = COLORMAP[ci * 4 + 1]; const b = COLORMAP[ci * 4 + 2];

      // Per ogni pixel nella cella (q, k)
      const baseX = k * cellSize;
      const baseY = q * cellSize;
      for (let dy = 0; dy < cellSize; dy++) {
        for (let dx = 0; dx < cellSize; dx++) {
          const pi = ((baseY + dy) * pixelW + (baseX + dx)) * 4;
          px[pi] = r; px[pi+1] = g; px[pi+2] = b; px[pi+3] = 255;
        }
      }

      // Bordo cella sottile
      if (cellSize >= 8) {
        const bx = baseX, by = baseY;
        for (let dy = 0; dy < cellSize; dy++) {
          const pi = ((by + dy) * pixelW + bx) * 4;
          if (pi + 4 < px.length) { px[pi] = 0; px[pi+1] = 0; px[pi+2] = 0; px[pi+3] = 40; }
        }
        for (let dx = 0; dx < cellSize; dx++) {
          const pi = (by * pixelW + bx + dx) * 4;
          if (pi + 4 < px.length) { px[pi] = 0; px[pi+1] = 0; px[pi+2] = 0; px[pi+3] = 40; }
        }
      }
    }
  }

  ctx.putImageData(imageData, margin.left, margin.top);

  // Salva per tooltip
  canvas._matrix = matrix;
  canvas._tokens = tokens;
  canvas._cellSize = cellSize;
  canvas._margin = margin;
  canvas._seqLen = seqLen;

  // Etichette row (query tokens)
  const labelSize = Math.max(9, Math.min(13, cellSize));
  ctx.fillStyle = isDark ? '#ccc' : '#333';
  ctx.font = `bold ${labelSize}px monospace`;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let i = 0; i < seqLen; i++) {
    const label = truncLabel(tokens[i], Math.max(2, Math.floor(margin.left / labelSize * 0.7)));
    ctx.fillText(label, margin.left - 6, margin.top + i * cellSize + cellSize / 2);
  }

  // Etichette colonna (key tokens) — ruotate
  ctx.textAlign = 'right';
  ctx.textBaseline = 'top';
  for (let i = 0; i < seqLen; i++) {
    const label = truncLabel(tokens[i], Math.max(2, Math.floor(cellSize / labelSize * 2)));
    const x = margin.left + i * cellSize + cellSize / 2;
    const y = margin.top - 4;
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(-Math.PI / 3);
    ctx.fillText(label, 0, 0);
    ctx.restore();
  }

  // Info
  const li = parseInt(el.attLayer.value, 10);
  const hi = parseInt(el.attHead.value, 10);
  const label = li >= 0 && hi >= 0 ? `Layer ${li} Head ${hi}`
              : li >= 0 ? `Layer ${li} (media head)`
              : hi >= 0 ? `Head ${hi} (media layer)`
              : 'Media tutti';
  el.attInfo.textContent = `${tokens.length}×${tokens.length} — ${label}  (zoom: ×${zoom.toFixed(1)}, rotella per zoomare)`;
}

// Tooltip + zoom/pan interattivo
function setupAttentionTooltip() {
  const tooltip = document.createElement('div');
  tooltip.id = 'att-tooltip';
  tooltip.style.cssText = 'position:absolute;display:none;padding:6px 10px;border-radius:6px;font-size:12px;pointer-events:none;z-index:200;background:var(--bg-3);color:var(--text-0);border:1px solid var(--border);box-shadow:0 2px 8px var(--shadow);max-width:300px;';
  document.body.appendChild(tooltip);

  let dragging = false, dragStartX = 0, dragStartY = 0;

  el.attCanvas.addEventListener('mousedown', (e) => {
    if (e.button === 0) { dragging = true; dragStartX = e.clientX; dragStartY = e.clientY; }
  });
  window.addEventListener('mouseup', () => { dragging = false; });
  el.attCanvas.addEventListener('mousemove', (e) => {
    const c = el.attCanvas;
    if (!c._matrix) return;

    if (dragging) {
      _attZoom.offsetX += (e.clientX - dragStartX);
      _attZoom.offsetY += (e.clientY - dragStartY);
      dragStartX = e.clientX; dragStartY = e.clientY;
      drawAttention();
      return;
    }

    // Tooltip
    const rect = c.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const cs = c._cellSize;
    const m = c._margin;
    const k = Math.floor((x - m.left) / cs);
    const q = Math.floor((y - m.top) / cs);
    const seq = c._seqLen || 0;

    if (q >= 0 && q < seq && k >= 0 && k < seq) {
      const val = c._matrix[q][k];
      tooltip.innerHTML = `<strong>Q[${q}]:</strong> ${escapeHtml(truncLabel(c._tokens[q], 16))}<br><strong>K[${k}]:</strong> ${escapeHtml(truncLabel(c._tokens[k], 16))}<br><strong>Attenzione:</strong> ${(val * 100).toFixed(1)}%`;
      tooltip.style.display = 'block';
      tooltip.style.left = (e.pageX + 14) + 'px';
      tooltip.style.top = (e.pageY + 14) + 'px';
    } else {
      tooltip.style.display = 'none';
    }
  });

  el.attCanvas.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; dragging = false; });

  // Zoom rotellina
  el.attCanvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const d = e.deltaY > 0 ? 0.85 : 1.15;
    _attZoom.scale = Math.max(0.3, Math.min(10, _attZoom.scale * d));
    drawAttention();
  }, {passive: false});
}

/* ── Panels ─────────────────────────────────────────────────────────────────── */
function openPanel(id) {
  document.getElementById(id).classList.add('open');
  const overlay = document.getElementById('overlay');
  if (!overlay) {
    const div = document.createElement('div');
    div.id = 'overlay';
    div.className = 'show';
    div.onclick = closeAllPanels;
    document.body.appendChild(div);
  } else {
    overlay.classList.add('show');
  }
}
function closeAllPanels() {
  document.querySelectorAll('.slide-panel').forEach(p => p.classList.remove('open'));
  const overlay = document.getElementById('overlay');
  if (overlay) overlay.classList.remove('show');
}

/* ── Event listeners ────────────────────────────────────────────────────────── */
function setupEventListeners() {
  // Tema
  el.btnTheme.onclick = toggleTheme;

  // Sidebar
  el.btnToggleSidebar.onclick = () => {
    el.sidebar.classList.toggle('open');
  };
  el.btnNewChat.onclick = createNewChat;

  // Panels
  el.btnSettings.onclick = () => openPanel('settings-panel');
  el.btnAttention.onclick = () => openPanel('attention-panel');
  document.querySelectorAll('.btn-close-panel').forEach(b => b.onclick = closeAllPanels);

  // Settings inputs
  [el.sTemp, el.sMaxTok, el.sTopK, el.sTopP, el.sRep].forEach(inp => {
    inp.oninput = () => {
      readSettingsFromUI();
      applySettingsToUI(); // aggiorna label
    };
  });
  [el.sStream, el.sGreedy, el.sChatMode].forEach(inp => {
    inp.onchange = readSettingsFromUI;
  });

  // Send / stop
  el.btnSend.onclick = sendMessage;
  el.btnStop.onclick = stopGeneration;
  el.msgInput.onkeydown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };
  el.msgInput.oninput = () => {
    el.msgInput.style.height = 'auto';
    el.msgInput.style.height = Math.min(200, el.msgInput.scrollHeight) + 'px';
  };

  // Import / export
  el.btnExport.onclick = exportChats;
  el.btnImport.onclick = () => el.fileImport.click();
  el.fileImport.onchange = (e) => {
    if (e.target.files[0]) importChats(e.target.files[0]);
  };

  // Attention
  el.btnAnalyze.onclick = analyzeAttention;
  el.btnAttClear.onclick = () => {
    el.attInput.value = '';
    el.attControls.style.display = 'none';
    el.attCanvas.width = 0;
    el.attCanvas.height = 0;
    el.attInfo.textContent = '';
    state.attentionData = null;
  };

  // Responsive sidebar close on main click (mobile)
  el.chatMessages.onclick = () => {
    if (window.innerWidth <= 768 && el.sidebar.classList.contains('open')) {
      el.sidebar.classList.remove('open');
    }
  };
}

/* ── Boot ───────────────────────────────────────────────────────────────────── */
function init() {
  loadTheme();
  loadSettings();
  loadChats();
  renderChatList();
  if (!state.currentChatId) createNewChat();
  else selectChat(state.currentChatId);
  setupEventListeners();
  setupAttentionTooltip();
  checkHealth();
  setInterval(checkHealth, 10000);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

})();
