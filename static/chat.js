/**
 * VAPSAI Chat UI — chat.js
 * Calls POST /webhook/chat (existing endpoint, no backend changes)
 * Features: session management, typing indicator, error/timeout handling,
 *           thumbs up/down feedback (localStorage), suggested questions,
 *           booking link, keyboard accessibility, new chat.
 */

'use strict';

/* ── Config ───────────────────────────────────────────────── */
const API_URL      = '/webhook/chat';   // Proxied from same origin; change if needed
const TIMEOUT_MS   = 30_000;
const SESSION_KEY  = 'vaps_session_id';
const FEEDBACK_KEY = 'vaps_feedback';

/* Starter questions shown on empty state */
const STARTERS = [
  'How do I create a new issue in VMS?',
  'What is the IssueManager workflow?',
  'How do I assign a ticket to a user?',
  'How do I close / resolve an issue?',
];

/* ── State ────────────────────────────────────────────────── */
let sessionId        = getOrCreateSession();
let isLoading        = false;
let lastUserMessage  = '';
let feedbackStore    = loadFeedback();

/* ── DOM Refs ─────────────────────────────────────────────── */
const $app          = document.getElementById('app');
const $msgList      = document.getElementById('message-list');
const $typingRow    = document.getElementById('typing-row');
const $chatForm     = document.getElementById('chat-form');
const $input        = document.getElementById('message-input');
const $sendBtn      = document.getElementById('send-btn');
const $sessionDisp  = document.getElementById('session-id-display');
const $newChatBtn   = document.getElementById('new-chat-btn');
const $welcomeState = document.getElementById('welcome-state');
const $toastCont    = document.getElementById('toast-container');

/* ── Init ─────────────────────────────────────────────────── */
(function init() {
  updateSessionDisplay();
  renderWelcome();

  $chatForm.addEventListener('submit', onSubmit);
  $input.addEventListener('keydown', onKeydown);
  $input.addEventListener('input', autoResize);
  $newChatBtn.addEventListener('click', newChat);
})();

/* ── Session ──────────────────────────────────────────────── */
function getOrCreateSession() {
  let id = sessionStorage.getItem(SESSION_KEY);
  if (!id) {
    id = crypto.randomUUID();
    sessionStorage.setItem(SESSION_KEY, id);
  }
  return id;
}

function updateSessionDisplay() {
  const short = sessionId.slice(0, 8) + '…';
  $sessionDisp.textContent = short;
  $sessionDisp.title       = `Session: ${sessionId}`;
}

function newChat() {
  if (isLoading) return;
  sessionStorage.removeItem(SESSION_KEY);
  sessionId = getOrCreateSession();
  feedbackStore = {};
  updateSessionDisplay();

  // Insert divider then clear
  const divider = document.createElement('div');
  divider.className = 'session-divider';
  divider.setAttribute('role', 'separator');
  divider.textContent = 'New conversation';

  // Clear messages
  $msgList.innerHTML = '';
  renderWelcome();
  $input.value = '';
  autoResize();
  $input.focus();
}

/* ── Welcome / Empty State ────────────────────────────────── */
function renderWelcome() {
  // $welcomeState is part of #message-list initially
  if (!document.getElementById('welcome-state')) {
    const ws = document.createElement('div');
    ws.id = 'welcome-state';
    ws.setAttribute('aria-label', 'Welcome screen');
    ws.innerHTML = `
      <div class="welcome-icon" aria-hidden="true">🤖</div>
      <h2>BBHS AI Assistant</h2>
      <p>Ask a question and I will answer using information from the active knowledge base.</p>
      <div class="starter-chips" role="list" aria-label="Suggested starter questions">
        ${STARTERS.map(q => `<button class="starter-chip" role="listitem" aria-label="Ask: ${q}">${q}</button>`).join('')}
      </div>`;
    $msgList.appendChild(ws);
  }

  // Wire starter chips
  $msgList.querySelectorAll('.starter-chip').forEach(btn => {
    btn.addEventListener('click', () => {
      $input.value = btn.textContent;
      autoResize();
      $chatForm.requestSubmit();
    });
  });
}

function hideWelcome() {
  const ws = document.getElementById('welcome-state');
  if (ws) ws.remove();
}

/* ── Form Submit ──────────────────────────────────────────── */
function onSubmit(e) {
  e.preventDefault();
  const text = $input.value.trim();
  if (!text || isLoading) return;
  sendMessage(text);
}

function onKeydown(e) {
  // Enter = send; Shift+Enter = newline
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    $chatForm.requestSubmit();
  }
}

function autoResize() {
  $input.style.height = 'auto';
  $input.style.height = Math.min($input.scrollHeight, 140) + 'px';
}

/* ── Send Message ─────────────────────────────────────────── */
async function sendMessage(text) {
  hideWelcome();
  isLoading        = true;
  lastUserMessage  = text;

  // Show user bubble
  appendMessage('user', text);
  $input.value = '';
  autoResize();
  setInputEnabled(false);
  showTyping(true);
  scrollToBottom();

  const msgId = crypto.randomUUID();

  try {
    const controller = new AbortController();
    const timer      = setTimeout(() => controller.abort(), TIMEOUT_MS);

    const res = await fetch(API_URL, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ message: text, session_id: sessionId }),
      signal:  controller.signal,
    });

    clearTimeout(timer);

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Server error ${res.status}`);
    }

    const data = await res.json();
    // Update session_id from server (server may normalize it)
    if (data.session_id) {
      sessionId = data.session_id;
      updateSessionDisplay();
    }

    showTyping(false);
    const botRow = appendMessage('bot', data.answer, msgId);

    // Suggested questions (if backend ever returns them)
    if (Array.isArray(data.suggested_questions) && data.suggested_questions.length) {
      appendSuggestions(data.suggested_questions);
    }

    // Booking URL (if backend returns it)
    if (data.booking_url) {
      appendBookingLink(data.booking_url, botRow);
    }

  } catch (err) {
    showTyping(false);
    if (err.name === 'AbortError') {
      appendError('⏱ Request timed out after 30 s. Please try again.', text);
    } else {
      appendError(`⚠ ${err.message || 'Something went wrong. Please try again.'}`, text);
    }
  } finally {
    isLoading = false;
    setInputEnabled(true);
    scrollToBottom();
    $input.focus();
  }
}

/* ── Bubble Helpers ───────────────────────────────────────── */
function appendMessage(role, text, msgId) {
  const row = document.createElement('div');
  row.className = `msg-row ${role}`;
  row.setAttribute('role', 'article');
  row.setAttribute('aria-label', `${role === 'user' ? 'You' : 'Assistant'}: ${text.slice(0, 60)}`);

  const avatarLabel = role === 'user' ? 'You' : 'BBHS AI Assistant';
  const avatarIcon  = role === 'user' ? '👤'  : '🤖';

  row.innerHTML = `
    <div class="msg-avatar ${role}" aria-label="${avatarLabel}" aria-hidden="true">${avatarIcon}</div>
    <div class="msg-content">
      <div class="msg-bubble">${escapeHtml(text)}</div>
      <div class="msg-time" aria-label="Sent at ${formatTime(new Date())}">${formatTime(new Date())}</div>
    </div>`;

  // Add feedback for bot messages
  if (role === 'bot' && msgId) {
    const fb = buildFeedback(msgId);
    row.querySelector('.msg-content').appendChild(fb);
  }

  $msgList.appendChild(row);
  return row;
}

function appendError(message, retryText) {
  const row = document.createElement('div');
  row.className = 'msg-row bot';
  row.setAttribute('role', 'alert');

  row.innerHTML = `
    <div class="msg-avatar bot" aria-hidden="true">🤖</div>
    <div class="msg-content">
      <div class="msg-bubble error">${escapeHtml(message)}
        <button class="retry-btn" aria-label="Retry sending your message">↺ Retry</button>
      </div>
    </div>`;

  row.querySelector('.retry-btn').addEventListener('click', () => {
    row.remove();
    sendMessage(retryText);
  });

  $msgList.appendChild(row);
}

function appendSuggestions(questions) {
  const wrap = document.createElement('div');
  wrap.className = 'suggestions-wrap';
  wrap.setAttribute('role', 'group');
  wrap.setAttribute('aria-label', 'Suggested follow-up questions');

  questions.forEach(q => {
    const btn = document.createElement('button');
    btn.className = 'suggestion-pill';
    btn.textContent = q;
    btn.setAttribute('aria-label', `Ask: ${q}`);
    btn.addEventListener('click', () => {
      $input.value = q;
      autoResize();
      $chatForm.requestSubmit();
      wrap.remove();
    });
    wrap.appendChild(btn);
  });

  $msgList.appendChild(wrap);
}

function appendBookingLink(url, parentRow) {
  const a = document.createElement('a');
  a.className   = 'booking-cta';
  a.href        = url;
  a.target      = '_blank';
  a.rel         = 'noopener noreferrer';
  a.textContent = '📅 Book a Demo';
  a.setAttribute('aria-label', 'Book a demo (opens in new tab)');
  parentRow.querySelector('.msg-content').appendChild(a);
}

/* ── Feedback ─────────────────────────────────────────────── */
function buildFeedback(msgId) {
  const wrap = document.createElement('div');
  wrap.className = 'msg-feedback';
  wrap.setAttribute('role', 'group');
  wrap.setAttribute('aria-label', 'Was this response helpful?');

  const up   = document.createElement('button');
  const down = document.createElement('button');

  up.className   = 'feedback-btn';
  down.className = 'feedback-btn';
  up.textContent   = '👍';
  down.textContent = '👎';
  up.setAttribute('aria-label',   'This response was helpful');
  down.setAttribute('aria-label', 'This response was not helpful');

  // If already voted
  const existing = feedbackStore[msgId];
  if (existing === 1)  { up.classList.add('active','up'); }
  if (existing === -1) { down.classList.add('active','down'); }

  function vote(rating, btn, other) {
    // Store: {timestamp, session_id, message_id, rating} — no PII
    feedbackStore[msgId] = rating;
    saveFeedback();

    btn.classList.add('active', rating === 1 ? 'up' : 'down');
    other.classList.remove('active','up','down');

    // Replace buttons with thank-you
    wrap.innerHTML = '<span class="feedback-thanks" aria-live="polite">Thanks for your feedback!</span>';
  }

  up.addEventListener('click',   () => vote(1,  up,   down));
  down.addEventListener('click', () => vote(-1, down, up));

  wrap.appendChild(up);
  wrap.appendChild(down);
  return wrap;
}

function loadFeedback() {
  try { return JSON.parse(localStorage.getItem(FEEDBACK_KEY) || '{}'); }
  catch { return {}; }
}

function saveFeedback() {
  try { localStorage.setItem(FEEDBACK_KEY, JSON.stringify(feedbackStore)); }
  catch { /* storage quota exceeded — ignore */ }
}

/* ── Typing Indicator ─────────────────────────────────────── */
function showTyping(visible) {
  $typingRow.classList.toggle('visible', visible);
  $typingRow.setAttribute('aria-hidden', String(!visible));
  if (visible) scrollToBottom();
}

/* ── UI Helpers ───────────────────────────────────────────── */
function setInputEnabled(enabled) {
  $input.disabled   = !enabled;
  $sendBtn.disabled = !enabled;
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    $msgList.scrollTo({ top: $msgList.scrollHeight, behavior: 'smooth' });
  });
}

function formatTime(date) {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function showToast(message, type = '') {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.setAttribute('role', 'status');
  toast.setAttribute('aria-live', 'polite');
  toast.textContent = message;
  $toastCont.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}
