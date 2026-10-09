// BBHS AI Assistant — Gemini-Style Chat Interface
(function() {
  'use strict';

  let sessionId = generateSessionId();
  let assistantName = 'BBHS AI Assistant';
  let activeAbortController = null;
  let lastUserMessage = '';

  const chatContainer = document.getElementById('chat-container');
  const messageList = document.getElementById('message-list');
  const emptyState = document.getElementById('empty-state');
  const chatForm = document.getElementById('chat-form');
  const userInput = document.getElementById('user-input');
  const actionBtn = document.getElementById('action-btn');
  const sendIcon = document.getElementById('send-icon');
  const stopIcon = document.getElementById('stop-icon');
  const loadingIndicator = document.getElementById('loading-indicator');
  const brandTitle = document.getElementById('brand-title');
  const footerNote = document.getElementById('footer-note');
  const newChatBtn = document.getElementById('new-chat-btn');
  const suggestionChips = document.getElementById('suggestion-chips');

  function renderSuggestions() {
    if (!suggestionChips) return;
    const suggestions = [
      'What can you help me with?',
      'Summarize the main topics in the knowledge base',
      'Help me find the right policy or procedure'
    ];
    suggestionChips.innerHTML = suggestions.map(text =>
      `<button type="button" class="chip-btn">${escapeHTML(text)}</button>`
    ).join('');
  }

  function generateSessionId() {
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
      try { return crypto.randomUUID(); } catch (e) {}
    }
    if (typeof crypto !== 'undefined' && typeof crypto.getRandomValues === 'function') {
      const bytes = new Uint8Array(16);
      crypto.getRandomValues(bytes);
      return Array.from(bytes, b => b.toString(16).padStart(2, '0')).join('');
    }
    return 'session-' + Date.now() + '-' + Math.random().toString(36).substring(2, 9);
  }

  // Load backend branding
  async function initConfig() {
    try {
      const res = await fetch('/chat/config');
      if (res.ok) {
        const data = await res.json();
        if (data.assistant_name) {
          assistantName = data.assistant_name;
          brandTitle.textContent = assistantName;
          document.title = assistantName;
          footerNote.textContent = `${assistantName} answers from its knowledge base and can make mistakes. Conversations are not saved.`;
          userInput.placeholder = `Message ${assistantName}...`;
        }
      }
    } catch (e) {
      // Graceful fallback to default branding
    }
  }

  // Escape HTML before markdown parsing for strict XSS prevention
  function escapeHTML(str) {
    return str.replace(/[&<>"']/g, function(tag) {
      const chars = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
      };
      return chars[tag] || tag;
    });
  }

  // Safe minimal Markdown parser
  function renderSafeMarkdown(rawText) {
    if (!rawText) return '';
    // 1. Fully escape all HTML tags/entities
    let safe = escapeHTML(rawText);

    // 2. Fenced code blocks ```lang\ncode\n```
    safe = safe.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, function(_, lang, code) {
      return `<pre><code>${code.trim()}</code></pre>`;
    });

    // 3. Inline code `code`
    safe = safe.replace(/`([^`\n]+)`/g, '<code>$1</code>');

    // 4. Bold **text** or __text__
    safe = safe.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    safe = safe.replace(/__([^_]+)__/g, '<strong>$1</strong>');

    // 5. Italic *text* or _text_
    safe = safe.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    safe = safe.replace(/_([^_]+)_/g, '<em>$1</em>');

    // 6. Safe links [text](http... or https...)
    safe = safe.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');

    // 7. Lists and paragraphs
    const lines = safe.split('\n');
    let inList = false;
    let listType = '';
    const formattedLines = [];

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const ulMatch = line.match(/^(\*|-)\s+(.*)/);
      const olMatch = line.match(/^(\d+)\.\s+(.*)/);

      if (ulMatch) {
        if (!inList || listType !== 'ul') {
          if (inList) formattedLines.push(`</${listType}>`);
          formattedLines.push('<ul>');
          inList = true;
          listType = 'ul';
        }
        formattedLines.push(`<li>${ulMatch[2]}</li>`);
      } else if (olMatch) {
        if (!inList || listType !== 'ol') {
          if (inList) formattedLines.push(`</${listType}>`);
          formattedLines.push('<ol>');
          inList = true;
          listType = 'ol';
        }
        formattedLines.push(`<li>${olMatch[2]}</li>`);
      } else {
        if (inList) {
          formattedLines.push(`</${listType}>`);
          inList = false;
        }
        if (line.trim().length === 0) {
          formattedLines.push('<br>');
        } else {
          formattedLines.push(`<p>${line}</p>`);
        }
      }
    }
    if (inList) {
      formattedLines.push(`</${listType}>`);
    }

    return formattedLines.join('');
  }

  function shouldAutoScroll() {
    const threshold = 120;
    return (chatContainer.scrollHeight - chatContainer.scrollTop - chatContainer.clientHeight) <= threshold;
  }

  function scrollToBottom(force = false) {
    if (force || shouldAutoScroll()) {
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }
  }

  function appendUserMessage(text) {
    emptyState.classList.add('d-none');
    const msgEl = document.createElement('div');
    msgEl.className = 'message user-message';

    const bubble = document.createElement('div');
    bubble.className = 'user-bubble';
    bubble.textContent = text; // textContent prevents any HTML insertion

    msgEl.appendChild(bubble);
    messageList.appendChild(msgEl);
    scrollToBottom(true);
  }

  function appendAssistantMessage(text) {
    const msgEl = document.createElement('div');
    msgEl.className = 'message assistant-message';

    const avatar = document.createElement('div');
    avatar.className = 'avatar logo-avatar';
    avatar.innerHTML = `<img src="/static/logo.png" alt="BBHS" class="avatar-img" width="24" height="24">`;

    const body = document.createElement('div');
    body.className = 'assistant-body';

    const textEl = document.createElement('div');
    textEl.className = 'assistant-text';
    textEl.innerHTML = renderSafeMarkdown(text);

    const actions = document.createElement('div');
    actions.className = 'message-actions';

    const copyBtn = document.createElement('button');
    copyBtn.className = 'action-btn';
    copyBtn.setAttribute('aria-label', 'Copy response');
    copyBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg><span>Copy</span>`;

    copyBtn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(text);
        copyBtn.querySelector('span').textContent = 'Copied!';
        setTimeout(() => {
          copyBtn.querySelector('span').textContent = 'Copy';
        }, 2000);
      } catch (err) {
        copyBtn.querySelector('span').textContent = 'Failed';
      }
    });

    actions.appendChild(copyBtn);
    body.appendChild(textEl);
    body.appendChild(actions);

    msgEl.appendChild(avatar);
    msgEl.appendChild(body);
    messageList.appendChild(msgEl);
    scrollToBottom(true);
  }

  function appendErrorMessage(friendlyText, retryPrompt) {
    const msgEl = document.createElement('div');
    msgEl.className = 'message assistant-message';

    const avatar = document.createElement('div');
    avatar.className = 'avatar logo-avatar';
    avatar.innerHTML = `<img src="/static/logo.png" alt="BBHS" class="avatar-img" width="24" height="24">`;

    const body = document.createElement('div');
    body.className = 'assistant-body';

    const card = document.createElement('div');
    card.className = 'error-card';

    const msg = document.createElement('span');
    msg.textContent = friendlyText;

    const retryBtn = document.createElement('button');
    retryBtn.className = 'retry-btn';
    retryBtn.textContent = 'Retry';
    retryBtn.addEventListener('click', () => {
      msgEl.remove();
      sendMessage(retryPrompt);
    });

    card.appendChild(msg);
    card.appendChild(retryBtn);
    body.appendChild(card);

    msgEl.appendChild(avatar);
    msgEl.appendChild(body);
    messageList.appendChild(msgEl);
    scrollToBottom(true);
  }

  function setInFlight(inFlight) {
    if (inFlight) {
      actionBtn.disabled = false;
      actionBtn.setAttribute('aria-label', 'Stop generating');
      actionBtn.setAttribute('title', 'Stop');
      sendIcon.classList.add('d-none');
      stopIcon.classList.remove('d-none');
      loadingIndicator.classList.remove('d-none');
    } else {
      activeAbortController = null;
      actionBtn.setAttribute('aria-label', 'Send message');
      actionBtn.setAttribute('title', 'Send (Enter)');
      sendIcon.classList.remove('d-none');
      stopIcon.classList.add('d-none');
      loadingIndicator.classList.add('d-none');
      updateSendButtonState();
      userInput.focus();
    }
  }

  async function sendMessage(text) {
    const trimmed = text.trim();
    if (!trimmed) return;
    lastUserMessage = trimmed;

    appendUserMessage(trimmed);
    userInput.value = '';
    autoResizeTextarea();

    activeAbortController = new AbortController();
    setInFlight(true);

    try {
      const response = await fetch('/webhook/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: trimmed,
          session_id: sessionId,
        }),
        signal: activeAbortController.signal,
      });

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }

      const data = await response.json();
      appendAssistantMessage(data.answer || 'No response returned.');
    } catch (err) {
      if (err.name === 'AbortError') {
        appendAssistantMessage('Request stopped.');
      } else {
        appendErrorMessage('Unable to connect to the assistant. Please try again.', trimmed);
      }
    } finally {
      setInFlight(false);
    }
  }

  function autoResizeTextarea() {
    userInput.style.height = 'auto';
    userInput.style.height = Math.min(userInput.scrollHeight, 180) + 'px';
  }

  function updateSendButtonState() {
    if (activeAbortController) {
      actionBtn.disabled = false;
    } else {
      actionBtn.disabled = userInput.value.trim().length === 0;
    }
  }

  // Handle Input events
  userInput.addEventListener('input', () => {
    autoResizeTextarea();
    updateSendButtonState();
  });

  userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      if (!actionBtn.disabled && !activeAbortController) {
        sendMessage(userInput.value);
      }
    }
  });

  chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    if (activeAbortController) {
      activeAbortController.abort();
    } else {
      sendMessage(userInput.value);
    }
  });

  // New Chat Action: reset memory, clear transcript, generate new session ID
  newChatBtn.addEventListener('click', () => {
    if (activeAbortController) {
      activeAbortController.abort();
    }
    sessionId = generateSessionId();
    messageList.innerHTML = '';
    emptyState.classList.remove('d-none');
    userInput.value = '';
    autoResizeTextarea();
    setInFlight(false);
    userInput.focus();
  });

  // Suggestion chips handler
  if (suggestionChips) {
    suggestionChips.addEventListener('click', (e) => {
      const chip = e.target.closest('.chip-btn');
      if (chip) {
        sendMessage(chip.textContent.trim());
      }
    });
  }

  initConfig();
  renderSuggestions();
  userInput.focus();
})();
