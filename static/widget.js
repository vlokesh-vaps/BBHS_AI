/**
 * VAPSAI Embeddable Chat Widget — widget.js
 * Drop a single <script> tag on any page to add a floating chat button.
 *
 * Configuration (data attributes on the <script> tag, or window.VAPSChatConfig):
 *   data-api-url       API base URL  (default: window.location.origin)
 *   data-title         Window title  (default: "BBHS AI Assistant")
 *   data-primary-color Accent colour (default: "#6C5CE7")
 *   data-welcome       Welcome msg   (default: "Hi! I am BBHS AI Assistant. How can I help?")
 *
 * Example:
 *   <script src="/static/widget.js"
 *           data-title="BBHS AI Assistant"
 *           data-primary-color="#6C5CE7"
 *           data-welcome="Hi! Ask me anything."
 *           defer></script>
 */

(function () {
  'use strict';

  /* ── Resolve config from data-* or window.VAPSChatConfig ── */
  const scriptEl = document.currentScript || (function () {
    const scripts = document.querySelectorAll('script[src*="widget.js"]');
    return scripts[scripts.length - 1];
  })();

  const ext     = (window.VAPSChatConfig || {});
  const attr    = (key, def) => (scriptEl && scriptEl.dataset[key]) || ext[key] || def;

  const CFG = {
    apiUrl:      attr('apiUrl',      window.location.origin),
    title:       attr('title',       'BBHS AI Assistant'),
    primary:     attr('primaryColor','#6C5CE7'),
    welcome:     attr('welcome',     'Hi! I am BBHS AI Assistant. How can I help?'),
  };

  const API_ENDPOINT = CFG.apiUrl.replace(/\/$/, '') + '/webhook/chat';
  const TIMEOUT_MS   = 30_000;

  /* ── State ─────────────────────────────────────────────── */
  let sessionId  = getOrCreateSession();
  let isLoading  = false;
  let isOpen     = false;

  /* ── Shadow DOM host ────────────────────────────────────── */
  const host   = document.createElement('div');
  host.id      = 'vaps-widget-host';
  host.setAttribute('aria-label', 'Chat widget');
  document.body.appendChild(host);

  const shadow = host.attachShadow({ mode: 'open' });

  /* ── Inline CSS ─────────────────────────────────────────── */
  const style = document.createElement('style');
  style.textContent = `
    :host{all:initial;font-family:'Inter',system-ui,sans-serif;}
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}

    #w-fab{
      position:fixed;bottom:24px;right:24px;z-index:2147483646;
      width:56px;height:56px;border-radius:50%;
      background:linear-gradient(135deg,${CFG.primary},#a29bfe);
      border:none;cursor:pointer;
      display:flex;align-items:center;justify-content:center;
      box-shadow:0 8px 28px ${CFG.primary}55;
      transition:transform .2s cubic-bezier(.34,1.56,.64,1),box-shadow .2s;
      color:#fff;font-size:24px;
    }
    #w-fab:hover{transform:scale(1.08);box-shadow:0 12px 36px ${CFG.primary}77;}
    #w-fab:focus-visible{outline:3px solid ${CFG.primary}99;outline-offset:3px;}

    #w-window{
      position:fixed;bottom:92px;right:24px;z-index:2147483645;
      width:380px;height:560px;max-height:80dvh;
      background:#0D0D1A;border:1px solid rgba(255,255,255,.09);
      border-radius:20px;box-shadow:0 24px 64px rgba(0,0,0,.6);
      display:flex;flex-direction:column;
      transform:scale(.92) translateY(20px);opacity:0;
      pointer-events:none;overflow:hidden;
      transition:transform .25s cubic-bezier(.34,1.56,.64,1),opacity .25s ease;
    }
    #w-window.open{transform:scale(1) translateY(0);opacity:1;pointer-events:all;}

    #w-header{
      display:flex;align-items:center;gap:10px;
      padding:13px 16px;border-bottom:1px solid rgba(255,255,255,.09);
      background:rgba(255,255,255,.04);flex-shrink:0;
    }
    #w-logo{
      width:32px;height:32px;border-radius:9px;
      background:linear-gradient(135deg,${CFG.primary},#a29bfe);
      display:flex;align-items:center;justify-content:center;font-size:16px;
    }
    #w-title{flex:1;font-size:14px;font-weight:600;color:#F0EEFF;}
    #w-close{
      background:none;border:none;cursor:pointer;color:#9B8FCE;
      font-size:20px;padding:2px 5px;border-radius:6px;line-height:1;
      transition:color .15s,background .15s;
    }
    #w-close:hover{background:rgba(255,255,255,.08);color:#F0EEFF;}

    #w-msgs{
      flex:1;overflow-y:auto;padding:14px 14px 6px;
      display:flex;flex-direction:column;gap:12px;scroll-behavior:smooth;
    }
    #w-msgs::-webkit-scrollbar{width:4px;}
    #w-msgs::-webkit-scrollbar-thumb{background:rgba(255,255,255,.1);border-radius:99px;}

    .w-welcome{
      display:flex;flex-direction:column;align-items:center;
      gap:10px;padding:20px 12px;text-align:center;color:#9B8FCE;font-size:13px;
      animation:w-in .3s ease both;
    }
    .w-welcome-icon{font-size:36px;}
    .w-welcome-title{font-size:15px;font-weight:600;color:#F0EEFF;}

    .w-row{display:flex;align-items:flex-end;gap:8px;animation:w-in .24s cubic-bezier(.34,1.56,.64,1) both;}
    .w-row.u{flex-direction:row-reverse;}
    @keyframes w-in{from{opacity:0;transform:translateY(10px);}to{opacity:1;transform:none;}}

    .w-avatar{
      width:26px;height:26px;border-radius:50%;flex-shrink:0;
      display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;
    }
    .w-avatar.bot{background:${CFG.primary};color:#fff;}
    .w-avatar.u  {background:rgba(255,255,255,.12);color:#9B8FCE;}

    .w-bubble{
      max-width:78%;padding:9px 13px;border-radius:16px;
      font-size:13.5px;line-height:1.6;word-break:break-word;white-space:pre-wrap;
    }
    .w-row.bot .w-bubble{
      background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.09);
      border-bottom-left-radius:5px;color:#F0EEFF;
    }
    .w-row.u .w-bubble{
      background:linear-gradient(135deg,${CFG.primary},#a29bfe);
      border-bottom-right-radius:5px;color:#fff;
    }
    .w-bubble.err{background:rgba(255,82,82,.12);border:1px solid rgba(255,82,82,.35);color:#FF6B6B;}

    .w-retry{
      display:block;margin-top:8px;padding:4px 10px;font-size:11.5px;
      border:1px solid rgba(255,82,82,.4);border-radius:6px;
      background:transparent;color:#FF6B6B;cursor:pointer;font-family:inherit;
    }

    #w-typing{display:flex;align-items:flex-end;gap:8px;opacity:0;transition:opacity .2s;}
    #w-typing.v{opacity:1;}
    .w-tdots{
      background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.09);
      border-radius:16px;border-bottom-left-radius:5px;
      padding:10px 14px;display:flex;gap:4px;align-items:center;
    }
    .wd{width:6px;height:6px;border-radius:50%;background:${CFG.primary};animation:wb 1.3s ease-in-out infinite;}
    .wd:nth-child(2){animation-delay:.18s;}
    .wd:nth-child(3){animation-delay:.36s;}
    @keyframes wb{0%,80%,100%{transform:translateY(0);opacity:.5;}40%{transform:translateY(-7px);opacity:1;}}

    #w-form{
      display:flex;align-items:flex-end;gap:8px;
      padding:10px 12px 12px;border-top:1px solid rgba(255,255,255,.09);
      background:rgba(255,255,255,.03);flex-shrink:0;
    }
    #w-input{
      flex:1;background:rgba(255,255,255,.07);
      border:1px solid rgba(255,255,255,.09);border-radius:12px;
      padding:8px 12px;color:#F0EEFF;font-family:inherit;
      font-size:13.5px;line-height:1.5;resize:none;
      max-height:90px;min-height:36px;outline:none;
      transition:border-color .2s;
    }
    #w-input::placeholder{color:#5C5580;}
    #w-input:focus{border-color:${CFG.primary}88;}

    #w-send{
      width:36px;height:36px;border-radius:50%;
      background:${CFG.primary};border:none;color:#fff;
      cursor:pointer;display:flex;align-items:center;justify-content:center;
      flex-shrink:0;box-shadow:0 3px 12px ${CFG.primary}55;
      transition:background .2s,transform .15s;
    }
    #w-send:hover{filter:brightness(1.15);}
    #w-send:active{transform:scale(.9);}
    #w-send:disabled{background:${CFG.primary}55;cursor:not-allowed;}
    #w-send svg{width:16px;height:16px;}

    .w-sugg{display:flex;flex-wrap:wrap;gap:5px;margin-top:6px;padding-left:34px;}
    .w-pill{
      background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.09);
      border-radius:99px;padding:4px 11px;font-size:11.5px;
      color:#9B8FCE;cursor:pointer;font-family:inherit;transition:all .18s;
    }
    .w-pill:hover{background:${CFG.primary};border-color:${CFG.primary};color:#fff;}

    .w-book{
      display:inline-flex;align-items:center;gap:5px;margin-top:8px;
      padding:6px 13px;border-radius:8px;text-decoration:none;
      background:rgba(108,92,231,.2);border:1px solid ${CFG.primary}66;
      color:#a29bfe;font-size:12px;font-family:inherit;transition:all .18s;
    }
    .w-book:hover{background:${CFG.primary};color:#fff;}

    .w-feedback{display:flex;gap:5px;margin-top:5px;}
    .w-fb-btn{
      background:none;border:1px solid rgba(255,255,255,.09);
      border-radius:6px;padding:2px 7px;cursor:pointer;
      font-size:12px;color:#5C5580;font-family:inherit;transition:all .18s;
    }
    .w-fb-btn:hover{border-color:${CFG.primary};color:#F0EEFF;}
    .w-fb-btn.up  .active{border-color:#00CE97;color:#00CE97;}
    .w-fb-btn.down.active{border-color:#FF6B6B;color:#FF6B6B;}

    @media(max-width:480px){
      #w-window{bottom:0;right:0;width:100%;height:100dvh;max-height:100dvh;border-radius:0;}
      #w-fab{bottom:16px;right:16px;}
    }
    @media(prefers-reduced-motion:reduce){*{animation-duration:.01ms !important;transition-duration:.01ms !important;}}
  `;
  shadow.appendChild(style);

  /* ── HTML ───────────────────────────────────────────────── */
  const tmpl = document.createElement('div');
  tmpl.innerHTML = `
    <!-- FAB -->
    <button id="w-fab" aria-label="Open ${CFG.title} chat" aria-expanded="false" aria-controls="w-window">
      💬
    </button>

    <!-- Chat Window -->
    <div id="w-window" role="dialog" aria-label="${CFG.title} chat" aria-modal="true">

      <div id="w-header">
        <div id="w-logo" aria-hidden="true">🤖</div>
        <div id="w-title">${escW(CFG.title)}</div>
        <button id="w-close" aria-label="Close chat">✕</button>
      </div>

      <div id="w-msgs" role="log" aria-live="polite" aria-label="Chat messages">
        <div class="w-welcome">
          <div class="w-welcome-icon" aria-hidden="true">🤖</div>
          <div class="w-welcome-title">${escW(CFG.title)}</div>
          <div>${escW(CFG.welcome)}</div>
        </div>
      </div>

      <div id="w-typing" aria-label="Assistant is typing" aria-hidden="true">
        <div class="w-avatar bot" aria-hidden="true">🤖</div>
        <div class="w-tdots">
          <span class="wd"></span><span class="wd"></span><span class="wd"></span>
        </div>
      </div>

      <form id="w-form" aria-label="Send a message" novalidate>
        <textarea
          id="w-input"
          rows="1"
          placeholder="Type a question…"
          aria-label="Type your question. Enter to send."
          autocomplete="off"
          maxlength="2000"
        ></textarea>
        <button id="w-send" type="submit" aria-label="Send">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <line x1="22" y1="2" x2="11" y2="13"/>
            <polygon points="22 2 15 22 11 13 2 9 22 2"/>
          </svg>
        </button>
      </form>
    </div>
  `;

  [...tmpl.children].forEach(c => shadow.appendChild(c));

  /* ── Refs ───────────────────────────────────────────────── */
  const $fab     = shadow.getElementById('w-fab');
  const $win     = shadow.getElementById('w-window');
  const $close   = shadow.getElementById('w-close');
  const $msgs    = shadow.getElementById('w-msgs');
  const $typing  = shadow.getElementById('w-typing');
  const $form    = shadow.getElementById('w-form');
  const $input   = shadow.getElementById('w-input');
  const $send    = shadow.getElementById('w-send');

  /* ── Toggle ─────────────────────────────────────────────── */
  $fab.addEventListener('click', toggle);
  $close.addEventListener('click', close);

  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && isOpen) close();
  });

  function toggle() { isOpen ? close() : open(); }

  function open() {
    isOpen = true;
    $win.classList.add('open');
    $fab.setAttribute('aria-expanded', 'true');
    $fab.innerHTML = '✕';
    requestAnimationFrame(() => $input.focus());
  }

  function close() {
    isOpen = false;
    $win.classList.remove('open');
    $fab.setAttribute('aria-expanded', 'false');
    $fab.innerHTML = '💬';
    $fab.focus();
  }

  /* ── Form ───────────────────────────────────────────────── */
  $form.addEventListener('submit', e => {
    e.preventDefault();
    const text = $input.value.trim();
    if (!text || isLoading) return;
    send(text);
  });

  $input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $form.requestSubmit(); }
  });

  $input.addEventListener('input', () => {
    $input.style.height = 'auto';
    $input.style.height = Math.min($input.scrollHeight, 90) + 'px';
  });

  /* ── Send ───────────────────────────────────────────────── */
  async function send(text) {
    isLoading = true;
    addBubble('u', text);
    $input.value = '';
    $input.style.height = 'auto';
    setEnabled(false);
    showTyping(true);
    scrollBot();

    const msgId = crypto.randomUUID();

    try {
      const ctrl  = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);

      const res = await fetch(API_ENDPOINT, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message:    text,
          session_id: sessionId,
          page_url:   window.location.href,   // widget sends current page URL
        }),
        signal: ctrl.signal,
      });

      clearTimeout(timer);

      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d.detail || `Server error ${res.status}`);
      }

      const data = await res.json();
      if (data.session_id) {
        sessionId = data.session_id;
      }

      showTyping(false);
      const row = addBubble('bot', data.answer, msgId);

      if (Array.isArray(data.suggested_questions) && data.suggested_questions.length) {
        addSuggestions(data.suggested_questions);
      }


    } catch (err) {
      showTyping(false);
      const msg = err.name === 'AbortError'
        ? '⏱ Request timed out. Please try again.'
        : `⚠ ${err.message || 'Something went wrong.'}`;
      addError(msg, text);
    } finally {
      isLoading = false;
      setEnabled(true);
      scrollBot();
      $input.focus();
    }
  }

  /* ── Bubble builders ────────────────────────────────────── */
  function addBubble(role, text, msgId) {
    const row = document.createElement('div');
    row.className = `w-row ${role}`;
    row.setAttribute('role', 'article');

    const av  = document.createElement('div');
    av.className = `w-avatar ${role}`;
    av.setAttribute('aria-hidden', 'true');
    av.textContent = role === 'u' ? '👤' : '🤖';

    const bubble = document.createElement('div');
    bubble.className = 'w-bubble';
    bubble.textContent = text;    // textContent = XSS-safe

    const col = document.createElement('div');
    col.style.cssText = 'display:flex;flex-direction:column;gap:4px;max-width:78%;';
    if (role === 'u') col.style.alignItems = 'flex-end';
    col.appendChild(bubble);

    if (role === 'bot' && msgId) col.appendChild(buildFeedback(msgId));

    row.appendChild(av);
    row.appendChild(col);
    $msgs.appendChild(row);
    return row;
  }

  function addError(msg, retryText) {
    const row = document.createElement('div');
    row.className = 'w-row bot';
    row.setAttribute('role', 'alert');

    const av = document.createElement('div');
    av.className = 'w-avatar bot';
    av.setAttribute('aria-hidden', 'true');
    av.textContent = '🤖';

    const bubble = document.createElement('div');
    bubble.className = 'w-bubble err';
    bubble.textContent = msg;

    const retry = document.createElement('button');
    retry.className = 'w-retry';
    retry.textContent = '↺ Retry';
    retry.setAttribute('aria-label', 'Retry sending your message');
    retry.addEventListener('click', () => { row.remove(); send(retryText); });
    bubble.appendChild(retry);

    row.appendChild(av);
    row.appendChild(bubble);
    $msgs.appendChild(row);
  }

  function addSuggestions(qs) {
    const wrap = document.createElement('div');
    wrap.className = 'w-sugg';
    wrap.setAttribute('role', 'group');
    wrap.setAttribute('aria-label', 'Suggested follow-up questions');
    qs.forEach(q => {
      const btn = document.createElement('button');
      btn.className = 'w-pill';
      btn.textContent = q;
      btn.setAttribute('aria-label', `Ask: ${q}`);
      btn.addEventListener('click', () => { $input.value = q; $form.requestSubmit(); wrap.remove(); });
      wrap.appendChild(btn);
    });
    $msgs.appendChild(wrap);
  }

  function addBooking(url, parentRow) {
    const a = document.createElement('a');
    a.className = 'w-book';
    a.href = url;
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    a.setAttribute('aria-label', 'Book a demo (opens in new tab)');
    a.textContent = '📅 Book a Demo';
    parentRow.querySelector('div > .w-bubble')?.after(a);
    if (!parentRow.querySelector('.w-book')) parentRow.appendChild(a);
  }

  function buildFeedback(msgId) {
    const wrap = document.createElement('div');
    wrap.className = 'w-feedback';
    wrap.setAttribute('role', 'group');
    wrap.setAttribute('aria-label', 'Was this helpful?');

    const up   = mkFbBtn('👍', 'Helpful');
    const down = mkFbBtn('👎', 'Not helpful');

    function vote(rating) {
      // Store {timestamp, session_id, message_id, rating} — no PII
      try {
      } catch (_) {}
      wrap.innerHTML = '<span style="font-size:11px;color:#00CE97;">Thanks!</span>';
    }

    up.addEventListener('click',   () => vote(1));
    down.addEventListener('click', () => vote(-1));
    wrap.appendChild(up);
    wrap.appendChild(down);
    return wrap;
  }

  function mkFbBtn(icon, label) {
    const btn = document.createElement('button');
    btn.className = 'w-fb-btn';
    btn.textContent = icon;
    btn.setAttribute('aria-label', label);
    return btn;
  }

  /* ── Helpers ────────────────────────────────────────────── */
  function showTyping(v) {
    $typing.classList.toggle('v', v);
    $typing.setAttribute('aria-hidden', String(!v));
    if (v) scrollBot();
  }

  function setEnabled(v) {
    $input.disabled = !v;
    $send.disabled  = !v;
  }

  function scrollBot() {
    requestAnimationFrame(() => $msgs.scrollTo({ top: $msgs.scrollHeight, behavior: 'smooth' }));
  }

  function escW(s) {
    return String(s)
      .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
      .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
  }

  function getOrCreateSession() {
    return crypto.randomUUID();
  }

  /* ── Bounce FAB on first load ───────────────────────────── */
  {
    setTimeout(() => {
      $fab.style.transform = 'scale(1.15)';
      setTimeout(() => { $fab.style.transform = ''; }, 300);
    }, 800);
  }

})();
