const state = { config: null, providers: [], kb: null, pollTimer: null };

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...options });
  if (response.status === 401) {
    throw new Error('Admin session expired; authenticate again');
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
  return data;
}

function showAlert(message, type = 'success') {
  document.getElementById('alert-area').innerHTML = `<div class="alert alert-${type} alert-dismissible fade show" role="alert">${message}<button type="button" class="btn-close" data-bs-dismiss="alert"></button></div>`;
}

function formatBytes(bytes) {
  if (!bytes || bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function renderKBActive(targetId, active, rebuild) {
  const container = document.getElementById(targetId);
  if (!container) return;
  container.innerHTML = '';

  const row = document.createElement('div');
  row.className = 'row g-2';

  const nameCol = document.createElement('div');
  nameCol.className = 'col-md-6';
  nameCol.innerHTML = '<strong>File:</strong> ';
  const nameSpan = document.createElement('span');
  nameSpan.className = 'text-break';
  nameSpan.textContent = active.display_name || 'None';
  nameCol.appendChild(nameSpan);

  const sizeCol = document.createElement('div');
  sizeCol.className = 'col-md-3';
  sizeCol.innerHTML = `<strong>Size:</strong> ${formatBytes(active.size_bytes)}`;

  const pagesCol = document.createElement('div');
  pagesCol.className = 'col-md-3';
  pagesCol.innerHTML = `<strong>Pages:</strong> ${active.pages || 0}`;

  const chunksCol = document.createElement('div');
  chunksCol.className = 'col-md-4';
  chunksCol.innerHTML = `<strong>Chunks:</strong> ${active.chunks || 0}`;

  const uploadedCol = document.createElement('div');
  uploadedCol.className = 'col-md-8';
  uploadedCol.innerHTML = `<strong>Uploaded:</strong> ${active.uploaded_at ? new Date(active.uploaded_at).toLocaleString() : 'Pre-loaded'}`;

  const statusCol = document.createElement('div');
  statusCol.className = 'col-12 mt-2 pt-2 border-top';
  const badgeClass = rebuild.state === 'done' ? 'success' : (rebuild.state === 'failed' ? 'danger' : (rebuild.state === 'idle' ? 'secondary' : 'warning'));
  statusCol.innerHTML = `<strong>Rebuild state:</strong> <span class="badge text-bg-${badgeClass}">${rebuild.state}</span> <small class="text-muted ms-2">${rebuild.message || ''}</small>`;

  row.appendChild(nameCol);
  row.appendChild(sizeCol);
  row.appendChild(pagesCol);
  row.appendChild(chunksCol);
  row.appendChild(uploadedCol);
  row.appendChild(statusCol);

  container.appendChild(row);
}

async function loadKBStatus() {
  try {
    const kb = await api('/admin/kb/status');
    state.kb = kb;
    renderKBActive('active-kb-details', kb.active, kb.rebuild);
    renderKBActive('dashboard-kb-card', kb.active, kb.rebuild);
    if (['validating', 'building', 'swapping'].includes(kb.rebuild.state)) {
      startKBPoll();
    }
  } catch (err) {
    if (err.message.includes('expired')) {
      showAlert('Session expired. Log in again.', 'warning');
    }
  }
}

function startKBPoll() {
  if (state.pollTimer) return;
  const alertEl = document.getElementById('kb-rebuild-alert');
  const spinner = document.getElementById('upload-spinner');
  const btnText = document.getElementById('upload-btn-text');
  const uploadBtn = document.getElementById('upload-rebuild-btn');

  uploadBtn.disabled = true;
  spinner.classList.remove('d-none');
  btnText.textContent = 'Rebuilding...';

  state.pollTimer = setInterval(async () => {
    try {
      const kb = await api('/admin/kb/status');
      state.kb = kb;
      renderKBActive('active-kb-details', kb.active, kb.rebuild);
      renderKBActive('dashboard-kb-card', kb.active, kb.rebuild);

      alertEl.classList.remove('d-none');
      alertEl.className = 'alert alert-info';
      alertEl.textContent = `[${kb.rebuild.state.toUpperCase()}] ${kb.rebuild.message}`;

      if (kb.rebuild.state === 'done') {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
        spinner.classList.add('d-none');
        btnText.textContent = 'Upload and Rebuild';
        uploadBtn.disabled = false;
        alertEl.className = 'alert alert-success';
        alertEl.textContent = 'Knowledge base rebuild completed successfully.';
        showAlert('Knowledge base rebuilt and activated.');
      } else if (kb.rebuild.state === 'failed') {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
        spinner.classList.add('d-none');
        btnText.textContent = 'Upload and Rebuild';
        uploadBtn.disabled = false;
        alertEl.className = 'alert alert-danger';
        alertEl.textContent = `Rebuild failed: ${kb.rebuild.message}`;
        showAlert(`Knowledge base rebuild failed: ${kb.rebuild.message}`, 'danger');
      }
    } catch (err) {
      if (err.message.includes('expired')) {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
        spinner.classList.add('d-none');
        btnText.textContent = 'Upload and Rebuild';
        alertEl.className = 'alert alert-warning';
        alertEl.textContent = 'Session expired. Log in again. The rebuild continues on the server.';
      }
    }
  }, 1500);
}

function input(label, name, value, type = 'text') {
  const checked = type === 'checkbox' ? (value ? ' checked' : '') : '';
  const display = type === 'checkbox' ? `<div class="form-check form-switch"><input class="form-check-input" name="${name}" type="checkbox"${checked}><label class="form-check-label">${label}</label></div>` : `<label class="form-label">${label}</label><input class="form-control" name="${name}" type="${type}" value="${String(value ?? '').replaceAll('"', '&quot;')}">`;
  return `<div class="col-md-6">${display}</div>`;
}

function populateDynamicForm(formId, section, labels) {
  const form = document.querySelector(`[data-section="${section}"]`);
  const target = document.getElementById(formId);
  target.innerHTML = Object.entries(labels).map(([name, label]) => {
    const value = state.config[section][name];
    const type = typeof value === 'boolean' ? 'checkbox' : (typeof value === 'number' ? 'number' : 'text');
    return type === 'checkbox' ? input(label, name, value, type) : `<div class="col-md-6">${input(label, name, value, type)}</div>`;
  }).join('');
}

function populatePrompts() {
  const prompts = state.config.prompts;
  const labels = { system_prompt: 'System prompt', rag_prompt: 'RAG prompt', no_answer_prompt: 'No-answer prompt', greeting_prompt: 'Greeting prompt', context_instructions: 'Context instructions' };
  document.getElementById('prompt-form').innerHTML = Object.entries(labels).map(([name, label]) => `<div class="mb-3"><label class="form-label">${label} <span class="text-muted">${prompts[name].length} chars</span></label><textarea class="form-control" name="${name}" maxlength="20000">${prompts[name]}</textarea></div>`).join('') + '<button class="btn btn-primary save-section">Save prompts</button>';
}

function fillForms() {
  const modelForm = document.querySelector('[data-section="models"]');
  Object.entries(state.config.models).forEach(([name, value]) => { const el = modelForm.elements[name]; if (el) el.type === 'checkbox' ? el.checked = value : el.value = value; });
  populatePrompts();
  populateDynamicForm('rag-fields', 'rag', { vector_top_k: 'Vector top-k', bm25_top_k: 'BM25 top-k', final_top_k: 'Final top-k', similarity_threshold: 'Similarity threshold', chunk_size: 'Chunk size', chunk_overlap: 'Chunk overlap', retrieval_strategy: 'Retrieval strategy' });
  populateDynamicForm('performance-fields', 'performance', { cache_enabled: 'Cache enabled', semantic_cache_threshold: 'Semantic cache threshold', max_history_messages: 'Maximum history messages', max_history_chars: 'Maximum history characters', request_timeout: 'Request timeout' });
  populateDynamicForm('guardrail-fields', 'guardrails', { strict_rag_mode: 'Strict RAG mode', allow_general_questions: 'Allow general questions', hallucination_protection: 'Hallucination protection', minimum_retrieval_score: 'Minimum retrieval score', out_of_scope_response: 'Out-of-scope response' });
}

function serialize(form) {
  const result = {};
  Array.from(form.elements).filter(el => el.name).forEach(el => { result[el.name] = el.type === 'checkbox' ? el.checked : (el.type === 'number' ? Number(el.value) : el.value); });
  return result;
}

async function loadAll() {
  const [config, providersResponse] = await Promise.all([api('/admin/config'), api('/admin/providers')]);
  state.config = config;
  state.providers = providersResponse.providers || [];
  const providers = state.providers;
  ['primary_provider', 'fallback_provider'].forEach(name => { const select = document.querySelector(`[name="${name}"]`); select.innerHTML = name === 'fallback_provider' ? '<option value="">None</option>' : ''; providers.forEach(p => { select.insertAdjacentHTML('beforeend', `<option value="${p.id}">${p.name}</option>`); }); select.value = state.config.models[name]; });
  const modelSelect = document.querySelector('[name="primary_model"]'); const primary = providers.find(p => p.id === state.config.models.primary_provider); modelSelect.innerHTML = (primary?.models || []).map(m => `<option>${m}</option>`).join(''); modelSelect.value = state.config.models.primary_model;
  fillForms(); renderActive(); await refreshHealth();
  await loadKBStatus();
}

function renderActive() { document.getElementById('active-config').innerHTML = `<div class="row"><div class="col-md-3"><strong>Model</strong><br>${state.config.models.primary_model}</div><div class="col-md-3"><strong>RAG</strong><br>Vector ${state.config.rag.vector_top_k} / BM25 ${state.config.rag.bm25_top_k} / Final ${state.config.rag.final_top_k}</div><div class="col-md-3"><strong>History</strong><br>${state.config.performance.max_history_messages} messages / ${state.config.performance.max_history_chars} chars</div><div class="col-md-3"><strong>Version</strong><br>${state.config.version}</div></div>`; }

async function refreshHealth() { const health = await api('/admin/health'); const cards = Object.entries(health).map(([name, item]) => `<div class="col-md-4 col-xl-2"><div class="card health-card"><div class="card-body"><div class="text-capitalize">${name.replaceAll('_', ' ')}</div><span class="badge text-bg-${item.status === 'healthy' ? 'success' : item.status === 'disabled' ? 'secondary' : 'warning'}">${item.status}</span><small class="d-block text-muted mt-2">${item.detail || ''}</small></div></div></div>`).join(''); document.getElementById('health-cards').innerHTML = cards; document.getElementById('header-status').textContent = Object.values(health).every(x => ['healthy', 'disabled'].includes(x.status)) ? 'System OK' : 'Attention needed'; }

// Drag & Drop / File Input setup
let selectedKBFile = null;

function setupKBFileHandling() {
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('kb-file-input');
  const label = document.getElementById('selected-file-label');
  const uploadBtn = document.getElementById('upload-rebuild-btn');

  if (!dropZone || !fileInput) return;

  dropZone.addEventListener('click', () => fileInput.click());

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('border-primary', 'bg-light');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('border-primary', 'bg-light');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('border-primary', 'bg-light');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files && fileInput.files.length > 0) {
      handleFileSelected(fileInput.files[0]);
    }
  });

  function handleFileSelected(file) {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      showAlert('Please select a valid PDF file.', 'danger');
      return;
    }
    selectedKBFile = file;
    label.textContent = `Selected: ${file.name} (${formatBytes(file.size)})`;
    uploadBtn.disabled = false;
  }

  uploadBtn.addEventListener('click', async () => {
    if (!selectedKBFile) return;
    if (!confirm('This replaces the current knowledge base. Do you want to proceed?')) {
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedKBFile);

    const alertEl = document.getElementById('kb-rebuild-alert');
    alertEl.classList.remove('d-none');
    alertEl.className = 'alert alert-info';
    alertEl.textContent = 'Uploading file and validating...';

    const spinner = document.getElementById('upload-spinner');
    const btnText = document.getElementById('upload-btn-text');
    uploadBtn.disabled = true;
    spinner.classList.remove('d-none');
    btnText.textContent = 'Uploading...';

    try {
      const response = await fetch('/admin/kb/upload', {
        method: 'POST',
        body: formData,
      });

      if (response.status === 401) {
        throw new Error('Admin session expired; authenticate again');
      }

      const data = await response.json().catch(() => ({}));

      if (response.status === 409) {
        throw new Error(data.detail || 'A rebuild is already in progress.');
      }

      if (!response.ok) {
        throw new Error(data.detail || `Upload failed (${response.status})`);
      }

      showAlert('PDF accepted. The assistant will switch to it after the index finishes building.');
      selectedKBFile = null;
      document.getElementById('kb-file-input').value = '';
      startKBPoll();
    } catch (err) {
      spinner.classList.add('d-none');
      btnText.textContent = 'Upload and Rebuild';
      uploadBtn.disabled = false;
      alertEl.className = 'alert alert-danger';
      alertEl.textContent = err.message;
      showAlert(err.message, 'danger');
    }
  });
}

document.addEventListener('click', async event => {
  const link = event.target.closest('.nav-link') || event.target.closest('#dashboard-manage-kb-link');
  if (link) {
    const targetHref = link.getAttribute('href');
    document.querySelectorAll('.admin-section').forEach(s => s.classList.toggle('d-none', `#${s.id}` !== targetHref));
    document.querySelectorAll('.nav-link').forEach(n => n.classList.toggle('active', n.getAttribute('href') === targetHref));
  }
  if (event.target.closest('#refresh-btn')) { try { await loadAll(); showAlert('Control Center refreshed.'); } catch (e) { showAlert(e.message, 'danger'); } }
  const save = event.target.closest('.save-section'); if (save) { const form = save.closest('form'); try { state.config = await api('/admin/config', { method: 'PUT', body: JSON.stringify({ [form.dataset.section]: serialize(form) }) }); fillForms(); renderActive(); showAlert('Configuration saved and applied.'); } catch (e) { showAlert(e.message, 'danger'); } }
  if (event.target.closest('#test-btn')) { try { const result = await api('/admin/config/test', { method: 'POST', body: JSON.stringify({ question: document.getElementById('test-question').value }) }); document.getElementById('test-result').textContent = JSON.stringify(result, null, 2); } catch (e) { showAlert(e.message, 'danger'); } }
});

setupKBFileHandling();
loadAll().catch(error => showAlert(error.message, 'danger'));
