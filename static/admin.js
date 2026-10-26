const state = { config: null, providers: [], kb: null, pollTimer: null, documents: [] };

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
  const area = document.getElementById('alert-area');
  if (!area) return;
  area.innerHTML = `<div class="alert alert-${type} alert-dismissible fade show" role="alert">${message}<button type="button" class="btn-close" data-bs-dismiss="alert"></button></div>`;
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function formatBytes(bytes) {
  if (!bytes || bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function getStatusBadge(status) {
  const map = {
    Published: '<span class="badge bg-success">Published</span>',
    Review: '<span class="badge bg-warning text-dark">In Review</span>',
    Approved: '<span class="badge bg-info text-dark">Approved</span>',
    Draft: '<span class="badge bg-secondary">Draft</span>',
    Disabled: '<span class="badge bg-dark">Disabled</span>',
    Archived: '<span class="badge bg-light text-dark border">Archived</span>',
  };
  return map[status] || `<span class="badge bg-secondary">${escapeHtml(status)}</span>`;
}

function getTypeBadge(type) {
  const map = {
    pdf: '<span class="badge bg-danger-subtle text-danger border border-danger-subtle"><i class="bi bi-file-earmark-pdf me-1"></i>PDF</span>',
    docx: '<span class="badge bg-primary-subtle text-primary border border-primary-subtle"><i class="bi bi-file-earmark-word me-1"></i>Word</span>',
    notice: '<span class="badge bg-warning-subtle text-dark border border-warning-subtle"><i class="bi bi-megaphone me-1"></i>Notice</span>',
    faq: '<span class="badge bg-success-subtle text-success border border-success-subtle"><i class="bi bi-patch-question me-1"></i>FAQ</span>',
    web_content: '<span class="badge bg-info-subtle text-info border border-info-subtle"><i class="bi bi-globe me-1"></i>Web</span>',
  };
  return map[type] || `<span class="badge bg-secondary">${escapeHtml(type)}</span>`;
}

// ============================================================================
// 1. KNOWLEDGE BASE DOCUMENTS REPOSITORY
// ============================================================================

async function loadDocuments() {
  const search = document.getElementById('kb-filter-search')?.value || '';
  const category = document.getElementById('kb-filter-category')?.value || '';
  const campus = document.getElementById('kb-filter-campus')?.value || '';
  const status = document.getElementById('kb-filter-status')?.value || '';

  const params = new URLSearchParams();
  if (search) params.append('search', search);
  if (category) params.append('category', category);
  if (campus) params.append('school_campus', campus);
  if (status) params.append('status', status);

  try {
    const docs = await api(`/admin/kb/documents?${params.toString()}`);
    state.documents = docs;
    renderDocumentsTable(docs);
    updateDashboardKBSummary(docs);
  } catch (err) {
    showAlert(`Failed to load knowledge base documents: ${err.message}`, 'danger');
  }
}

function renderDocumentsTable(docs) {
  const tbody = document.getElementById('documents-table-body');
  if (!tbody) return;

  if (docs.length === 0) {
    tbody.innerHTML = '<tr><td colspan="9" class="text-center py-4 text-muted">No documents found matching the filter criteria.</td></tr>';
    return;
  }

  tbody.innerHTML = docs.map(doc => {
    const validity = doc.effective_date
      ? `${doc.effective_date} ${doc.expiry_date ? 'to ' + doc.expiry_date : '(Active)'}`
      : (doc.expiry_date ? 'Until ' + doc.expiry_date : 'Continuous');

    let actionButtons = '';
    if (doc.status === 'Draft') {
      actionButtons = `
        <button class="btn btn-sm btn-outline-warning doc-action-btn" data-id="${doc.id}" data-action="submit_for_review" title="Submit to Review"><i class="bi bi-send"></i></button>
        <button class="btn btn-sm btn-outline-danger doc-delete-btn" data-id="${doc.id}" title="Delete"><i class="bi bi-trash"></i></button>
      `;
    } else if (doc.status === 'Review') {
      actionButtons = `
        <button class="btn btn-sm btn-success doc-action-btn" data-id="${doc.id}" data-action="publish" title="Approve & Publish Immediately"><i class="bi bi-check-circle"></i> Publish</button>
        <button class="btn btn-sm btn-outline-info doc-action-btn" data-id="${doc.id}" data-action="approve" title="Approve"><i class="bi bi-check"></i></button>
        <button class="btn btn-sm btn-outline-danger doc-reject-btn" data-id="${doc.id}" title="Request Changes"><i class="bi bi-x-circle"></i></button>
      `;
    } else if (doc.status === 'Approved') {
      actionButtons = `
        <button class="btn btn-sm btn-primary doc-action-btn" data-id="${doc.id}" data-action="publish" title="Publish to Live Assistant"><i class="bi bi-cloud-upload"></i> Publish</button>
        <button class="btn btn-sm btn-outline-secondary doc-action-btn" data-id="${doc.id}" data-action="disable" title="Disable"><i class="bi bi-pause"></i></button>
      `;
    } else if (doc.status === 'Published') {
      actionButtons = `
        <button class="btn btn-sm btn-outline-secondary doc-action-btn" data-id="${doc.id}" data-action="disable" title="Disable document retrieval"><i class="bi bi-pause"></i> Disable</button>
        <button class="btn btn-sm btn-outline-primary doc-replace-btn" data-id="${doc.id}" title="Upload new version"><i class="bi bi-arrow-repeat"></i> Replace</button>
      `;
    } else if (doc.status === 'Disabled') {
      actionButtons = `
        <button class="btn btn-sm btn-outline-success doc-action-btn" data-id="${doc.id}" data-action="publish" title="Re-enable and Publish"><i class="bi bi-play"></i> Enable</button>
        <button class="btn btn-sm btn-outline-danger doc-delete-btn" data-id="${doc.id}" title="Delete"><i class="bi bi-trash"></i></button>
      `;
    } else {
      actionButtons = `<button class="btn btn-sm btn-outline-danger doc-delete-btn" data-id="${doc.id}" title="Delete"><i class="bi bi-trash"></i></button>`;
    }

    const rejectionHtml = doc.rejection_reason
      ? `<div class="small text-danger mt-1"><i class="bi bi-info-circle me-1"></i>Feedback: ${escapeHtml(doc.rejection_reason)}</div>`
      : '';

    return `
      <tr class="doc-row">
        <td>
          <div class="fw-bold text-dark">${escapeHtml(doc.title)}</div>
          <small class="text-muted">${doc.file_name ? escapeHtml(doc.file_name) : (doc.source_url ? escapeHtml(doc.source_url) : 'Direct Content')}</small>
          ${rejectionHtml}
        </td>
        <td>${getTypeBadge(doc.content_type)}</td>
        <td><span class="badge text-bg-light border">${escapeHtml(doc.category)}</span></td>
        <td>
          <div>${escapeHtml(doc.school_campus)}</div>
          <small class="text-muted">${escapeHtml(doc.academic_year)}</small>
        </td>
        <td>
          <div>v${escapeHtml(doc.version)}</div>
          <small class="text-muted">${escapeHtml(doc.owner)}</small>
        </td>
        <td><small class="text-muted">${escapeHtml(validity)}</small></td>
        <td>${getStatusBadge(doc.status)}</td>
        <td><span class="badge bg-secondary-subtle text-secondary">${doc.chunks_count || 0}</span></td>
        <td class="text-end">
          <div class="btn-group btn-group-sm">${actionButtons}</div>
        </td>
      </tr>
    `;
  }).join('');
}

function updateDashboardKBSummary(docs) {
  let pub = 0, rev = 0, dft = 0;
  docs.forEach(d => {
    if (d.status === 'Published') pub++;
    else if (d.status === 'Review') rev++;
    else if (d.status === 'Draft') dft++;
  });
  const pubEl = document.getElementById('dash-published-count');
  const revEl = document.getElementById('dash-review-count');
  const dftEl = document.getElementById('dash-draft-count');
  if (pubEl) pubEl.textContent = pub;
  if (revEl) revEl.textContent = rev;
  if (dftEl) dftEl.textContent = dft;

  const appBadge = document.getElementById('approval-badge');
  if (appBadge) {
    appBadge.textContent = rev;
    appBadge.classList.toggle('d-none', rev === 0);
  }
}

// ============================================================================
// 2. APPROVAL QUEUE
// ============================================================================

async function loadApprovals() {
  try {
    const docs = await api('/admin/kb/documents?status=Review');
    const tbody = document.getElementById('approvals-table-body');
    if (!tbody) return;

    if (docs.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7" class="text-center py-4 text-muted"><i class="bi bi-check2-all text-success fs-4 d-block mb-1"></i>No pending documents in review queue.</td></tr>';
      return;
    }

    tbody.innerHTML = docs.map(doc => `
      <tr>
        <td>
          <div class="fw-bold">${escapeHtml(doc.title)}</div>
          <small class="text-muted">${getTypeBadge(doc.content_type)} | ${doc.file_name ? escapeHtml(doc.file_name) : 'Text Content'}</small>
        </td>
        <td><span class="badge text-bg-light border">${escapeHtml(doc.category)}</span></td>
        <td>${escapeHtml(doc.school_campus)} <br><small class="text-muted">${escapeHtml(doc.academic_year)}</small></td>
        <td>${escapeHtml(doc.owner)}</td>
        <td>v${escapeHtml(doc.version)}</td>
        <td><small class="text-muted">${new Date(doc.updated_at).toLocaleDateString()}</small></td>
        <td class="text-end">
          <div class="btn-group btn-group-sm">
            <button class="btn btn-success doc-action-btn" data-id="${doc.id}" data-action="publish" title="Approve & Publish to Assistant"><i class="bi bi-check-circle"></i> Approve & Publish</button>
            <button class="btn btn-outline-info doc-action-btn" data-id="${doc.id}" data-action="approve" title="Approve without publishing"><i class="bi bi-check"></i> Approve</button>
            <button class="btn btn-outline-danger doc-reject-btn" data-id="${doc.id}" title="Request changes"><i class="bi bi-x-circle"></i> Reject</button>
          </div>
        </td>
      </tr>
    `).join('');
  } catch (err) {
    showAlert(`Failed to load approval queue: ${err.message}`, 'danger');
  }
}


// ============================================================================
// 4. ACTION HANDLERS & MODAL FORMS
// ============================================================================

function setupActionHandlers() {
  // Workflow state action
  document.addEventListener('click', async event => {
    const actionBtn = event.target.closest('.doc-action-btn');
    if (actionBtn) {
      const docId = actionBtn.dataset.id;
      const action = actionBtn.dataset.action;
      try {
        actionBtn.disabled = true;
        await api(`/admin/kb/documents/${docId}/action`, {
          method: 'POST',
          body: JSON.stringify({ action: action, actor: 'Administrator' }),
        });
        showAlert(`Document updated successfully (${action}).`);
        await loadDocuments();
        await loadApprovals();
      } catch (err) {
        showAlert(err.message, 'danger');
      } finally {
        actionBtn.disabled = false;
      }
    }

    // Reject / request changes button
    const rejectBtn = event.target.closest('.doc-reject-btn');
    if (rejectBtn) {
      const docId = rejectBtn.dataset.id;
      document.getElementById('rejection-doc-id').value = docId;
      document.getElementById('rejection-comment').value = '';
      const modal = new bootstrap.Modal(document.getElementById('rejectionModal'));
      modal.show();
    }

    // Delete document button
    const deleteBtn = event.target.closest('.doc-delete-btn');
    if (deleteBtn) {
      const docId = deleteBtn.dataset.id;
      if (confirm('Are you sure you want to delete this document? Any indexed chunks will be purged immediately.')) {
        try {
          deleteBtn.disabled = true;
          await api(`/admin/kb/documents/${docId}`, { method: 'DELETE' });
          showAlert('Document deleted.');
          await loadDocuments();
          await loadApprovals();
        } catch (err) {
          showAlert(err.message, 'danger');
        }
      }
    }

    // Replace document file
    const replaceBtn = event.target.closest('.doc-replace-btn');
    if (replaceBtn) {
      const docId = replaceBtn.dataset.id;
      document.getElementById('replace-doc-id').value = docId;
      const modal = new bootstrap.Modal(document.getElementById('replaceModal'));
      modal.show();
    }


    // Sync all indexes button
    const syncBtn = event.target.closest('#sync-indexes-btn');
    if (syncBtn) {
      try {
        syncBtn.disabled = true;
        syncBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Syncing...';
        const res = await api('/admin/kb/sync-all', { method: 'POST' });
        showAlert(`Search indexes synchronized. Total active chunks: ${res.total_indexed_chunks}`);
        await loadDocuments();
      } catch (err) {
        showAlert(err.message, 'danger');
      } finally {
        syncBtn.disabled = false;
        syncBtn.innerHTML = '<i class="bi bi-arrow-repeat"></i> Sync Indexes';
      }
    }

    // Filter resets
    if (event.target.closest('#kb-reset-filters-btn')) {
      document.getElementById('kb-filter-search').value = '';
      document.getElementById('kb-filter-category').value = '';
      document.getElementById('kb-filter-campus').value = '';
      document.getElementById('kb-filter-status').value = '';
      loadDocuments();
    }
  });

  // Filter input listeners
  ['kb-filter-search', 'kb-filter-category', 'kb-filter-campus', 'kb-filter-status'].forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener('change', () => loadDocuments());
      if (el.tagName === 'INPUT') {
        el.addEventListener('input', () => {
          clearTimeout(window._searchTimer);
          window._searchTimer = setTimeout(() => loadDocuments(), 300);
        });
      }
    }
  });

  // File Upload Form
  const fileForm = document.getElementById('file-upload-form');
  if (fileForm) {
    fileForm.addEventListener('submit', async e => {
      e.preventDefault();
      const fileInput = document.getElementById('form-file-input');
      if (!fileInput.files || fileInput.files.length === 0) return;

      const formData = new FormData();
      formData.append('file', fileInput.files[0]);
      formData.append('title', document.getElementById('form-file-title').value);
      formData.append('category', document.getElementById('form-file-category').value);
      formData.append('school_campus', document.getElementById('form-file-campus').value);
      formData.append('academic_year', document.getElementById('form-file-year').value);
      formData.append('language', document.getElementById('form-file-language').value);
      formData.append('owner', document.getElementById('form-file-owner').value);
      formData.append('version', document.getElementById('form-file-version').value);
      formData.append('effective_date', document.getElementById('form-file-effective').value || '');
      formData.append('expiry_date', document.getElementById('form-file-expiry').value || '');
      formData.append('submit_for_review', document.getElementById('form-file-review').checked);

      const spinner = document.getElementById('file-spinner');
      const submitBtn = document.getElementById('file-submit-btn');
      try {
        spinner.classList.remove('d-none');
        submitBtn.disabled = true;

        const response = await fetch('/admin/kb/documents/file', { method: 'POST', body: formData });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.detail || `Upload failed (${response.status})`);

        showAlert(`Document "${data.title}" uploaded successfully with status: ${data.status}.`);
        bootstrap.Modal.getInstance(document.getElementById('ingestionModal')).hide();
        fileForm.reset();
        await loadDocuments();
        await loadApprovals();
      } catch (err) {
        showAlert(err.message, 'danger');
      } finally {
        spinner.classList.add('d-none');
        submitBtn.disabled = false;
      }
    });
  }

  // Notice Form
  const noticeForm = document.getElementById('notice-form');
  if (noticeForm) {
    noticeForm.addEventListener('submit', async e => {
      e.preventDefault();
      try {
        const payload = {
          title: document.getElementById('form-notice-title').value,
          content_type: 'notice',
          category: document.getElementById('form-notice-category').value,
          school_campus: document.getElementById('form-notice-campus').value,
          academic_year: document.getElementById('form-notice-year').value,
          content_text: document.getElementById('form-notice-text').value,
          effective_date: document.getElementById('form-notice-effective').value || null,
          expiry_date: document.getElementById('form-notice-expiry').value || null,
          submit_for_review: document.getElementById('form-notice-review').checked,
        };
        const doc = await api('/admin/kb/documents/text', { method: 'POST', body: JSON.stringify(payload) });
        showAlert(`Notice "${doc.title}" created with status: ${doc.status}.`);
        bootstrap.Modal.getInstance(document.getElementById('ingestionModal')).hide();
        noticeForm.reset();
        await loadDocuments();
        await loadApprovals();
      } catch (err) {
        showAlert(err.message, 'danger');
      }
    });
  }

  // FAQ Form
  const faqForm = document.getElementById('faq-form');
  if (faqForm) {
    faqForm.addEventListener('submit', async e => {
      e.preventDefault();
      try {
        const q = document.getElementById('form-faq-question').value;
        const a = document.getElementById('form-faq-answer').value;
        const payload = {
          title: `FAQ: ${q.substring(0, 80)}`,
          content_type: 'faq',
          category: document.getElementById('form-faq-category').value,
          school_campus: document.getElementById('form-faq-campus').value,
          academic_year: '2026-2027',
          content_text: `Q: ${q}\nA: ${a}`,
          submit_for_review: document.getElementById('form-faq-review').checked,
        };
        const doc = await api('/admin/kb/documents/text', { method: 'POST', body: JSON.stringify(payload) });
        showAlert(`FAQ saved successfully with status: ${doc.status}.`);
        bootstrap.Modal.getInstance(document.getElementById('ingestionModal')).hide();
        faqForm.reset();
        await loadDocuments();
        await loadApprovals();
      } catch (err) {
        showAlert(err.message, 'danger');
      }
    });
  }

  // Web Content Form
  const webForm = document.getElementById('web-form');
  if (webForm) {
    webForm.addEventListener('submit', async e => {
      e.preventDefault();
      try {
        const payload = {
          title: document.getElementById('form-web-title').value,
          content_type: 'web_content',
          category: document.getElementById('form-web-category').value,
          source_url: document.getElementById('form-web-url').value || null,
          content_text: document.getElementById('form-web-text').value || null,
          submit_for_review: document.getElementById('form-web-review').checked,
        };
        const doc = await api('/admin/kb/documents/text', { method: 'POST', body: JSON.stringify(payload) });
        showAlert(`Web content "${doc.title}" saved.`);
        bootstrap.Modal.getInstance(document.getElementById('ingestionModal')).hide();
        webForm.reset();
        await loadDocuments();
        await loadApprovals();
      } catch (err) {
        showAlert(err.message, 'danger');
      }
    });
  }

  // Rejection Feedback Form
  const rejForm = document.getElementById('rejection-form');
  if (rejForm) {
    rejForm.addEventListener('submit', async e => {
      e.preventDefault();
      const docId = document.getElementById('rejection-doc-id').value;
      const comment = document.getElementById('rejection-comment').value;
      try {
        await api(`/admin/kb/documents/${docId}/action`, {
          method: 'POST',
          body: JSON.stringify({ action: 'reject', comment: comment, actor: 'Administrator' }),
        });
        showAlert('Document returned to Draft with reviewer feedback.');
        bootstrap.Modal.getInstance(document.getElementById('rejectionModal')).hide();
        await loadDocuments();
        await loadApprovals();
      } catch (err) {
        showAlert(err.message, 'danger');
      }
    });
  }


  // Replace Document File Form
  const repForm = document.getElementById('replace-file-form');
  if (repForm) {
    repForm.addEventListener('submit', async e => {
      e.preventDefault();
      const docId = document.getElementById('replace-doc-id').value;
      const fileInput = document.getElementById('replace-file-input');
      const versionInput = document.getElementById('replace-version-input').value;

      if (!fileInput.files || fileInput.files.length === 0) return;

      const formData = new FormData();
      formData.append('file', fileInput.files[0]);
      formData.append('new_version', versionInput);

      const spinner = document.getElementById('replace-spinner');
      const submitBtn = document.getElementById('replace-submit-btn');
      try {
        spinner.classList.remove('d-none');
        submitBtn.disabled = true;

        const response = await fetch(`/admin/kb/documents/${docId}/replace`, {
          method: 'POST',
          body: formData,
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.detail || `Replace failed (${response.status})`);

        showAlert(`Document updated to version ${data.version}. Re-indexing applied.`);
        bootstrap.Modal.getInstance(document.getElementById('replaceModal')).hide();
        repForm.reset();
        await loadDocuments();
      } catch (err) {
        showAlert(err.message, 'danger');
      } finally {
        spinner.classList.add('d-none');
        submitBtn.disabled = false;
      }
    });
  }
}

// ============================================================================
// 5. EXISTING CONFIG FORMS & HEALTH MANAGEMENT
// ============================================================================

function renderKBActive(targetId, active, rebuild) {
  const container = document.getElementById(targetId);
  if (!container) return;
  container.innerHTML = `
    <div class="row g-2">
      <div class="col-md-6"><strong>File:</strong> <span class="text-break">${escapeHtml(active.display_name || 'None')}</span></div>
      <div class="col-md-3"><strong>Size:</strong> ${formatBytes(active.size_bytes)}</div>
      <div class="col-md-3"><strong>Pages:</strong> ${active.pages || 0}</div>
      <div class="col-md-4"><strong>Chunks:</strong> ${active.chunks || 0}</div>
      <div class="col-md-8"><strong>Uploaded:</strong> ${active.uploaded_at ? new Date(active.uploaded_at).toLocaleString() : 'Pre-loaded'}</div>
      <div class="col-12 mt-2 pt-2 border-top">
        <strong>Rebuild State:</strong> <span class="badge text-bg-${rebuild.state === 'done' ? 'success' : (rebuild.state === 'failed' ? 'danger' : (rebuild.state === 'idle' ? 'secondary' : 'warning'))}">${rebuild.state}</span>
        <small class="text-muted ms-2">${escapeHtml(rebuild.message || '')}</small>
      </div>
    </div>
  `;
}

async function loadKBStatus() {
  try {
    const kb = await api('/admin/kb/status');
    state.kb = kb;
    renderKBActive('active-kb-details', kb.active, kb.rebuild);
    renderKBActive('dashboard-kb-card', kb.active, kb.rebuild);
  } catch (err) {}
}

function input(label, name, value, type = 'text') {
  if (type === 'checkbox') {
    return `<div class="form-check form-switch"><input class="form-check-input" name="${name}" type="checkbox" ${value ? 'checked' : ''}><label class="form-check-label">${label}</label></div>`;
  }
  return `<label class="form-label">${label}</label><input class="form-control" name="${name}" type="${type}" value="${value ?? ''}">`;
}

function populateDynamicForm(formId, section, labels) {
  const target = document.getElementById(formId);
  if (!target || !state.config || !state.config[section]) return;
  target.innerHTML = Object.entries(labels).map(([name, label]) => {
    const value = state.config[section][name];
    const type = typeof value === 'boolean' ? 'checkbox' : (typeof value === 'number' ? 'number' : 'text');
    return type === 'checkbox' ? input(label, name, value, type) : `<div class="col-md-6">${input(label, name, value, type)}</div>`;
  }).join('');
}

function populatePrompts() {
  if (!state.config || !state.config.prompts) return;
  const prompts = state.config.prompts;
  const labels = { system_prompt: 'System prompt', rag_prompt: 'RAG prompt', no_answer_prompt: 'No-answer prompt', greeting_prompt: 'Greeting prompt', context_instructions: 'Context instructions' };
  const formEl = document.getElementById('prompt-form');
  if (formEl) {
    formEl.innerHTML = Object.entries(labels).map(([name, label]) => `<div class="mb-3"><label class="form-label">${label} <span class="text-muted">${prompts[name].length} chars</span></label><textarea class="form-control" name="${name}" maxlength="20000">${escapeHtml(prompts[name])}</textarea></div>`).join('') + '<button class="btn btn-primary save-section">Save prompts</button>';
  }
}

function fillForms() {
  if (!state.config) return;
  const modelForm = document.querySelector('[data-section="models"]');
  if (modelForm) {
    Object.entries(state.config.models).forEach(([name, value]) => {
      const el = modelForm.elements[name];
      if (el) el.type === 'checkbox' ? el.checked = value : el.value = value;
    });
  }
  populatePrompts();
  populateDynamicForm('rag-fields', 'rag', { vector_top_k: 'Vector top-k', bm25_top_k: 'BM25 top-k', final_top_k: 'Final top-k', similarity_threshold: 'Similarity threshold', chunk_size: 'Chunk size', chunk_overlap: 'Chunk overlap', retrieval_strategy: 'Retrieval strategy' });
  populateDynamicForm('performance-fields', 'performance', { cache_enabled: 'Cache enabled', semantic_cache_threshold: 'Semantic cache threshold', max_history_messages: 'Maximum history messages', max_history_chars: 'Maximum history characters', request_timeout: 'Request timeout' });
  populateDynamicForm('guardrail-fields', 'guardrails', { strict_rag_mode: 'Strict RAG mode', allow_general_questions: 'Allow general questions', hallucination_protection: 'Hallucination protection', minimum_retrieval_score: 'Minimum retrieval score', out_of_scope_response: 'Out-of-scope response' });
}

function serialize(form) {
  const result = {};
  Array.from(form.elements).filter(el => el.name).forEach(el => {
    result[el.name] = el.type === 'checkbox' ? el.checked : (el.type === 'number' ? Number(el.value) : el.value);
  });
  return result;
}

function renderActive() {
  const act = document.getElementById('active-config');
  if (!act || !state.config) return;
  act.innerHTML = `
    <div class="row">
      <div class="col-md-3"><strong>Primary Model</strong><br>${escapeHtml(state.config.models.primary_model)}</div>
      <div class="col-md-3"><strong>RAG Settings</strong><br>Vector ${state.config.rag.vector_top_k} / BM25 ${state.config.rag.bm25_top_k} / Final ${state.config.rag.final_top_k}</div>
      <div class="col-md-3"><strong>History Limit</strong><br>${state.config.performance.max_history_messages} msgs / ${state.config.performance.max_history_chars} chars</div>
      <div class="col-md-3"><strong>Config Version</strong><br>v${state.config.version}</div>
    </div>
  `;
}

async function refreshHealth() {
  try {
    const health = await api('/admin/health');
    const cards = Object.entries(health).map(([name, item]) => `
      <div class="col-md-4 col-xl-2">
        <div class="card health-card border-0 shadow-sm">
          <div class="card-body">
            <div class="text-capitalize small fw-semibold text-muted">${name.replaceAll('_', ' ')}</div>
            <span class="badge text-bg-${item.status === 'healthy' ? 'success' : (item.status === 'disabled' ? 'secondary' : 'warning')} mt-1">${item.status}</span>
            <small class="d-block text-muted mt-2 text-truncate" title="${escapeHtml(item.detail || '')}">${escapeHtml(item.detail || '')}</small>
          </div>
        </div>
      </div>
    `).join('');
    document.getElementById('health-cards').innerHTML = cards;
    document.getElementById('header-status').textContent = Object.values(health).every(x => ['healthy', 'disabled'].includes(x.status)) ? 'System OK' : 'Attention needed';
  } catch (err) {}
}

async function loadAll() {
  const [config, providersResponse] = await Promise.all([api('/admin/config'), api('/admin/providers')]);
  state.config = config;
  state.providers = providersResponse.providers || [];

  const providers = state.providers;
  ['primary_provider', 'fallback_provider'].forEach(name => {
    const select = document.querySelector(`[name="${name}"]`);
    if (select) {
      select.innerHTML = name === 'fallback_provider' ? '<option value="">None</option>' : '';
      providers.forEach(p => { select.insertAdjacentHTML('beforeend', `<option value="${p.id}">${escapeHtml(p.name)}</option>`); });
      select.value = state.config.models[name];
    }
  });

  const modelSelect = document.querySelector('[name="primary_model"]');
  if (modelSelect) {
    const primary = providers.find(p => p.id === state.config.models.primary_provider);
    modelSelect.innerHTML = (primary?.models || []).map(m => `<option>${escapeHtml(m)}</option>`).join('');
    modelSelect.value = state.config.models.primary_model;
  }

  fillForms();
  renderActive();
  await refreshHealth();
  await loadKBStatus();
  await loadDocuments();
  await loadApprovals();
}

// Navigation & Click Listeners
document.addEventListener('click', async event => {
  const link = event.target.closest('.nav-link') || event.target.closest('a[href^="#"]');
  if (link && link.getAttribute('href') && link.getAttribute('href').startsWith('#')) {
    const targetHref = link.getAttribute('href');
    if (['#dashboard', '#kb', '#approval', '#models', '#prompts', '#rag', '#performance', '#guardrails', '#system'].includes(targetHref)) {
      event.preventDefault();
      document.querySelectorAll('.admin-section').forEach(s => s.classList.toggle('d-none', `#${s.id}` !== targetHref));
      document.querySelectorAll('.nav-link').forEach(n => n.classList.toggle('active', n.getAttribute('href') === targetHref));
      window.location.hash = targetHref;

      if (targetHref === '#kb') await loadDocuments();
      if (targetHref === '#approval') await loadApprovals();
    }
  }

  if (event.target.closest('#refresh-btn')) {
    try {
      await loadAll();
      showAlert('Control Center data refreshed.');
    } catch (e) {
      showAlert(e.message, 'danger');
    }
  }
  if (event.target.closest('#refresh-approvals-btn')) await loadApprovals();

  const save = event.target.closest('.save-section');
  if (save) {
    const form = save.closest('form');
    try {
      state.config = await api('/admin/config', {
        method: 'PUT',
        body: JSON.stringify({ [form.dataset.section]: serialize(form) }),
      });
      fillForms();
      renderActive();
      showAlert('Configuration saved and applied.');
    } catch (e) {
      showAlert(e.message, 'danger');
    }
  }

  if (event.target.closest('#test-btn')) {
    try {
      const result = await api('/admin/config/test', {
        method: 'POST',
        body: JSON.stringify({ question: document.getElementById('test-question').value }),
      });
      document.getElementById('test-result').textContent = JSON.stringify(result, null, 2);
    } catch (e) {
      showAlert(e.message, 'danger');
    }
  }
});

// Setup on load
setupActionHandlers();
loadAll().catch(err => showAlert(err.message, 'danger'));
