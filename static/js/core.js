// Core helpers shared by every page: API, DOM, icons, toasts, sheets, pickers.

// ───────────────────────── API ─────────────────────────
async function parse(r, what) {
  let body = null;
  const text = await r.text();
  try { body = text ? JSON.parse(text) : null; } catch { body = text; }
  if (!r.ok) {
    let msg = (body && body.detail) || (typeof body === 'string' && body) || `${what} failed (${r.status})`;
    if (typeof msg !== 'string') msg = JSON.stringify(msg);
    throw new Error(msg.slice(0, 400));
  }
  return body;
}
export const api = {
  get: (url) => fetch(url).then((r) => parse(r, 'GET ' + url)),
  post: (url, data) => fetch(url, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data ?? {}),
  }).then((r) => parse(r, 'POST ' + url)),
  form: (url, fd) => fetch(url, { method: 'POST', body: fd }).then((r) => parse(r, 'POST ' + url)),
};

export const qs = (o) => Object.entries(o)
  .filter(([, v]) => v !== undefined && v !== null && v !== '')
  .map(([k, v]) => encodeURIComponent(k) + '=' + encodeURIComponent(v)).join('&');

// Domain calls
export const Parts = {
  search: (q, limit = 10, offset = 0) => api.get('/api/parts?' + qs({ q, limit, offset })),
  detail: (pk) => api.get('/api/part/detail?pk=' + encodeURIComponent(pk)),
  byIpn: (ipn) => api.get('/api/part/by-ipn?ipn=' + encodeURIComponent(ipn)),
  update: (pk, data) => api.post('/api/part/update-meta', { pk: Number(pk), ...data }),
  create: (data) => {
    const fd = new FormData();
    Object.entries(data).forEach(([k, v]) => { if (v !== undefined && v !== null && v !== '') fd.append(k, v); });
    return api.form('/api/part/create', fd);
  },
  uploadImage: (pk, file) => {
    const fd = new FormData(); fd.append('part_id', String(pk)); fd.append('image', file);
    return api.form('/api/upload', fd);
  },
  imgUrl: (pk, thumb = false, bust = '') => `/api/part/image-proxy?pk=${pk}${thumb ? '&thumb=true' : ''}${bust ? '&_=' + bust : ''}`,
};
export const Stock = {
  byPart: (pk) => api.get('/api/stock/by-part?part_id=' + pk),
  add: (id, quantity, notes) => api.post('/api/stock/add_qty', { stock_id: Number(id), quantity, notes }),
  remove: (id, quantity, notes) => api.post('/api/stock/remove_qty', { stock_id: Number(id), quantity, notes }),
  transfer: (id, location_id, quantity, notes) => api.post('/api/stock/transfer', { stock_id: Number(id), location_id: Number(location_id), quantity, notes: notes || undefined }),
  create: (part_id, quantity, location_id, notes) => api.post('/api/stock/create', { part_id: Number(part_id), quantity, location_id: Number(location_id), notes: notes || undefined }),
};
export const Locations = { search: (q, limit = 12) => api.get('/api/locations?' + qs({ q, limit })) };

// ───────────────────────── Storage ─────────────────────────
export const store = {
  get(k, d = null) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* ignore */ } },
  // raw string values (kept compatible with the v1 app's keys)
  raw(k, d = '') { try { const v = localStorage.getItem(k); return v === null ? d : v; } catch { return d; } },
  setRaw(k, v) { try { localStorage.setItem(k, String(v)); } catch { /* ignore */ } },
};
export const settings = {
  get printer() { return store.raw('hw_label_printer', ''); },
  set printer(v) { store.setRaw('hw_label_printer', v); },
  get cutMargin() { return parseFloat(store.raw('hw_cut_margin', '0')) || 0; },
  set cutMargin(v) { store.setRaw('hw_cut_margin', v); },
  get partTemplate() { return Number(store.raw('default_part_template', '11')) || 11; },
  set partTemplate(v) { store.setRaw('default_part_template', v); },
  get autoImageSearch() { return store.get('auto_image_search', true); },
  set autoImageSearch(v) { store.set('auto_image_search', !!v); },
  get askPrintAfterCreate() { return store.get('ask_print_after_create', true); },
  set askPrintAfterCreate(v) { store.set('ask_print_after_create', !!v); },
};
export function rememberPart(p) {
  if (!p || !p.pk) return;
  const list = store.get('recent_parts', []).filter((x) => x.pk !== p.pk);
  list.unshift({ pk: p.pk, name: p.name, ipn: p.ipn || '', has_image: p.has_image !== false, t: Date.now() });
  store.set('recent_parts', list.slice(0, 12));
}
export const recentParts = () => store.get('recent_parts', []);

// ───────────────────────── DOM ─────────────────────────
export const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (m) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[m]));
class Raw { constructor(s) { this.s = s; } toString() { return this.s; } }
export const raw = (s) => new Raw(s);
// Tagged template: interpolations are escaped unless wrapped in raw() or are arrays of raw.
export function html(strings, ...vals) {
  let out = '';
  strings.forEach((s, i) => {
    out += s;
    if (i < vals.length) {
      const v = vals[i];
      if (v instanceof Raw) out += v.s;
      else if (Array.isArray(v)) out += v.map((x) => (x instanceof Raw ? x.s : esc(x))).join('');
      else if (v === false || v === null || v === undefined) out += '';
      else out += esc(v);
    }
  });
  return raw(out);
}
export function el(markup) {
  const t = document.createElement('template');
  t.innerHTML = String(markup).trim();
  return t.content.firstElementChild;
}
export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
export const debounce = (fn, ms = 200) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
export const fmtQty = (q) => {
  const n = Number(q);
  if (!Number.isFinite(n)) return '—';
  return Number.isInteger(n) ? String(n) : String(+n.toFixed(3));
};
export const num = (v) => { const n = parseFloat(String(v ?? '').replace(',', '.')); return Number.isFinite(n) ? n : NaN; };
export const partLabel = (p) => (p.name || 'Unnamed') + (p.ipn ? ` · ${p.ipn}` : '');

// ───────────────────────── Icons ─────────────────────────
const P = {
  home: '<path d="M3 10.5 12 3l9 7.5"/><path d="M5 9.5V21h14V9.5"/><path d="M10 21v-6h4v6"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
  scan: '<path d="M3 7V5a2 2 0 0 1 2-2h2M17 3h2a2 2 0 0 1 2 2v2M21 17v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2"/><rect x="7" y="7" width="4" height="4" rx=".5"/><rect x="13" y="13" width="4" height="4" rx=".5"/><path d="M13 7h4v3M7 13v4h3"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  minus: '<path d="M5 12h14"/>',
  tag: '<path d="M20.6 13.4 13.4 20.6a2 2 0 0 1-2.8 0L3 13V3h10l7.6 7.6a2 2 0 0 1 0 2.8z"/><circle cx="7.5" cy="7.5" r="1.5"/>',
  sparkles: '<path d="M11 3l1.9 5.1L18 10l-5.1 1.9L11 17l-1.9-5.1L4 10l5.1-1.9z"/><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z"/>',
  settings: '<path d="M4 6h9M17 6h3M4 12h3M11 12h9M4 18h11M19 18h1"/><circle cx="15" cy="6" r="2"/><circle cx="9" cy="12" r="2"/><circle cx="17" cy="18" r="2"/>',
  box: '<path d="M21 8 12 3 3 8v8l9 5 9-5z"/><path d="m3 8 9 5 9-5M12 13v8"/>',
  back: '<path d="m15 18-6-6 6-6"/>',
  chev: '<path d="m9 18 6-6-6-6"/>',
  x: '<path d="M18 6 6 18M6 6l12 12"/>',
  camera: '<path d="M3 8a2 2 0 0 1 2-2h2l2-2h6l2 2h2a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><circle cx="12" cy="13" r="3.5"/>',
  image: '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-5-5L5 21"/>',
  edit: '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/>',
  printer: '<path d="M6 9V3h12v6"/><rect x="2" y="9" width="20" height="9" rx="2"/><path d="M6 14h12v7H6z"/>',
  move: '<path d="M8 3 4 7l4 4M4 7h16M16 21l4-4-4-4M20 17H4"/>',
  check: '<path d="M20 6 9 17l-5-5"/>',
  grid: '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
  list: '<path d="M8 6h13M8 12h13M8 18h13M3.5 6h.01M3.5 12h.01M3.5 18h.01"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  moon: '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>',
  download: '<path d="M12 3v12M7 10l5 5 5-5M5 21h14"/>',
  upload: '<path d="M12 21V9M7 14l5-5 5 5M5 3h14"/>',
  refresh: '<path d="M21 12a9 9 0 1 1-3-6.7L21 8"/><path d="M21 3v5h-5"/>',
  link: '<path d="M10 13a5 5 0 0 0 7.5.5l3-3a5 5 0 0 0-7-7l-1.7 1.7"/><path d="M14 11a5 5 0 0 0-7.5-.5l-3 3a5 5 0 0 0 7 7l1.7-1.7"/>',
  trash: '<path d="M3 6h18M8 6V4h8v2M6 6l1 15h10l1-15"/>',
  pin: '<path d="M12 21s-7-6.2-7-12a7 7 0 0 1 14 0c0 5.8-7 12-7 12z"/><circle cx="12" cy="9" r="2.5"/>',
  wand: '<path d="M15 4V2M15 16v-2M8 9h2M20 9h2M17.8 11.8 19 13M17.8 6.2 19 5M3 21l9-9M12.2 6.2 11 5"/>',
  plug: '<path d="M9 2v6M15 2v6M6 8h12v4a6 6 0 0 1-12 0zM12 18v4"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 16v-4M12 8h.01"/>',
  layers: '<path d="m12 3 9 5-9 5-9-5z"/><path d="m3 13 9 5 9-5"/>',
  hash: '<path d="M4 9h16M4 15h16M10 3 8 21M16 3l-2 18"/>',
  globe: '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/>',
  more: '<circle cx="5" cy="12" r="1.3"/><circle cx="12" cy="12" r="1.3"/><circle cx="19" cy="12" r="1.3"/>',
};
export const icon = (name, cls = '') => raw(`<svg class="icon ${cls}" viewBox="0 0 24 24" aria-hidden="true">${P[name] || ''}</svg>`);

// Thumbnail element markup with graceful fallback
export function thumb(p, cls = '', bust = '') {
  if (p && p.pk && p.has_image !== false) {
    return html`<div class="thumb ${cls}"><img loading="lazy" alt="" src="${Parts.imgUrl(p.pk, true, bust)}" onerror="this.replaceWith(Object.assign(document.createElement('span'),{className:'faint',innerHTML:'&#9633;'}))"></div>`;
  }
  return html`<div class="thumb ${cls}">${icon('box')}</div>`;
}

// ───────────────────────── Toasts ─────────────────────────
let toastHost;
export function toast(msg, type = 'ok', ms = 2800) {
  if (!toastHost) { toastHost = el('<div class="toasts" aria-live="polite"></div>'); document.body.appendChild(toastHost); }
  const t = el(html`<div class="toast ${type}">${type === 'ok' ? icon('check') : type === 'err' ? icon('info') : ''}<span>${msg}</span></div>`);
  toastHost.appendChild(t);
  setTimeout(() => { t.classList.add('out'); setTimeout(() => t.remove(), 220); }, type === 'err' ? Math.max(ms, 5000) : ms);
}
export const toastErr = (e) => toast(e && e.message ? e.message : String(e), 'err');

// Inline status line
export function status(node, msg, type = '') {
  if (!node) return;
  node.className = 'status' + (type ? ' ' + type : '');
  node.textContent = msg || '';
}

// Button busy wrapper
export async function busy(btn, fn) {
  if (!btn) return fn();
  btn.classList.add('busy'); btn.disabled = true;
  try { return await fn(); } finally { btn.classList.remove('busy'); btn.disabled = false; }
}

// ───────────────────────── Sheets ─────────────────────────
const openSheets = [];
export function sheet({ title, body, foot, wide = false, onClose } = {}) {
  const ov = el(html`<div class="overlay" role="dialog" aria-modal="true">
    <div class="sheet ${wide ? 'wide' : ''}">
      <div class="sheet-head"><h2>${title || ''}</h2><button class="iconbtn" data-close aria-label="Close">${icon('x')}</button></div>
      <div class="sheet-body"></div>
      ${foot ? raw('<div class="sheet-foot"></div>') : ''}
    </div></div>`);
  const bodyEl = $('.sheet-body', ov);
  if (body instanceof Node) bodyEl.appendChild(body); else if (body) bodyEl.innerHTML = String(body);
  const footEl = $('.sheet-foot', ov);
  if (foot && footEl) { if (foot instanceof Node) footEl.appendChild(foot); else footEl.innerHTML = String(foot); }
  let closed = false;
  const api_ = {
    el: ov, body: bodyEl, foot: footEl,
    close(result) {
      if (closed) return; closed = true;
      ov.remove(); openSheets.splice(openSheets.indexOf(api_), 1);
      if (!openSheets.length) document.body.style.overflow = '';
      onClose && onClose(result);
    },
  };
  ov.addEventListener('mousedown', (e) => { if (e.target === ov) api_.close(); });
  $('[data-close]', ov).onclick = () => api_.close();
  document.body.appendChild(ov);
  document.body.style.overflow = 'hidden';
  openSheets.push(api_);
  setTimeout(() => {
    if (ov.contains(document.activeElement)) return; // user already focused something
    const f = $('input:not([type=checkbox]):not([type=file]),textarea', bodyEl);
    if (f && window.matchMedia('(min-width: 641px)').matches) f.focus();
  }, 60);
  return api_;
}
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    const lb = $('.lightbox'); if (lb) { lb.remove(); return; }
    const top = openSheets[openSheets.length - 1]; if (top) top.close();
  }
});
export const closeAllSheets = () => [...openSheets].forEach((s) => s.close());

export function confirmDialog({ title, message, ok = 'OK', cancel = 'Cancel', danger = false }) {
  return new Promise((resolve) => {
    const foot = el('<div style="display:contents"></div>');
    const s = sheet({ title, body: html`<p class="muted" style="margin:0">${message}</p>`, foot, onClose: (r) => resolve(!!r) });
    foot.innerHTML = String(html`<button class="btn" data-no>${cancel}</button><button class="btn ${danger ? 'danger' : 'primary'}" data-yes>${ok}</button>`);
    $('[data-no]', foot).onclick = () => s.close(false);
    $('[data-yes]', foot).onclick = () => s.close(true);
  });
}

export function lightbox(url) {
  if (!url) return;
  const lb = el(html`<div class="lightbox"><img alt="" src="${url}"></div>`);
  lb.onclick = () => lb.remove();
  document.body.appendChild(lb);
}

// ───────────────────────── Typeahead ─────────────────────────
// Attach a dropdown to an <input> inside a .field (position:relative).
export function typeahead(input, { fetch: fetcher, render, onPick, minChars = 1, delay = 180 }) {
  const host = input.parentElement;
  let list = null; let items = []; let hl = -1; let seq = 0;
  const close = () => { if (list) { list.remove(); list = null; } hl = -1; };
  const show = (rows) => {
    items = rows;
    if (!list) { list = el('<div class="ta-list" role="listbox"></div>'); host.appendChild(list); }
    list.innerHTML = rows.length
      ? rows.map((r, i) => `<div class="ta-item" data-i="${i}">${render(r)}</div>`).join('')
      : '<div class="ta-empty">No matches</div>';
  };
  const run = debounce(async () => {
    const q = input.value.trim();
    if (q.length < minChars) { close(); return; }
    const my = ++seq;
    try { const rows = await fetcher(q); if (my === seq && document.activeElement === input) show(rows || []); } catch (e) { if (my === seq) { show([]); list.innerHTML = `<div class="ta-empty" style="color:var(--err)">${esc(e.message)}</div>`; } }
  }, delay);
  const pick = (i) => { const r = items[i]; if (!r) return; close(); onPick(r); };
  input.setAttribute('autocomplete', 'off');
  input.addEventListener('input', run);
  input.addEventListener('focus', () => { if (input.value.trim().length >= minChars) run(); });
  input.addEventListener('keydown', (e) => {
    if (!list || !items.length) return;
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      hl = (hl + (e.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length;
      $$('.ta-item', list).forEach((n, i) => n.classList.toggle('hl', i === hl));
    } else if (e.key === 'Enter' && hl >= 0) { e.preventDefault(); e.stopPropagation(); pick(hl); }
  });
  host.addEventListener('mousedown', (e) => { const it = e.target.closest('.ta-item'); if (it) { e.preventDefault(); pick(Number(it.dataset.i)); } });
  input.addEventListener('blur', () => setTimeout(close, 120));
  return { close };
}

const partRow = (p) => `<span class="t1">${esc(p.name || 'Unnamed')}</span><span class="t2">${esc([p.ipn ? 'IPN ' + p.ipn : '', '#' + p.pk, p.in_stock != null ? fmtQty(p.in_stock) + ' in stock' : ''].filter(Boolean).join(' · '))}</span>`;
const locRow = (l) => `<span class="t1">${esc(l.name || l.path)}</span>${l.path && l.path !== l.name ? `<span class="t2">${esc(l.path)}</span>` : ''}`;

// A self-contained "pick a part" / "pick a location" field.
// Returns { el, get value(), set(value), clear() }; value is the picked object or null.
function pickerField({ label, placeholder, fetcher, render, display, onChange, iconName }) {
  const root = el(html`<div class="field">
    ${label ? html`<label>${label}</label>` : ''}
    <div class="picked" hidden>${icon(iconName)}<div class="grow ellipsis" data-name></div><button type="button" class="iconbtn" data-clear aria-label="Clear" style="width:30px;height:30px">${icon('x', 'sm')}</button></div>
    <div class="field" data-search><input class="input" placeholder="${placeholder}"></div>
  </div>`);
  const input = $('input', root); const pickedEl = $('.picked', root); const searchEl = $('[data-search]', root);
  let value = null;
  const set = (v, silent) => {
    value = v || null;
    pickedEl.hidden = !value; searchEl.hidden = !!value;
    if (value) $('[data-name]', root).textContent = display(value);
    input.value = '';
    if (!silent && onChange) onChange(value);
  };
  typeahead(input, { fetch: fetcher, render, onPick: (r) => set(r) });
  $('[data-clear]', root).onclick = () => { set(null); setTimeout(() => input.focus(), 0); };
  return { el: root, input, get value() { return value; }, set, clear: () => set(null) };
}
export const partField = (o = {}) => pickerField({
  label: o.label ?? 'Part', placeholder: o.placeholder || 'Search name or IPN…', iconName: 'box',
  fetcher: (q) => Parts.search(q, 10), render: partRow, display: partLabel, onChange: o.onChange,
});
export const locationField = (o = {}) => pickerField({
  label: o.label ?? 'Location', placeholder: o.placeholder || 'Search location…', iconName: 'pin',
  fetcher: (q) => Locations.search(q, 12), render: locRow, display: (l) => l.path || l.name, onChange: o.onChange,
});

// Sheet that lets the user search for and pick a part. Resolves with the part or null.
export function pickPart(title = 'Choose a part') {
  return new Promise((resolve) => {
    const body = el(html`<div class="stack">
      <div class="searchbox field">${icon('search')}<input class="input" placeholder="Search name, IPN or #ID…"></div>
      <div class="card flush"><div class="list" data-results><div class="empty">${icon('search')}<span>Start typing to search</span></div></div></div>
    </div>`);
    let result = null;
    const s = sheet({ title, body, onClose: () => resolve(result) });
    const input = $('input', body); const out = $('[data-results]', body);
    const run = debounce(async () => {
      const q = input.value.trim();
      if (!q) { out.innerHTML = String(html`<div class="empty">${icon('search')}<span>Start typing to search</span></div>`); return; }
      try {
        let rows = await Parts.search(q, 15);
        if (/^#?\d+$/.test(q)) { try { const p = await Parts.detail(q.replace('#', '')); if (!rows.find((r) => r.pk === p.pk)) rows = [p, ...rows]; } catch { /* not an id */ } }
        out.innerHTML = rows.length ? rows.map((p, i) => String(html`<div class="list-item" data-i="${i}">${thumb(p)}<div class="grow"><div class="t-title ellipsis">${p.name}</div><div class="t-sub">${[p.ipn ? 'IPN ' + p.ipn : '', '#' + p.pk].filter(Boolean).join(' · ')}</div></div><span class="qty ${Number(p.in_stock) ? '' : 'zero'}">${p.in_stock != null ? fmtQty(p.in_stock) : ''}</span></div>`)).join('')
          : String(html`<div class="empty">${icon('box')}<span>No parts found</span></div>`);
        $$('.list-item', out).forEach((n) => { n.onclick = () => { result = rows[Number(n.dataset.i)]; s.close(); }; });
      } catch (e) { out.innerHTML = String(html`<div class="empty" style="color:var(--err)">${e.message}</div>`); }
    }, 200);
    input.addEventListener('input', run);
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') { const first = $('.list-item', out); if (first) first.click(); } });
    setTimeout(() => input.focus(), 80);
  });
}

// ───────────────────────── Image search sheet ─────────────────────────
// Lets the user search the web for an image and apply it to a part.
// Resolves true if an image was applied.
const IMG_PAGE = 18;
export function imageSearch({ partId, name = '', description = '' }) {
  return new Promise((resolve) => {
    const body = el(html`<div class="stack">
      <div class="row wrap">
        <span class="label">Search using</span>
        <label class="check"><input type="checkbox" data-n checked> Name</label>
        <label class="check"><input type="checkbox" data-d ${description ? raw('checked') : ''}> Description</label>
      </div>
      <div class="row">
        <div class="field grow"><input class="input" data-q placeholder="Search query"></div>
        <button class="btn primary icon-only" data-go aria-label="Search">${icon('search')}</button>
      </div>
      <div class="status" data-st></div>
      <div class="img-grid" data-grid></div>
      <div class="row" style="justify-content:center"><button class="btn" data-more hidden>Load more</button></div>
    </div>`);
    const foot = el(html`<div style="display:contents"><button class="btn" data-cancel>Close</button><button class="btn primary" data-use disabled>${icon('check')} Use image</button></div>`);
    let applied = false; let sel = null; let query = ''; let offset = 0;
    const s = sheet({ title: 'Find an image online', body, foot, wide: true, onClose: () => resolve(applied) });
    const qEl = $('[data-q]', body); const grid = $('[data-grid]', body); const st = $('[data-st]', body);
    const more = $('[data-more]', body); const use = $('[data-use]', foot);
    const composed = () => [$('[data-n]', body).checked ? name : '', $('[data-d]', body).checked ? description : ''].filter(Boolean).join(' ').trim();
    const syncQ = () => { qEl.value = composed(); };
    $('[data-n]', body).onchange = () => { syncQ(); run(); };
    $('[data-d]', body).onchange = () => { syncQ(); run(); };
    const append = (urls) => {
      urls.forEach((u) => {
        const b = el(html`<button type="button" class="img-opt"><img loading="lazy" alt="" referrerpolicy="no-referrer" src="${u}"></button>`);
        $('img', b).onerror = () => b.remove();
        b.onclick = () => { $$('.img-opt', grid).forEach((x) => x.classList.remove('sel')); b.classList.add('sel'); sel = u; use.disabled = false; };
        grid.appendChild(b);
      });
    };
    async function run(reset = true) {
      const q = qEl.value.trim();
      if (!q) { status(st, 'Enter a search query', 'err'); return; }
      if (reset) { query = q; offset = 0; grid.innerHTML = ''; sel = null; use.disabled = true; }
      status(st, 'Searching…', 'loading'); more.hidden = true;
      try {
        const r = await api.post('/api/part/image-search', { query, offset });
        const imgs = r.images || [];
        append(imgs);
        status(st, grid.children.length ? '' : 'No images found', '');
        more.hidden = imgs.length === 0;
      } catch (e) { status(st, e.message, 'err'); }
    }
    $('[data-go]', body).onclick = () => run(true);
    qEl.addEventListener('keydown', (e) => { if (e.key === 'Enter') run(true); });
    more.onclick = () => { offset += IMG_PAGE; run(false); };
    $('[data-cancel]', foot).onclick = () => s.close();
    use.onclick = () => busy(use, async () => {
      try {
        await api.post('/api/part/image-download', { part_id: String(partId), image_url: sel });
        applied = true; toast('Image applied'); s.close();
      } catch (e) { status(st, e.message + ' — try another image', 'err'); }
    });
    syncQ(); run(true);
  });
}

// Upload a picked File to a part, with toast feedback. Returns true on success.
export async function uploadPartImage(pk, file) {
  try { await Parts.uploadImage(pk, file); toast('Image uploaded'); return true; } catch (e) { toastErr(e); return false; }
}
