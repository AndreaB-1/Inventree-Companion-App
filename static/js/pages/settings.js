// Settings: appearance, printing defaults, hardware lists (v1 "Label Config"), server info.
import { $, $$, html, raw, el, icon, api, settings, status, toast, toastErr, busy, confirmDialog, num } from '../core.js';
import { setTheme, themePref, appInfo } from '../app.js';
import { HW_TYPE_KEYS, HW_TYPES } from '../hardware.js';

export const title = 'Settings';
export const narrow = true;

export async function render(view) {
  const info = await appInfo;
  view.innerHTML = String(html`
    <div class="section-title">Appearance</div>
    <div class="card"><div class="row between wrap"><span class="t-title">Theme</span>
      <div class="segmented" data-theme>
        <button data-v="system">System</button><button data-v="light">${icon('sun', 'sm')} Light</button><button data-v="dark">${icon('moon', 'sm')} Dark</button>
      </div></div></div>

    <div class="section-title">After creating a part</div>
    <div class="card" style="padding-top:6px;padding-bottom:6px">
      <label class="switch"><span><span class="t-title">Ask to print a part label</span><br><span class="small faint">Uses the default template below</span></span><input type="checkbox" data-askprint></label>
      <label class="switch"><span><span class="t-title">Suggest an image from the web</span><br><span class="small faint">When no photo was taken</span></span><input type="checkbox" data-autoimg></label>
    </div>

    <div class="section-title">InvenTree label printing</div>
    <div class="card stack">
      <div class="field"><label>Default part label template</label>
        <div class="row"><select class="grow" data-tpl><option value="${settings.partTemplate}">#${settings.partTemplate}</option></select><button class="btn icon-only" data-tplreload aria-label="Reload templates">${icon('refresh')}</button></div>
        <span class="hint">Used by “Print label” on a part and after creating a part.</span></div>
    </div>

    <div class="section-title">Brother QL label printer</div>
    <div class="card stack">
      <div class="grid-2">
        <div class="field"><label>Printer address</label><input class="input mono" data-printer value="${settings.printer}" placeholder="192.168.1.x or tcp://…"></div>
        <div class="field"><label>Cut margin per side (mm)</label><input class="input" type="number" step="0.5" min="0" max="15" data-margin value="${settings.cutMargin}"></div>
      </div>
      <div class="row wrap"><button class="btn" data-test>${icon('plug')} Test connection</button><div class="status" data-tst></div></div>
    </div>

    <div class="section-title">Hardware lists <span class="chip" style="text-transform:none;letter-spacing:0">used by the label maker</span></div>
    <div class="card stack">
      <div class="segmented" data-cfgtab><button data-v="sizes" class="on">Thread sizes</button><button data-v="standards">Standards</button><button data-v="materials">Materials</button></div>
      <div data-cfgbody><div class="skel" style="height:120px"></div></div>
      <div class="row wrap" style="border-top:1px solid var(--line);padding-top:14px">
        <button class="btn primary" data-save>${icon('check')} Save lists</button>
        <button class="btn ghost" data-reset>${icon('refresh', 'sm')} Reset to defaults</button>
        <div class="status" data-cst></div>
      </div>
    </div>

    <div class="section-title">Server</div>
    <div class="card"><dl class="kv">
      <dt>InvenTree</dt><dd class="mono">${info.inventree_url || '—'}</dd>
      <dt>AI naming</dt><dd>${info.ai_enabled ? html`<span class="chip ok">On</span> <span class="mono small">${info.ai_model || ''}</span>` : html`<span class="chip warn">Off</span> <span class="small faint">set OPENAI_API_KEY</span>`}</dd>
      <dt>Label plugin</dt><dd class="mono">${info.label_plugin_key || 'auto-detect'}</dd>
      <dt>Label machine</dt><dd>${info.label_machine_configured ? 'Configured' : html`<span class="faint">Not set (INVENTREE_LABEL_MACHINE_ID)</span>`}</dd>
    </dl></div>
  `);
  const q = (s) => $(s, view);

  // Theme
  const syncTheme = () => $$('[data-theme] button', view).forEach((b) => b.classList.toggle('on', b.dataset.v === themePref()));
  q('[data-theme]').onclick = (e) => { const b = e.target.closest('[data-v]'); if (b) { setTheme(b.dataset.v); syncTheme(); } };
  syncTheme();

  // Toggles
  q('[data-askprint]').checked = settings.askPrintAfterCreate;
  q('[data-askprint]').onchange = (e) => { settings.askPrintAfterCreate = e.target.checked; };
  q('[data-autoimg]').checked = settings.autoImageSearch;
  q('[data-autoimg]').onchange = (e) => { settings.autoImageSearch = e.target.checked; };

  // Template list
  async function loadTpl() {
    const sel = q('[data-tpl]');
    try {
      const rows = await api.get('/api/labels/templates?model_type=part&enabled=true');
      const cur = settings.partTemplate;
      sel.innerHTML = rows.map((t) => `<option value="${t.pk}">${String(html`#${t.pk} · ${t.name}`)}</option>`).join('')
        + (rows.find((t) => t.pk === cur) ? '' : `<option value="${cur}">#${cur} (not found)</option>`);
      sel.value = String(cur);
    } catch (e) { toastErr(e); }
  }
  q('[data-tpl]').onchange = (e) => { settings.partTemplate = e.target.value; toast('Default template saved'); };
  q('[data-tplreload]').onclick = loadTpl;
  loadTpl();

  // Printer
  q('[data-printer]').onchange = (e) => { settings.printer = e.target.value.trim(); };
  q('[data-margin]').onchange = (e) => { settings.cutMargin = num(e.target.value) || 0; };
  q('[data-test]').onclick = (e) => busy(e.currentTarget, async () => {
    const addr = q('[data-printer]').value.trim(); settings.printer = addr;
    if (!addr) { status(q('[data-tst]'), 'Enter an address first', 'err'); return; }
    status(q('[data-tst]'), 'Testing…', 'loading');
    try { const r = await api.get('/api/labels/hardware/printer-test?printer=' + encodeURIComponent(addr)); status(q('[data-tst]'), 'Reachable — ' + r.printer, 'ok'); } catch (err) { status(q('[data-tst]'), err.message, 'err'); }
  });

  // ── Hardware config editor ──
  let cfg = null; let tab = 'sizes'; let dirty = false; let stdFilter = '';
  const body = q('[data-cfgbody]'); const cst = q('[data-cst]');
  const markDirty = () => { dirty = true; status(cst, 'Unsaved changes', ''); };
  try { cfg = JSON.parse(JSON.stringify(await api.get('/api/config/hardware'))); } catch (e) { body.innerHTML = String(html`<div class="note warn">${e.message}</div>`); return null; }
  cfg.thread_sizes = cfg.thread_sizes || { metric: [], imperial: [], custom: [] };
  cfg.standards = cfg.standards || []; cfg.materials = cfg.materials || [];

  const pills = (arr, key) => html`<div class="pills">${arr.map((v, i) => html`<span class="pill">${v}<button data-del="${key}" data-i="${i}" aria-label="Remove ${v}">${icon('x', 'sm')}</button></span>`)}</div>`;
  const adder = (key, ph) => html`<form class="row" data-add="${key}" style="margin-top:10px"><input class="input grow" placeholder="${ph}"><button class="btn sm">${icon('plus', 'sm')} Add</button></form>`;

  function paintCfg() {
    if (!cfg) return;
    if (tab === 'sizes') {
      body.innerHTML = String(html`
        ${['metric', 'imperial', 'custom'].map((k) => html`<div style="margin-bottom:16px"><div class="label" style="margin-bottom:8px;text-transform:capitalize">${k} <span class="faint">(${(cfg.thread_sizes[k] || []).length})</span></div>
          ${pills(cfg.thread_sizes[k] || [], 'ts:' + k)}${adder('ts:' + k, k === 'metric' ? 'e.g. M5.5' : k === 'imperial' ? 'e.g. 3/8"-16' : 'e.g. TR10')}</div>`)}`);
    } else if (tab === 'materials') {
      body.innerHTML = String(html`${pills(cfg.materials, 'mat')}${adder('mat', 'e.g. Inconel 625')}`);
    } else {
      const types = [...new Set([...HW_TYPE_KEYS, ...cfg.standards.map((s) => s.type).filter(Boolean)])];
      const tlabel = (t) => (HW_TYPES[t] ? HW_TYPES[t].label : t);
      body.innerHTML = String(html`
        <div class="field" style="max-width:220px;margin-bottom:10px"><select data-stdf><option value="">All types</option>${types.map((t) => html`<option value="${t}" ${t === stdFilter ? raw('selected') : ''}>${tlabel(t)}</option>`)}</select></div>
        <div style="overflow-x:auto"><table class="table"><thead><tr><th>Standard</th><th>Type</th><th class="hide-sm">Description</th><th></th></tr></thead><tbody>
        ${cfg.standards.map((s, i) => (stdFilter && s.type !== stdFilter ? '' : html`<tr><td class="mono" style="white-space:nowrap">${s.name}</td><td>${tlabel(s.type || '')}</td><td class="hide-sm muted">${s.desc || ''}</td>
          <td style="text-align:right"><button class="iconbtn" style="width:30px;height:30px" data-del="std" data-i="${i}" aria-label="Remove">${icon('trash', 'sm')}</button></td></tr>`))}
        </tbody></table></div>
        <form class="grid-3" data-addstd style="margin-top:12px">
          <input class="input" name="name" placeholder="Standard, e.g. DIN 125" required>
          <select class="input" name="type">${types.map((t) => html`<option value="${t}">${tlabel(t)}</option>`)}</select>
          <div class="row"><input class="input grow" name="desc" placeholder="Description"><button class="btn sm">${icon('plus', 'sm')}</button></div>
        </form>`);
      const f = $('[data-stdf]', body); f.onchange = () => { stdFilter = f.value; paintCfg(); };
      $('[data-addstd]', body).onsubmit = (e) => {
        e.preventDefault(); const fd = new FormData(e.target);
        const name = String(fd.get('name') || '').trim(); const type = String(fd.get('type') || ''); const desc = String(fd.get('desc') || '').trim();
        if (!name || cfg.standards.find((s) => s.name === name && s.type === type)) return;
        cfg.standards.push({ name, type, desc }); markDirty(); paintCfg();
      };
    }
    $$('[data-add]', body).forEach((form) => {
      form.onsubmit = (e) => {
        e.preventDefault(); const v = $('input', form).value.trim(); if (!v) return;
        const arr = listFor(form.dataset.add); if (!arr.includes(v)) { arr.push(v); markDirty(); }
        paintCfg(); const again = $(`[data-add="${form.dataset.add}"] input`, body); if (again) again.focus();
      };
    });
    $$('[data-del]', body).forEach((b) => { b.onclick = () => { listFor(b.dataset.del).splice(Number(b.dataset.i), 1); markDirty(); paintCfg(); }; });
  }
  function listFor(key) {
    if (key === 'mat') return cfg.materials;
    if (key === 'std') return cfg.standards;
    const k = key.split(':')[1]; cfg.thread_sizes[k] = cfg.thread_sizes[k] || []; return cfg.thread_sizes[k];
  }
  q('[data-cfgtab]').onclick = (e) => { const b = e.target.closest('[data-v]'); if (!b) return; tab = b.dataset.v; $$('[data-cfgtab] button', view).forEach((x) => x.classList.toggle('on', x === b)); paintCfg(); };
  q('[data-save]').onclick = (e) => busy(e.currentTarget, async () => {
    try { await api.post('/api/config/hardware', cfg); dirty = false; status(cst, 'Saved', 'ok'); toast('Hardware lists saved'); } catch (err) { status(cst, err.message, 'err'); }
  });
  q('[data-reset]').onclick = async () => {
    if (!(await confirmDialog({ title: 'Reset hardware lists?', message: 'All thread sizes, standards and materials go back to the factory defaults. This cannot be undone.', ok: 'Reset', danger: true }))) return;
    try { cfg = await api.post('/api/config/hardware/reset', {}); dirty = false; paintCfg(); status(cst, 'Reset to defaults', 'ok'); } catch (err) { status(cst, err.message, 'err'); }
  };
  paintCfg();
  const warn = (e) => { if (dirty) { e.preventDefault(); e.returnValue = ''; } };
  window.addEventListener('beforeunload', warn);
  return () => { window.removeEventListener('beforeunload', warn); if (dirty) toast('Hardware list changes were not saved', 'err'); };
}
export { el };
