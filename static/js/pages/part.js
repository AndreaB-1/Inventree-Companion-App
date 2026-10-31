// Part hub: details, image, stock (adjust / move / add) and InvenTree label printing.
import {
  $, $$, html, raw, el, icon, api, Parts, Stock, fmtQty, num, toast, toastErr, status, busy,
  sheet, lightbox, imageSearch, uploadPartImage, locationField, rememberPart, store, confirmDialog,
} from '../core.js';
import { chrome, go, appInfo } from '../app.js';

export const title = 'Part';

export async function render(view, { id }, query) {
  const pk = Number(id);
  let part = null; let stock = []; let bust = '';
  let tab = query.tab === 'labels' ? 'labels' : 'stock';
  const selected = new Set();

  view.innerHTML = String(html`<div class="part-hero">
    <div class="part-img skel"></div>
    <div class="stack tight"><div class="skel" style="height:28px;width:70%"></div><div class="skel" style="height:16px;width:40%"></div><div class="skel" style="height:60px"></div></div>
  </div>`);

  try {
    [part, stock] = await Promise.all([Parts.detail(pk), Stock.byPart(pk)]);
  } catch (e) {
    view.innerHTML = String(html`<div class="card"><div class="empty">${icon('info')}<span>${e.message}</span><a class="btn" href="#/search">Back to search</a></div></div>`);
    return;
  }
  rememberPart(part);
  chrome.title(part.name || 'Part #' + pk);

  // ── Layout ──
  view.innerHTML = String(html`
    <div class="part-hero">
      <div class="part-img" data-img></div>
      <div class="part-info">
        <div class="row wrap" style="gap:6px">
          <span class="chip mono">#${pk}</span>
          ${part.ipn ? html`<span class="chip accent mono">IPN ${part.ipn}</span>` : ''}
          ${part.category ? html`<span class="chip">${part.category}</span>` : ''}
          ${part.active === false ? html`<span class="chip warn">Inactive</span>` : ''}
        </div>
        <h2 class="part-name" data-name></h2>
        <p class="part-desc" data-desc></p>
      </div>
      <div class="kpis" data-kpis></div>
      <div class="part-actions">
        <button class="btn" data-edit>${icon('edit')} Edit</button>
        <label class="btn filebtn">${icon('camera')} Photo<input type="file" accept="image/*" capture="environment" data-photo></label>
        <button class="btn" data-web>${icon('globe')} Web image</button>
        <button class="btn" data-quickprint>${icon('printer')} Print label</button>
        <a class="btn" href="#/labels?part=${pk}">${icon('tag')} Hardware label</a>
      </div>
    </div>
    <div class="tabs" role="tablist">
      <button data-tab="stock">${icon('layers', 'sm')} Stock <span class="count" data-count></span></button>
      <button data-tab="labels">${icon('printer', 'sm')} Print labels</button>
    </div>
    <div data-pane="stock"></div>
    <div data-pane="labels" hidden></div>
  `);

  const imgBox = $('[data-img]', view);
  function paintHeader() {
    $('[data-name]', view).textContent = part.name || 'Unnamed';
    const d = $('[data-desc]', view); d.textContent = part.description || ''; d.hidden = !part.description;
    imgBox.innerHTML = '';
    const url = Parts.imgUrl(pk, false, bust);
    if (part.has_image || bust) {
      const img = el('<img alt="">'); img.src = url;
      img.onerror = () => { imgBox.innerHTML = String(icon('image', 'lg')); };
      imgBox.appendChild(img);
      imgBox.onclick = () => lightbox(url);
    } else { imgBox.innerHTML = String(icon('image', 'lg')); imgBox.onclick = null; }
    const total = stock.reduce((a, s) => a + (Number(s.quantity) || 0), 0);
    const locs = new Set(stock.map((s) => s.location_id).filter(Boolean)).size;
    $('[data-kpis]', view).innerHTML = String(html`
      <div class="kpi"><div class="k-v">${fmtQty(total)}</div><div class="k-l">In stock${part.units ? ' (' + part.units + ')' : ''}</div></div>
      <div class="kpi"><div class="k-v">${stock.length}</div><div class="k-l">Stock items</div></div>
      <div class="kpi"><div class="k-v">${locs}</div><div class="k-l">Locations</div></div>`);
    $('[data-count]', view).textContent = stock.length;
  }

  // ── Tabs ──
  function showTab(t) {
    tab = t;
    $$('[data-tab]', view).forEach((b) => b.classList.toggle('on', b.dataset.tab === t));
    $$('[data-pane]', view).forEach((p) => { p.hidden = p.dataset.pane !== t; });
    if (t === 'labels') renderLabels();
    history.replaceState(null, '', `#/part/${pk}${t === 'labels' ? '?tab=labels' : ''}`);
  }
  $$('[data-tab]', view).forEach((b) => { b.onclick = () => showTab(b.dataset.tab); });

  // ── Stock pane ──
  const stockPane = $('[data-pane="stock"]', view);
  async function reloadStock() {
    try { stock = await Stock.byPart(pk); } catch (e) { toastErr(e); }
    [...selected].forEach((sid) => { if (!stock.find((s) => s.pk === sid)) selected.delete(sid); });
    paintHeader(); renderStock();
  }
  function renderStock() {
    stockPane.innerHTML = String(html`
      <div class="row between" style="margin-bottom:10px">
        <span class="small faint">${stock.length ? 'Tick items to print their labels' : ''}</span>
        <button class="btn sm" data-addstock>${icon('plus', 'sm')} New stock item</button>
      </div>
      <div class="card flush" data-list>
        ${stock.length ? stock.map((s) => html`<div class="stock-card" data-sid="${s.pk}">
          <input type="checkbox" class="check" style="width:18px;height:18px;accent-color:var(--accent)" data-sel ${selected.has(s.pk) ? raw('checked') : ''} aria-label="Select for printing">
          <div class="loc">${icon('pin', 'sm')}<span>${s.location_path || s.location_name || 'No location'}</span></div>
          <div class="qty ${Number(s.quantity) ? '' : 'zero'}">${fmtQty(s.quantity)}</div>
          <div class="meta">${['Stock #' + s.pk, s.batch ? 'Batch ' + s.batch : '', s.serial ? 'Serial ' + s.serial : ''].filter(Boolean).join(' · ')}</div>
          <div class="ops">
            <div class="stepper">
              <button class="minus" data-op="remove" aria-label="Remove">${icon('minus')}</button>
              <input inputmode="decimal" value="1" aria-label="Amount" data-amt>
              <button class="plus" data-op="add" aria-label="Add">${icon('plus')}</button>
            </div>
            <button class="btn sm" data-move>${icon('move', 'sm')} Move</button>
          </div>
        </div>`) : html`<div class="empty">${icon('layers')}<span>No stock for this part yet.</span></div>`}
      </div>`);
    $('[data-addstock]', stockPane).onclick = openAddStock;
    $$('.stock-card', stockPane).forEach((row) => {
      const sid = Number(row.dataset.sid); const s = stock.find((x) => x.pk === sid);
      $('[data-sel]', row).onchange = (e) => { if (e.target.checked) selected.add(sid); else selected.delete(sid); updateTargets(); };
      $$('[data-op]', row).forEach((b) => {
        b.onclick = () => busy(b, async () => {
          const q = num($('[data-amt]', row).value);
          if (!(q > 0)) { toast('Enter an amount above 0', 'err'); return; }
          const op = b.dataset.op;
          if (op === 'remove' && q > Number(s.quantity) && !(await confirmDialog({ title: 'Remove more than in stock?', message: `This item only has ${fmtQty(s.quantity)}.`, ok: 'Remove anyway', danger: true }))) return;
          try {
            await (op === 'add' ? Stock.add(sid, q, 'Adjusted via companion app') : Stock.remove(sid, q, 'Adjusted via companion app'));
            toast(`${op === 'add' ? 'Added' : 'Removed'} ${fmtQty(q)}`);
            await reloadStock();
          } catch (e) { toastErr(e); }
        });
      });
      $('[data-move]', row).onclick = () => openMove(s);
    });
  }

  function openMove(s) {
    const loc = locationField({ label: 'New location' });
    const body = el(html`<div class="stack">
      <div class="note">${icon('pin', 'sm')} From <b>${s.location_path || s.location_name || 'No location'}</b> · ${fmtQty(s.quantity)} available</div>
      <div class="field"><label>Quantity to move</label><input class="input" inputmode="decimal" data-q value="${fmtQty(s.quantity)}"><span class="hint">Less than the full amount splits the stock item.</span></div>
      <div data-loc></div>
      <div class="field"><label>Notes (optional)</label><input class="input" data-notes placeholder="e.g. Reorganising shelf B3"></div>
    </div>`);
    $('[data-loc]', body).replaceWith(loc.el);
    const foot = el(html`<div style="display:contents"><button class="btn" data-c>Cancel</button><button class="btn primary" data-ok>${icon('move')} Move stock</button></div>`);
    const sh = sheet({ title: 'Move stock', body, foot });
    $('[data-c]', foot).onclick = () => sh.close();
    $('[data-ok]', foot).onclick = (ev) => busy(ev.currentTarget, async () => {
      const q = num($('[data-q]', body).value);
      if (!loc.value) { toast('Choose a destination location', 'err'); return; }
      if (!(q > 0)) { toast('Enter a valid quantity', 'err'); return; }
      if (q > Number(s.quantity)) { toast(`Quantity exceeds available stock (${fmtQty(s.quantity)})`, 'err'); return; }
      try {
        await Stock.transfer(s.pk, loc.value.pk, q, $('[data-notes]', body).value.trim());
        sh.close(); toast('Moved to ' + (loc.value.path || loc.value.name)); await reloadStock();
      } catch (e) { toastErr(e); }
    });
  }

  function openAddStock() {
    const loc = locationField({ label: 'Location' });
    const body = el(html`<div class="stack">
      <div class="field"><label>Quantity</label><input class="input" inputmode="decimal" data-q placeholder="e.g. 100"></div>
      <div data-loc></div>
      <div class="field"><label>Notes (optional)</label><input class="input" data-notes></div>
    </div>`);
    $('[data-loc]', body).replaceWith(loc.el);
    const foot = el(html`<div style="display:contents"><button class="btn" data-c>Cancel</button><button class="btn primary" data-ok>${icon('plus')} Create</button></div>`);
    const sh = sheet({ title: 'New stock item', body, foot });
    $('[data-c]', foot).onclick = () => sh.close();
    $('[data-ok]', foot).onclick = (ev) => busy(ev.currentTarget, async () => {
      const q = num($('[data-q]', body).value);
      if (!(q > 0)) { toast('Enter a quantity', 'err'); return; }
      if (!loc.value) { toast('Choose a location', 'err'); return; }
      try { await Stock.create(pk, q, loc.value.pk, $('[data-notes]', body).value.trim()); sh.close(); toast('Stock item created'); await reloadStock(); } catch (e) { toastErr(e); }
    });
  }

  // ── Labels pane (InvenTree templates) ──
  const labelsPane = $('[data-pane="labels"]', view);
  let labelsBuilt = false;
  async function renderLabels() {
    if (labelsBuilt) { updateTargets(); return; }
    labelsBuilt = true;
    const model0 = store.raw('label_model', 'stockitem');
    labelsPane.innerHTML = String(html`<div class="card stack">
      <div class="grid-3">
        <div class="field"><label>Label for</label><select data-model>
          <option value="stockitem">Selected stock items</option><option value="part">This part</option><option value="stocklocation">Locations of selected items</option>
        </select></div>
        <div class="field"><label>Template</label><select data-tpl><option value="">Loading…</option></select></div>
        <div class="field"><label>Printer plugin</label><select data-plugin><option value="">Server default</option></select></div>
      </div>
      <div class="note" data-targets></div>
      <div class="row wrap"><button class="btn primary" data-print>${icon('printer')} Print</button><button class="btn ghost" data-refresh>${icon('refresh', 'sm')} Reload templates</button><div class="status" data-st></div></div>
    </div>`);
    const modelEl = $('[data-model]', labelsPane); modelEl.value = model0;
    const tplEl = $('[data-tpl]', labelsPane); const plEl = $('[data-plugin]', labelsPane); const st = $('[data-st]', labelsPane);
    async function loadTemplates() {
      tplEl.innerHTML = '<option value="">Loading…</option>';
      try {
        const rows = await api.get(`/api/labels/templates?model_type=${modelEl.value}&enabled=true`);
        const last = store.raw('label_tpl_' + modelEl.value, '');
        tplEl.innerHTML = rows.length ? rows.map((t) => `<option value="${t.pk}">${String(html`${t.name}`)}</option>`).join('') : '<option value="">No templates for this type</option>';
        if (last && rows.find((t) => String(t.pk) === last)) tplEl.value = last;
      } catch (e) { tplEl.innerHTML = '<option value="">Error</option>'; status(st, e.message, 'err'); }
    }
    async function loadPlugins() {
      try {
        const rows = await api.get('/api/plugins/label?active=true');
        const info = await appInfo;
        plEl.innerHTML = `<option value="">Server default${info.label_plugin_key ? ' (' + info.label_plugin_key + ')' : ''}</option>` + rows.map((p) => `<option value="${String(html`${p.key}`)}">${String(html`${p.name}`)}</option>`).join('');
        plEl.value = store.raw('label_plugin', '');
      } catch { /* optional */ }
    }
    modelEl.onchange = () => { store.setRaw('label_model', modelEl.value); updateTargets(); loadTemplates(); };
    tplEl.onchange = () => store.setRaw('label_tpl_' + modelEl.value, tplEl.value);
    plEl.onchange = () => store.setRaw('label_plugin', plEl.value);
    $('[data-refresh]', labelsPane).onclick = loadTemplates;
    $('[data-print]', labelsPane).onclick = (ev) => busy(ev.currentTarget, async () => {
      const items = targetItems();
      if (!tplEl.value) { status(st, 'Pick a template first', 'err'); return; }
      if (!items.length) { status(st, modelEl.value === 'part' ? 'No part' : 'Tick at least one stock item on the Stock tab', 'err'); return; }
      status(st, 'Sending…', 'loading');
      try {
        const body = { template_id: Number(tplEl.value), items };
        if (plEl.value) body.plugin_key = plEl.value;
        await api.post('/api/labels/print', body);
        status(st, 'Print job sent', 'ok'); toast('Print job sent');
      } catch (e) { status(st, e.message, 'err'); }
    });
    updateTargets(); loadTemplates(); loadPlugins();
  }
  function targetItems() {
    const model = $('[data-model]', labelsPane)?.value;
    if (model === 'part') return [pk];
    const sel = stock.filter((s) => selected.has(s.pk));
    if (model === 'stocklocation') return [...new Set(sel.map((s) => s.location_id).filter(Boolean))];
    return sel.map((s) => s.pk);
  }
  function updateTargets() {
    const t = $('[data-targets]', labelsPane); if (!t) return;
    const model = $('[data-model]', labelsPane).value; const n = targetItems().length;
    t.textContent = model === 'part' ? `Prints one label for “${part.name}”.`
      : n ? `Will print ${n} label${n === 1 ? '' : 's'} (${model === 'stocklocation' ? 'locations' : 'stock items'} ticked on the Stock tab).`
        : 'Nothing selected — tick stock items on the Stock tab first.';
  }

  // ── Header actions ──
  $('[data-photo]', view).onchange = async (e) => {
    const f = e.target.files[0]; e.target.value = ''; if (!f) return;
    toast('Uploading…', '');
    if (await uploadPartImage(pk, f)) { bust = Date.now(); part.has_image = true; paintHeader(); }
  };
  $('[data-web]', view).onclick = async () => {
    if (await imageSearch({ partId: pk, name: part.name, description: part.description })) { bust = Date.now(); part.has_image = true; paintHeader(); }
  };
  $('[data-quickprint]', view).onclick = (ev) => busy(ev.currentTarget, async () => {
    const tpl = Number(store.raw('default_part_template', '11')) || 11;
    try {
      const body = { template_id: tpl, items: [pk] };
      const plugin = store.raw('label_plugin', ''); if (plugin) body.plugin_key = plugin;
      await api.post('/api/labels/print', body); toast('Part label sent to printer');
    } catch (e) { toastErr(e); }
  });
  $('[data-edit]', view).onclick = openEdit;

  async function openEdit() {
    const info = await appInfo;
    const body = el(html`<div class="stack">
      <div class="field"><label>Name</label><input class="input" data-n></div>
      <div class="field"><label>IPN</label><input class="input mono" data-i placeholder="Internal part number"></div>
      <div class="field"><label>Description</label><textarea data-d></textarea></div>
      ${info.ai_enabled !== false ? html`<div class="row wrap"><button class="btn sm" data-ai>${icon('sparkles', 'sm')} Suggest with AI</button><span class="status small" data-aist></span></div>` : ''}
    </div>`);
    $('[data-n]', body).value = part.name || ''; $('[data-i]', body).value = part.ipn || ''; $('[data-d]', body).value = part.description || '';
    const foot = el(html`<div style="display:contents"><button class="btn" data-c>Cancel</button><button class="btn primary" data-ok>${icon('check')} Save</button></div>`);
    const sh = sheet({ title: 'Edit part', body, foot });
    const ai = $('[data-ai]', body);
    if (ai) ai.onclick = () => busy(ai, async () => {
      const prompt = [$('[data-n]', body).value, $('[data-d]', body).value].filter(Boolean).join(' — ');
      status($('[data-aist]', body), 'Thinking…', 'loading');
      try {
        const r = await api.post('/api/ai/suggest-part', { prompt, strict: false, language: (navigator.language || 'en').slice(0, 2) });
        if (r.name) $('[data-n]', body).value = r.name; if (r.description) $('[data-d]', body).value = r.description;
        status($('[data-aist]', body), 'Suggestion applied — review and save', 'ok');
      } catch (e) { status($('[data-aist]', body), e.message, 'err'); }
    });
    $('[data-c]', foot).onclick = () => sh.close();
    $('[data-ok]', foot).onclick = (ev) => busy(ev.currentTarget, async () => {
      const data = { name: $('[data-n]', body).value.trim(), ipn: $('[data-i]', body).value.trim(), description: $('[data-d]', body).value.trim() };
      if (!data.name) { toast('Name is required', 'err'); return; }
      try { await Parts.update(pk, data); Object.assign(part, data); rememberPart(part); paintHeader(); chrome.title(part.name); sh.close(); toast('Saved'); } catch (e) { toastErr(e); }
    });
  }

  paintHeader(); renderStock(); showTab(tab);

  if (query.action === 'move') {
    if (stock.length === 1) openMove(stock[0]);
    else if (stock.length > 1) toast('Choose which stock item to move', '');
    else toast('This part has no stock to move', 'err');
  }
  return null;
}
export { go };
