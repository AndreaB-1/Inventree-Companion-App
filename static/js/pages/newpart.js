// Create a part: AI naming, initial stock, photo, then optional label print + web image.
import {
  $, html, el, icon, api, Parts, num, toast, toastErr, status, busy, sheet, imageSearch,
  uploadPartImage, locationField, rememberPart, settings, store,
} from '../core.js';
import { go, appInfo } from '../app.js';

export const title = 'New part';
export const narrow = true;

export async function render(view, _p, query) {
  const info = await appInfo;
  const loc = locationField({ label: 'Location (for initial stock)' });
  view.innerHTML = String(html`
    <div class="card stack">
      <div class="card-title">${icon('sparkles')}<span class="grow">Describe it</span><span class="chip accent">AI</span></div>
      ${info.ai_enabled === false ? html`<div class="note warn">AI is not configured on the server (OPENAI_API_KEY). You can still fill the fields manually.</div>` : ''}
      <div class="field"><textarea data-prompt placeholder="e.g. M3×22 A2 DIN 912 · BC548 transistor · RJ45 CAT6 shielded pass-through"></textarea></div>
      <div class="row wrap">
        <button class="btn primary" data-ai>${icon('sparkles')} Suggest name & description</button>
        <label class="check"><input type="checkbox" data-nameonly> Name only</label>
        <label class="check"><input type="checkbox" data-strict checked> Strict</label>
      </div>
      <div class="status" data-aist></div>
    </div>

    <div class="card stack">
      <div class="card-title">${icon('box')}<span class="grow">Part details</span></div>
      <div class="grid-2">
        <div class="field"><label>Name *</label><input class="input" data-name placeholder="M3 × 22 – Socket Head Cap Screw A2"></div>
        <div class="field"><label>IPN</label><input class="input mono" data-ipn placeholder="Optional"></div>
      </div>
      <div class="field"><label>Description</label><textarea data-desc placeholder="M3 × 22 – Socket Head Cap Screw – DIN 912 – A2 Stainless Steel"></textarea></div>
      <div class="grid-2">
        <div class="field"><label>Initial quantity</label><input class="input" data-qty inputmode="decimal" placeholder="e.g. 100"></div>
        <div data-loc></div>
      </div>
    </div>

    <div class="card stack">
      <div class="card-title">${icon('camera')}<span class="grow">Photo</span><span class="chip">optional</span></div>
      <div class="row wrap">
        <label class="btn filebtn">${icon('camera')} Take / choose photo<input type="file" accept="image/*" capture="environment" data-file></label>
        <span class="small muted grow" data-fname>No photo — you can search the web after creating.</span>
      </div>
    </div>

    <div class="row wrap" style="margin-top:16px">
      <button class="btn primary lg" data-create>${icon('check')} Create part</button>
      <div class="status" data-st></div>
    </div>
    <div data-done></div>
  `);
  $('[data-loc]', view).replaceWith(loc.el);
  const f = (k) => $(`[data-${k}]`, view);
  if (query.name) f('name').value = query.name;
  if (query.desc) f('desc').value = query.desc;

  f('file').onchange = () => { const file = f('file').files[0]; f('fname').textContent = file ? file.name : 'No photo — you can search the web after creating.'; };

  f('ai').onclick = () => busy(f('ai'), async () => {
    const prompt = f('prompt').value.trim();
    if (!prompt) { status(f('aist'), 'Type a description first', 'err'); return; }
    status(f('aist'), 'Thinking…', 'loading');
    try {
      const r = await api.post('/api/ai/suggest-part', {
        prompt, language: (navigator.language || 'en').slice(0, 2),
        name_only: f('nameonly').checked, strict: f('strict').checked,
      });
      f('name').value = r.name || '';
      if (!f('nameonly').checked) f('desc').value = r.description || '';
      status(f('aist'), 'Suggested — review below, then create', 'ok');
    } catch (e) { status(f('aist'), e.message, 'err'); }
  });

  f('create').onclick = () => busy(f('create'), async () => {
    const name = f('name').value.trim(); const description = f('desc').value.trim();
    const ipn = f('ipn').value.trim(); const qty = num(f('qty').value);
    if (!name) { status(f('st'), 'Name is required', 'err'); f('name').focus(); return; }
    if (qty > 0 && !loc.value) { status(f('st'), 'Choose a location for the initial quantity', 'err'); return; }
    status(f('st'), 'Creating…', 'loading');
    let created;
    try {
      created = await Parts.create({ name, description, ipn, initial_qty: qty > 0 ? qty : '', location_id: qty > 0 ? loc.value.pk : '' });
    } catch (e) { status(f('st'), e.message, 'err'); return; }
    status(f('st'), created.message || 'Part created', 'ok');
    rememberPart({ pk: created.pk, name: created.name, ipn, has_image: false });

    const file = f('file').files[0];
    let hasImage = false;
    if (file) hasImage = await uploadPartImage(created.pk, file);

    showDone(created, { name, description, hasImage });
    if (settings.askPrintAfterCreate) await askPrint(created.pk);
    if (!hasImage && settings.autoImageSearch) await imageSearch({ partId: created.pk, name, description });
  });

  function showDone(created, { name, description }) {
    const box = el(html`<div class="card" style="margin-top:16px;border-color:var(--ok)">
      <div class="row top"><span class="thumb" style="background:var(--ok-soft);color:var(--ok);border:0">${icon('check')}</span>
        <div class="grow"><div class="t-title">Created “${created.name}”</div><div class="t-sub">Part #${created.pk}</div></div></div>
      <div class="btn-row" style="margin-top:14px">
        <a class="btn primary" href="#/part/${created.pk}">Open part</a>
        <button class="btn" data-print>${icon('printer')} Print label</button>
        <button class="btn" data-img>${icon('globe')} Find image</button>
        <a class="btn" href="#/labels?part=${created.pk}">${icon('tag')} Hardware label</a>
        <button class="btn ghost" data-again>${icon('plus')} Create another</button>
      </div></div>`);
    $('[data-print]', box).onclick = (e) => busy(e.currentTarget, () => printPart(created.pk));
    $('[data-img]', box).onclick = () => imageSearch({ partId: created.pk, name, description });
    $('[data-again]', box).onclick = () => { go('#/new?_=' + Date.now()); };
    f('done').innerHTML = ''; f('done').appendChild(box);
    box.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
}

async function printPart(pk) {
  try {
    const body = { template_id: settings.partTemplate, items: [Number(pk)] };
    const plugin = store.raw('label_plugin', ''); if (plugin) body.plugin_key = plugin;
    await api.post('/api/labels/print', body); toast('Print job sent');
  } catch (e) { toastErr(e); }
}

function askPrint(pk) {
  return new Promise((resolve) => {
    const foot = el(html`<div style="display:contents"><button class="btn" data-no>Skip</button><button class="btn primary" data-yes>${icon('printer')} Print</button></div>`);
    const sh = sheet({ title: 'Print a label?', body: html`<p class="muted" style="margin:0">Send a part label for the new item to the default printer (template #${settings.partTemplate})?</p>`, foot, onClose: () => resolve() });
    $('[data-no]', foot).onclick = () => sh.close();
    $('[data-yes]', foot).onclick = (e) => busy(e.currentTarget, async () => { await printPart(pk); sh.close(); });
  });
}
