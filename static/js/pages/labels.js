// Hardware label maker (Brother QL, 12 mm tape) with live preview.
import {
  $, $$, html, raw, el, icon, api, Parts, num, toast, toastErr, status, busy, sheet, debounce,
  partField, locationField, imageSearch, uploadPartImage, settings, partLabel, rememberPart,
} from '../core.js';
import { HW_TYPES, HW_TYPE_KEYS, STANDARD_FILL, sizeAutoFill, suggestName, suggestDesc, configLists } from '../hardware.js';

export const title = 'Label maker';

// State survives navigation within the session.
const S = {
  type: 'screw',
  specs: {}, // type -> {field: value}
  opts: { show_diagram: true, show_topview: true, show_sideview: true, show_material: true, show_standard: true, show_grade: true },
  link: 'qr', // 'qr' | 'text' | 'none'
  qrPart: null,
  refPart: null,
  length: 36,
  copies: 1,
  lastB64: null,
};
let cfg = null;

export async function render(view, _p, query) {
  try { cfg = await api.get('/api/config/hardware'); } catch { cfg = cfg || {}; }
  if (query.part) {
    try { const p = await Parts.detail(query.part); S.qrPart = p; S.refPart = p; if (S.link === 'none') S.link = 'qr'; } catch (e) { toastErr(e); }
  }

  view.innerHTML = String(html`<div class="lm">
    <div class="stack">
      <div class="card">
        <div class="card-title">${icon('layers')}<span class="grow">Hardware type</span></div>
        <div class="type-grid" data-types>
          ${HW_TYPE_KEYS.map((k) => html`<button type="button" class="type-btn" data-type="${k}"><img alt="" data-tv="${k}">${HW_TYPES[k].label}</button>`)}
        </div>
      </div>
      <div class="card">
        <div class="card-title">${icon('settings')}<span class="grow">Specifications</span><span class="chip accent" data-typename></span></div>
        <div class="grid-2" data-form></div>
      </div>
      <div class="card stack">
        <div class="card-title" style="margin:0">${icon('tag')}<span class="grow">Label content</span></div>
        <div class="opts-grid">
          ${[['show_diagram', 'Diagrams'], ['show_topview', 'Top view'], ['show_sideview', 'Side view'], ['show_material', 'Material / finish'], ['show_standard', 'Standard'], ['show_grade', 'Grade']]
            .map(([k, l]) => html`<label class="check"><input type="checkbox" data-opt="${k}" ${S.opts[k] ? raw('checked') : ''}> ${l}</label>`)}
        </div>
        <div>
          <div class="label" style="margin-bottom:8px">Link to an InvenTree part</div>
          <div class="segmented" data-linkmode>
            <button data-v="qr">${icon('scan', 'sm')} QR code</button>
            <button data-v="text">Sideways IPN</button>
            <button data-v="none">None</button>
          </div>
        </div>
        <div data-linkbody></div>
      </div>
      <div class="card">
        <div class="card-title">${icon('hash')}<span class="grow">Tape</span><span class="chip">12 mm</span></div>
        <div class="grid-3">
          <div class="field"><label>Total length (mm)</label><input class="input" type="number" min="12" max="200" step="1" data-len value="${S.length}"></div>
          <div class="field"><label>Cut margin / side</label><input class="input" type="number" min="0" max="15" step="0.5" data-margin value="${settings.cutMargin}"></div>
          <div class="field"><label>Copies</label><input class="input" type="number" min="1" max="99" step="1" data-copies value="${S.copies}"></div>
        </div>
        <div class="hint" data-tapehint style="margin-top:8px"></div>
      </div>
    </div>

    <aside class="lm-side">
      <div class="card stack">
        <div class="card-title" style="margin:0">${icon('image')}<span class="grow">Preview</span><span class="tiny faint" data-dim></span></div>
        <div class="tape" data-tape><span class="faint small">Generating…</span></div>
        <div class="views" data-views>
          <figure><img alt="Top view" data-vt><figcaption>Top</figcaption></figure>
          <figure><img alt="Side view" data-vs><figcaption>Side</figcaption></figure>
        </div>
        <div class="status small" data-pst></div>
      </div>
      <div class="card stack">
        <div class="card-title" style="margin:0">${icon('printer')}<span class="grow">Print</span><span class="chip">Brother QL-710W</span></div>
        <div class="field"><label>Printer address</label>
          <div class="row"><input class="input grow mono" data-printer placeholder="192.168.1.x or tcp://…" value="${settings.printer}"><button class="btn icon-only" data-test aria-label="Test connection" title="Test connection">${icon('plug')}</button></div>
        </div>
        <div class="status small" data-tst></div>
        <div class="row">
          <button class="btn primary lg grow" data-print>${icon('printer')} Print</button>
          <button class="btn lg icon-only" data-dl aria-label="Download PNG" title="Download PNG">${icon('download')}</button>
        </div>
      </div>
    </aside>
  </div>`);

  const q = (s) => $(s, view);
  const form = q('[data-form]');

  // ── Type & spec form ──
  function buildForm() {
    const def = HW_TYPES[S.type]; const lists = configLists(cfg, S.type);
    const cur = S.specs[S.type] || (S.specs[S.type] = {});
    q('[data-typename]').textContent = def.label;
    $$('[data-type]', view).forEach((b) => b.classList.toggle('on', b.dataset.type === S.type));
    form.innerHTML = def.fields.map((f) => {
      let opts = f.opts;
      if (f.dyn === 'sizes' && lists.sizes.length) opts = lists.sizes;
      if (f.dyn === 'materials' && lists.materials.length) opts = lists.materials;
      if (f.dyn === 'standards' && lists.standards.length) opts = lists.standards;
      const val = cur[f.id] ?? f.def ?? '';
      const ctl = f.type === 'select'
        ? `<select data-f="${f.id}">${opts.map((o) => `<option${String(o) === String(val) ? ' selected' : ''}>${String(html`${o}`)}</option>`).join('')}</select>`
        : String(html`<input class="input" type="number" data-f="${f.id}" value="${val}" ${f.min != null ? raw(`min="${f.min}"`) : ''} ${f.max != null ? raw(`max="${f.max}"`) : ''} ${f.step != null ? raw(`step="${f.step}"`) : ''}>`);
      return String(html`<div class="field"><label>${f.label}</label>${raw(ctl)}${f.hint ? html`<span class="hint">${f.hint}</span>` : ''}</div>`);
    }).join('');
    // First build for a type: apply size-derived defaults
    if (!cur.__init) { cur.__init = true; const sz = specs().thread_size || specs().fits_size; applyFill(sizeAutoFill(S.type, sz)); }
    collect();
  }
  const specs = () => { const o = {}; $$('[data-f]', form).forEach((n) => { o[n.dataset.f] = n.value; }); return o; };
  const collect = () => { Object.assign(S.specs[S.type], specs()); };
  function applyFill(vals) { Object.entries(vals).forEach(([k, v]) => { const n = $(`[data-f="${k}"]`, form); if (n && v != null) n.value = v; }); }

  form.addEventListener('change', (e) => {
    const id = e.target.dataset.f;
    if (id === 'thread_size' || id === 'fits_size') applyFill(sizeAutoFill(S.type, e.target.value));
    if (id === 'standard') {
      const fill = STANDARD_FILL[e.target.value];
      if (fill) {
        const h = $('[data-f="head_type"]', form); const d = $('[data-f="drive"]', form);
        if (h && [...h.options].some((o) => o.value === fill.head)) h.value = fill.head;
        if (d && [...d.options].some((o) => o.value === fill.drive)) d.value = fill.drive;
      }
    }
    collect(); refresh();
  });
  form.addEventListener('input', () => { collect(); refresh(); });
  $$('[data-type]', view).forEach((b) => { b.onclick = () => { S.type = b.dataset.type; buildForm(); refresh(true); }; });

  // Type icons: render each type's default top view once
  HW_TYPE_KEYS.forEach(async (k) => {
    const img = $(`[data-tv="${k}"]`, view);
    const cached = sessionStorage.getItem('tv_' + k);
    if (cached) { img.src = cached; return; }
    try {
      const sp = {}; HW_TYPES[k].fields.forEach((f) => { sp[f.id] = f.def; });
      const r = await api.post('/api/labels/hardware/generate', { type: k, specs: sp, label_length_mm: 36, options: {} });
      const src = 'data:image/png;base64,' + r.topview; img.src = src; try { sessionStorage.setItem('tv_' + k, src); } catch { /* full */ }
    } catch { img.remove(); }
  });

  // ── Options ──
  $$('[data-opt]', view).forEach((c) => { c.onchange = () => { S.opts[c.dataset.opt] = c.checked; refresh(); }; });

  // ── Part link ──
  function buildLink() {
    $$('[data-linkmode] button', view).forEach((b) => b.classList.toggle('on', b.dataset.v === S.link));
    const box = q('[data-linkbody]'); box.innerHTML = '';
    if (S.link === 'none') { box.innerHTML = '<p class="hint" style="margin:0">The label will have no QR code or part reference.</p>'; return; }
    if (S.link === 'qr') {
      const pf = partField({ label: 'Part encoded in the QR code', onChange: (p) => { S.qrPart = p; refresh(); } });
      if (S.qrPart) pf.set(S.qrPart, true);
      box.appendChild(pf.el);
      const row = el(html`<div class="row wrap" style="margin-top:10px"><span class="hint grow">QR contains {"part": id} — scanning it opens the part.</span><button class="btn sm" data-newpart>${icon('plus', 'sm')} Create part from specs</button></div>`);
      $('[data-newpart]', row).onclick = () => createFromSpecs((p) => { S.qrPart = p; pf.set(p, true); refresh(); });
      box.appendChild(row);
    } else {
      const pf = partField({ label: 'Part reference printed sideways', onChange: (p) => { S.refPart = p; refresh(); } });
      if (S.refPart) pf.set(S.refPart, true);
      box.appendChild(pf.el);
      box.appendChild(el('<p class="hint" style="margin:8px 0 0">The IPN (or name if no IPN) is printed vertically where the QR would be.</p>'));
    }
  }
  q('[data-linkmode]').onclick = (e) => { const b = e.target.closest('[data-v]'); if (!b) return; S.link = b.dataset.v; buildLink(); refresh(); };

  // ── Tape ──
  function tapeHint() {
    const len = num(q('[data-len]').value) || 36; const m = num(q('[data-margin]').value) || 0;
    q('[data-tapehint]').textContent = m > 0 ? `Printable content ≈ ${Math.max(10, len - 2 * m).toFixed(1)} mm (${m} mm dead zone each end; QL-710W ≈ 3 mm).` : `Printable content ${len} mm. Tip: set the cut margin to 3 mm for a QL-710W.`;
  }
  q('[data-len]').oninput = () => { S.length = num(q('[data-len]').value) || 36; tapeHint(); refresh(); };
  q('[data-margin]').oninput = () => { settings.cutMargin = num(q('[data-margin]').value) || 0; tapeHint(); refresh(); };
  q('[data-copies]').oninput = () => { S.copies = Math.max(1, parseInt(q('[data-copies]').value, 10) || 1); };

  // ── Payload / preview ──
  function payload() {
    return {
      type: S.type, specs: { ...S.specs[S.type], __init: undefined }, label_length_mm: S.length,
      options: {
        ...S.opts,
        show_qr: S.link === 'qr' && !!S.qrPart,
        qr_data: S.link === 'qr' && S.qrPart ? JSON.stringify({ part: Number(S.qrPart.pk) }) : '',
        part_ref: S.link === 'text' && S.refPart ? (S.refPart.ipn || S.refPart.name || '') : '',
        cut_margin_mm_per_side: settings.cutMargin,
      },
    };
  }
  let seq = 0;
  const doPreview = async () => {
    const my = ++seq; const pst = q('[data-pst]'); if (!pst) return;
    try {
      const r = await api.post('/api/labels/hardware/generate', payload());
      if (my !== seq) return;
      S.lastB64 = r.image;
      const img = el('<img alt="Label preview">'); img.src = 'data:image/png;base64,' + r.image;
      img.style.width = Math.min(r.width * 3, 2000) + 'px';
      const tape = q('[data-tape]'); tape.innerHTML = ''; tape.appendChild(img);
      q('[data-vt]').src = 'data:image/png;base64,' + r.topview; q('[data-vs]').src = 'data:image/png;base64,' + r.sideview;
      q('[data-dim]').textContent = `${r.width}×${r.height}px`;
      const warn = S.link === 'qr' && !S.qrPart ? 'Pick a part to add the QR code.' : '';
      status(pst, warn, warn ? '' : '');
    } catch (e) { if (my === seq) status(pst, e.message, 'err'); }
  };
  const refreshDeb = debounce(doPreview, 300);
  function refresh(now) { if (now) doPreview(); else refreshDeb(); }

  // ── Print ──
  q('[data-printer]').onchange = () => { settings.printer = q('[data-printer]').value.trim(); };
  q('[data-test]').onclick = (e) => busy(e.currentTarget, async () => {
    const addr = q('[data-printer]').value.trim(); const st = q('[data-tst]');
    if (!addr) { status(st, 'Enter the printer address first', 'err'); return; }
    settings.printer = addr; status(st, 'Testing…', 'loading');
    try { const r = await api.get('/api/labels/hardware/printer-test?printer=' + encodeURIComponent(addr)); status(st, 'Reachable — ' + r.printer, 'ok'); } catch (err) { status(st, err.message, 'err'); }
  });
  q('[data-print]').onclick = (e) => busy(e.currentTarget, async () => {
    const addr = q('[data-printer]').value.trim(); const st = q('[data-tst]');
    if (!addr) { status(st, 'Enter the printer address first', 'err'); q('[data-printer]').focus(); return; }
    settings.printer = addr; status(st, 'Printing…', 'loading');
    try { await api.post('/api/labels/hardware/print', { ...payload(), printer: addr, copies: S.copies }); status(st, `Sent ${S.copies} label${S.copies > 1 ? 's' : ''} to printer`, 'ok'); toast('Sent to printer'); } catch (err) { status(st, err.message, 'err'); }
  });
  q('[data-dl]').onclick = () => {
    if (!S.lastB64) return;
    const s = S.specs[S.type] || {};
    const sz = s.thread_size || s.fits_size || s.diameter_mm || s.shaft_diameter_mm || '';
    const a = document.createElement('a'); a.href = 'data:image/png;base64,' + S.lastB64;
    a.download = `label_${S.type}_${sz}.png`.replace(/\s+/g, '_'); a.click();
  };

  // ── Create part from specs (then link it) ──
  function createFromSpecs(onLinked) {
    collect();
    const s = S.specs[S.type];
    const loc = locationField({ label: 'Location' });
    const body = el(html`<div class="stack">
      <div class="field"><label>Name *</label><input class="input" data-n></div>
      <div class="field"><label>IPN</label><input class="input mono" data-i placeholder="Optional"></div>
      <div class="field"><label>Description</label><textarea data-d></textarea></div>
      <div class="grid-2"><div class="field"><label>Initial quantity</label><input class="input" inputmode="decimal" data-q placeholder="0"></div><div data-loc></div></div>
      <div class="row wrap"><label class="btn sm filebtn">${icon('camera', 'sm')} Photo<input type="file" accept="image/*" capture="environment" data-file></label><span class="small faint" data-fname>Optional — otherwise you can pick a web image.</span>
      <button class="btn sm ghost" data-refill>${icon('refresh', 'sm')} Refill from specs</button></div>
    </div>`);
    $('[data-loc]', body).replaceWith(loc.el);
    const fill = () => { $('[data-n]', body).value = suggestName(S.type, s); $('[data-d]', body).value = suggestDesc(S.type, s); };
    fill();
    $('[data-refill]', body).onclick = fill;
    $('[data-file]', body).onchange = (e) => { $('[data-fname]', body).textContent = e.target.files[0]?.name || ''; };
    const foot = el(html`<div style="display:contents"><button class="btn" data-c>Cancel</button><button class="btn primary" data-ok>${icon('check')} Create & link</button></div>`);
    const sh = sheet({ title: 'Create part from specs', body, foot });
    $('[data-c]', foot).onclick = () => sh.close();
    $('[data-ok]', foot).onclick = (ev) => busy(ev.currentTarget, async () => {
      const name = $('[data-n]', body).value.trim(); const description = $('[data-d]', body).value.trim();
      const qty = num($('[data-q]', body).value);
      if (!name) { toast('Name is required', 'err'); return; }
      if (qty > 0 && !loc.value) { toast('Choose a location for the initial quantity', 'err'); return; }
      try {
        const ipn = $('[data-i]', body).value.trim();
        const r = await Parts.create({ name, description, ipn, initial_qty: qty > 0 ? qty : '', location_id: qty > 0 ? loc.value.pk : '' });
        const p = { pk: r.pk, name: r.name, ipn, has_image: false };
        rememberPart(p); onLinked(p); sh.close(); toast(`Created & linked: ${partLabel(p)}`);
        const file = $('[data-file]', body).files[0];
        if (file) await uploadPartImage(r.pk, file);
        else if (settings.autoImageSearch) await imageSearch({ partId: r.pk, name, description });
      } catch (e) { toastErr(e); }
    });
  }

  buildForm(); buildLink(); tapeHint(); refresh(true);
}
