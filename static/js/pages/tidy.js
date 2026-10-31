// AI review of part names & descriptions (v1 "Naming Standards").
import { $, $$, html, el, icon, api, Parts, thumb, status, toast, toastErr } from '../core.js';

export const title = 'Tidy names';

const BATCH = 30;

export function render(view) {
  const parts = new Map(); // pk -> {name, description, ipn, has_image}
  const sugg = new Map(); // pk -> {name, description}
  let offset = 0; let running = false; let stopReq = false;

  view.innerHTML = String(html`
    <div class="card stack">
      <p class="muted" style="margin:0">Loads parts ${BATCH} at a time. Analyze them one by one or all at once — the AI suggests a name and description following the naming convention. Nothing is saved until you accept.</p>
      <div class="row wrap">
        <button class="btn primary" data-load>${icon('refresh')} Load parts</button>
        <button class="btn" data-all disabled>${icon('sparkles')} Analyze all loaded</button>
        <button class="btn" data-stop hidden>Stop</button>
        <button class="btn ok" data-acceptall hidden>${icon('check')} Accept all suggestions</button>
        <label class="check"><input type="checkbox" data-onlychg> Only show suggested changes</label>
      </div>
      <div class="progress" data-prog hidden><div style="width:0"></div></div>
      <div class="status" data-st></div>
    </div>
    <div class="card flush" data-listcard hidden style="margin-top:14px"><div data-list></div></div>
    <div class="row" style="justify-content:center;margin-top:14px"><button class="btn" data-more hidden>Load ${BATCH} more</button></div>
  `);
  const q = (s) => $(s, view);
  const list = q('[data-list]'); const st = q('[data-st]');

  function rowHtml(pk) {
    const p = parts.get(pk); const s = sugg.get(pk);
    const field = (lbl, cur, nw, state) => html`<div class="diff"><span class="lbl">${lbl}</span>
      <span class="old ${nw ? 'strike' : ''}">${cur || '—'}</span>
      ${nw ? html`<span class="new">→ ${nw}</span>` : state === 'same' ? html`<span class="same">Already follows convention</span>` : ''}</div>`;
    const done = p._state;
    return html`${thumb({ pk, has_image: p.has_image })}
      <div class="stack tight" style="min-width:0">
        <div class="row wrap" style="gap:6px"><a class="t-title" href="#/part/${pk}" style="color:inherit;text-decoration:none">#${pk}</a>${p.ipn ? html`<span class="chip mono">IPN ${p.ipn}</span>` : ''}
          ${done === 'saved' ? html`<span class="chip ok">Updated</span>` : done === 'skipped' ? html`<span class="chip">Skipped</span>` : done === 'same' ? html`<span class="chip ok">OK</span>` : ''}</div>
        ${field('Name', p.name, s && s.name !== p.name ? s.name : '', done)}
        ${field('Description', p.description, s && s.description !== p.description ? s.description : '', done)}
        <div class="status small" data-rst></div>
      </div>
      <div class="tidy-actions">
        <button class="btn sm" data-an title="Analyze">${icon('sparkles', 'sm')}<span>Analyze</span></button>
        ${s ? html`<button class="btn sm ok" data-ac>${icon('check', 'sm')}<span>Accept</span></button><button class="btn sm ghost" data-sk>Skip</button>` : ''}
      </div>`;
  }
  function paint(pk) {
    let row = $(`[data-pk="${pk}"]`, list);
    if (!row) { row = el(`<div class="tidy-row" data-pk="${pk}"></div>`); list.appendChild(row); }
    row.innerHTML = String(rowHtml(pk));
    $('[data-an]', row).onclick = () => analyze(pk);
    const ac = $('[data-ac]', row); if (ac) ac.onclick = () => accept(pk);
    const sk = $('[data-sk]', row); if (sk) sk.onclick = () => { sugg.delete(pk); parts.get(pk)._state = 'skipped'; paint(pk); sync(); };
    applyFilter(row, pk);
  }
  function applyFilter(row, pk) { row.hidden = q('[data-onlychg]').checked && !sugg.has(pk); }
  function sync() {
    q('[data-acceptall]').hidden = !sugg.size || running;
    q('[data-acceptall]').innerHTML = String(html`${icon('check')} Accept all (${sugg.size})`);
    q('[data-all]').disabled = !parts.size || running;
  }

  async function load(reset) {
    const btn = reset ? q('[data-load]') : q('[data-more]');
    if (reset) { parts.clear(); sugg.clear(); offset = 0; list.innerHTML = ''; }
    btn.classList.add('busy'); status(st, 'Loading…', 'loading');
    try {
      const rows = await Parts.search('', BATCH, offset);
      rows.forEach((p) => { parts.set(p.pk, { name: p.name || '', description: p.description || '', ipn: p.ipn || '', has_image: p.has_image }); paint(p.pk); });
      offset += rows.length;
      q('[data-listcard]').hidden = !parts.size;
      q('[data-more]').hidden = rows.length < BATCH;
      status(st, `${parts.size} part${parts.size === 1 ? '' : 's'} loaded`, 'ok');
    } catch (e) { status(st, e.message, 'err'); }
    finally { btn.classList.remove('busy'); sync(); }
  }

  async function analyze(pk) {
    const row = $(`[data-pk="${pk}"]`, list); const p = parts.get(pk);
    const b = $('[data-an]', row); b.classList.add('busy');
    status($('[data-rst]', row), 'Analyzing…', 'loading');
    try {
      const r = await api.post('/api/ai/suggest-part', { prompt: [p.name, p.description].filter(Boolean).join(' — '), strict: false });
      const name = (r.name || '').trim() || p.name; const description = (r.description || '').trim() || p.description;
      if (name !== p.name || description !== p.description) { sugg.set(pk, { name, description }); p._state = ''; } else { sugg.delete(pk); p._state = 'same'; }
      paint(pk);
    } catch (e) { b.classList.remove('busy'); status($('[data-rst]', row), e.message, 'err'); }
    sync();
  }
  async function accept(pk) {
    const s = sugg.get(pk); if (!s) return;
    const row = $(`[data-pk="${pk}"]`, list); const b = $('[data-ac]', row); if (b) b.classList.add('busy');
    try {
      await Parts.update(pk, { name: s.name, description: s.description });
      Object.assign(parts.get(pk), s, { _state: 'saved' }); sugg.delete(pk); paint(pk);
      return true;
    } catch (e) { if (b) b.classList.remove('busy'); status($('[data-rst]', row), e.message, 'err'); toastErr(e); return false; }
    finally { sync(); }
  }

  async function runAll(fn, pks, label) {
    running = true; stopReq = false; sync();
    q('[data-stop]').hidden = false; const prog = q('[data-prog]'); prog.hidden = false;
    let i = 0;
    for (const pk of pks) {
      if (stopReq) break;
      i++; status(st, `${label} ${i}/${pks.length}…`, 'loading'); prog.firstElementChild.style.width = (i / pks.length * 100) + '%';
      await fn(pk);
      await new Promise((r) => setTimeout(r, 150));
    }
    running = false; q('[data-stop]').hidden = true; prog.hidden = true;
    status(st, stopReq ? `Stopped after ${i}` : `Done — ${i} part${i === 1 ? '' : 's'}`, 'ok'); sync();
  }

  q('[data-load]').onclick = () => load(true);
  q('[data-more]').onclick = () => load(false);
  q('[data-all]').onclick = () => runAll(analyze, [...parts.keys()], 'Analyzing');
  q('[data-acceptall]').onclick = () => runAll(accept, [...sugg.keys()], 'Saving').then(() => toast('Suggestions applied'));
  q('[data-stop]').onclick = () => { stopReq = true; };
  q('[data-onlychg]').onchange = () => $$('.tidy-row', list).forEach((r) => applyFilter(r, Number(r.dataset.pk)));
  return () => { stopReq = true; };
}
