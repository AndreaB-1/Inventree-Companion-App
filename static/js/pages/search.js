import { $, html, raw, icon, el, Parts, store, fmtQty, thumb, status, debounce, lightbox } from '../core.js';
import { chrome } from '../app.js';

export const title = 'Search';

const PAGE = 24;
const words = (q) => (q || '').toLowerCase().split(/\s+/).filter(Boolean);
const matchesAll = (p, ws) => {
  if (!ws.length) return true;
  const blob = `${p.name || ''} ${p.ipn || ''} ${p.description || ''} #${p.pk}`.toLowerCase();
  return ws.every((w) => blob.includes(w));
};

export function render(view, _params, query) {
  let mode = store.get('search_view', 'grid');
  const st = { q: query.q || '', offset: 0, loading: false, done: false, shown: 0, seq: 0 };

  const actions = chrome.actions(html`<div class="segmented" data-mode>
    <button data-v="grid" aria-label="Grid view">${icon('grid', 'sm')}</button>
    <button data-v="list" aria-label="List view">${icon('list', 'sm')}</button></div>`);
  const syncMode = () => actions.querySelectorAll('[data-v]').forEach((b) => b.classList.toggle('on', b.dataset.v === mode));
  syncMode();

  view.innerHTML = String(html`
    <form class="searchbox field" data-form style="margin:4px 0 6px">
      ${icon('search')}<input class="input" name="q" value="${st.q}" placeholder="All words must match, any order — blank to browse" enterkeyhint="search">
    </form>
    <div class="status" data-st style="margin:8px 2px 12px"></div>
    <div data-results></div>
    <div class="row" style="justify-content:center;margin-top:16px"><button class="btn" data-more hidden>Load more</button></div>
  `);
  const input = $('input', view); const out = $('[data-results]', view);
  const stEl = $('[data-st]', view); const more = $('[data-more]', view);

  const container = () => {
    let c = out.firstElementChild;
    if (!c) {
      c = mode === 'grid' ? el('<div class="pgrid"></div>') : el('<div class="card flush"><div class="list"></div></div>');
      out.appendChild(c);
    }
    return mode === 'grid' ? c : c.firstElementChild;
  };
  const card = (p) => {
    const stock = p.in_stock != null ? html`<span class="qty ${Number(p.in_stock) ? '' : 'zero'}">${fmtQty(p.in_stock)}</span>` : '';
    if (mode === 'grid') {
      const n = el(String(html`<a class="pcard" href="#/part/${p.pk}">
        <div class="pimg">${p.has_image ? html`<img loading="lazy" alt="" src="${Parts.imgUrl(p.pk, true)}">` : icon('box', 'lg')}</div>
        <div class="pbody"><div class="pname">${p.name || 'Unnamed'}</div>
        <div class="pfoot"><span class="faint ellipsis">${p.ipn || '#' + p.pk}</span>${stock}</div></div></a>`));
      const img = n.querySelector('img');
      if (img) img.onerror = () => { img.replaceWith(el(String(icon('box', 'lg')))); };
      return n;
    }
    return el(String(html`<a class="list-item" href="#/part/${p.pk}">${thumb(p)}
      <div class="grow"><div class="t-title ellipsis">${p.name || 'Unnamed'}</div>
      <div class="t-sub ellipsis">${[p.ipn ? 'IPN ' + p.ipn : '', '#' + p.pk, p.description].filter(Boolean).join(' · ')}</div></div>${stock}</a>`));
  };

  async function load(reset) {
    if (reset) { st.offset = 0; st.done = false; st.shown = 0; out.innerHTML = ''; st.seq++; st.loading = false; }
    if (st.loading || st.done) return;
    const my = st.seq; st.loading = true; more.hidden = true;
    const ws = words(st.q);
    status(stEl, ws.length ? `Matching: ${ws.join(' + ')}` : 'Browsing all parts', 'loading');
    try {
      // Server search uses the first word; remaining words are filtered client-side.
      // Keep fetching pages until we have something to show or run out.
      let added = 0;
      for (let guard = 0; guard < 6 && !st.done && added === 0; guard++) {
        const page = await Parts.search(ws[0] || '', PAGE, st.offset);
        if (my !== st.seq) return;
        st.offset += page.length;
        if (page.length < PAGE) st.done = true;
        const c = container();
        page.filter((p) => matchesAll(p, ws)).forEach((p) => { c.appendChild(card(p)); added++; });
      }
      st.shown += added;
      if (!st.shown) {
        out.innerHTML = String(html`<div class="card"><div class="empty">${icon('search')}<span>No parts found${st.q ? raw(` for “${String(st.q).replace(/[<>&]/g, '')}”`) : ''}</span></div></div>`);
      }
      status(stEl, st.shown ? `${st.shown} part${st.shown === 1 ? '' : 's'}${st.done ? '' : '+'}` : '', '');
      more.hidden = st.done;
    } catch (e) { status(stEl, e.message, 'err'); }
    finally { if (my === st.seq) st.loading = false; }
  }

  const submit = () => {
    st.q = input.value.trim();
    history.replaceState(null, '', '#/search' + (st.q ? '?q=' + encodeURIComponent(st.q) : ''));
    load(true);
  };
  $('[data-form]', view).onsubmit = (e) => { e.preventDefault(); submit(); };
  input.addEventListener('input', debounce(submit, 400));
  more.onclick = () => load(false);
  actions.querySelector('[data-mode]').onclick = (e) => {
    const b = e.target.closest('[data-v]'); if (!b || b.dataset.v === mode) return;
    mode = b.dataset.v; store.set('search_view', mode); syncMode(); load(true);
  };
  // Infinite scroll
  const io = new IntersectionObserver((es) => { if (es.some((x) => x.isIntersecting) && !more.hidden) load(false); }, { rootMargin: '300px' });
  io.observe(more);
  load(true);
  if (!st.q && window.matchMedia('(min-width: 900px)').matches) input.focus();
  return () => io.disconnect();
}
export { lightbox };
