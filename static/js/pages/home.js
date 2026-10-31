import { $, $$, html, icon, raw, recentParts, thumb, pickPart, fmtQty } from '../core.js';
import { go } from '../app.js';

export const title = 'Home';

const ACTIONS = [
  { key: 'scan', title: 'Scan QR', sub: 'Open a part from its label', icon: 'scan', primary: true },
  { key: 'new', title: 'New part', sub: 'Create with AI naming', icon: 'plus' },
  { key: 'adjust', title: 'Adjust stock', sub: 'Add or remove quantity', icon: 'layers' },
  { key: 'move', title: 'Move stock', sub: 'Transfer to a location', icon: 'move' },
  { key: 'labels', title: 'Label maker', sub: 'Hardware tape labels', icon: 'tag' },
  { key: 'tidy', title: 'Tidy names', sub: 'AI naming review', icon: 'wand' },
];

export function render(view) {
  const recent = recentParts();
  view.innerHTML = String(html`
    <form class="searchbox field hero-search" data-search>
      ${icon('search')}<input class="input" name="q" placeholder="Search parts by name, IPN or #ID" enterkeyhint="search">
    </form>
    <div class="actions-grid">
      ${ACTIONS.map((a) => html`<button type="button" class="action ${a.primary ? 'primary' : ''}" data-act="${a.key}">
        <span class="a-ic">${icon(a.icon)}</span><span class="a-t">${a.title}</span><span class="a-s">${a.sub}</span></button>`)}
    </div>
    <div class="section-title">Recently viewed</div>
    ${recent.length ? html`<div class="card flush"><div class="list">
      ${recent.slice(0, 8).map((p) => html`<a class="list-item" href="#/part/${p.pk}">${thumb(p)}
        <div class="grow"><div class="t-title ellipsis">${p.name}</div><div class="t-sub">${[p.ipn ? 'IPN ' + p.ipn : '', '#' + p.pk].filter(Boolean).join(' · ')}</div></div>
        ${icon('chev', 'chev')}</a>`)}
    </div></div>` : html`<div class="card"><div class="empty">${icon('box')}<span>Parts you open will show up here.</span></div></div>`}
  `);

  $('[data-search]', view).onsubmit = (e) => {
    e.preventDefault();
    const q = e.target.q.value.trim();
    if (/^#\d+$/.test(q)) go('#/part/' + q.slice(1));
    else go('#/search' + (q ? '?q=' + encodeURIComponent(q) : ''));
  };
  $$('[data-act]', view).forEach((b) => {
    b.onclick = async () => {
      const k = b.dataset.act;
      if (k === 'adjust' || k === 'move') {
        const p = await pickPart(k === 'move' ? 'Move stock — choose part' : 'Adjust stock — choose part');
        if (p) go(`#/part/${p.pk}?tab=stock${k === 'move' ? '&action=move' : ''}`);
      } else go('#/' + k);
    };
  });
}
export { fmtQty, raw };
