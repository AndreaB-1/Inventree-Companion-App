import { html, icon } from '../core.js';

export const title = 'More';
export const narrow = true;

const ITEMS = [
  { href: '#/new', icon: 'plus', t: 'New part', s: 'Create a part, set stock, add a photo' },
  { href: '#/labels', icon: 'tag', t: 'Label maker', s: 'Brother QL hardware labels' },
  { href: '#/tidy', icon: 'wand', t: 'Tidy names', s: 'AI review of names & descriptions' },
  { href: '#/settings', icon: 'settings', t: 'Settings', s: 'Theme, printers, hardware lists' },
];

export function render(view) {
  view.innerHTML = String(html`<div class="card flush"><div class="list">
    ${ITEMS.map((i) => html`<a class="list-item" href="${i.href}">
      <span class="thumb" style="background:var(--accent-soft);color:var(--accent);border:0">${icon(i.icon)}</span>
      <div class="grow"><div class="t-title">${i.t}</div><div class="t-sub">${i.s}</div></div>${icon('chev', 'chev')}</a>`)}
  </div></div>`);
}
