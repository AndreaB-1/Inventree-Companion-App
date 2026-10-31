// App shell: navigation, hash router, theme.
import { $, $$, html, icon, raw, api, store, closeAllSheets, toastErr } from './core.js';

const NAV = [
  { href: '#/', key: 'home', label: 'Home', icon: 'home' },
  { href: '#/search', key: 'search', label: 'Search', icon: 'search' },
  { href: '#/scan', key: 'scan', label: 'Scan', icon: 'scan' },
  { href: '#/new', key: 'new', label: 'New part', icon: 'plus' },
  { group: 'Tools' },
  { href: '#/labels', key: 'labels', label: 'Label maker', icon: 'tag' },
  { href: '#/tidy', key: 'tidy', label: 'Tidy names', icon: 'wand' },
  { href: '#/settings', key: 'settings', label: 'Settings', icon: 'settings' },
];
const TABS = [
  { href: '#/', key: 'home', label: 'Home', icon: 'home' },
  { href: '#/search', key: 'search', label: 'Search', icon: 'search' },
  { href: '#/scan', key: 'scan', label: 'Scan', icon: 'scan', scan: true },
  { href: '#/labels', key: 'labels', label: 'Labels', icon: 'tag' },
  { href: '#/more', key: 'more', label: 'More', icon: 'grid' },
];

// route pattern -> page module
const ROUTES = [
  { re: /^\/$/, page: 'home', nav: 'home' },
  { re: /^\/search$/, page: 'search', nav: 'search' },
  { re: /^\/scan$/, page: 'scan', nav: 'scan' },
  { re: /^\/new$/, page: 'newpart', nav: 'new', tab: 'home' },
  { re: /^\/part\/(\d+)$/, page: 'part', nav: 'search', back: true, keys: ['id'] },
  { re: /^\/labels$/, page: 'labels', nav: 'labels' },
  { re: /^\/tidy$/, page: 'tidy', nav: 'tidy', tab: 'more' },
  { re: /^\/settings$/, page: 'settings', nav: 'settings', tab: 'more' },
  { re: /^\/more$/, page: 'more', nav: 'more' },
];

function buildNav() {
  $('#railLinks').innerHTML = NAV.map((n) => (n.group
    ? `<div class="rail-group">${n.group}</div>`
    : String(html`<a href="${n.href}" data-key="${n.key}">${icon(n.icon)}<span>${n.label}</span></a>`))).join('');
  $('#tabbar').innerHTML = TABS.map((t) => String(t.scan
    ? html`<a href="${t.href}" data-key="${t.key}" class="scan-tab"><span class="scan-bubble">${icon(t.icon)}</span><span>${t.label}</span></a>`
    : html`<a href="${t.href}" data-key="${t.key}">${icon(t.icon)}<span>${t.label}</span></a>`)).join('');
  $('#backBtn').innerHTML = String(icon('back'));
  $('#backBtn').onclick = () => { if (history.length > 1) history.back(); else location.hash = '#/'; };
}

// ── Theme ──
function applyTheme(pref) {
  const t = pref === 'system' ? (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light') : pref;
  document.documentElement.setAttribute('data-theme', t);
  $('#themeToggle').innerHTML = String(icon(t === 'dark' ? 'sun' : 'moon'));
  const meta = document.querySelector('meta[name=theme-color]');
  if (meta) meta.content = t === 'dark' ? '#141413' : '#f4f2ee';
}
export function setTheme(pref) { store.setRaw('theme', pref); applyTheme(pref); }
export const themePref = () => store.raw('theme', 'system');
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { if (themePref() === 'system') applyTheme('system'); });

// ── Page chrome API (used by pages) ──
export const chrome = {
  title(t) { $('#pageTitle').textContent = t; document.title = t + ' · InvenTree Companion'; },
  actions(markup) { $('#appbarActions').innerHTML = markup ? String(markup) : ''; return $('#appbarActions'); },
};

// ── Router ──
let cleanup = null; let navSeq = 0;
export function parseHash() {
  const h = location.hash.replace(/^#/, '') || '/';
  const [path, q = ''] = h.split('?');
  return { path: path || '/', query: Object.fromEntries(new URLSearchParams(q)) };
}
export const go = (hash) => { location.hash = hash; };

async function route() {
  const my = ++navSeq;
  const { path, query } = parseHash();
  const r = ROUTES.find((x) => x.re.test(path)) || ROUTES[0];
  const m = path.match(r.re) || [];
  const params = {}; (r.keys || []).forEach((k, i) => { params[k] = m[i + 1]; });

  closeAllSheets();
  if (cleanup) { try { cleanup(); } catch { /* ignore */ } cleanup = null; }
  $$('#railLinks a').forEach((a) => a.classList.toggle('active', a.dataset.key === r.nav));
  $$('#tabbar a').forEach((a) => a.classList.toggle('active', a.dataset.key === (r.tab || r.nav)));
  $('#backBtn').hidden = !r.back;
  chrome.actions('');

  const view = $('#view');
  let mod;
  try { mod = await import(`./pages/${r.page}.js`); } catch (e) { toastErr(e); return; }
  if (my !== navSeq) return;
  view.className = 'view view-enter' + (mod.narrow ? ' narrow' : '');
  view.innerHTML = '';
  chrome.title(mod.title || 'InvenTree Companion');
  window.scrollTo(0, 0);
  try { cleanup = (await mod.render(view, params, query)) || null; } catch (e) { console.error(e); toastErr(e); }
}

// ── Boot ──
buildNav();
applyTheme(themePref());
$('#themeToggle').onclick = () => setTheme(document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
window.addEventListener('hashchange', route);
window.addEventListener('scroll', () => $('#appbar').classList.toggle('scrolled', window.scrollY > 4), { passive: true });
// Server info (non-blocking)
export const appInfo = api.get('/api/app/info').catch(() => ({}));
route();
appInfo.then((i) => { if (i.inventree_url) $('#serverBadge').textContent = i.inventree_url.replace(/^https?:\/\//, '').replace(/\/$/, ''); });
export { raw };
