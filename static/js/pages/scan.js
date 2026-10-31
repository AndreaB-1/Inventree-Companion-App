// Scan a QR label (photo → server decode, or live camera where the browser allows it).
import { $, html, el, icon, api, Parts, status, toast } from '../core.js';
import { go } from '../app.js';

export const title = 'Scan';
export const narrow = true;

// Turn decoded QR text into a destination. Returns true if handled.
export async function resolveCode(text, payload) {
  let obj = payload || null;
  if (!obj) { try { const j = JSON.parse(text); if (j && typeof j === 'object') obj = j; } catch { /* not JSON */ } }
  if (obj) {
    if (obj.part != null) { go('#/part/' + Number(obj.part)); return true; }
    if (obj.stockitem != null) {
      const s = await api.get('/api/stock/item?pk=' + Number(obj.stockitem));
      go(`#/part/${s.part}?tab=stock`); return true;
    }
    if (obj.stocklocation != null) { go('#/search'); toast('That is a location label (#' + obj.stocklocation + ')', ''); return true; }
    if (obj.query) text = String(obj.query);
  }
  const t = String(text || '').trim();
  let m = t.match(/\/part\/(\d+)/i); if (m) { go('#/part/' + m[1]); return true; }
  m = t.match(/\/stock\/item\/(\d+)/i);
  if (m) { const s = await api.get('/api/stock/item?pk=' + m[1]); go(`#/part/${s.part}?tab=stock`); return true; }
  return lookup(t);
}

// Manual lookup: #ID / number → part; else IPN; else search.
export async function lookup(q) {
  q = String(q || '').trim(); if (!q) return false;
  if (/^#?\d+$/.test(q)) { try { const p = await Parts.detail(q.replace('#', '')); go('#/part/' + p.pk); return true; } catch { /* fall through */ } }
  try { const p = await Parts.byIpn(q); go('#/part/' + p.pk); return true; } catch { /* fall through */ }
  go('#/search?q=' + encodeURIComponent(q)); return true;
}

export function render(view) {
  const live = 'BarcodeDetector' in window && window.isSecureContext && navigator.mediaDevices?.getUserMedia;
  view.innerHTML = String(html`
    <label class="scan-drop" data-drop>
      <span class="big">${icon('camera')}</span>
      <span class="t-title" style="font-size:17px">Take a photo of the QR code</span>
      <span class="small faint">Fill the frame with the code. Works over plain HTTP on your LAN.</span>
      <input type="file" accept="image/*" capture="environment" data-file>
    </label>
    <div class="status" data-st style="margin:12px 2px"></div>
    ${live ? html`<button class="btn block" data-live>${icon('scan')} Use live camera instead</button>` : ''}
    <div class="section-title">Or look it up</div>
    <form class="row" data-form>
      <div class="field grow"><input class="input" name="q" placeholder="Part #ID, IPN or name" enterkeyhint="go"></div>
      <button class="btn primary">Go</button>
    </form>
  `);
  const st = $('[data-st]', view); const drop = $('[data-drop]', view);
  let stream = null; let raf = 0;

  $('[data-file]', view).onchange = async (e) => {
    const file = e.target.files[0]; e.target.value = ''; if (!file) return;
    const prev = $('img.preview', drop); if (prev) prev.remove();
    const img = el('<img class="preview" alt="">'); img.src = URL.createObjectURL(file); drop.prepend(img);
    status(st, 'Decoding…', 'loading');
    const fd = new FormData(); fd.append('image', file);
    try {
      const r = await api.form('/api/decode_qr', fd);
      status(st, 'Found: ' + (r.text || '').slice(0, 80), 'ok');
      await resolveCode(r.text, r.payload);
    } catch (err) { status(st, (err.message || 'No QR found') + ' — try again closer, with more light.', 'err'); }
  };

  $('[data-form]', view).onsubmit = async (e) => { e.preventDefault(); status(st, 'Looking up…', 'loading'); await lookup(e.target.q.value); };

  const liveBtn = $('[data-live]', view);
  if (liveBtn) liveBtn.onclick = async () => {
    try {
      const det = new window.BarcodeDetector({ formats: ['qr_code'] });
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
      const video = el('<video playsinline muted></video>'); video.srcObject = stream; drop.prepend(video); await video.play();
      liveBtn.hidden = true; status(st, 'Point the camera at a QR code…', 'loading');
      const tick = async () => {
        if (!stream) return;
        try { const codes = await det.detect(video); if (codes.length) { const text = codes[0].rawValue; stop(); status(st, 'Found: ' + text.slice(0, 80), 'ok'); await resolveCode(text); return; } } catch { /* frame not ready */ }
        raf = requestAnimationFrame(tick);
      };
      tick();
    } catch (err) { status(st, 'Camera unavailable: ' + err.message, 'err'); }
  };
  function stop() { cancelAnimationFrame(raf); if (stream) stream.getTracks().forEach((t) => t.stop()); stream = null; }
  return stop;
}
