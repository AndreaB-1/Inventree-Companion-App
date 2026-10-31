// End-to-end smoke test of the UI against the dev stack (mock InvenTree).
//   docker compose -f docker-compose.dev.yml --profile mock up -d --build
//   node tests/e2e_smoke.js [baseUrl] [screenshotDir]
// Requires the `playwright` npm package with Chromium.
const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

const BASE = process.argv[2] || 'http://localhost:8570';
const SHOTS = process.argv[3] || path.join(__dirname, 'screenshots');
fs.mkdirSync(SHOTS, { recursive: true });

let failures = 0;
const ok = (m) => console.log('  ✓ ' + m);
const bad = (m) => { failures++; console.log('  ✗ ' + m); };
async function step(name, fn) {
  console.log('• ' + name);
  try { await fn(); } catch (e) { bad(name + ': ' + e.message.split('\n')[0]); }
}

(async () => {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 860 } });
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error' && !/Failed to load resource/.test(m.text())) errors.push('console: ' + m.text()); });
  page.on('dialog', (d) => d.accept());
  const shot = (n) => page.screenshot({ path: path.join(SHOTS, n + '.png'), fullPage: true });

  await step('Home renders', async () => {
    await page.goto(BASE + '/#/');
    await page.waitForSelector('.actions-grid .action');
    ok((await page.locator('.action').count()) + ' quick actions');
    await shot('desktop-home');
  });

  await step('Search: all words must match', async () => {
    await page.goto(BASE + '/#/search?q=M3%20A2');
    await page.waitForSelector('.pcard');
    const n = await page.locator('.pcard').count();
    n === 2 ? ok('2 results for "M3 A2"') : bad('expected 2 results, got ' + n);
    await shot('desktop-search');
    await page.locator('.pcard').first().click();
    await page.waitForSelector('.part-name');
    ok('opened part: ' + (await page.textContent('.part-name')));
  });

  await step('Part page: stock add / remove', async () => {
    await page.goto(BASE + '/#/part/1');
    await page.waitForSelector('.stock-card');
    const first = page.locator('.stock-card').first();
    await first.locator('[data-amt]').fill('5');
    await first.locator('[data-op=add]').click();
    await page.waitForFunction(() => document.querySelector('.stock-card .qty')?.textContent === '125');
    ok('added 5 → 125');
    await page.locator('.stock-card').first().locator('[data-amt]').fill('25');
    await page.locator('.stock-card').first().locator('[data-op=remove]').click();
    await page.waitForFunction(() => document.querySelector('.stock-card .qty')?.textContent === '100');
    ok('removed 25 → 100');
    await shot('desktop-part');
  });

  await step('Part page: move part of a stock item', async () => {
    await page.locator('.stock-card').first().locator('[data-move]').click();
    await page.waitForSelector('.sheet');
    await page.fill('.sheet [data-q]', '10');
    await page.fill('.sheet .field[data-search] input', 'drawer');
    await page.waitForSelector('.ta-item');
    await page.locator('.ta-item').first().click();
    await page.click('.sheet [data-ok]');
    await page.waitForFunction(() => document.querySelectorAll('.stock-card').length === 3);
    const locs = await page.locator('.stock-card .loc').allTextContents();
    locs.some((l) => l.includes('Electronics Drawer')) ? ok('split 10 into Electronics Drawer') : bad('moved item not found: ' + locs);
  });

  await step('Part page: new stock item', async () => {
    await page.click('[data-addstock]');
    await page.fill('.sheet [data-q]', '7');
    await page.fill('.sheet .field[data-search] input', 'shelf');
    await page.waitForSelector('.ta-item');
    await page.locator('.ta-item').first().click();
    await page.click('.sheet [data-ok]');
    await page.waitForFunction(() => document.querySelectorAll('.stock-card').length === 4);
    ok('4 stock items');
  });

  await step('Part page: edit details', async () => {
    await page.click('[data-edit]');
    await page.fill('.sheet [data-n]', 'M3 × 12 – Socket Head Cap Screw A2 (edited)');
    await page.click('.sheet [data-ok]');
    await page.waitForFunction(() => document.querySelector('.part-name').textContent.includes('(edited)'));
    ok('name saved');
  });

  await step('Part page: photo upload', async () => {
    const png = path.join(SHOTS, '_upload.png');
    fs.writeFileSync(png, Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAEklEQVR4nGP4z8CAFWEXHbQSACj/P8Fu7N9hAAAAAElFTkSuQmCC', 'base64'));
    await page.setInputFiles('[data-photo]', png);
    await page.waitForSelector('[data-img] img');
    await page.waitForFunction(() => { const i = document.querySelector('[data-img] img'); return i && i.complete && i.naturalWidth > 0; });
    ok('image uploaded and displayed via proxy');
  });

  await step('Part page: InvenTree label print (stock items)', async () => {
    await page.click('[data-tab=labels]');
    await page.waitForSelector('[data-tpl] option[value="21"]', { state: 'attached' });
    await page.click('[data-tab=stock]');
    await page.locator('.stock-card [data-sel]').nth(0).check();
    await page.locator('.stock-card [data-sel]').nth(1).check();
    await page.click('[data-tab=labels]');
    const t = await page.textContent('[data-targets]');
    t.includes('2 labels') ? ok(t) : bad('targets: ' + t);
    await page.click('[data-pane=labels] [data-print]');
    await page.waitForSelector('[data-pane=labels] .status.ok');
    ok('print job sent');
    await shot('desktop-part-labels');
  });

  await step('New part: create with stock, print prompt', async () => {
    await page.goto(BASE + '/#/new');
    await page.waitForSelector('[data-name]');
    await page.fill('[data-name]', 'M6 × 20 – Hex Bolt 8.8');
    await page.fill('[data-desc]', 'M6 × 20 – Hex Bolt – DIN 933 – Zinc Plated 8.8');
    await page.fill('[data-ipn]', 'BLT-M6-20');
    await page.fill('[data-qty]', '50');
    await page.fill('.field[data-search] input', 'A1');
    await page.waitForSelector('.ta-item');
    await page.locator('.ta-item').first().click();
    await page.click('[data-create]');
    await page.waitForSelector('.sheet:has-text("Print a label")');
    ok('print prompt shown');
    await page.click('.sheet [data-yes]');
    // image search sheet opens afterwards (web search may be offline in CI)
    await page.waitForSelector('.sheet:has-text("Find an image")', { timeout: 15000 });
    ok('image search sheet opened');
    await page.click('.sheet [data-cancel]');
    await page.waitForSelector('text=Created “M6 × 20 – Hex Bolt 8.8”');
    ok('created panel shown');
    await shot('desktop-newpart');
  });

  await step('Label maker: live preview, types, QR link', async () => {
    await page.goto(BASE + '/#/labels?part=2');
    await page.waitForSelector('[data-tape] img');
    ok('preview rendered ' + (await page.textContent('[data-dim]')));
    await page.click('[data-type=nut]');
    await page.waitForFunction(() => document.querySelector('[data-f=nut_type]'));
    const af = await page.inputValue('[data-f=af_mm]');
    af === '7' ? ok('nut AF auto-filled 7') : bad('AF=' + af);
    await page.selectOption('[data-f=thread_size]', 'M8');
    await page.waitForFunction(() => document.querySelector('[data-f=af_mm]').value === '13');
    ok('M8 → AF 13');
    await page.click('[data-type=screw]');
    await page.selectOption('[data-f=standard]', 'ISO 10642');
    const head = await page.inputValue('[data-f=head_type]');
    head === 'Countersunk' ? ok('standard → Countersunk head') : bad('head=' + head);
    const linked = await page.locator('[data-linkbody] .picked').isVisible();
    linked ? ok('part #2 pre-linked for QR') : bad('QR part not linked');
    await page.waitForTimeout(800);
    await shot('desktop-labels');
    const [dl] = await Promise.all([page.waitForEvent('download'), page.click('[data-dl]')]);
    ok('download: ' + dl.suggestedFilename());
  });

  await step('Label maker: create part from specs', async () => {
    await page.click('[data-newpart]');
    const nm = await page.inputValue('.sheet [data-n]');
    nm.includes('Countersunk') ? ok('name from specs: ' + nm) : bad('name=' + nm);
    await page.click('.sheet [data-ok]');
    await page.waitForSelector('.sheet:has-text("Find an image")', { timeout: 15000 });
    await page.click('.sheet [data-cancel]');
    ok('created & linked');
  });

  await step('Tidy names: load + analyze (AI off → error shown)', async () => {
    await page.goto(BASE + '/#/tidy');
    await page.click('[data-load]');
    await page.waitForSelector('.tidy-row');
    ok((await page.locator('.tidy-row').count()) + ' rows');
    await page.locator('.tidy-row [data-an]').first().click();
    await page.waitForSelector('.tidy-row .status.err');
    ok('error surfaced: ' + (await page.locator('.tidy-row .status.err').first().textContent()).slice(0, 60));
    await shot('desktop-tidy');
  });

  await step('Settings: hardware lists persist', async () => {
    await page.goto(BASE + '/#/settings');
    await page.click('[data-cfgtab] [data-v=materials]');
    await page.fill('[data-add=mat] input', 'Inconel 625');
    await page.click('[data-add=mat] button');
    await page.click('[data-save]');
    await page.waitForSelector('[data-cst].ok');
    await page.reload();
    await page.click('[data-cfgtab] [data-v=materials]');
    (await page.locator('.pill:has-text("Inconel 625")').count()) ? ok('material saved') : bad('material not persisted');
    await page.waitForSelector('[data-tpl] option[value="12"]', { state: 'attached' });
    ok('templates loaded into default selector');
    await shot('desktop-settings');
  });

  await step('Scan: QR photo opens part', async () => {
    const qr = process.env.QR_PNG;
    if (!qr) { ok('skipped (set QR_PNG to a QR image encoding {"part": 3})'); return; }
    await page.goto(BASE + '/#/scan');
    await page.setInputFiles('[data-file]', qr);
    await page.waitForURL(/#\/part\/3/, { timeout: 20000 });
    ok('navigated to part 3');
  });

  await step('Scan: manual lookup by IPN', async () => {
    await page.goto(BASE + '/#/scan');
    await page.fill('[data-form] input', 'NUT-M4');
    await page.click('[data-form] button');
    await page.waitForURL(/#\/part\/3/);
    ok('IPN → part 3');
  });

  await step('Mobile layout', async () => {
    const m = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, deviceScaleFactor: 2 });
    const mp = await m.newPage();
    mp.on('pageerror', (e) => errors.push('mobile pageerror: ' + e.message));
    for (const [name, hash, sel] of [['home', '#/', '.action'], ['search', '#/search', '.pcard'], ['part', '#/part/1', '.stock-card'], ['labels', '#/labels', '[data-tape] img'], ['more', '#/more', '.list-item'], ['scan', '#/scan', '.scan-drop']]) {
      await mp.goto(BASE + '/' + hash); await mp.waitForSelector(sel); await mp.waitForTimeout(300);
      const overflow = await mp.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);
      overflow ? bad(name + ': horizontal overflow') : ok(name + ': fits 390px');
      await mp.screenshot({ path: path.join(SHOTS, 'mobile-' + name + '.png'), fullPage: false });
    }
    await mp.goto(BASE + '/#/part/1'); await mp.waitForSelector('.stock-card');
    await mp.locator('.stock-card [data-move]').first().click(); await mp.waitForSelector('.sheet');
    await mp.waitForTimeout(300);
    await mp.screenshot({ path: path.join(SHOTS, 'mobile-move-sheet.png') });
    ok('move sheet (bottom sheet) shown');
    // dark theme
    await mp.evaluate(() => { localStorage.setItem('theme', 'dark'); });
    await mp.goto(BASE + '/#/'); await mp.reload(); await mp.waitForSelector('.action'); await mp.waitForTimeout(200);
    (await mp.getAttribute('html', 'data-theme')) === 'dark' ? ok('dark theme applied') : bad('dark theme not applied');
    await mp.screenshot({ path: path.join(SHOTS, 'mobile-home-dark.png') });
    ok('dark theme screenshot');
    await m.close();
  });

  if (errors.length) { errors.forEach((e) => bad(e)); }
  await browser.close();
  console.log(failures ? `\n${failures} FAILURE(S)` : '\nALL PASSED');
  process.exit(failures ? 1 : 0);
})();
