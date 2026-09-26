// Headless-browser smoke test for a generated tetris.html: load it, click START (or the first start/play button),
// press game keys for ~7 s, and report runtime errors plus whether the in-game timer and main canvas changed.
// Keyword checks and `node --check` miss runtime failures; this catches a game that loads but never runs.
//
//   docker run --rm -u $(id -u):$(id -g) -v <dir with tetris.html and this file>:/data --entrypoint node \
//     minlag/mermaid-cli /data/smoke_tetris.cjs tetris.html
// (minlag/mermaid-cli ships puppeteer and Chromium; on arm64 the browser is /usr/bin/chromium-browser.)
const puppeteer = require('/home/mermaidcli/node_modules/puppeteer');
const fs = require('fs');
(async () => {
  const file = process.argv[2] || 'tetris.html';
  const exe = ['/usr/bin/chromium-browser', '/usr/bin/chromium'].find(p => fs.existsSync(p));
  const b = await puppeteer.launch({executablePath: exe, args: ['--no-sandbox']});
  const p = await b.newPage();
  const errors = [];
  p.on('pageerror', e => errors.push(`${e.message} @ ${(e.stack.split('\n')[1] || '').trim()}`));
  await p.goto('file:///data/' + file);
  await new Promise(r => setTimeout(r, 800));
  const state = () => p.evaluate(() => ({
    hud: (document.body.innerText.replace(/\s+/g, ' ').match(/(SCORE|TIME)[^A-Z]*\S+/g) || []).join(' '),
    canvas: document.querySelector('canvas') ? document.querySelector('canvas').toDataURL().length : null}));
  const before = await state();
  const clicked = await p.evaluate(() => {
    const b = Array.from(document.querySelectorAll('button')).find(x => /start|play/i.test(x.textContent));
    if (b) { b.click(); return b.textContent.trim(); } return null; });
  await p.keyboard.press('Enter');
  const samples = [];
  for (let i = 0; i < 60; i++) {
    await p.keyboard.press(['ArrowLeft', 'ArrowRight', 'ArrowUp', 'Space', 'ArrowDown', 'KeyC'][i % 6]);
    await new Promise(r => setTimeout(r, 120));
    if (i % 20 === 19) samples.push(await state());
  }
  const after = await state();
  console.log(JSON.stringify({file, clicked, before, samples, after,
    canvas_changed: before.canvas !== after.canvas, hud_changed: before.hud !== after.hud, errors}, null, 1));
  await b.close();
})();
