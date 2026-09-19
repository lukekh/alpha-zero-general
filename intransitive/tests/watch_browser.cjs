/* Optional integration test against a dedicated local play server.
 * PLAYWRIGHT_MODULE=/path/to/playwright node intransitive/tests/watch_browser.cjs
 * Server: python -m intransitive.play --port 8879 --opponent local
 */
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');

(async () => {
  const browser = await chromium.launch({headless: true,
    executablePath: process.env.CHROME_BINARY || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
  const page = await browser.newPage({viewport: {width: 1200, height: 1000}});
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  try {
    await page.goto('http://127.0.0.1:8879/');
    await page.waitForFunction(() => typeof live !== 'undefined' && !busy);
    await page.selectOption('#game-mode', 'watch');
    await page.selectOption('#blue-bot', 'random');
    await page.selectOption('#red-bot', 'random');
    await page.fill('#pause-seconds', '60');
    await page.click('#restart');
    await page.waitForFunction(() => live.mode === 'watch' && live.ply >= 8);
    assert.equal(await page.evaluate(() => state.ply), 0, 'UI delay must not throttle production');
    await page.click('#generation-toggle');
    await page.waitForFunction(() => !generating && !busy);
    assert.equal(await page.locator('#board button:disabled').count(), 81);
    await page.click('#next-move');
    assert.equal(await page.evaluate(() => state.ply), 1);
    await page.keyboard.press('ArrowRight');
    assert.equal(await page.evaluate(() => state.ply), 2);
    await page.keyboard.press('ArrowLeft');
    assert.equal(await page.evaluate(() => state.ply), 1);
    await page.click('#pause-seconds');
    await page.keyboard.press('ArrowLeft');
    assert.equal(await page.evaluate(() => state.ply), 1, 'Input keyboard controls must not seek');
    await page.click('#first-move');
    assert.equal(await page.evaluate(() => state.ply), 0);
    await page.click('#last-move');
    assert.equal(await page.evaluate(() => state.ply === live.ply), true);
    await page.keyboard.press('Home');
    assert.equal(await page.evaluate(() => state.ply), 0);
    await page.keyboard.press('End');
    assert.equal(await page.evaluate(() => state.ply === live.ply), true);
    await page.fill('#pause-seconds', '0.5');
    await page.click('#first-move');
    await page.click('#playback-toggle');
    await page.waitForFunction(() => state.ply >= 1, {timeout: 4000});
    assert.equal(await page.evaluate(() => state.ply < live.ply), true, 'Replay should not jump to live');
    await page.click('#playback-toggle');
    fs.mkdirSync('checkpoints/play-watch-20260916', {recursive: true});
    await page.screenshot({path:'checkpoints/play-watch-20260916/watch.png',fullPage:true});
    const oldGame = await page.evaluate(() => live.game_id);
    await page.reload();
    await page.waitForFunction(() => typeof live !== 'undefined' && live.mode === 'watch');
    assert.equal(await page.evaluate(() => playback.frames[0].ply), 0, 'Reload must restore opening/history');
    await page.selectOption('#game-mode', 'play');
    await page.selectOption('#opponent', 'local');
    await page.click('#restart');
    await page.waitForFunction(id => live.game_id !== id && live.mode === 'local' && !busy, oldGame);
    assert.equal(await page.evaluate(() => playback.frames.length), 1, 'Restart clears old buffered frames');
    const move = await page.evaluate(() => live.legal[0]);
    const square = xy => '#square-' + 'ABCDEFGHI'[xy[0]] + (xy[1]+1);
    await page.click(square(move.source));
    await page.click(square(move.target));
    await page.waitForFunction(() => live.ply === 1 && !busy);
    await page.click('#previous-move');
    assert.equal(await page.evaluate(() => state.ply), 0);
    assert.equal(await page.locator('#board button:disabled').count(), 81);
    await page.click('#last-move');
    await page.click('#undo');
    await page.waitForFunction(() => live.ply === 0 && !busy);
    assert.equal(await page.evaluate(() => playback.frames.length), 1);
    await page.setViewportSize({width:375,height:812});
    await page.selectOption('#game-mode','watch');
    await page.selectOption('#blue-bot','greedy');
    await page.selectOption('#red-bot','alphabeta');
    await page.fill('#max-depth','1');
    await page.fill('#time-limit','1');
    await page.click('#restart');
    await page.waitForFunction(() => live.mode==='watch' && live.ply>=2);
    await page.click('#generation-toggle');
    await page.waitForFunction(() => !busy && !generating);
    await page.click('#last-move');
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true,
      'Mobile controls must not overflow horizontally');
    assert.equal(await page.evaluate(() => playback.frames.some(frame => frame.analysis)), true,
      'Real alpha-beta bot should produce analysis');
    assert.deepEqual(errors, []);
    console.log('PASS: independent generation/playback, history buttons/keys, input focus, reload, restart, local moves/undo; no browser errors.');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
