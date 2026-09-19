/* Dedicated server on 8879 with --checkpoint ... --opponent local --simulations 128. */
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');

(async () => {
  const browser = await chromium.launch({headless: true,
    executablePath: process.env.CHROME_BINARY || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
  const page = await browser.newPage({viewport: {width: 375, height: 812}});
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  try {
    await page.goto('http://127.0.0.1:8879/');
    await page.waitForFunction(() => typeof live !== 'undefined' && !busy);
    assert.equal(await page.locator('#model-options').isVisible(), false);
    await page.selectOption('#opponent', 'model');
    assert.equal(await page.locator('#model-options').isVisible(), true);
    assert.equal(await page.locator('#mcts-simulations').inputValue(), '128');
    const originalId = await page.evaluate(() => live.game_id);
    await page.fill('#mcts-simulations', '1');
    await page.click('#restart');
    assert.equal(await page.evaluate(() => live.game_id), originalId);
    await page.fill('#mcts-simulations', '256');
    await page.click('#restart');
    await page.waitForFunction(() => live.opponent === 'model' && live.model_simulations === 256 && !busy);
    assert.match(await page.locator('#model').innerText(), /256 MCTS simulations/);
    const move = await page.evaluate(() => live.legal[0]);
    const square = xy => '#square-' + 'ABCDEFGHI'[xy[0]] + (xy[1]+1);
    await page.click(square(move.source));
    await page.click(square(move.target));
    await page.waitForFunction(() => live.ply === 2 && !busy);
    await page.reload();
    await page.waitForFunction(() => typeof live !== 'undefined' && !busy);
    assert.equal(await page.locator('#mcts-simulations').inputValue(), '256');
    await page.fill('#mcts-simulations', '512');
    assert.equal(await page.evaluate(() => live.model_simulations), 256, 'Editing must not change the active game');
    await page.click('#restart');
    await page.waitForFunction(() => live.model_simulations === 512 && live.ply === 0 && !busy);
    assert.match(await page.locator('#model').innerText(), /512 MCTS simulations/);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.selectOption('#game-mode', 'watch');
    await page.selectOption('#blue-bot', 'model');
    await page.selectOption('#red-bot', 'model');
    await page.click('#restart');
    await page.waitForFunction(() => live.mode === 'watch' && live.ply >= 2);
    await page.click('#generation-toggle');
    await page.waitForFunction(() => !busy && !generating);
    assert.equal(await page.evaluate(() => live.model_simulations), 512);
    assert.equal(await page.evaluate(() => live.bot_labels.every(label => label.includes('512 MCTS simulations'))), true);
    assert.deepEqual(errors, []);
    console.log('PASS: MCTS input validation, 256-simulation move, reload, 512 restart, two model bots, mobile layout.');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
