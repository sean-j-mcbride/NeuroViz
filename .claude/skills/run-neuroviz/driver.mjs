// Headless-Chrome driver for NeuroViz. Reads one command per line from stdin.
//
//   node .claude/skills/run-neuroviz/driver.mjs [url] [--dark] [--out DIR] [--browser chrome|chromium] < script
//
// --browser chrome (default) uses the installed Google Chrome and falls back to
// Playwright's bundled Chromium if Chrome can't launch; --browser chromium forces
// the bundled one (install: npx --prefix .claude/skills/run-neuroviz playwright-core install chromium).
// url defaults to http://localhost:5173 (the dev server). Screenshots go to
// --out (default $TMPDIR/neuroviz-shots). Results are printed as JSON lines.
// Blank lines and lines starting with # are ignored. See SKILL.md for commands.
import { mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createInterface } from 'node:readline';
import { chromium } from 'playwright-core';

const args = process.argv.slice(2);
const flag = (name) => {
  const i = args.indexOf(name);
  if (i < 0) return undefined;
  const [, value] = args.splice(i, 2);
  return value;
};
const dark = args.includes('--dark') && args.splice(args.indexOf('--dark'), 1);
const out = flag('--out') ?? join(tmpdir(), 'neuroviz-shots');
const browserKind = flag('--browser') ?? 'chrome';
const url = args[0] ?? 'http://localhost:5173';
mkdirSync(out, { recursive: true });

async function launch() {
  if (browserKind === 'chrome') {
    try {
      return ['chrome', await chromium.launch({ channel: 'chrome', headless: true })];
    } catch (e) {
      console.error(
        `Chrome failed to launch (${String(e).split('\n')[0]}); trying bundled Chromium.`,
      );
    }
  } else if (browserKind !== 'chromium') {
    console.error(`--browser must be chrome or chromium, got ${browserKind}`);
    process.exit(1);
  }
  try {
    return ['chromium', await chromium.launch({ headless: true })];
  } catch (e) {
    console.error(
      `Bundled Chromium failed to launch (${String(e).split('\n')[0]}). Install it with: ` +
        'npx --prefix .claude/skills/run-neuroviz playwright-core install chromium',
    );
    process.exit(1);
  }
}
const [launched, browser] = await launch();
// Stderr, so stdout stays one JSON result per command.
console.error(`browser: ${launched} ${browser.version()}`);
const ctx = await browser.newContext({
  viewport: { width: 1400, height: 1000 },
  colorScheme: dark ? 'dark' : 'light',
});
const page = await ctx.newPage();
const errors = [];
page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
page.on('pageerror', (e) => errors.push(String(e)));
try {
  await page.goto(url);
} catch (e) {
  console.error(
    `Nothing is serving ${url} (${String(e).split('\n')[0]}). Start the dev server or preview first.`,
  );
  await browser.close();
  process.exit(1);
}
// The last panel to render on either tab: its first snapshot has arrived.
const READY = '.inside-panel:visible, #page-mnist .training-panel';
await page.waitForSelector(READY);

const print = (cmd, result) => console.log(JSON.stringify({ cmd, ...result }));
const epoch = async () => Number((await page.textContent('.readout-value')).replace(/,/g, ''));
const tooltip = async () =>
  (await page.locator('.tooltip').count())
    ? (await page.innerText('.tooltip')).replace(/\s+/g, ' ')
    : null;

const noticeText = async () =>
  (await page.locator('.notice').count()) ? await page.innerText('.notice span') : null;

const commands = {
  /** shot NAME [SELECTOR] — full page, or one element. */
  async shot(name, ...sel) {
    const path = join(out, `${name}.png`);
    if (sel.length) await page.locator(sel.join(' ')).screenshot({ path });
    else await page.screenshot({ path, fullPage: true });
    return { path };
  },
  async click(...sel) {
    await page.click(sel.join(' '));
    return {};
  },
  /** select SELECTOR VALUE — last word is the value. */
  async select(...words) {
    const value = words.pop();
    await page.selectOption(words.join(' '), value);
    return {};
  },
  /** hover SELECTOR — moves the mouse to the element's centre; prints any tooltip. */
  async hover(...sel) {
    const box = await page.locator(sel.join(' ')).first().boundingBox();
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.waitForTimeout(200);
    return { tooltip: await tooltip() };
  },
  /** move SELECTOR FX FY — move the mouse to (FX, FY) as fractions of the element's box; prints any tooltip. */
  async move(...words) {
    const fy = Number(words.pop());
    const fx = Number(words.pop());
    const box = await page.locator(words.join(' ')).first().boundingBox();
    await page.mouse.move(box.x + fx * box.width, box.y + fy * box.height);
    await page.waitForTimeout(200);
    return { tooltip: await tooltip() };
  },
  /** clickat SELECTOR FX FY — click at (FX, FY) as fractions of the element's box (e.g. one feature-map pixel). */
  async clickat(...words) {
    const fy = Number(words.pop());
    const fx = Number(words.pop());
    const el = page.locator(words.join(' ')).first();
    await el.scrollIntoViewIfNeeded(); // mouse coordinates are viewport coordinates
    const box = await el.boundingBox();
    await page.mouse.click(box.x + fx * box.width, box.y + fy * box.height);
    await page.waitForTimeout(200);
    return {};
  },
  /**
   * draw SELECTOR FX,FY FX,FY … [| FX,FY …] — press, drag through the points (fractions of the
   * element's box) and release; `|` starts a new stroke. For the MNIST digit pad.
   */
  async draw(...words) {
    const at = words.findIndex((w) => /^[\d.]+,[\d.]+$/.test(w));
    const box = await page.locator(words.slice(0, at).join(' ')).first().boundingBox();
    const strokes = words
      .slice(at)
      .join(' ')
      .split('|')
      .map((s) => s.trim().split(/\s+/));
    for (const stroke of strokes) {
      const pts = stroke.map((p) => p.split(',').map(Number));
      const xy = ([fx, fy]) => [box.x + fx * box.width, box.y + fy * box.height];
      await page.mouse.move(...xy(pts[0]));
      await page.mouse.down();
      for (const p of pts.slice(1)) await page.mouse.move(...xy(p), { steps: 8 });
      await page.mouse.up();
    }
    await page.waitForTimeout(300);
    return {};
  },
  async wait(ms) {
    await page.waitForTimeout(Number(ms));
    return {};
  },
  async text(...sel) {
    return { text: (await page.innerText(sel.join(' '))).replace(/\s+/g, ' ') };
  },
  /** play MS — Play, wait MS, Pause, then let the final snapshot land. */
  async play(ms = '2000') {
    await page.click('button:has-text("Play")');
    await page.waitForTimeout(Number(ms));
    await page.click('button:has-text("Pause")');
    await page.waitForTimeout(400);
    return { epoch: await epoch(), stats: (await page.innerText('.stats')).replace(/\s+/g, ' ') };
  },
  /** measure — while training: epochs/s, readout updates/s (≈ snapshots/s), main-thread fps over 2 s. */
  async measure() {
    const e1 = await epoch();
    const r = await page.evaluate(
      () =>
        new Promise((resolve) => {
          const el = document.querySelector('.readout-value');
          let updates = 0;
          let frames = 0;
          const mo = new MutationObserver(() => updates++);
          mo.observe(el, { childList: true, characterData: true, subtree: true });
          const t0 = performance.now();
          const f = () => {
            frames++;
            if (performance.now() - t0 < 2000) requestAnimationFrame(f);
            else {
              mo.disconnect();
              resolve({ snapshotsPerSec: updates / 2, fps: frames / 2 });
            }
          };
          requestAnimationFrame(f);
        }),
    );
    return { ...r, epochsPerSec: ((await epoch()) - e1) / 2 };
  },
  /** sweep [MS] — move the mouse back and forth across the network graph (hover cost): fps, worst frame. */
  async sweep(ms = '2000') {
    const box = await page.locator('.network-canvas').boundingBox();
    await page.evaluate((duration) => {
      window.__sweep = new Promise((resolve) => {
        const gaps = [];
        let prev = performance.now();
        const t0 = prev;
        const f = (now) => {
          gaps.push(now - prev);
          prev = now;
          if (now - t0 < duration) requestAnimationFrame(f);
          else resolve({ fps: gaps.length / (duration / 1000), worstFrameMs: Math.max(...gaps) });
        };
        requestAnimationFrame(f);
      });
    }, Number(ms));
    const end = Date.now() + Number(ms);
    let tooltips = 0;
    let moves = 0;
    for (let i = 0; Date.now() < end; i++, moves++) {
      const f = (i % 100) / 100;
      const x = box.x + box.width * (i % 200 < 100 ? f : 1 - f);
      await page.mouse.move(x, box.y + box.height * (0.3 + 0.4 * f));
      if (i % 20 === 0 && (await page.locator('.tooltip').count())) tooltips++;
    }
    const r = await page.evaluate(() => window.__sweep);
    return { ...r, worstFrameMs: Math.round(r.worstFrameMs), moves, tooltipChecksHit: tooltips };
  },
  /** throttle RATE — slow the page's CPU by RATE× (1 = off), to mimic a slower machine. Chrome/Chromium only. */
  async throttle(rate) {
    const cdp = await ctx.newCDPSession(page);
    await cdp.send('Emulation.setCPUThrottlingRate', { rate: Number(rate) });
    return {};
  },
  /**
   * pick [FX FY] — open step-through and pick the data point drawn nearest to
   * (FX, FY), fractions of the output plot from its top-left (default 0.75 0.3:
   * upper right, well away from the origin, whose zero inputs make a dull trace).
   * Finds points by their fill colour in the overlay canvas, so it never misses.
   */
  async pick(fx = '0.75', fy = '0.3') {
    if (!(await page.locator('.step-through').count()))
      await page.click('button:has-text("Step through")');
    const canvas = page.locator('.decision-boundary .points');
    const target = await canvas.evaluate(
      (el, [tx, ty]) => {
        const { width: w, height: h } = el;
        const px = el.getContext('2d').getImageData(0, 0, w, h).data;
        // Point fills: orange (label 0) and blue (label 1), as in viz/colour.ts.
        const fills = [
          [245, 147, 34],
          [8, 119, 189],
        ];
        const isFill = (i) =>
          px[i + 3] > 200 &&
          fills.some(
            ([r, g, b]) =>
              Math.abs(px[i] - r) + Math.abs(px[i + 1] - g) + Math.abs(px[i + 2] - b) < 30,
          );
        let best = null;
        let bestD = Infinity;
        for (let y = 0; y < h; y += 2) {
          for (let x = 0; x < w; x += 2) {
            if (!isFill((y * w + x) * 4)) continue;
            const d = (x / w - tx) ** 2 + (y / h - ty) ** 2;
            if (d < bestD) [best, bestD] = [{ fx: x / w, fy: y / h }, d];
          }
        }
        return best;
      },
      [Number(fx), Number(fy)],
    );
    if (!target) return { picked: null, error: 'no data points drawn on the output plot' };
    const box = await canvas.boundingBox();
    await page.mouse.click(box.x + box.width * target.fx, box.y + box.height * target.fy);
    await page.waitForTimeout(300);
    return {
      picked: (await page.locator('.step-through-point').count())
        ? await page.innerText('.step-through-point')
        : null,
    };
  },
  /** next [N|end] — advance step-through N stages (default 1) or to the last one. */
  async next(n = '1') {
    const btn = page.locator('[aria-label="Next stage"]');
    for (let i = 0; n === 'end' || i < Number(n); i++) {
      if (await btn.isDisabled()) break;
      await btn.click();
    }
    await page.waitForTimeout(100);
    return {
      stage: await page.innerText('.step-through strong'),
      badges: await page.locator('.probe-value').allTextContents(),
    };
  },
  /** deep ACTIVATION — circle data, 6 hidden layers × 8 units, all with ACTIVATION. */
  async deep(activation) {
    await page.click('button:has-text("Circle")');
    while (!(await page.locator('[aria-label="Add a hidden layer"]').isDisabled()))
      await page.click('[aria-label="Add a hidden layer"]');
    for (let l = 1; l <= 6; l++) {
      const add = page.locator(`[aria-label="Add a neuron to hidden layer ${l}"]`);
      while (!(await add.isDisabled())) await add.click();
      await page.selectOption(`[aria-label="Activation of hidden layer ${l}"]`, activation);
    }
    await page.waitForTimeout(300);
    return { network: await page.innerText('.network-toolbar') };
  },
  /** download NAME SELECTOR — click SELECTOR and save the file it downloads to the output dir as NAME. */
  async download(name, ...sel) {
    const [dl] = await Promise.all([page.waitForEvent('download'), page.click(sel.join(' '))]);
    const path = join(out, name);
    await dl.saveAs(path);
    await page.waitForTimeout(200);
    return { path, suggested: dl.suggestedFilename() };
  },
  /** upload FILE SELECTOR — give FILE to a (possibly hidden) file input, as if chosen in the picker. */
  async upload(file, ...sel) {
    await page.setInputFiles(sel.join(' '), file);
    await page.waitForTimeout(500);
    return { notice: await noticeText() };
  },
  /** goto URL — load a URL (e.g. a share link), wait for the first snapshot. */
  async goto(target) {
    await page.goto(target);
    await page.waitForSelector(READY);
    return { url: page.url(), notice: await noticeText() };
  },
  /** reload — reload the page (keeping the hash), wait for the first snapshot. */
  async reload() {
    await page.reload();
    await page.waitForSelector(READY);
    return { url: page.url() };
  },
  /** url — the current address, including the settings hash. */
  async url() {
    return { url: page.url() };
  },
  async workers() {
    return { workers: page.workers().map((w) => w.url().split('/').pop()) };
  },
  async errors() {
    return { errors };
  },
};

for await (const raw of createInterface({ input: process.stdin })) {
  const line = raw.trim();
  if (!line || line.startsWith('#')) continue;
  const [cmd, ...rest] = line.split(/\s+/);
  const fn = commands[cmd];
  if (!fn) {
    print(cmd, { error: `unknown command; try: ${Object.keys(commands).join(', ')}` });
    continue;
  }
  try {
    print(line, await fn(...rest));
  } catch (e) {
    print(line, { error: String(e).split('\n')[0] });
  }
}
await browser.close();
