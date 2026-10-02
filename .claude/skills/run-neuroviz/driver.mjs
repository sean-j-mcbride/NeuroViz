// Headless-Chrome driver for NeuroViz. Reads one command per line from stdin.
//
//   node .claude/skills/run-neuroviz/driver.mjs [url] [--dark] [--out DIR] < script
//
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
const url = args[0] ?? 'http://localhost:5173';
mkdirSync(out, { recursive: true });

const browser = await chromium.launch({ channel: 'chrome', headless: true });
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
await page.waitForSelector('.inside-panel'); // last panel to render: first snapshot has arrived

const print = (cmd, result) => console.log(JSON.stringify({ cmd, ...result }));
const epoch = async () => Number((await page.textContent('.readout-value')).replace(/,/g, ''));
const tooltip = async () =>
  (await page.locator('.tooltip').count())
    ? (await page.innerText('.tooltip')).replace(/\s+/g, ' ')
    : null;

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
  /** pick — open step-through and click the output plot until a data point is picked. */
  async pick() {
    if (!(await page.locator('.step-through').count()))
      await page.click('button:has-text("Step through")');
    const db = await page.locator('.decision-boundary .points').boundingBox();
    const spots = [
      [0.5, 0.5],
      [0.45, 0.5],
      [0.55, 0.45],
      [0.4, 0.4],
      [0.6, 0.6],
      [0.3, 0.5],
      [0.7, 0.5],
      [0.5, 0.3],
    ];
    for (const [fx, fy] of spots) {
      await page.mouse.click(db.x + db.width * fx, db.y + db.height * fy);
      await page.waitForTimeout(250);
      if (await page.locator('.step-through-point').count())
        return { picked: await page.innerText('.step-through-point') };
    }
    return { picked: null };
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
