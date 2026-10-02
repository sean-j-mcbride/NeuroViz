---
name: run-neuroviz
description: Build, run, and drive the NeuroViz neural-network visualiser. Use when asked to start the app or dev server, take a screenshot of the UI, check that training/step-through/tooltips/histograms work in a real browser, demo deep sigmoid vs ReLU, measure training speed or frame rate, run the tests, or call engine/session code directly.
---

NeuroViz is a static Vite + React app; training runs in a Web Worker. To drive it, start the dev
server (or the static preview), then pipe a command script to
`.claude/skills/run-neuroviz/driver.mjs`. That is a Playwright driver running headless **system
Google Chrome**, or Playwright's **bundled Chromium** when Chrome is missing or you ask for it.
All paths below are relative to the repo root.

Verified on macOS (Darwin, Apple silicon, Node ≥ 20) with both Chrome 154 and bundled Chromium 153;
both give the same results. It was not tried on Linux (no container runtime on the machine it was
written on).

## Setup

```bash
npm install                                         # app deps
npm --prefix .claude/skills/run-neuroviz install    # driver's only dep: playwright-core
```

Only if Google Chrome isn't installed (or you want `--browser chromium`): download Playwright's
headless Chromium, ~94 MB, into `~/Library/Caches/ms-playwright` on macOS:

```bash
npx --prefix .claude/skills/run-neuroviz playwright-core install chromium
```

## Run (agent path)

Start the dev server and poll until it serves. macOS has no `timeout`, so use a loop:

```bash
lsof -ti:5173 -sTCP:LISTEN | xargs kill 2>/dev/null
(npm run dev -- --port 5173 --strictPort > /tmp/neuroviz-dev.log 2>&1 &)
for i in $(seq 1 60); do curl -sf http://localhost:5173 >/dev/null && echo up && break; sleep 0.5; done
```

Drive it. One command per line on stdin, and each result prints as a JSON line:

```bash
node .claude/skills/run-neuroviz/driver.mjs <<'EOF'
workers
click button:has-text("Play")
wait 1000
measure
click button:has-text("Pause")
wait 400
text .stats
hover .edge-hit >> nth=20
shot network .network-panel
pick
next 3
shot forward .network-panel
next end
shot backward .network-panel
errors
EOF
```

Expected: `workers` lists `training.worker.ts?worker_file&type=module`. `measure` reports about
300 epochs/s, ~12 snapshots/s and 60 fps. `hover` prints the edge's tooltip (weight, ∂L/∂w,
sparkline range). `errors` is `[]`. **Open the screenshots and look at them.**

Arguments: `driver.mjs [url] [--dark] [--out DIR] [--browser chrome|chromium]`. The URL defaults
to `http://localhost:5173`. Screenshots go to `$TMPDIR/neuroviz-shots/<name>.png` unless you pass
`--out`, and every `shot` prints its absolute path. `--browser chrome` (the default) falls back to
bundled Chromium if Chrome won't launch; `--browser chromium` forces it. The first stderr line
says which one ran, e.g. `browser: chrome 154.0.8037.95`.

| command                    | what it does                                                                             |
| -------------------------- | ---------------------------------------------------------------------------------------- |
| `shot NAME [SELECTOR]`     | full-page screenshot, or just one element                                                |
| `click SELECTOR`           | Playwright selector, e.g. `button:has-text("Step")`, `[aria-label="Add a hidden layer"]` |
| `select SELECTOR VALUE`    | choose a `<select>` option (the last word is the value)                                  |
| `hover SELECTOR`           | move the mouse to the element's centre and print any tooltip text                        |
| `text SELECTOR`            | print the element's text                                                                 |
| `wait MS`                  | sleep                                                                                    |
| `play MS`                  | Play, wait, Pause, let the last snapshot land; prints the epoch and loss/accuracy table  |
| `measure`                  | while training: epochs/s, readout updates/s (≈ snapshots/s) and main-thread fps over 2 s |
| `sweep [MS]`               | sweep the mouse over the network graph (hover cost): fps, worst frame, number of moves   |
| `throttle RATE`            | slow the page CPU RATE× via CDP (1 = off), to mimic a slower machine                     |
| `pick`                     | open step-through and click the output plot until a data point is picked                 |
| `next [N\|end]`            | advance step-through; prints the stage name and the neuron value badges                  |
| `deep relu\|sigmoid\|tanh` | circle data, 6 hidden layers × 8 units, all one activation (the Phase 3 demo)            |
| `workers`                  | URLs of running Web Workers (proves training is off the main thread)                     |
| `errors`                   | console errors and page errors collected so far                                          |

Deep sigmoid vs ReLU, on the static build in dark mode:

```bash
npm run build > /tmp/neuroviz-build.log 2>&1 && tail -1 /tmp/neuroviz-build.log
lsof -ti:4173 -sTCP:LISTEN | xargs kill 2>/dev/null
(npx vite preview --port 4173 --strictPort > /tmp/neuroviz-preview.log 2>&1 &)
for i in $(seq 1 60); do curl -sf http://localhost:4173 >/dev/null && echo up && break; sleep 0.5; done
node .claude/skills/run-neuroviz/driver.mjs http://localhost:4173 --dark <<'EOF'
workers
deep sigmoid
play 3000
shot deep-sigmoid .inside-panel
click button:has-text("Reset")
deep relu
play 3000
shot deep-relu .inside-panel
errors
EOF
lsof -ti:4173 -sTCP:LISTEN | xargs kill 2>/dev/null
```

Expected after ~900 epochs: sigmoid at ~52 % train accuracy, ReLU at 100 %. In the sigmoid
screenshot, the gradient-RMS lines fan out over about four decades.

Stop the dev server: `lsof -ti:5173 -sTCP:LISTEN | xargs kill`.

## Direct invocation (engine / session code, no browser)

There is no `tsx`. To call `src/engine` or `src/worker` code directly, write a throwaway test
file and run it with output enabled. Vitest hides `console.log` from passing tests unless you
pass `--silent=false`.

```bash
cat > src/worker/zz_probe.test.ts <<'EOF'
import { it } from 'vitest';
import { TrainingSession } from './session';
it('probe', () => {
  const s = new TrainingSession({
    dataset: { kind: 'circle', n: 400, noise: 0.1, seed: 1 },
    network: { hidden: [{ units: 8, activation: 'tanh' }] },
    lr: 0.03, batchSize: 10, optimiser: 'sgd', l2: 0, dropout: 0, seed: 1, gridSize: 4,
  });
  for (let e = 0; e < 50; e++) s.trainEpoch();
  console.log('train accuracy', s.snapshot().trainAccuracy);
});
EOF
npx vitest run src/worker/zz_probe.test.ts --silent=false --reporter=verbose | grep 'train accuracy'
rm src/worker/zz_probe.test.ts
```

## Run (human path)

`npm run dev`, open http://localhost:5173, then Ctrl-C to stop.

## Test

```bash
npm test && npm run typecheck && npm run lint && npm run build
```

All of these must pass at the end of each phase (CLAUDE.md).

## Gotchas

- **Read the epoch readout only after the last snapshot lands.** Snapshots are pulled, one in
  flight at a time. Right after Pause the readout can trail by one snapshot, so `play` waits
  400 ms before reading.
- **Edges are hovered through `.edge-hit`,** an invisible 8 px stroke over each visible `.edge`.
  Tooltips are `position: fixed` and appear only on mousemove, so use `hover` (it moves the
  mouse). Playwright's `locator.hover()` also works for `.neuron`.
- **Picking a step-through point needs a click within 10 px of a data point.** `pick` tries a
  fixed list of spots. On the spirals and circle the centre spot (0, 0) usually hits. That point
  sits at the input origin, so input → H1 edges show w·a = 0 (thin), and that's expected.
- **`.transport select >> nth=2` is the Speed select.** The optimiser row also has class
  `transport`, so its selects come after it (nth 3–5). The selector `.optimiser-controls select`
  is less fragile for those.
- **Every architecture or dataset change rebuilds the session at epoch 0.** Hyperparameter
  changes (learning rate, batch size, optimiser, L2, dropout) apply live.
- **The worker URL differs per mode:** `training.worker.ts?worker_file&type=module` on the dev
  server, `training.worker-<hash>.js` on the preview build.
- **Kill servers by port** (`lsof -ti:PORT -sTCP:LISTEN | xargs kill`). `$!` after `npm run dev &`
  is only the npm wrapper.

## Troubleshooting

- **`Chrome failed to launch (… is not found at …); trying bundled Chromium.`**: Chrome isn't
  installed where Playwright looks. Harmless if the bundled Chromium is installed; the run
  continues on it.
- **`Bundled Chromium failed to launch (… Executable doesn't exist at …)`**: run the
  `playwright-core install chromium` command from Setup.
- **`Nothing is serving http://localhost:5173 (Error: page.goto: net::ERR_CONNECTION_REFUSED …)`**: the
  dev server isn't running (or you meant the preview on 4173). Start it with the block above.
- **`(eval):1: command not found: timeout`**: macOS has no `timeout`. Use the
  `for i in $(seq 1 60)` polling loop above.
- **Driver prints nothing for `console.log` in a throwaway test**: add
  `--silent=false --reporter=verbose` to `npx vitest run`.
