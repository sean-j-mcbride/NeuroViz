# NeuroViz roadmap

Each phase should end with something you can run and look at.
Move to the next phase only when the current one feels solid.

---

## Phase 0 — Scaffold

- Vite + React + TS (strict), Vitest, ESLint, Prettier, Zustand.
- Folder structure from CLAUDE.md, empty modules with stub exports.
- Seeded RNG (e.g. mulberry32) in `engine/random.ts`.
- GitHub Actions: typecheck + test on push.

**Done when:** `npm run dev` shows a blank app shell, `npm test` passes a trivial test.

---

## Phase 1 — Engine core (no UI)

- `Tensor` with: create, zeros, randn (seeded), matmul, transpose, add
  (with row broadcast), elementwise map, sum over axis.
- Layers: `Dense`, `ReLU`, `Tanh`, `Sigmoid`.
- Losses: MSE, softmax + cross-entropy (combined, numerically stable).
- Optimiser: SGD.
- `Sequential` model + `trainStep(x, y)` returning loss.
- Init: Xavier/He.
- Tests: gradient checks for every layer/loss; an XOR test that trains
  to near-zero loss with a fixed seed.

**Done when:** a Vitest test trains a 2-4-1 net on XOR deterministically.

---

## Phase 2 — The 2D playground (first visual milestone)

- Datasets: circle, XOR, two spirals, two Gaussians; noise + size sliders.
- Architecture builder: add/remove hidden layers, neurons per layer,
  activation per layer.
- Controls: play / pause / single-step / reset, learning rate, batch size.
- Visualisations:
  - **Network graph** (SVG): edge thickness = |weight|, colour = sign.
  - **Per-neuron mini heatmaps**: what each hidden neuron responds to
    across the 2D input space.
  - **Decision boundary** (canvas) with the data points on top.
  - **Loss curve** (train and test).

**Done when:** you can watch a 2-8-8-1 tanh net untangle the spirals.

---

## Phase 3 — See inside training

- Training in a **Web Worker**; Snapshot protocol; UI throttling.
- "Step-through" mode: pick one input, animate the forward pass
  (activations light up) then the backward pass (gradients flow back).
- Hover a neuron/edge → tooltip with value, gradient, recent history.
- Optimisers: momentum, Adam. L2 regularisation, dropout.
- Weight and gradient histograms per layer over time
  (spot vanishing/exploding gradients).

**Done when:** you can visibly show why sigmoid on a deep net trains slower than ReLU.

---

## Phase 4 — Save, load, compare

- Export/import model config + weights as JSON.
- Shareable URL encoding the architecture + hyperparameters.
- Side-by-side comparison of two runs (loss curves overlaid).
- Presets ("Underfitting", "Overfitting", "Dead ReLUs", "Too high LR").

**Done when:** you can save mid-run, reload the page, load the file and train on as if nothing
happened (bitwise identical, by test); a pasted link reproduces the set-up; and each preset shows
its failure mode next to a pinned run with the fix.

---

## Phase 5 — MNIST with an MLP

- Load a subset of MNIST (bundle a compressed ~10k-image subset; no server).
- Performance pass: tighter matmul, typed-array reuse, maybe batch in worker.
- Visualisations:
  - First-layer weights rendered as 28×28 images.
  - Draw-a-digit canvas → live prediction bars.
  - Confusion matrix; gallery of most-confidently-wrong examples.

**Done when:** ~95%+ test accuracy in a couple of minutes in the browser.

---

## Phase 6 — CNNs

- Layers: `Conv2D`, `MaxPool2D`, `Flatten` (gradient checks!).
- Visualisations:
  - Filter viewer (learned kernels).
  - Feature-map viewer per layer for a chosen input.
  - Receptive-field highlight: click a feature-map pixel, see the input region.
- Architecture builder supports 3D tensor shapes with shape validation.

---

## Stretch ideas

- WebGPU backend for matmul/conv behind the same `Tensor` API.
- Saliency maps / gradient × input.
- Small RNN on character data; attention heatmaps for a tiny transformer.
- Embedding projector (PCA/t-SNE of hidden activations).
- Optional TensorFlow.js backend to compare against your engine.

---

## Phase log

<!-- Claude Code appends "Done / decisions / known issues" notes here. -->

### Phase 0 — Scaffold (2026-10-02)

**Done:** Vite + React + TS (strict) shell rendering `<h1>NeuroViz</h1>`; Vitest, ESLint (flat config),
Prettier, Zustand stub store; folder structure with stub modules; seeded `Rng` (mulberry32 +
Box–Muller) with tests; GitHub Actions running typecheck, lint and test.

**Decisions:**

- App renamed from NeuroScope to NeuroViz.
- `tsconfig.engine.json` typechecks `src/engine` + `src/data` with `lib: ["ES2022"]` and no DOM,
  so DOM use in the engine fails `npm run typecheck`. ESLint `no-restricted-imports` also stops
  engine/data importing React, Zustand or the app layers (viz, ui, state, worker).
- `vite.config.ts` uses `base: './'` so the static build works from a GitHub Pages subpath.
- CI also runs lint (the roadmap only asked for typecheck + test).

**Known issues:** none.

### Phase 1 — Engine core (2026-10-02)

**Done:** `Tensor` + ops (`matmul` with transpose flags, `transpose`, `add` with row broadcast,
`map`, `map2`, `sumAxis`, `sumAll`); `Dense`, `ReLU`, `Tanh`, `Sigmoid`; Xavier/He normal init;
`MSELoss` and fused, numerically stable `SoftmaxCrossEntropyLoss`; `SGD`; `Sequential`;
`Trainer.trainStep(x, y)`; gradient-check helper (`gradcheck.ts`). A 2-4-1 tanh/sigmoid net trains
XOR to MSE 2.3e-4 in 3000 full-batch steps (seed 42, lr 1), bitwise identically on every run.
39 tests.

**Decisions:**

- Ops are free functions with an optional `out` argument. Layers own their output/gradient
  buffers and reallocate only when the batch size changes. Consequence: tensors returned by
  `forward`/`backward` are overwritten by the next call — clone to keep them.
- `matmul` folds transposes into its indexing, so backward (`xᵀ·g`, `g·Wᵀ`) allocates nothing.
- `backward` **overwrites** parameter grads (it does not accumulate). Revisit if we ever need
  gradient accumulation across micro-batches.
- `Sequential` implements `Layer` and names params by layer index (`"0.W"`, `"2.b"`).
- Gradient check: central differences dividing by the perturbation actually stored in float32;
  normwise relative error ‖a−n‖ / (‖a‖+‖n‖) per tensor; scalar objective Σ y ⊙ R with a fixed random
  R for layers. Default eps = 5e-3 (measured optimum between truncation and float32 noise).
  A meta-test confirms a 1% error in dW is caught.
- Tolerances: per-layer and per-loss checks use 1e-4 (all measure ≤ 2.4e-5). End-to-end model
  checks use **1e-3**: a sweep over eps showed the first-layer W of dense→tanh→dense→sigmoid+MSE
  has a float32 noise floor of ~1.2e-4 at the best eps (its gradients are tiny), so 1e-4 is not
  reachable there in float32. This is not a bug — the error is U-shaped in eps.
- XOR hyperparameters are not seed-sensitive: seeds 1–20 all reach final MSE < 2.5e-4.
- `tsconfig.engine.json` excludes `*.test.ts` (tests run under Node and may use `console`);
  production engine code is still checked without DOM types.

**Known issues:**

- Matmul is a naive triple loop — fine for now; the Phase 5 performance pass should revisit it.
- The ReLU gradient check keeps inputs at |x| ≥ 0.1 to stay away from the kink at 0.

### Phase 2 — The 2D playground (2026-10-02)

**Done:** Datasets (circle, XOR, two spirals, two Gaussians) with noise and size sliders and a
seeded 70/30 train/test split. An architecture builder (0–6 hidden layers, 1–8 neurons each,
tanh / ReLU / sigmoid / linear per layer). Play / pause / step (one epoch) / reset, with learning
rate, batch size, speed and weight seed. Visualisations: SVG network graph (edge width = |w|,
colour = sign, hover for values) whose neurons are drawn as mini heatmaps over the input plane,
a canvas decision boundary with train (and optionally test) points, a hand-drawn train/test loss
curve with a log-scale toggle, and a loss/accuracy table. Engine additions:
`BCEWithLogitsLoss` (gradient-checked) and `layerFromConfig`. A seeded test trains 2-8-8-1 tanh
on the spirals to ≥ 95 % train accuracy in 2000 epochs. In the browser this takes a few seconds
(99.6 % train / 91.7 % test after ~7000 epochs). 76 tests.

**Decisions:**

- Training runs on the **main thread** this phase, through a DOM-free `TrainingSession`
  (`src/worker/session.ts`) that emits structured-cloneable `Snapshot`s. The UI hook
  (`ui/useTrainingLoop.ts`) drives it from `requestAnimationFrame` (N epochs/frame, or an 8 ms
  budget at "Max") and publishes at most ~15 snapshots/s. Phase 3 only has to move the session
  into a Worker. `src/worker` is now typechecked without DOM types and has the same import
  boundary as engine/data. engine/data may not import from worker.
- Binary head = linear `Dense(→1)` + fused `BCEWithLogitsLoss`, so the boundary shows σ(logit).
- `NetworkSpec` (the compact hidden-layer description) and `networkToLayerConfig` live in
  `worker/network.ts` rather than `state/`, because the session needs them and the boundary
  rule forbids worker → state. Init is chosen per layer: He for ReLU, Xavier otherwise.
- Domain ±6 (TF Playground convention), data radius 5. Noise is Gaussian position jitter
  (std = 2·noise) with labels kept, so classes genuinely overlap. Clamped to the domain.
- Defaults (spirals, lr 0.03, batch size 10, noise 0) come from a sweep. With plain SGD, lr 0.03 / batch
  10 was the only setting where every seed tried (5 seeds, noise 0 and 0.1) reached ≥ 95 %.
  Higher learning rates fit faster but oscillate.
- One heatmap grid (G = 50, 2500 points) feeds both the neuron tiles and the decision boundary;
  the canvas is drawn at native resolution and the browser's bilinear scaling smooths it.
- Neuron colour normalisation is registered by column kind (`viz/colour.ts` `NORMALISERS`):
  fixed ranges for tanh / sigmoid / output, per-neuron max-|v| for ReLU / linear.
- Added a **Speed** control (not in the roadmap): at ~3000 epochs/s the spirals would otherwise
  untangle too fast to watch. Default 5 epochs/frame.
- Reset rebuilds from the same seed (reproducible); "New weights" and "Regenerate data" draw a
  new seed with `Math.random` in the UI. That is the only non-seeded randomness, and the seed it
  picks is shown.

**Known issues:**

- Layer buffers reallocate whenever the forward batch size changes (training batches vs. the
  full-train/test evaluation each epoch vs. the grid). Harmless at this scale (~3300 epochs/s at
  60 fps on spirals). Revisit in the Phase 5 performance pass.
- Each snapshot copies the full loss history and the data points. Fine for tens of thousands
  of epochs; downsample or send the data once when the worker protocol lands in Phase 3.
- With noisy data the test loss can climb steeply while test accuracy stays high (over-confident
  logits on overlapping points). This is real model behaviour, but it can surprise people. Phase 3's
  L2 regularisation is the natural fix to demonstrate.
- No component tests (no DOM test environment). The UI was checked by driving the dev and preview
  builds in headless Chrome.

### Phase 3 — See inside training (2026-10-02)

**Done:** Training runs in a **Web Worker** (`worker/training.worker.ts` → DOM-free
`TrainingController`). The UI pulls snapshots, with at most one in flight and ≤ 15 per second.
Engine additions: `Momentum` and `Adam` optimisers, L2 regularisation (`addL2`,
gradient-checked) and an inverted `Dropout` layer (gradient-checked with a frozen mask). All three
are live controls in a second header row. **Step-through mode:** pick a data point on the output
plot, then step or animate the forward pass (activations light up, edges show w·a), the loss, and
the backward pass (∂L/∂a per neuron, edges show the per-example ∂L/∂w). **Hover cards** on every
edge and neuron show the value, the full-batch gradient, a sparkline of the last 100 epochs, and
the probe values while step-through is open. **Inside training panel:** gradient RMS per layer
over the run (log axis), plus per-layer weight and log-|gradient| histograms over time.
Done-when: on the circle, a 6×8 sigmoid net's first-layer gradient is ~3e-4 of the output
layer's, against ~0.1 for ReLU. ReLU reaches 90 % in ~9 epochs; sigmoid is still at chance after 600. A seeded test asserts both, and in the browser the fan-out of the gradient lines and the
thinning backward edges make it visible. 160 tests.

**Decisions:**

- **Snapshots are pulled.** A rAF loop requests one when none is in flight and ≥ 66 ms have
  passed, so a slow UI asks less often instead of building a backlog. Buffers are sent in the
  transfer list (zero copy). The UI chooses `sessionId` on `init`, so replies from a previous
  session are dropped. Point sets are sent once per session in `ready` (fixes the Phase 2 known
  issue). Measured in Chrome: ~12 snapshots per second at 60 fps; Max speed reaches ~4,600
  epochs/s on the spirals (the main-thread loop managed ~3,300).
- **Speed is now epochs per second** (30 / 100 / 300 / 1000 / Max; default 300 ≈ the old
  5 per frame). The worker trains in ≤ 12 ms slices paced to that rate, yields through a
  `MessageChannel` (avoiding the 4 ms clamp on nested `setTimeout(0)`), and drops any backlog over
  250 ms rather than racing to catch up. The controller takes an injected clock and scheduler, so
  pacing is unit-tested with a fake clock.
- **Observing never changes a run.** Snapshots, gradient measurements and probes run in evaluation
  mode: dropout is the identity and makes no RNG draws, and the optimiser is untouched. A test
  interleaves snapshots and probes with training and checks the weights stay bitwise identical.
- **Dropout layers are always in the model** (after each hidden activation), so the rate can change
  live. At rate 0 a Dropout layer is the identity and draws nothing, and it takes no draws from the
  init RNG at construction, so seeded runs are unchanged (there is a test for this).
- **Optimiser state** is keyed on each parameter's value tensor, which is stable, unlike the `Param`
  objects `Sequential` rebuilds. Switching optimiser mid-run starts it with fresh state.
- **L2 is coupled** (λ·w added to the gradient before the optimiser, so Adam + L2 is not AdamW) and
  applies to weights only. Loss curves show the **data** loss without the penalty.
- **Gradients shown in hover cards and histograms** are full-batch (whole training set, evaluation
  mode, including the L2 term), not the noisy mini-batch gradients. They are measured at every
  snapshot and at every timeline record.
- **The histogram timeline** keeps 128 columns. When full it **drops every other column and doubles
  the interval** instead of averaging pairs (a change from the plan), so each column is an exact
  moment and the extra measurement cost grows only as O(log epochs). Weights use 40 bins on a
  **signed-log axis**, asinh(w / 0.05), covering ±50. It is linear below |w| ≈ 0.05 and about
  ×1.5 per bin above 0.1, with dotted guides at ±1. A first version used linear bins over ±4, but
  the spirals' output-layer weights (~±10) piled into the edge bins. Gradients use 32 bins over
  log10|g| ∈ [−10, 1], with zeros in the bottom bin; dead ReLUs show up there.
- **Every chart has a table view** (Charts / Table toggle on the Inside training panel), per the
  dataviz rule that identity and values never depend on colour or hover alone. The tables show
  per-layer weight RMS, gradient RMS and the share of |w| > 1 and of gradients below 10⁻⁶ (read
  from the histograms, so accurate to about one bin), plus gradient RMS over time with a
  first ÷ last ratio column.
- **Snapshots carry the L2 strength**, so the edge hover card says ∂(L + L2)/∂w when L2 is on. Bias
  gradients never include L2.
- **The worker hook's logic lives in a DOM-free `TrainingClient`** (`ui/trainingClient.ts`); the
  hook only wires it to React. Its tests cover the one-in-flight rule, the queued request, the
  66 ms throttle, dropping stale-session replies and probe attachment. They also run it against the
  real `TrainingController` through a simulated channel that structured-clones every message.
  Mutation checks (removing the stale-session guard or the in-flight guard) make them fail.
- **Component tests** use jsdom and Testing Library, opted into per file with
  `// @vitest-environment jsdom`. Everything else stays in the faster Node environment.
  `src/test/setup.ts` stubs ResizeObserver, matchMedia and canvas for jsdom.
- **Hover cost was measured, not assumed.** The driver's `sweep` command moves the mouse across a
  6×8 graph (~700 SVG paths, re-rendered on every move) at about 60 moves/s. Paused or training at
  Max, and even at 4× CPU throttle, the page held 60 fps with no frame over 17 ms, so there is no
  memoisation yet.
- **The probe** is one example in evaluation mode with data loss only. At the output the view
  shows ∂L/∂z = p − y; `dA` there is ∂L/∂p with p clamped. In step-through, backward edge widths
  are logarithmic over four decades, relative to the largest ∂L/∂w in the whole trace rather than
  per layer, so vanishing gradients stay visible. There are 2C stages: C forward, the loss, then
  C − 1 backward.
- **Colour:** layer depth uses an ordinal blue ramp, from faint for the first layer to strong for
  the output. The stock reference steps failed the adjacent-ΔL check at 7 layers, so the ramp is
  interpolated in OKLab between steps 250 → 700 (light) and 600 → 100 (dark). Both versions pass
  the dataviz palette validator. Histogram density uses a sequential surface → blue ramp, scaled
  by √fraction so thin tails stay visible.

**Known issues:**

- ~~Each snapshot still copies the full loss history~~ (bounded since the Phase 4 follow-up) and
  runs one extra full-batch forward/backward. Cheap at this scale; revisit for MNIST in Phase 5.
- Weights beyond ±50 still land in the edge bins (the tooltip says "and below" / "and above").
- With plain SGD, the full-batch gradient RMS line is spiky: weights move between records. This
  is real behaviour, not a rendering artefact.
- Right after Pause, the epoch readout can trail the true stopping point by one snapshot. A
  request may already be in flight, and the final snapshot follows it.
- Momentum β and Adam β₁/β₂ are fixed in the UI (configurable in the engine).
- Component tests cover the network graph, step-through bar, tables, point picking and store
  logic, but not canvas drawing (jsdom has no 2D context, so heatmaps, the decision boundary and
  charts are only checked in the browser driver's screenshots).

### Phase 4 — Save, load, compare (2026-10-02)

**Done:** **Save model / Load model** writes and reads a readable JSON file (settings plus a
checkpoint). Loading resumes **exactly**: train N epochs, save, load and train M more is bitwise
identical to training N + M, for SGD, momentum and Adam with dropout and L2. Tests cover this
through the full file round-trip. **Share links:** the settings live in a readable URL hash
(`#data=spirals&points=400&…&layers=8tanh,8tanh&lr=0.03&…`). It stays in step as you change
things, it is applied before the first render, and **Copy link** copies it. **Presets:**
Underfitting, Overfitting, Dead ReLUs and Too high LR, plus Default settings. Each has a note
(what it is, what to look for, what to try) and a **Pin this run and try the fix** button.
**Compare:** pin the current run (or load a saved model) as a reference. Its loss curves are
overlaid on the live ones, and a Compare panel shows both decision boundaries side by side, a
table of scores and the settings that changed. 235 tests.

**Decisions:**

- **What a checkpoint holds:** epoch, step, loss histories, parameters, optimiser state, both RNG
  states, and the **training-set order**. Each epoch shuffles the previous order in place, so the
  next one depends on it. Mutation tests show that dropping the order, either RNG or the optimiser
  state each breaks exact resume. Optimiser state is keyed by parameter name (`"0.W"`), not by
  tensor identity. Engine hooks: `Rng.getState/setState`, `Optimiser.saveState/loadState`,
  `exportParams/importParams`.
- **The file format** is `{ format: "neuroviz-model", version: 1, savedAt, config, checkpoint }`.
  Each array goes on one line, and each float32 is written as the shortest decimal that reads back
  as the same float32 (`0.1`, not `0.10000000149011612`). That is lossless (checked on 10,000
  random bit patterns) and readable. A diverged loss is written as `"NaN"` or `"Infinity"`.
  Loading checks the structure, holds the settings to the choices the controls offer, then builds
  the session the file describes. Any mismatch (another network, another dataset size) becomes a
  readable error **before** anything is replaced.
- **The settings type and every control's choices moved to `state/config.ts`.** The UI, the
  validator (`state/validate.ts`), the link codec and the presets all share it. Settings from a
  file must be valid. A link is lenient: missing values take defaults, and invalid ones are
  ignored with a notice that names what is allowed.
- **Reset after a load returns to the loaded checkpoint.** Changing the data, network or seed
  drops the checkpoint; hyperparameters still apply live. Loading a model or a preset always
  starts a fresh, paused session.
- **A link reproduces the set-up, not the weights.** Everything is seeded, so training from it
  repeats the run exactly. The hash is written with `history.replaceState`, debounced to 300 ms,
  because Safari throws after 100 calls in 30 s while a slider is dragged. A `hashchange`
  listener applies links pasted into an open tab.
- **Worker protocol:** `checkpoint` request/reply, and `init` takes `resume`. Errors now carry the
  `requestId` of a failed request, so a failed save rejects the right promise without disturbing
  the snapshot in flight. A save sees any hyperparameter change sent before it, because messages
  are handled in order.
- **Presets were picked by a seeded sweep** (seeds 1–5) and each has a **fix**. Tests check both
  the effect and the fix for seed 1; the 5-seed ranges are quoted in `presets.test.ts`. Default
  spirals converges too slowly to be a quick "good fit" baseline, so each preset is paired with its
  own fix instead. The fixes are: Underfitting (circle, 1 tanh neuron, 65 % → 3 neurons, 99 %);
  Overfitting (150 noisy points, 6×8 ReLU, Adam; test loss climbs to 1.2–4.5 → L2 0.03 keeps it at
  0.32–0.40); Dead ReLUs (4×8 ReLU, Adam lr 0.1, 50–69 % of neurons dead → lr 0.01, ≤ 16 %);
  Too high LR (the default with lr 3; the loss rises in ~44 % of epochs → lr 0.03, ≤ 2.4 %).
- **Comparing is "pin a reference"** (agreed in planning), not two live runs. One worker. A pinned
  run keeps its snapshot's arrays by reference (snapshots are never mutated). A reference loaded
  from a file rebuilds that session on the main thread just long enough for one snapshot.
- **Reference colour:** violet (`#4a3aa7` light, `#ab9ff5` dark). Hue shows which run, and the
  dashes show test vs train, as before. Against the existing ink and grey lines, the palette
  validator gives normal-vision ΔE ≥ 17 in both modes. A lighter violet for the reference's test
  line failed against the grey test line (ΔE 8.5), so both reference lines use the one violet. The
  reference is drawn thinner, behind the live run. The x-axis spans the longer run. The loss curve
  gained a crosshair tooltip with every value at the hovered epoch, and lines now break at
  non-finite values instead of joining across them.
- `HistogramTimeline.due` counts from the first recorded epoch rather than 0, so a run resumed at
  an odd epoch keeps evenly spaced columns.
- The run-neuroviz driver gained `download`, `upload`, `goto`, `reload` and `url`.

**Known issues:**

All six were fixed in the follow-up below.

- ~~After a load, the histogram timeline and the hover sparklines restart at the loaded epoch.~~
- ~~Model files grow with the loss history: ~18 KB at 600 epochs, roughly 2 MB at 100,000.~~
- ~~A pinned run records its final settings only.~~
- ~~One noisy Adam spike can squash the linear loss axis.~~
- ~~Dead ReLUs are visible as blank tiles and in the bottom gradient bin, but no neuron is
  explicitly flagged as dead.~~
- ~~Speed and "Show test data" are not part of the link.~~

### Phase 4 follow-up — the six known issues (2026-10-02)

**Done** (one commit each, in this order):

1. **Links carry the view.** `speed=…&showTest=0|1` join the hash. A link whose settings are a
   preset (or its fix) opens with that preset's note.
2. **One spike no longer squashes the loss chart.** On a linear scale the top of the axis is
   1.25 × the 98th percentile, but never below any curve's starting or latest value, so the
   early descent and the current state always show. Anything higher is drawn along the top, and
   the chart says "Clipped above 4.6 (peak 13)". Log scale keeps the full range.
3. **Dead ReLUs are flagged.** A ReLU neuron that outputs 0 for every training point is marked in
   the snapshot, found during the existing full-batch gradient pass at no extra cost. Dead
   neurons are hatched in the graph, counted in the column label ("Hidden 1 · 5 dead"),
   explained on hover and counted in the Inside training table.
4. **Mid-run setting changes are recorded.** The session logs the settings in force from each
   epoch, and the log is in snapshots, checkpoints and files. The loss curve marks changes with
   ticks (described on hover), and the Compare panel lists each run's changes.
5. **The charts' history survives save and load,** bitwise, including across a timeline
   compaction.
6. **Bounded loss history.** Every epoch is kept for the first 4,096; then neighbouring buckets
   merge (mean, min, max) and the width doubles. The latest and lowest losses stay exact. The
   chart draws at most ~2 points per pixel, each with a faint min–max band, so spikes survive on
   long runs. The tooltip gives a merged bucket's mean, range and epochs. 290 tests.

**Decisions:**

- **Model-file format 2** brings the settings log, the `history` block and the bucketed `losses`.
  Version 1 files still load: their per-epoch losses are replayed, which is exact because the
  session now always records float32-rounded losses (the values it always stored). The settings
  are taken as unchanged through the run, and the charts restart.
- **The `history` block is compact by choice** (agreed in planning). Histograms are whole counts
  per bin, 1, 2 or 4 bytes wide by layer size. RMS values and the sparkline ring are float32. All
  are little-endian base64. Everything else in the file stays readable. For that to be exact,
  `histogramInto` now counts whole numbers and then divides, instead of adding 1/n repeatedly.
- **Measured file sizes** at 600 epochs: ~97 KB for the default 2×8 net (80 KB of it chart
  history) and ~284 KB for 6×8, where ~210 KB is the 100-epoch sparkline ring. Loss curves top
  out near 270 KB whatever the run length (a 100,000-epoch run used to need ~2.3 MB). Until
  buckets merge, only the means are written, since min = max = mean.
- **"Dead" means dead on the training set**, not on the plotted plane, which also covers points
  outside the data. A hatched tile can therefore still show some colour away from the data. The
  Dead ReLUs preset test now uses these flags. Re-measured on seeds 1–5: 53–72 % dead, and
  3–19 % with the fix, so its "Then try" text now says "under a fifth" rather than "almost every
  neuron".
- **The settings log de-duplicates.** The UI re-sends unchanged settings. Several changes within
  one epoch keep only the last, and changing back removes the entry. A resume with different
  settings logs the change at the checkpoint's epoch.
- **Long-run speed is unchanged:** about 4,600 epochs/s at Max on the spirals, as in Phase 3.
- The run-neuroviz driver gained `move SELECTOR FX FY`, which points at a spot on a canvas (used
  to check the change-tick tooltip).

**Known issues:**

- The 6×8 network's sparkline history dominates its file size (~210 KB). Shortening the hover
  history, or saving it at lower precision, would shrink it if that matters.
- The clipping rule can still cut a genuine late climb taller than 1.25 × the 98th percentile
  that isn't at the very end; the label says so and the log scale shows it all.
- Reference ticks sit slightly above this run's ticks so both stay visible. With many changes,
  the x-axis can get busy.
- Loading a reference from a file still rebuilds its session on the main thread (milliseconds at
  this scale).

### Phase 5 — MNIST with an MLP (2026-10-03)

**Done:** A **Playground | MNIST** tab switch. The MNIST tab trains an MLP on a bundled, seeded,
digit-stratified subset:

- 10,000 images from the official training set and 2,000 from the official test set;
- 1,000 and 200 of each digit;
- `public/mnist-subset-v1.bin.gz`, 1.97 MB, rebuilt byte for byte by `npm run data:mnist`.

It shows:

- the **first-layer weights as 28×28 tiles**, scaled per tile or shared, with hover, click to
  enlarge and a table view; with no hidden layer they are the 10 class templates;
- a **draw-a-digit pad** with live prediction bars, and the most active units outlined on the tiles;
- a **confusion matrix** with every count written in; clicking a mistake filters the gallery;
- a **gallery of the most confidently wrong** test images, each of which can be tried on the pad;
- loss and error-rate curves.

MNIST runs **save and load** and resume exactly, even mid-epoch. **Performance pass:** an exact,
faster `matmul`, a skipped input gradient for the first layer, and grow-only layer buffers.

**Done-when:** the default 784 → 128 ReLU → 10 network (Adam, lr 0.001, batch 64) reaches **95 %
test accuracy about 4 s after Play** in the browser (static build, ~2.2 epochs/s). After 20 s it
is at ~96 %. A seeded test asserts ≥ 95 % within 6 epochs. Seeds 1–5 first reach it after 5–6
epochs (about 2.6 s in Node) and are at 95.4–96.2 % after 10 epochs. 349 tests.

**Decisions:**

- **MNIST is its own session, snapshot and page** (agreed in planning). The playground's session
  cannot stretch to 784 inputs:
  - every snapshot runs a full-batch gradient;
  - the sparkline ring would be ~40 MB;
  - the graph would need ~100k SVG edges.

  `MnistSession` shares the engine, `LossHistory`, the optimisers' save/load, the controller and
  `TrainingClient`. The controller drives either kind of session through a small `WorkerSession`
  interface, picked by `config.task`.

- **The unit of work is one mini-batch,** not an epoch. An epoch (~0.4 s) is far longer than the
  worker's 12 ms slice, so Pause and snapshots would lag. Step trains one batch, and Speed is in
  batches/s (30 / 100 / 300 / Max; Max by default).
- **Metrics are recorded every 5,000 training examples** (twice an epoch), so the x-axis is linear
  in examples even if the batch size changes. Each record runs in evaluation mode:
  - loss and error on a fixed 1,000-image training subset (the same for every run) and on all
    2,000 test images;
  - each test image's predicted class and confidence.

  That costs ~21 % of training time. Every 2,000 examples cost 41 %, which is why the interval is
  5,000. Snapshots then take ~0.03 ms. Error rate (1 − accuracy) reuses `LossHistory` unchanged,
  since lower is better there as for loss.

- **Each snapshot carries every test image's prediction and confidence** (10 KB), rather than a
  top-24 list as planned. The confusion matrix, the gallery and its per-cell filter all come from
  that one source, in `viz/mnistResults.ts`.
- **Pixels stay as bytes** (9.4 MB, not 37 MB as float32). `gatherImages` scales them into each
  batch. The parsed subset goes to the worker in `init` (structured clone); the UI keeps its own
  copy for the gallery.
- **`matmul` is bitwise identical to before.** The new kernels:
  - reorder the loops so the inner loop is contiguous;
  - accumulate each element in float64 in the original order (a float32 × float32 product is exact
    in float64);
  - skip zero a-values, only when b is finite, so 0 · Inf still gives NaN.

  Small products (k · n < 256) keep the plain loop, whose set-up costs less at the playground's
  sizes. Every transpose combination is compared bitwise against the old loop, and a mutation
  (float32 accumulators) fails that test. All existing tests, including the seeded preset numbers,
  pass unchanged.

  Measured in Node:
  - a 784-128-10 epoch: 2.5 s → 0.33 s;
  - the spirals: unchanged at ~4,200 epochs/s;
  - in the browser at Max: still ~4,600 epochs/s.

- **`Dense.inputGrad = false`** on a model's first layer skips the unused ∂L/∂x, a third of that
  layer's backward cost at 784 inputs. It is gradient-checked: the parameter gradients are identical
  and the input gradient is zeros.
- **`TensorBuffer`** gives layers and losses grow-only output buffers with views, so alternating
  batch sizes no longer reallocates. This fixes the Phase 2 known issue.
- **Default: 1 × 128 ReLU, Adam, lr 0.001, batch 64**, from a sweep of six settings × 3 seeds over
  30 epochs. Wider nets (256) or two layers gain ≤ 0.5 points at up to twice the cost.
- **Drawn digits are preprocessed like MNIST** (`data/drawing.ts`):
  - crop to the ink;
  - area-average the longer side to 20 px;
  - centre the centre of mass in 28×28.

  "Try" from the gallery passes the exact test image instead, so the pad agrees with the gallery.

- **Model-file format `neuroviz-mnist-model` v1.** Settings and curves are readable. Weights,
  optimiser state, order and test predictions are base64. The default net with Adam is ~1.6 MB.
  Loading:
  - checks the data checksum and the settings;
  - builds the session the file describes on the main thread (milliseconds: resuming skips
    evaluation) before anything is replaced.

  Each tab's loader sends the other tab's files to the right tab.

- **Tabs:** leaving a tab pauses its run and keeps it. The MNIST worker and data start on the first
  visit. The hash is `#mnist` while that tab shows, and the playground's link otherwise.
- **Colour:** the weight tiles use the playground's sign colours, so sign reads the same on both
  tabs. The confusion matrix colours only the mistakes, on the validated sequential ramp scaled to
  the largest mistake, with the diagonal outlined. All its numbers are written in, so it doubles as
  its own table.
- The run-neuroviz driver gained `draw` for the pad. `goto` and `reload` now work on either tab.

**Known issues:**

- No MNIST share links, presets or compare (out of scope by agreement), and no gradient histograms,
  dead-unit flags or settings-change ticks for MNIST.
- The first record comes after 5,000 images, so the curves jump from chance to ~90 % in one step.
- The default net overfits (100 % train, ~96 % test by epoch 20). Border pixels are almost always
  blank, so their weights keep their random start and the tiles look noisy. L2 0.001 clears both:
  96.7 % test and clean stroke detectors after 15 s, though a few ReLU tiles go blank (dead).
- The gallery's confidences come from the last record, while "Try" uses the current weights, so the
  two can differ a little mid-record.
- With 2,000 test images, one image is 0.05 %, so the test accuracy wobbles by a few tenths between
  records.
- Only checked in headless Chrome. `DecompressionStream` and pointer events are standard, but
  Firefox and Safari were not tried.
- Canvas drawing (tiles, pad, digit images) has no component tests, because jsdom has no 2D
  context. It was checked through driver screenshots in light and dark mode.
