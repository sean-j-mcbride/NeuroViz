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
thinning backward edges make it visible. 123 tests.

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

- Each snapshot still copies the full loss history and runs one extra full-batch
  forward/backward. Both are cheap at this scale; revisit for MNIST in Phase 5.
- Weights beyond ±50 still land in the edge bins (the tooltip says "and below" / "and above").
- With plain SGD, the full-batch gradient RMS line is spiky: weights move between records. This
  is real behaviour, not a rendering artefact.
- Right after Pause, the epoch readout can trail the true stopping point by one snapshot. A
  request may already be in flight, and the final snapshot follows it.
- Momentum β and Adam β₁/β₂ are fixed in the UI (configurable in the engine).
- There are still no component tests. The UI was checked by driving the dev server and the static
  preview build (light and dark) in headless Chrome.
