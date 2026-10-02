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
