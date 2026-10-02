# NeuroViz — neural network visualiser

A browser app for building, training and _watching_ neural networks learn.
It starts with tiny MLPs on 2D toy data and grows towards MNIST and CNNs.
The NN engine is written from scratch in TypeScript so every number is
inspectable — no ML libraries in the engine.

See `docs/ROADMAP.md` for phases. Work on ONE phase at a time.

## Stack

- Vite + React + TypeScript (strict mode)
- Vitest for tests
- Rendering: SVG for the network graph, `<canvas>` for heatmaps / decision
  boundaries / feature maps (SVG gets slow past a few thousand elements)
- Charts: lightweight, hand-rolled on canvas or uPlot — no heavy chart libs
- No backend. Must build to static files (deployable to GitHub Pages).

## Architecture (keep these boundaries strict)

```
src/
  engine/      Pure TS. No DOM, no React. Tensor, layers, losses, optimisers,
               model, training loop. Must run in a Web Worker unchanged.
  data/        Dataset generators/loaders (2D toys, later MNIST).
  worker/      Training worker: runs engine, posts Snapshots to the UI.
  viz/         Visual components. Read Snapshots; never mutate the model.
  ui/          Controls: architecture builder, hyperparameters, play/pause.
  state/       App state (Zustand). Model config is serialisable JSON.
```

### Core contracts

- `Tensor`: `Float32Array` data + `shape: number[]`. Row-major.
- Every layer implements:
  ```ts
  interface Layer {
    kind: string; // "dense" | "relu" | "conv2d" ...
    forward(x: Tensor, train: boolean): Tensor;
    backward(gradOut: Tensor): Tensor; // stores param grads internally
    params(): { name: string; value: Tensor; grad: Tensor }[];
    toJSON(): LayerConfig;
  }
  ```
  Layers cache what they need from `forward` for `backward`. Explicit
  per-layer backward (not a general autograd graph) — easier to visualise.
- `Snapshot`: a plain, structured-cloneable object the worker sends to the
  UI (epoch, loss history, weights, activations/gradients for a probe batch,
  decision-boundary grid). Visualisations only ever consume Snapshots.
- Adding a new layer type = engine class + gradient-check test + a viz
  renderer registered by `kind`. Nothing else should need to change.

## Rules

- Every layer and loss gets a **numerical gradient check** test
  (central differences, tolerance ~1e-4 relative) before it is used in the UI.
- Seeded RNG everywhere (`engine/random.ts`) so runs are reproducible.
- Engine code: no `any`, no allocation inside hot inner loops where avoidable.
- Keep UI responsive: training runs in the worker; UI throttles snapshot
  requests (~10–20 per second max).
- Use British spelling in UI text.

## Commands

- `npm run dev` — dev server
- `npm test` — Vitest
- `npm run build` — static build
- `npm run lint` / `npm run typecheck`

## Workflow

- Start each phase in plan mode; agree the plan before writing code.
- Finish a phase with: tests green, typecheck clean, a short note added to
  `docs/ROADMAP.md` under that phase ("Done / decisions / known issues").
- Commit at the end of each phase.
