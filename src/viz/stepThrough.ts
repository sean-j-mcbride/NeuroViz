/**
 * Step-through stages for a network with C columns (input, hidden…, output):
 *
 *   0 … C−1      forward: column s has just been computed
 *   C            loss: the prediction is compared with the label
 *   C+1 … 2C−1   backward: the gradient has flowed back to column 2C−1−s
 */
export type Phase = 'forward' | 'loss' | 'backward';

export interface StageView {
  phase: Phase;
  /** Columns 0 … forwardThrough show their forward value (forward and loss phases). */
  forwardThrough: number;
  /** Columns gradFrom … C−1 show their gradient (backward phase); C when none do. */
  gradFrom: number;
}

export function stageCount(columns: number): number {
  return 2 * columns;
}

export function stageView(stage: number, columns: number): StageView {
  const last = columns - 1;
  if (stage < columns) return { phase: 'forward', forwardThrough: stage, gradFrom: columns };
  if (stage === columns) return { phase: 'loss', forwardThrough: last, gradFrom: columns };
  return {
    phase: 'backward',
    forwardThrough: last,
    gradFrom: Math.max(0, 2 * columns - 1 - stage),
  };
}

/** Whether dense layer k (column k → k+1) carries forward contributions w·a. */
export function forwardLayer(v: StageView, k: number): boolean {
  return v.phase !== 'backward' && k + 1 <= v.forwardThrough;
}

/** Whether dense layer k carries per-example gradients ∂L/∂w (the gradient has passed through it). */
export function gradLayer(v: StageView, k: number): boolean {
  return v.phase === 'backward' && k >= v.gradFrom;
}

export function stageLabel(stage: number, columnNames: string[]): string {
  const c = columnNames.length;
  const v = stageView(stage, c);
  if (v.phase === 'forward') return `Forward pass · ${columnNames[stage]}`;
  if (v.phase === 'loss') return 'Loss';
  return `Backward pass · ${columnNames[v.gradFrom]}`;
}
