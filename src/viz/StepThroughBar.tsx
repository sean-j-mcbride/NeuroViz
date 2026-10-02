import { useEffect, useState } from 'react';
import type { ProbeTrace } from '../worker';
import { formatPrecise } from './colour';
import { stageCount, stageLabel, stageView } from './stepThrough';

const ANIMATION_MS = 700;

interface StepThroughBarProps {
  /** The trace for the picked point; null until one is picked (or while it loads). */
  probe: ProbeTrace | null;
  columnNames: string[];
  stage: number;
  onStage: (stage: number) => void;
  onClose: () => void;
}

const PHASE_HINT = {
  forward: 'Each neuron computes a = f(Σ w·a_prev + b). Edges show each input’s contribution w·a.',
  loss: 'The prediction p is compared with the label: loss = −log p for blue, −log(1 − p) for orange.',
  backward:
    'The gradient ∂L/∂a flows back through each layer (chain rule). Edges show ∂L/∂w for this point; width is logarithmic, spanning four orders of magnitude.',
};

/** Controls and commentary for step-through mode. */
export function StepThroughBar({
  probe,
  columnNames,
  stage,
  onStage,
  onClose,
}: StepThroughBarProps) {
  const [playing, setPlaying] = useState(false);
  const last = stageCount(columnNames.length) - 1;
  const animating = playing && probe !== null && stage < last;

  useEffect(() => {
    if (!animating) return;
    const id = setTimeout(() => onStage(stage + 1), ANIMATION_MS);
    return () => clearTimeout(id);
  }, [animating, stage, onStage]);

  if (!probe) {
    return (
      <div className="step-through">
        <div className="step-through-row">
          <strong>Step through</strong>
          <span className="step-through-hint">
            Click a data point on the output plot to trace it through the network.
          </span>
          <button type="button" className="close" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    );
  }

  const phase = stageView(stage, columnNames.length).phase;
  return (
    <div className="step-through">
      <div className="step-through-row">
        <strong>{stageLabel(stage, columnNames)}</strong>
        <span className="step-through-point">
          ({probe.x[0]!.toFixed(2)}, {probe.x[1]!.toFixed(2)}) · label{' '}
          {probe.label === 1 ? 'blue' : 'orange'} · p {probe.p.toFixed(3)} · loss{' '}
          {formatPrecise(probe.loss)}
        </span>
        <div className="step-through-buttons">
          <button
            type="button"
            onClick={() => onStage(Math.max(0, stage - 1))}
            disabled={stage === 0}
            aria-label="Previous stage"
          >
            ◀
          </button>
          <button
            type="button"
            className="primary"
            onClick={() => {
              if (animating) return setPlaying(false);
              if (stage >= last) onStage(0);
              setPlaying(true);
            }}
          >
            {animating ? '❚❚ Pause' : '▶ Animate'}
          </button>
          <button
            type="button"
            onClick={() => onStage(Math.min(last, stage + 1))}
            disabled={stage >= last}
            aria-label="Next stage"
          >
            ▶
          </button>
          <button type="button" className="close" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
      <progress max={last} value={stage} aria-label="Stage" />
      <p className="step-through-hint">{PHASE_HINT[phase]}</p>
    </div>
  );
}
