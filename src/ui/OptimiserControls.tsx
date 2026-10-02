import { DROPOUT_RATES, L2_STRENGTHS, OPTIMISER_NAMES, useAppStore } from '../state/store';
import type { Hyperparams } from '../worker';

const OPTIMISERS: { value: Hyperparams['optimiser']; title: string }[] = [
  { value: 'sgd', title: 'Plain stochastic gradient descent' },
  { value: 'momentum', title: 'SGD with momentum (β = 0.9)' },
  { value: 'adam', title: 'Adam (β₁ = 0.9, β₂ = 0.999)' },
];

/** Optimiser and regularisation settings; all apply live without resetting training. */
export function OptimiserControls() {
  const { optimiser, l2, dropout } = useAppStore((s) => s.config.training);
  const setTraining = useAppStore.getState().setTraining;

  return (
    <div className="transport optimiser-controls">
      <label className="field">
        <span>Optimiser</span>
        <select
          value={optimiser}
          onChange={(e) => setTraining({ optimiser: e.target.value as Hyperparams['optimiser'] })}
          title={OPTIMISERS.find((o) => o.value === optimiser)?.title}
        >
          {OPTIMISERS.map((o) => (
            <option key={o.value} value={o.value} title={o.title}>
              {OPTIMISER_NAMES[o.value]}
            </option>
          ))}
        </select>
      </label>

      <label className="field">
        <span>L2 regularisation</span>
        <select
          value={l2}
          onChange={(e) => setTraining({ l2: Number(e.target.value) })}
          title="Penalty λ/2 · Σw² on the weights (not biases)"
        >
          {L2_STRENGTHS.map((v) => (
            <option key={v} value={v}>
              {v === 0 ? 'None' : v}
            </option>
          ))}
        </select>
      </label>

      <label className="field">
        <span>Dropout</span>
        <select
          value={dropout}
          onChange={(e) => setTraining({ dropout: Number(e.target.value) })}
          title="Fraction of hidden neurons switched off at random on each training step"
        >
          {DROPOUT_RATES.map((v) => (
            <option key={v} value={v}>
              {v === 0 ? 'None' : `${v * 100} %`}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}
