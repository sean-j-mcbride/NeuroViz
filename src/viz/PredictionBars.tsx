import { MNIST_CLASSES } from '../data';

const pct = (p: number) => `${(p * 100).toFixed(p >= 0.995 || p < 0.005 ? 0 : 1)} %`;

/** The probability of each digit for the drawn input, as bars with the values written out. */
export function PredictionBars({ probs }: { probs: Float32Array | null }) {
  const best = probs ? probs.indexOf(Math.max(...probs)) : -1;
  return (
    <div className="prediction-bars" role="list" aria-label="Predicted probability of each digit">
      {Array.from({ length: MNIST_CLASSES }, (_, d) => {
        const p = probs?.[d] ?? 0;
        return (
          <div
            key={d}
            role="listitem"
            className={`prediction-row ${d === best ? 'best' : ''}`}
            aria-label={probs ? `${d}: ${pct(p)}${d === best ? ', the prediction' : ''}` : `${d}`}
          >
            <span className="prediction-digit">{d}</span>
            <span className="prediction-track">
              <span className="prediction-bar" style={{ width: `${p * 100}%` }} />
            </span>
            <span className="prediction-value">{probs ? pct(p) : ''}</span>
          </div>
        );
      })}
    </div>
  );
}
