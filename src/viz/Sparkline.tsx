const W = 180;
const H = 36;
const PAD = 3;

interface SparklineProps {
  values: Float32Array;
  /** Epoch of each value, for the end labels. */
  epochs: Float32Array;
}

/** A small line of recent values, with the range and epoch span labelled. */
export function Sparkline({ values, epochs }: SparklineProps) {
  const n = values.length;
  if (n < 2) return null;
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  const span = hi - lo || 1;
  const x = (i: number) => PAD + (i / (n - 1)) * (W - 2 * PAD);
  const y = (v: number) => PAD + (1 - (v - lo) / span) * (H - 2 * PAD);
  const points = Array.from(values, (v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  return (
    <figure className="sparkline">
      <svg width={W} height={H} role="img" aria-label="Recent history">
        <polyline points={points} />
        <circle cx={x(n - 1)} cy={y(values[n - 1]!)} r={2.5} />
      </svg>
      <figcaption>
        <span>epoch {epochs[0]}</span>
        <span>
          range {lo.toPrecision(3)} … {hi.toPrecision(3)}
        </span>
        <span>{epochs[n - 1]}</span>
      </figcaption>
    </figure>
  );
}
