import type { TimelineSnapshot } from '../worker';
import { formatPrecise } from './colour';
import { layerLabel } from './labels';
import { shareWhere } from './timelineStats';

const pct = (f: number) => `${(f * 100).toFixed(1)} %`;

/** The Inside-training charts as tables: the latest state per layer, and gradient size over time. */
export function InsideTables({ timeline }: { timeline: TimelineSnapshot }) {
  const { epochs, layers, weightBins, gradBins } = timeline;
  const n = layers.length;
  const T = epochs.length;
  if (T === 0) return null;
  const last = T - 1;
  const rows = Array.from({ length: T }, (_, i) => T - 1 - i); // newest first

  return (
    <div className="inside-tables">
      <div>
        <h3>Per layer at epoch {epochs[last]}</h3>
        <table className="data-table">
          <thead>
            <tr>
              <th scope="col">Layer</th>
              <th scope="col">Weight RMS</th>
              <th scope="col">Weights with |w| &gt; 1</th>
              <th scope="col">Gradient RMS</th>
              <th scope="col">Gradients below 10⁻⁶</th>
            </tr>
          </thead>
          <tbody>
            {layers.map((l, k) => (
              <tr key={k}>
                <th scope="row">{layerLabel(k, n)}</th>
                <td>{formatPrecise(l.weightRms[last]!)}</td>
                <td>≈ {pct(shareWhere(l.weightHist, last, weightBins, (c) => Math.abs(c) > 1))}</td>
                <td>{formatPrecise(l.gradRms[last]!)}</td>
                <td>≈ {pct(shareWhere(l.gradHist, last, gradBins, (c) => c < 1e-6))}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="hint">
          Shares are read from the histograms, so they are accurate to about one bin.
        </p>
      </div>

      <div>
        <h3>Gradient RMS over time</h3>
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th scope="col">Epoch</th>
                {layers.map((_, k) => (
                  <th key={k} scope="col">
                    {layerLabel(k, n)}
                  </th>
                ))}
                {n > 1 && <th scope="col">First ÷ last</th>}
              </tr>
            </thead>
            <tbody>
              {rows.map((t) => (
                <tr key={t}>
                  <th scope="row">{epochs[t]}</th>
                  {layers.map((l, k) => (
                    <td key={k}>{formatPrecise(l.gradRms[t]!)}</td>
                  ))}
                  {n > 1 && (
                    <td>{formatPrecise(layers[0]!.gradRms[t]! / layers[n - 1]!.gradRms[t]!)}</td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
