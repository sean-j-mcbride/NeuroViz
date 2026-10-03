import type { NeuronColumn, TimelineSnapshot } from '../worker';
import { formatPrecise } from './colour';
import { EPOCH_AXIS, type TimelineAxis } from './labels';
import { shareWhere } from './timelineStats';

const pct = (f: number) => `${(f * 100).toFixed(1)} %`;

/** The Inside-training charts as tables: the latest state per layer, and gradient size over time. */
/** "3 of 8" dead neurons in a ReLU column; "–" for other kinds. */
function deadText(column: NeuronColumn | undefined): string {
  if (!column?.dead) return '–';
  const n = column.dead.reduce((a, d) => a + d, 0);
  return `${n} of ${column.units}`;
}

export function InsideTables({
  timeline,
  columns,
  axis = EPOCH_AXIS,
  deadOf,
  deadTitle = 'ReLU neurons fed by this layer that output 0 for every training point',
}: {
  timeline: TimelineSnapshot;
  /** The snapshot's neuron columns, for the dead-neuron counts. */
  columns?: NeuronColumn[];
  axis?: TimelineAxis;
  /** Overrides the dead count shown for layer k (e.g. "2 of 8 channels"). */
  deadOf?: (k: number) => string;
  deadTitle?: string;
}) {
  const { epochs, layers, weightBins, gradBins } = timeline;
  const n = layers.length;
  const T = epochs.length;
  if (T === 0) return null;
  const last = T - 1;
  const rows = Array.from({ length: T }, (_, i) => T - 1 - i); // newest first

  return (
    <div className="inside-tables">
      <div>
        <h3>Per layer at {axis.at(epochs[last]!)}</h3>
        <table className="data-table">
          <thead>
            <tr>
              <th scope="col">Layer</th>
              <th scope="col">Weight RMS</th>
              <th scope="col">Weights with |w| &gt; 1</th>
              <th scope="col">Gradient RMS</th>
              <th scope="col">Gradients below 10⁻⁶</th>
              <th scope="col" title={deadTitle}>
                Dead outputs
              </th>
            </tr>
          </thead>
          <tbody>
            {layers.map((l, k) => (
              <tr key={k}>
                <th scope="row">{axis.layerName(k, n)}</th>
                <td>{formatPrecise(l.weightRms[last]!)}</td>
                <td>≈ {pct(shareWhere(l.weightHist, last, weightBins, (c) => Math.abs(c) > 1))}</td>
                <td>{formatPrecise(l.gradRms[last]!)}</td>
                <td>≈ {pct(shareWhere(l.gradHist, last, gradBins, (c) => c < 1e-6))}</td>
                <td>{deadOf ? deadOf(k) : deadText(columns?.[k + 1])}</td>
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
                <th scope="col">{axis.column}</th>
                {layers.map((_, k) => (
                  <th key={k} scope="col">
                    {axis.layerName(k, n)}
                  </th>
                ))}
                {n > 1 && <th scope="col">First ÷ last</th>}
              </tr>
            </thead>
            <tbody>
              {rows.map((t) => (
                <tr key={t}>
                  <th scope="row">{axis.cell(epochs[t]!)}</th>
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
