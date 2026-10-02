import type { Snapshot } from '../worker';

function pct(v: number): string {
  return `${(v * 100).toFixed(1)} %`;
}

export function Stats({ snapshot }: { snapshot: Snapshot }) {
  const { trainLoss, testLoss, trainAccuracy, testAccuracy } = snapshot;
  return (
    <table className="stats">
      <thead>
        <tr>
          <th />
          <th>Loss</th>
          <th>Accuracy</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Train</th>
          <td>{trainLoss.at(-1)?.toFixed(4)}</td>
          <td>{pct(trainAccuracy)}</td>
        </tr>
        <tr>
          <th>Test</th>
          <td>{testLoss.at(-1)?.toFixed(4)}</td>
          <td>{pct(testAccuracy)}</td>
        </tr>
      </tbody>
    </table>
  );
}
