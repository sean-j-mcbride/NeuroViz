import { useRef } from 'react';
import { type RunChange, changesDuringRun, diffConfigs } from '../state/configDiff';
import { referenceFromFile } from '../state/reference';
import { useAppStore } from '../state/store';
import { DecisionBoundary } from '../viz';
import { type LossHistorySnapshot, outputGrid } from '../worker';
import { errorText, readModelFile } from './files';

interface RunStats {
  epoch: number;
  losses: LossHistorySnapshot;
  trainAccuracy: number;
  testAccuracy: number;
}

const loss = (v: number) => (Number.isFinite(v) ? v.toFixed(4) : String(v));
const pct = (v: number) => `${(v * 100).toFixed(1)} %`;

const ROWS: [string, (r: RunStats) => string][] = [
  ['Epoch', (r) => r.epoch.toLocaleString('en-GB')],
  ['Train loss', (r) => loss(r.losses.train.latest)],
  ['Test loss', (r) => loss(r.losses.test.latest)],
  [
    'Lowest test loss',
    ({ losses: { test } }) =>
      test.bestEpoch < 0
        ? '–'
        : `${loss(test.best)} (epoch ${test.bestEpoch.toLocaleString('en-GB')})`,
  ],
  ['Train accuracy', (r) => pct(r.trainAccuracy)],
  ['Test accuracy', (r) => pct(r.testAccuracy)],
];

/** The live run side by side with a pinned (or loaded) reference run. */
export function ComparePanel() {
  const reference = useAppStore((s) => s.reference);
  const snapshot = useAppStore((s) => s.snapshot);
  const sessionData = useAppStore((s) => s.sessionData);
  const config = useAppStore((s) => s.config);
  const { pinReference, setReference, setNotice } = useAppStore.getState();
  const fileInput = useRef<HTMLInputElement>(null);

  const loadReference = async (file: File | undefined) => {
    if (!file) return;
    try {
      setReference(referenceFromFile(file.name, await readModelFile(file)));
    } catch (e) {
      setNotice({ kind: 'error', text: `Couldn’t load ${file.name}: ${errorText(e)}` });
    }
  };

  const toolbar = (
    <div className="compare-actions">
      <button
        type="button"
        onClick={() => pinReference()}
        disabled={!snapshot}
        title="Freeze the current run as the reference"
      >
        {reference ? 'Pin current run instead' : 'Pin current run'}
      </button>
      <button type="button" onClick={() => fileInput.current?.click()}>
        Load reference from file…
      </button>
      <input
        ref={fileInput}
        type="file"
        accept=".json,application/json"
        hidden
        aria-label="Model file to compare against"
        onChange={(e) => {
          void loadReference(e.target.files?.[0]);
          e.target.value = '';
        }}
      />
      {reference && (
        <button type="button" onClick={() => setReference(null)}>
          Clear
        </button>
      )}
    </div>
  );

  if (!reference || !snapshot) {
    return (
      <section className="panel compare-panel">
        <div className="panel-toolbar">
          <h2>Compare</h2>
          {toolbar}
        </div>
        <p className="hint">
          Pin the current run (or load a saved model) as a reference, then change something and
          train again. The reference’s loss curves appear under Output, and its decision boundary,
          scores and settings appear here beside the live run’s.
        </p>
      </section>
    );
  }

  const changes = diffConfigs(reference.config, config);
  const referenceChanges = changesDuringRun(reference.hyperparamLog);
  const runChanges = changesDuringRun(snapshot.hyperparamLog);
  return (
    <section className="panel compare-panel">
      <div className="panel-toolbar">
        <h2>Compare</h2>
        {toolbar}
      </div>
      <div className="compare-grid">
        <div className="compare-boundaries">
          <figure>
            <DecisionBoundary
              boundary={reference.boundary}
              data={reference.data}
              showTestData={false}
              title={`Reference (${reference.label}): probability of the blue class`}
            />
            <figcaption>
              <span className="swatch reference" /> Reference: {reference.label}
            </figcaption>
          </figure>
          <figure>
            <DecisionBoundary
              boundary={outputGrid(snapshot)}
              data={sessionData}
              showTestData={false}
              title="This run: probability of the blue class"
            />
            <figcaption>
              <span className="swatch train" /> This run
            </figcaption>
          </figure>
        </div>

        <table className="data-table compare-table">
          <thead>
            <tr>
              <th />
              <th scope="col">Reference</th>
              <th scope="col">This run</th>
            </tr>
          </thead>
          <tbody>
            {ROWS.map(([label, show]) => (
              <tr key={label}>
                <th scope="row">{label}</th>
                <td>{show(reference)}</td>
                <td>{show(snapshot)}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <div className="compare-diff">
          <h3>Settings changed since the reference</h3>
          {changes.length === 0 ? (
            <p className="hint">None: both runs use the same settings.</p>
          ) : (
            <ul>
              {changes.map((c) => (
                <li key={c.label}>
                  <span className="diff-label">{c.label}</span> {c.from} → <strong>{c.to}</strong>
                </li>
              ))}
            </ul>
          )}
          {(referenceChanges.length > 0 || runChanges.length > 0) && (
            <>
              <h3 className="compare-subhead">Changed during the run</h3>
              <ChangeList who="Reference" changes={referenceChanges} />
              <ChangeList who="This run" changes={runChanges} />
            </>
          )}
        </div>
      </div>
    </section>
  );
}

/** One run's mid-run setting changes (also marked with ticks on the loss curve). */
function ChangeList({ who, changes }: { who: string; changes: RunChange[] }) {
  if (changes.length === 0) return <p className="hint">{who}: none.</p>;
  return (
    <ul aria-label={`${who}: changes during the run`}>
      {changes.map((c) => (
        <li key={c.epoch}>
          <span className="diff-label">
            {who}, epoch {c.epoch.toLocaleString('en-GB')}:
          </span>{' '}
          {c.text}
        </li>
      ))}
    </ul>
  );
}
