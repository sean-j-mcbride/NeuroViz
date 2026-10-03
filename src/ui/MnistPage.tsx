import { useMemo, useRef, useState } from 'react';
import { MNIST_PIXELS, type MnistSet } from '../data';
import { useMnistStore } from '../state/mnistStore';
import { MNIST_BATCH_SIZES, mnistNetworkError } from '../state/mnistConfig';
import {
  ConfusionMatrix,
  DigitImage,
  DigitPad,
  type DigitPadHandle,
  FeatureMaps,
  FilterViewer,
  LossCurve,
  PredictionBars,
  WeightImages,
  WeightTile,
  WrongGallery,
  confusionCounts,
  formatPrecise,
  perClassAccuracy,
  recordEpoch,
  topUnits,
  unitStats,
} from '../viz';
import { type FirstLayerWeights, type MnistSnapshot, type Speed, TRAIN_EVAL_SIZE } from '../worker';
import { HyperparamControls } from './OptimiserControls';
import { MnistArchitecture } from './MnistArchitecture';
import { MnistProjectBar } from './MnistProjectBar';
import { TransportControls } from './TransportBar';
import { useMnistWorker } from './useMnistWorker';

const MNIST_SPEEDS: readonly Speed[] = [30, 100, 300, 'max'];

const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)} %`;

function MnistControls({
  onStep,
  requestCheckpoint,
}: {
  onStep: () => void;
  requestCheckpoint: ReturnType<typeof useMnistWorker>['requestCheckpoint'];
}) {
  const running = useMnistStore((s) => s.running);
  const speed = useMnistStore((s) => s.speed);
  const { training, seed, network } = useMnistStore((s) => s.config);
  const epoch = useMnistStore((s) => s.snapshot?.epoch ?? 0);
  const { setRunning, setSpeed, setTraining, setSeed, reset } = useMnistStore.getState();
  return (
    <div className="app-header page-controls">
      <TransportControls
        running={running}
        onRunning={setRunning}
        onReset={reset}
        onStep={onStep}
        stepUnit="one mini-batch"
        speedUnit="batches"
        readout={{
          label: 'Epoch',
          value: epoch.toLocaleString('en-GB', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
          }),
        }}
        lr={training.lr}
        batchSize={training.batchSize}
        batchSizes={MNIST_BATCH_SIZES}
        onTraining={setTraining}
        speed={speed}
        speeds={MNIST_SPEEDS}
        onSpeed={setSpeed}
        seed={seed}
        onSeed={setSeed}
        blocked={mnistNetworkError(network) && 'Fix the network first (see Network)'}
      />
      <HyperparamControls
        optimiser={training.optimiser}
        l2={training.l2}
        dropout={training.dropout}
        onTraining={setTraining}
      />
      <MnistProjectBar requestCheckpoint={requestCheckpoint} />
    </div>
  );
}

function WeightsPanel({ layer, hidden }: { layer: FirstLayerWeights; hidden?: Float32Array }) {
  const [scale, setScale] = useState<'tile' | 'shared'>('tile');
  const [asTable, setAsTable] = useState(false);
  const [selected, setSelected] = useState<number | null>(null);
  // No hidden layer: the first layer is the output (hidden layers are never 10 wide).
  const classes = layer.outFeatures === 10;
  const unitName = (j: number) => (classes ? `Digit ${j}` : `Unit ${j + 1}`);
  const stats = useMemo(() => (asTable ? unitStats(layer) : []), [asTable, layer]);
  const active = useMemo(() => new Set(hidden ? topUnits(hidden, 8) : []), [hidden]);
  const sel = selected !== null && selected < layer.outFeatures ? selected : null;

  return (
    <section className="panel weights-panel">
      <div className="panel-toolbar">
        <h2>{classes ? 'Weights for each digit' : 'What the first layer looks for'}</h2>
        <div className="segmented" role="group" aria-label="Show as">
          <button type="button" aria-pressed={!asTable} onClick={() => setAsTable(false)}>
            Images
          </button>
          <button type="button" aria-pressed={asTable} onClick={() => setAsTable(true)}>
            Table
          </button>
        </div>
      </div>
      {asTable ? (
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th scope="col">{classes ? 'Digit' : 'Unit'}</th>
                <th scope="col">Lowest weight</th>
                <th scope="col">Highest weight</th>
                <th scope="col">RMS</th>
                <th scope="col">Bias</th>
                {hidden && <th scope="col">For your digit</th>}
              </tr>
            </thead>
            <tbody>
              {stats.map((s, j) => (
                <tr key={j} className={active.has(j) ? 'highlight' : undefined}>
                  <th scope="row">{classes ? j : j + 1}</th>
                  <td>{formatPrecise(s.min)}</td>
                  <td>{formatPrecise(s.max)}</td>
                  <td>{formatPrecise(s.rms)}</td>
                  <td>{formatPrecise(s.bias)}</td>
                  {hidden && <td>{formatPrecise(hidden[j]!)}</td>}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <>
          <div className="weights-options">
            <div className="segmented" role="group" aria-label="Colour scale">
              <button
                type="button"
                aria-pressed={scale === 'tile'}
                onClick={() => setScale('tile')}
              >
                Each tile
              </button>
              <button
                type="button"
                aria-pressed={scale === 'shared'}
                onClick={() => setScale('shared')}
              >
                Shared scale
              </button>
            </div>
            <span className="legend">
              <span className="legend-item">
                <span className="swatch-box neg" /> Negative
              </span>
              <span className="legend-item">
                <span className="swatch-box mid" /> 0
              </span>
              <span className="legend-item">
                <span className="swatch-box pos" /> Positive
              </span>
            </span>
          </div>
          <WeightImages
            layer={layer}
            scale={scale}
            hidden={hidden}
            unitName={unitName}
            selected={sel}
            onSelect={setSelected}
          />
          {sel !== null && (
            <div className="weight-detail">
              <WeightTile layer={layer} unit={sel} />
              <div>
                <h3>{unitName(sel)}</h3>
                <p className="hint">
                  Blue pixels raise this {classes ? 'digit’s score' : 'unit’s input'} when inked,
                  orange ones lower it.{' '}
                  {hidden && `For your digit it gives ${formatPrecise(hidden[sel]!)}.`}
                </p>
                <button type="button" onClick={() => setSelected(null)}>
                  Close
                </button>
              </div>
            </div>
          )}
        </>
      )}
      <p className="hint">
        {classes
          ? 'With no hidden layer, each digit’s score is its template’s match with the image.'
          : 'Each tile is one unit’s 784 incoming weights laid out as the image. They start as random noise, and training draws strokes into the middle. Pixels near the edge are almost always blank, so their weights get no gradient and keep the noise: try L2 regularisation 0.001 to clear it.'}{' '}
        {hidden && !classes ? 'Outlined: the units your digit excites most. ' : ''}Click a tile to
        enlarge it.
      </p>
    </section>
  );
}

function TryPanel({
  padRef,
  probs,
  onTestImage,
}: {
  padRef: React.RefObject<DigitPadHandle | null>;
  probs: Float32Array | null;
  onTestImage: (() => void) | null;
}) {
  const drawn = useMnistStore((s) => s.drawn);
  const setDrawn = useMnistStore.getState().setDrawn;
  return (
    <section className="panel try-panel">
      <h2>Try it</h2>
      <div className="pad-wrap">
        <DigitPad ref={padRef} onChange={setDrawn} />
        {!drawn && <span className="pad-placeholder">Draw a digit here</span>}
      </div>
      <div className="pad-row">
        <button type="button" onClick={() => padRef.current?.clear()} disabled={!drawn}>
          Clear
        </button>
        {onTestImage && (
          <button type="button" onClick={onTestImage} title="Show a random image from the test set">
            Test image
          </button>
        )}
        {drawn && (
          <span className="pad-preview">
            <DigitImage pixels={drawn} size={42} label="What the network sees" />
            <span className="hint">
              What the network sees: cropped, scaled to 20 px and centred
            </span>
          </span>
        )}
      </div>
      <PredictionBars probs={drawn ? probs : null} />
      <p className="hint">The prediction updates as the network trains.</p>
    </section>
  );
}

function TrainingPanel({ snapshot }: { snapshot: MnistSnapshot }) {
  // Log scale by default: past the first epoch the interesting part is a few per cent of error.
  const [logScale, setLogScale] = useState(true);
  const { recordEvery, trainSize } = snapshot;
  const xName = useMemo(
    () => (i: number) => `epoch ${recordEpoch(i, recordEvery, trainSize)}`,
    [recordEvery, trainSize],
  );
  const xRange = useMemo(
    () => (a: number, b: number) =>
      `epochs ${recordEpoch(a, recordEvery, trainSize)}–${recordEpoch(b, recordEvery, trainSize)}`,
    [recordEvery, trainSize],
  );
  const percent = (v: number) => pct(v, 2);
  const percentTick = (v: number) => `${+(v * 100).toPrecision(2)} %`;
  return (
    <section className="panel training-panel">
      <h2>Training</h2>
      <table className="stats">
        <tbody>
          <tr>
            <th scope="row">Test accuracy</th>
            <td className="big">{pct(snapshot.testAccuracy)}</td>
          </tr>
          <tr>
            <th scope="row">Train accuracy</th>
            <td>{pct(snapshot.trainAccuracy)}</td>
          </tr>
          <tr>
            <th scope="row">Test loss</th>
            <td>{snapshot.losses.test.latest.toFixed(4)}</td>
          </tr>
          <tr>
            <th scope="row">Images seen</th>
            <td>{snapshot.examples.toLocaleString('en-GB')}</td>
          </tr>
        </tbody>
      </table>
      <p className="hint">
        Test: all {snapshot.testSize.toLocaleString('en-GB')} test images. Train: a fixed{' '}
        {TRAIN_EVAL_SIZE.toLocaleString('en-GB')} of the {trainSize.toLocaleString('en-GB')}{' '}
        training images. Measured every {recordEvery.toLocaleString('en-GB')} images.
      </p>
      <div className="loss-header">
        <h3>Loss</h3>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={logScale}
            onChange={(e) => setLogScale(e.target.checked)}
          />
          Log scale
        </label>
      </div>
      <div className="legend loss-legend">
        <span className="legend-item">
          <span className="swatch train" /> Train
          <span className="swatch test" /> Test
        </span>
      </div>
      <LossCurve
        losses={snapshot.losses}
        logScale={logScale}
        xName={xName}
        xRange={xRange}
        label="Loss over training"
      />
      <div className="loss-header error-header">
        <h3>Error rate (1 − accuracy)</h3>
      </div>
      <LossCurve
        losses={snapshot.errors}
        logScale={logScale}
        xName={xName}
        xRange={xRange}
        formatValue={percent}
        formatTick={percentTick}
        label="Error rate over training"
      />
    </section>
  );
}

function ResultsPanels({
  snapshot,
  test,
  onTry,
}: {
  snapshot: MnistSnapshot;
  test: MnistSet;
  onTry: (i: number) => void;
}) {
  const filter = useMnistStore((s) => s.galleryFilter);
  const setFilter = useMnistStore.getState().setGalleryFilter;
  const [rowShares, setRowShares] = useState(false);
  const counts = useMemo(
    () => confusionCounts(test.labels, snapshot.evaluation.predicted),
    [test, snapshot.evaluation],
  );
  const perClass = useMemo(() => perClassAccuracy(counts), [counts]);
  const worst = perClass.indexOf(Math.min(...perClass));
  return (
    <>
      <section className="panel confusion-panel">
        <div className="panel-toolbar">
          <h2>Confusion matrix</h2>
          <label className="checkbox">
            <input
              type="checkbox"
              checked={rowShares}
              onChange={(e) => setRowShares(e.target.checked)}
            />
            % of each row
          </label>
        </div>
        <ConfusionMatrix
          counts={counts}
          rowShares={rowShares}
          selected={filter}
          onSelect={setFilter}
        />
        <p className="hint">
          Rows are the true digit, columns the network’s answer; the diagonal is correct. Colour
          marks the mistakes, darker for more. Hardest digit now: {worst} ({pct(perClass[worst]!)}{' '}
          right). Click a mistake to see those images.
        </p>
      </section>
      <section className="panel gallery-panel">
        <div className="panel-toolbar">
          <h2>
            {filter ? `${filter.label}s taken for ${filter.predicted}s` : 'Most confidently wrong'}
          </h2>
          {filter && (
            <button type="button" onClick={() => setFilter(null)}>
              Show all mistakes
            </button>
          )}
        </div>
        <WrongGallery test={test} evaluation={snapshot.evaluation} filter={filter} onTry={onTry} />
      </section>
    </>
  );
}

/** The MNIST tab: train an MLP on 10,000 handwritten digits and look inside it. */
export function MnistPage() {
  const worker = useMnistWorker();
  const data = useMnistStore((s) => s.data);
  const snapshot = useMnistStore((s) => s.snapshot);
  const drawn = useMnistStore((s) => s.drawn);
  const padRef = useRef<DigitPadHandle | null>(null);
  const subset = data.status === 'ready' ? data.subset : null;

  const tryImage = (i: number) => {
    if (!subset) return;
    const image = subset.test.images.subarray(i * MNIST_PIXELS, (i + 1) * MNIST_PIXELS);
    padRef.current?.show(image);
    useMnistStore.getState().setDrawn(Float32Array.from(image, (v) => v / 255));
  };

  return (
    <div className="mnist-page">
      <MnistControls onStep={worker.step} requestCheckpoint={worker.requestCheckpoint} />
      {data.status === 'loading' && <p className="panel hint">Loading the MNIST images (2 MB)…</p>}
      {data.status === 'error' && (
        <div className="notice error" role="alert">
          <span>Couldn’t load the MNIST images: {data.message}.</span>
          <button
            type="button"
            onClick={() => useMnistStore.getState().setData({ status: 'idle' })}
          >
            Try again
          </button>
        </div>
      )}
      <main className="mnist-grid">
        <MnistArchitecture />
        {snapshot?.filters ? (
          <section className="panel weights-panel">
            <h2>What the filters look for</h2>
            <FilterViewer filters={snapshot.filters} rows={snapshot.network.conv} />
            <p className="hint">
              Each tile is one kernel: the weights a filter slides across its input. They start as
              random noise; training turns the first layer’s into small stroke and edge detectors.
              Click a tile to see its numbers.
            </p>
          </section>
        ) : snapshot?.firstLayer ? (
          <WeightsPanel layer={snapshot.firstLayer} hidden={snapshot.prediction?.hidden} />
        ) : (
          <section className="panel" />
        )}
        <TryPanel
          padRef={padRef}
          probs={snapshot?.prediction?.probs ?? null}
          onTestImage={
            subset ? () => tryImage(Math.floor(Math.random() * subset.test.labels.length)) : null
          }
        />
      </main>
      {snapshot?.filters && (
        <section className="panel feature-maps-panel">
          {drawn && snapshot.prediction?.maps ? (
            <FeatureMaps
              rows={snapshot.network.conv}
              maps={snapshot.prediction.maps}
              input={drawn}
            />
          ) : (
            <>
              <h2>Feature maps</h2>
              <p className="hint">
                Draw a digit, or press “Test image”, to see what every layer makes of it.
              </p>
            </>
          )}
        </section>
      )}
      {snapshot && subset && (
        <div className="mnist-results">
          <TrainingPanel snapshot={snapshot} />
          <ResultsPanels snapshot={snapshot} test={subset.test} onTry={tryImage} />
        </div>
      )}
      <p className="credit">
        MNIST by Yann LeCun, Corinna Cortes and Christopher J. C. Burges (CC BY-SA 3.0): 10,000
        training and 2,000 test images, 1,000 and 200 of each digit, are bundled with NeuroViz.
      </p>
    </div>
  );
}
