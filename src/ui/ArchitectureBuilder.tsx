import { MAX_HIDDEN_LAYERS, MAX_UNITS, useAppStore } from '../state/store';
import { NetworkGraph } from '../viz';
import { ACTIVATIONS, type Activation, type HiddenLayerSpec } from '../worker';

const ACTIVATION_NAMES: Record<Activation, string> = {
  tanh: 'Tanh',
  relu: 'ReLU',
  sigmoid: 'Sigmoid',
  linear: 'Linear',
};

/** The network graph, with controls to edit the hidden layers above it. */
export function ArchitectureBuilder() {
  const snapshot = useAppStore((s) => s.snapshot);
  const hidden = useAppStore((s) => s.config.network.hidden);
  const setNetwork = useAppStore.getState().setNetwork;

  const update = (next: HiddenLayerSpec[]) => setNetwork({ hidden: next });
  const patchLayer = (i: number, patch: Partial<HiddenLayerSpec>) =>
    update(hidden.map((l, k) => (k === i ? { ...l, ...patch } : l)));
  const addLayer = () =>
    update([...hidden, { units: 4, activation: hidden.at(-1)?.activation ?? 'tanh' }]);

  return (
    <section className="panel network-panel">
      <div className="network-toolbar">
        <h2>Network</h2>
        <div className="stepper" aria-label="Hidden layers">
          <button
            type="button"
            onClick={() => update(hidden.slice(0, -1))}
            disabled={hidden.length === 0}
            aria-label="Remove a hidden layer"
          >
            −
          </button>
          <span>
            {hidden.length} hidden layer{hidden.length === 1 ? '' : 's'}
          </span>
          <button
            type="button"
            onClick={addLayer}
            disabled={hidden.length >= MAX_HIDDEN_LAYERS}
            aria-label="Add a hidden layer"
          >
            +
          </button>
        </div>
      </div>

      {snapshot && snapshot.columns.length === hidden.length + 2 && (
        <NetworkGraph
          snapshot={snapshot}
          renderHeader={(column, kind) => {
            if (kind === 'input' || kind === 'output') return null;
            const i = column - 1;
            const layer = hidden[i]!;
            return (
              <div className="layer-controls">
                <div className="stepper small">
                  <button
                    type="button"
                    onClick={() => patchLayer(i, { units: layer.units - 1 })}
                    disabled={layer.units <= 1}
                    aria-label={`Remove a neuron from hidden layer ${column}`}
                  >
                    −
                  </button>
                  <span>{layer.units}</span>
                  <button
                    type="button"
                    onClick={() => patchLayer(i, { units: layer.units + 1 })}
                    disabled={layer.units >= MAX_UNITS}
                    aria-label={`Add a neuron to hidden layer ${column}`}
                  >
                    +
                  </button>
                </div>
                <select
                  value={layer.activation}
                  onChange={(e) => patchLayer(i, { activation: e.target.value as Activation })}
                  aria-label={`Activation of hidden layer ${column}`}
                >
                  {ACTIVATIONS.map((a) => (
                    <option key={a} value={a}>
                      {ACTIVATION_NAMES[a]}
                    </option>
                  ))}
                </select>
              </div>
            );
          }}
        />
      )}

      <p className="hint">
        Each neuron shows what it computes across the input plane. Edge thickness is the size of a
        weight; colour is its sign (blue positive, orange negative). Hover for values.
      </p>
    </section>
  );
}
