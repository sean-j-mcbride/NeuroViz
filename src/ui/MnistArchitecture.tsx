import { ACTIVATION_NAMES } from '../state/config';
import {
  MNIST_MAX_HIDDEN_LAYERS,
  MNIST_NEW_LAYER,
  MNIST_UNITS,
  describeMnistNetwork,
  mnistParamCount,
  useMnistStore,
} from '../state/mnistStore';
import { ACTIVATIONS, type Activation, type HiddenLayerSpec } from '../worker';

/** The MNIST network's hidden layers: add, remove, resize, change activation. */
export function MnistArchitecture() {
  const network = useMnistStore((s) => s.config.network);
  const setNetwork = useMnistStore.getState().setNetwork;
  const { hidden } = network;
  const update = (next: HiddenLayerSpec[]) => setNetwork({ hidden: next });
  const patch = (i: number, p: Partial<HiddenLayerSpec>) =>
    update(hidden.map((l, k) => (k === i ? { ...l, ...p } : l)));

  return (
    <section className="panel mnist-architecture">
      <h2>Network</h2>
      <ol className="layer-list">
        <li className="layer-fixed">784 inputs (28 × 28 pixels)</li>
        {hidden.map((l, i) => (
          <li key={i} className="layer-row">
            <select
              value={l.units}
              onChange={(e) => patch(i, { units: Number(e.target.value) })}
              aria-label={`Units in hidden layer ${i + 1}`}
            >
              {MNIST_UNITS.map((u) => (
                <option key={u} value={u}>
                  {u}
                </option>
              ))}
            </select>
            <select
              value={l.activation}
              onChange={(e) => patch(i, { activation: e.target.value as Activation })}
              aria-label={`Activation of hidden layer ${i + 1}`}
            >
              {ACTIVATIONS.map((a) => (
                <option key={a} value={a}>
                  {ACTIVATION_NAMES[a]}
                </option>
              ))}
            </select>
            <button
              type="button"
              onClick={() => update(hidden.filter((_, k) => k !== i))}
              aria-label={`Remove hidden layer ${i + 1}`}
              title="Remove this layer"
            >
              ✕
            </button>
          </li>
        ))}
        <li className="layer-fixed">10 outputs (softmax over the digits)</li>
      </ol>
      <button
        type="button"
        onClick={() => update([...hidden, MNIST_NEW_LAYER])}
        disabled={hidden.length >= MNIST_MAX_HIDDEN_LAYERS}
      >
        + Add a hidden layer
      </button>
      <p className="hint">
        {describeMnistNetwork(network)}: {mnistParamCount(network).toLocaleString('en-GB')} weights
        and biases. Changing the network starts a new run; the settings above apply live.
      </p>
    </section>
  );
}
