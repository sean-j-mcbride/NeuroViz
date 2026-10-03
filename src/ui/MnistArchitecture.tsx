import type { Shape } from '../engine';
import { ACTIVATION_NAMES } from '../state/config';
import {
  CONV_FILTERS,
  CONV_KERNELS,
  CONV_PADDINGS,
  CONV_STRIDES,
  MNIST_MAX_CONV_LAYERS,
  MNIST_MAX_HIDDEN_LAYERS,
  MNIST_NEW_CONV,
  MNIST_NEW_LAYER,
  MNIST_NEW_POOL,
  MNIST_TEMPLATES,
  MNIST_UNITS,
  describeMnistNetwork,
  mnistNetworkCost,
  useMnistStore,
} from '../state/mnistStore';
import {
  ACTIVATIONS,
  type Activation,
  type ConvLayerSpec,
  type HiddenLayerSpec,
  type MnistNetworkSpec,
  checkConvStack,
} from '../worker';

const times = (s: Shape) => s.join(' × ');

const PADDING_NAMES = { valid: 'valid', same: 'same' } as const;
const PADDING_TITLE =
  'Valid: no padding, so each 5×5 kernel shrinks the map by 4. Same: zeros around the edge keep the size (at stride 1).';

const TEMPLATE_NAMES = { mlp: 'MLP', cnn: 'Small CNN' } as const;

function ConvRow({
  row,
  index,
  shape,
  error,
  onChange,
  onRemove,
}: {
  row: ConvLayerSpec;
  index: number;
  /** Output shape, or null when an earlier row already doesn't fit. */
  shape: Shape | null;
  error: string | null;
  onChange(row: ConvLayerSpec): void;
  onRemove(): void;
}) {
  const n = index + 1;
  const out = error ? (
    <span className="layer-error">{error}</span>
  ) : (
    <span className="layer-shape">{shape ? `→ ${times(shape)}` : '—'}</span>
  );
  return (
    <li className={`layer-block${error ? ' invalid' : ''}`}>
      <div className="layer-row">
        {row.kind === 'pool' ? (
          <span className="layer-kind">
            Max pool {row.size} × {row.size}
          </span>
        ) : (
          <>
            <span className="layer-kind">Conv</span>
            <select
              value={row.filters}
              onChange={(e) => onChange({ ...row, filters: Number(e.target.value) })}
              aria-label={`Filters in conv layer ${n}`}
            >
              {CONV_FILTERS.map((f) => (
                <option key={f} value={f}>
                  {f} filters
                </option>
              ))}
            </select>
            <select
              value={row.kernel}
              onChange={(e) => onChange({ ...row, kernel: Number(e.target.value) })}
              aria-label={`Kernel size of conv layer ${n}`}
            >
              {CONV_KERNELS.map((k) => (
                <option key={k} value={k}>
                  {k}×{k}
                </option>
              ))}
            </select>
          </>
        )}
        <button
          type="button"
          onClick={onRemove}
          aria-label={`Remove conv layer ${n}`}
          title="Remove this layer"
        >
          ✕
        </button>
      </div>
      <div className="layer-row layer-detail">
        {row.kind === 'conv' && (
          <>
            <select
              value={row.activation}
              onChange={(e) => onChange({ ...row, activation: e.target.value as Activation })}
              aria-label={`Activation of conv layer ${n}`}
            >
              {ACTIVATIONS.map((a) => (
                <option key={a} value={a}>
                  {ACTIVATION_NAMES[a]}
                </option>
              ))}
            </select>
            <select
              value={row.stride}
              onChange={(e) => onChange({ ...row, stride: Number(e.target.value) })}
              aria-label={`Stride of conv layer ${n}`}
            >
              {CONV_STRIDES.map((s) => (
                <option key={s} value={s}>
                  stride {s}
                </option>
              ))}
            </select>
            <select
              value={row.padding}
              onChange={(e) =>
                onChange({ ...row, padding: e.target.value as (typeof CONV_PADDINGS)[number] })
              }
              aria-label={`Padding of conv layer ${n}`}
              title={PADDING_TITLE}
            >
              {CONV_PADDINGS.map((p) => (
                <option key={p} value={p}>
                  {PADDING_NAMES[p]}
                </option>
              ))}
            </select>
          </>
        )}
      </div>
      <div className="layer-out">{out}</div>
    </li>
  );
}

const sameNetwork = (a: MnistNetworkSpec, b: MnistNetworkSpec) =>
  JSON.stringify(a) === JSON.stringify(b);

/**
 * The MNIST network: optional conv/pool rows (each showing the shape it
 * outputs, or why it doesn't fit), an automatic flatten, then dense layers.
 */
export function MnistArchitecture() {
  const network = useMnistStore((s) => s.config.network);
  const setNetwork = useMnistStore.getState().setNetwork;
  const { conv, hidden } = network;
  const check = checkConvStack(conv);
  const cost = mnistNetworkCost(network);
  const updateHidden = (next: HiddenLayerSpec[]) => setNetwork({ ...network, hidden: next });
  const patchHidden = (i: number, p: Partial<HiddenLayerSpec>) =>
    updateHidden(hidden.map((l, k) => (k === i ? { ...l, ...p } : l)));
  const updateConv = (next: ConvLayerSpec[]) => setNetwork({ ...network, conv: next });
  const cnn = conv.length > 0;

  return (
    <section className="panel mnist-architecture">
      <div className="panel-toolbar">
        <h2>Network</h2>
        <div className="segmented" role="group" aria-label="Start from">
          {(['mlp', 'cnn'] as const).map((t) => (
            <button
              key={t}
              type="button"
              aria-pressed={sameNetwork(network, MNIST_TEMPLATES[t].network)}
              title={`${describeMnistNetwork(MNIST_TEMPLATES[t].network)}, learning rate ${MNIST_TEMPLATES[t].training.lr}, batch size ${MNIST_TEMPLATES[t].training.batchSize}`}
              onClick={() => {
                setNetwork(MNIST_TEMPLATES[t].network);
                useMnistStore.getState().setTraining(MNIST_TEMPLATES[t].training);
              }}
            >
              {TEMPLATE_NAMES[t]}
            </button>
          ))}
        </div>
      </div>
      <ol className="layer-list" aria-label="Image layers">
        <li className="layer-fixed">
          {cnn ? 'Input image → 1 × 28 × 28' : '784 inputs (28 × 28 pixels)'}
        </li>
        {conv.map((row, i) => (
          <ConvRow
            key={i}
            row={row}
            index={i}
            shape={check.shapes[i] ?? null}
            error={!check.ok && check.row === i ? check.message : null}
            onChange={(r) => updateConv(conv.map((x, k) => (k === i ? r : x)))}
            onRemove={() => updateConv(conv.filter((_, k) => k !== i))}
          />
        ))}
      </ol>
      <div className="layer-add">
        <button
          type="button"
          onClick={() => updateConv([...conv, MNIST_NEW_CONV])}
          disabled={conv.length >= MNIST_MAX_CONV_LAYERS}
        >
          + Conv
        </button>
        <button
          type="button"
          onClick={() => updateConv([...conv, MNIST_NEW_POOL])}
          disabled={conv.length >= MNIST_MAX_CONV_LAYERS}
        >
          + Pool
        </button>
      </div>
      <ol className="layer-list" aria-label="Dense layers">
        {cnn && (
          <li className="layer-fixed">
            {check.ok ? `Flatten → ${check.flat.toLocaleString('en-GB')} values` : 'Flatten'}
          </li>
        )}
        {hidden.map((l, i) => (
          <li key={i} className="layer-row">
            <span className="layer-kind">Dense</span>
            <select
              value={l.units}
              onChange={(e) => patchHidden(i, { units: Number(e.target.value) })}
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
              onChange={(e) => patchHidden(i, { activation: e.target.value as Activation })}
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
              onClick={() => updateHidden(hidden.filter((_, k) => k !== i))}
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
        onClick={() => updateHidden([...hidden, MNIST_NEW_LAYER])}
        disabled={hidden.length >= MNIST_MAX_HIDDEN_LAYERS}
      >
        + Add a hidden layer
      </button>
      {check.ok && cost ? (
        <p className="hint">
          {describeMnistNetwork(network)}: {cost.params.toLocaleString('en-GB')} weights and biases,{' '}
          {(cost.multiplyAdds / 1e6).toFixed(2)} M multiply-adds per image (the default MLP: 0.10
          M), a rough guide to speed. Changing the network starts a new run; the settings above
          apply live.{cnn && ' Dropout applies to the dense layers only.'}
        </p>
      ) : (
        <p className="hint" role="alert">
          This network can’t take a 28 × 28 image: a layer’s input has become smaller than its
          window. Remove a pool or conv layer, or use same padding. Until then the last network that
          worked stays on screen.
        </p>
      )}
    </section>
  );
}
