import { useRef, useState } from 'react';
import { parseMnistModelFile, serialiseMnistModelFile } from '../state/mnistModelFile';
import { useMnistStore } from '../state/mnistStore';
import { useAppStore } from '../state/store';
import type { MnistCheckpoint } from '../worker';
import { downloadText, errorText } from './files';

const epochs = (c: MnistCheckpoint, trainSize: number) =>
  (c.examples / trainSize).toLocaleString('en-GB', { maximumFractionDigits: 2 });

/** Save and load MNIST runs. */
export function MnistProjectBar({
  requestCheckpoint,
}: {
  requestCheckpoint: () => Promise<MnistCheckpoint>;
}) {
  const data = useMnistStore((s) => s.data);
  const setNotice = useAppStore.getState().setNotice;
  const fileInput = useRef<HTMLInputElement>(null);
  const [saving, setSaving] = useState(false);
  const subset = data.status === 'ready' ? data.subset : null;

  const save = async () => {
    if (!subset) return;
    const { config } = useMnistStore.getState();
    setSaving(true);
    try {
      const checkpoint = await requestCheckpoint();
      const at = epochs(checkpoint, subset.train.labels.length);
      const name = `neuroviz-mnist-epoch-${at}.json`;
      downloadText(name, serialiseMnistModelFile(config, subset, checkpoint));
      setNotice({ kind: 'info', text: `Saved ${name}` });
    } catch (e) {
      setNotice({ kind: 'error', text: `Couldn’t save the model: ${errorText(e)}` });
    } finally {
      setSaving(false);
    }
  };

  const load = async (file: File | undefined) => {
    if (!file || !subset) return;
    try {
      const model = parseMnistModelFile(await file.text(), subset);
      useMnistStore.getState().loadModel(model);
      setNotice({
        kind: 'info',
        text: `Loaded ${file.name}: paused at epoch ${epochs(model.checkpoint, subset.train.labels.length)}`,
      });
    } catch (e) {
      setNotice({ kind: 'error', text: `Couldn’t load ${file.name}: ${errorText(e)}` });
    }
  };

  return (
    <div className="transport project-bar">
      <div className="transport-buttons">
        <button
          type="button"
          onClick={() => void save()}
          disabled={saving || !subset}
          title="Download the settings and weights, to resume exactly where training is now"
        >
          Save model
        </button>
        <button
          type="button"
          onClick={() => fileInput.current?.click()}
          disabled={!subset}
          title="Resume a saved MNIST model from where it was saved"
        >
          Load model…
        </button>
        <input
          ref={fileInput}
          type="file"
          accept=".json,application/json"
          hidden
          aria-label="MNIST model file to load"
          onChange={(e) => {
            void load(e.target.files?.[0]);
            e.target.value = '';
          }}
        />
      </div>
    </div>
  );
}
