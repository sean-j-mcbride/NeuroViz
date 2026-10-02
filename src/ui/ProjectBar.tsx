import { useRef, useState } from 'react';
import { serialiseModelFile } from '../state/modelFile';
import { PRESETS, type PresetId } from '../state/presets';
import { encodeLink } from '../state/shareUrl';
import { DEFAULT_CONFIG, sameConfig, useAppStore } from '../state/store';
import type { Checkpoint } from '../worker';
import { downloadText, errorText, readModelFile } from './files';

const DEFAULT_OPTION = 'default';

interface ProjectBarProps {
  requestCheckpoint: () => Promise<Checkpoint>;
}

/** Presets, saving and loading models, and share links. */
export function ProjectBar({ requestCheckpoint }: ProjectBarProps) {
  const config = useAppStore((s) => s.config);
  const speed = useAppStore((s) => s.speed);
  const showTestData = useAppStore((s) => s.showTestData);
  const presetId = useAppStore((s) => s.presetId);
  const { loadPreset, setConfig, loadModel, setNotice } = useAppStore.getState();
  const fileInput = useRef<HTMLInputElement>(null);
  const [saving, setSaving] = useState(false);

  // The select names the preset only while the set-up is still that preset (or its fix).
  const preset = PRESETS.find((p) => p.id === presetId);
  const selected =
    preset && (sameConfig(config, preset.config) || sameConfig(config, preset.fix))
      ? preset.id
      : sameConfig(config, DEFAULT_CONFIG)
        ? DEFAULT_OPTION
        : '';

  const choose = (value: string) => {
    if (value === DEFAULT_OPTION) setConfig(DEFAULT_CONFIG);
    else if (value) loadPreset(value as PresetId);
  };

  const save = async () => {
    // The config as of the request: the worker applies any change sent before it.
    const { config } = useAppStore.getState();
    setSaving(true);
    try {
      const checkpoint = await requestCheckpoint();
      const name = `neuroviz-${config.dataset.kind}-epoch-${checkpoint.epoch}.json`;
      downloadText(name, serialiseModelFile(config, checkpoint));
      setNotice({ kind: 'info', text: `Saved ${name}` });
    } catch (e) {
      setNotice({ kind: 'error', text: `Couldn’t save the model: ${errorText(e)}` });
    } finally {
      setSaving(false);
    }
  };

  const load = async (file: File | undefined) => {
    if (!file) return;
    try {
      const model = await readModelFile(file);
      loadModel(model);
      setNotice({
        kind: 'info',
        text: `Loaded ${file.name}: paused at epoch ${model.checkpoint.epoch.toLocaleString('en-GB')}`,
      });
    } catch (e) {
      setNotice({ kind: 'error', text: `Couldn’t load ${file.name}: ${errorText(e)}` });
    }
  };

  const copyLink = async () => {
    const url = `${location.origin}${location.pathname}${location.search}${encodeLink(config, { speed, showTestData })}`;
    try {
      await navigator.clipboard.writeText(url);
      setNotice({ kind: 'info', text: 'Link copied. It sets up this run from scratch.' });
    } catch {
      setNotice({ kind: 'info', text: `Copy this link: ${url}` });
    }
  };

  return (
    <div className="transport project-bar">
      <label className="field">
        <span>Preset</span>
        <select value={selected} onChange={(e) => choose(e.target.value)}>
          <option value="" disabled>
            {selected === '' ? 'Custom' : 'Choose…'}
          </option>
          {PRESETS.map((p) => (
            <option key={p.id} value={p.id} title={p.summary}>
              {p.name}
            </option>
          ))}
          <option value={DEFAULT_OPTION}>Default settings</option>
        </select>
      </label>

      <div className="transport-buttons">
        <button
          type="button"
          onClick={() => void save()}
          disabled={saving}
          title="Download the settings and weights, to resume exactly where training is now"
        >
          Save model
        </button>
        <button
          type="button"
          onClick={() => fileInput.current?.click()}
          title="Resume a saved model from where it was saved"
        >
          Load model…
        </button>
        <input
          ref={fileInput}
          type="file"
          accept=".json,application/json"
          hidden
          aria-label="Model file to load"
          onChange={(e) => {
            void load(e.target.files?.[0]);
            e.target.value = ''; // so choosing the same file again still loads it
          }}
        />
        <button
          type="button"
          onClick={() => void copyLink()}
          title="Copy a link that sets up these settings. Everything is seeded, so training from it repeats this run."
        >
          Copy link
        </button>
      </div>
    </div>
  );
}
