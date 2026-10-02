// @vitest-environment jsdom
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { parseModelFile, serialiseModelFile } from '../state/modelFile';
import { PRESETS } from '../state/presets';
import { DEFAULT_CONFIG, toSessionConfig, useAppStore } from '../state/store';
import { TrainingSession } from '../worker';
import { ProjectBar } from './ProjectBar';

const initial = useAppStore.getState();
const s = () => useAppStore.getState();

function trained(epochs = 3) {
  const t = new TrainingSession({ ...toSessionConfig(DEFAULT_CONFIG), gridSize: 2 });
  for (let e = 0; e < epochs; e++) t.trainEpoch();
  return t;
}

beforeEach(() => useAppStore.setState(initial, true));
afterEach(() => vi.restoreAllMocks());

const presetSelect = () => screen.getByRole('combobox', { name: 'Preset' }) as HTMLSelectElement;
const fileInput = () => screen.getByLabelText('Model file to load') as HTMLInputElement;

describe('ProjectBar', () => {
  it('loads a preset and names it until the settings change', () => {
    render(<ProjectBar requestCheckpoint={vi.fn()} />);
    expect(presetSelect().value).toBe('default');
    fireEvent.change(presetSelect(), { target: { value: 'overfitting' } });
    expect(s().config).toEqual(PRESETS.find((p) => p.id === 'overfitting')!.config);
    expect(presetSelect().value).toBe('overfitting');
    act(() => s().setTraining({ dropout: 0.3 }));
    expect(presetSelect().value).toBe('');
    expect(screen.getByRole('option', { name: 'Custom' })).toBeTruthy();
    fireEvent.change(presetSelect(), { target: { value: 'default' } });
    expect(s().config).toEqual(DEFAULT_CONFIG);
  });

  it('saves the current settings with the worker’s checkpoint as a downloadable file', async () => {
    let blob: Blob | undefined;
    URL.createObjectURL = vi.fn((b: Blob) => {
      blob = b;
      return 'blob:model';
    });
    URL.revokeObjectURL = vi.fn();
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
    const checkpoint = trained().checkpoint();
    render(<ProjectBar requestCheckpoint={() => Promise.resolve(checkpoint)} />);

    fireEvent.click(screen.getByRole('button', { name: 'Save model' }));
    await waitFor(() => expect(click).toHaveBeenCalled());
    const a = click.mock.contexts[0] as HTMLAnchorElement;
    expect(a.download).toBe('neuroviz-spirals-epoch-3.json');
    expect(parseModelFile(await blob!.text())).toMatchObject({
      config: DEFAULT_CONFIG,
      checkpoint,
    });
    expect(s().notice).toEqual({ kind: 'info', text: 'Saved neuroviz-spirals-epoch-3.json' });
  });

  it('reports a failed save', async () => {
    render(<ProjectBar requestCheckpoint={() => Promise.reject(new Error('worker gone'))} />);
    fireEvent.click(screen.getByRole('button', { name: 'Save model' }));
    await waitFor(() =>
      expect(s().notice).toEqual({
        kind: 'error',
        text: 'Couldn’t save the model: worker gone',
      }),
    );
  });

  it('loads a saved model to resume, and explains a bad file', async () => {
    render(<ProjectBar requestCheckpoint={vi.fn()} />);
    const checkpoint = trained(4).checkpoint();
    const text = serialiseModelFile(DEFAULT_CONFIG, checkpoint);
    fireEvent.change(fileInput(), { target: { files: [new File([text], 'run.json')] } });
    await waitFor(() => expect(s().resume).toEqual(checkpoint));
    expect(s().notice?.text).toBe('Loaded run.json: paused at epoch 4');

    fireEvent.change(fileInput(), { target: { files: [new File(['{}'], 'notes.json')] } });
    await waitFor(() => expect(s().notice?.kind).toBe('error'));
    expect(s().notice?.text).toBe('Couldn’t load notes.json: This is not a saved NeuroViz model.');
    expect(s().resume).toEqual(checkpoint); // unchanged
  });

  it('copies a link carrying the settings', async () => {
    const writeText = vi.fn((_text: string) => Promise.resolve());
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
    s().setTraining({ lr: 1 });
    render(<ProjectBar requestCheckpoint={vi.fn()} />);
    fireEvent.click(screen.getByRole('button', { name: 'Copy link' }));
    await waitFor(() => expect(writeText).toHaveBeenCalled());
    expect(writeText.mock.calls[0]![0]).toMatch(/#data=spirals&.*&lr=1&/);
    expect(s().notice?.text).toMatch(/^Link copied/);
  });
});
