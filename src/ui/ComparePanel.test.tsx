// @vitest-environment jsdom
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';
import { serialiseModelFile } from '../state/modelFile';
import { DEFAULT_CONFIG, toSessionConfig, useAppStore } from '../state/store';
import { TrainingSession } from '../worker';
import { ComparePanel } from './ComparePanel';

const initial = useAppStore.getState();
const s = () => useAppStore.getState();

function trained(epochs: number, config = DEFAULT_CONFIG) {
  const t = new TrainingSession({ ...toSessionConfig(config), gridSize: 4 });
  for (let e = 0; e < epochs; e++) t.trainEpoch();
  return t;
}

/** Sets the live run's snapshot and data, as the worker would. */
function show(t: TrainingSession) {
  s().setSnapshot(t.snapshot());
  s().setSessionData(t.data());
}

beforeEach(() => useAppStore.setState(initial, true));

describe('ComparePanel', () => {
  it('explains itself until there is a reference; pinning needs a snapshot', () => {
    render(<ComparePanel />);
    expect(screen.getByText(/Pin the current run/)).toBeTruthy();
    expect(
      (screen.getByRole('button', { name: 'Pin current run' }) as HTMLButtonElement).disabled,
    ).toBe(true);
  });

  it('shows both runs’ scores and the settings changed since the reference', () => {
    show(trained(3));
    s().pinReference();
    s().setTraining({ lr: 0.1, optimiser: 'adam' });
    show(trained(5, s().config));
    render(<ComparePanel />);

    const rows = screen.getAllByRole('row');
    const epochRow = rows.find((r) => within(r).queryByRole('rowheader')?.textContent === 'Epoch')!;
    expect(
      within(epochRow)
        .getAllByRole('cell')
        .map((c) => c.textContent),
    ).toEqual(['3', '5']);
    expect(screen.getAllByRole('listitem').map((li) => li.textContent)).toEqual([
      'Learning rate 0.03 → 0.1',
      'Optimiser SGD → Adam',
    ]);
    expect(screen.queryByText('Changed during the run')).toBeNull();
    expect(screen.getByText(/Reference: Pinned run/)).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: 'Clear' }));
    expect(s().reference).toBeNull();
  });

  it('loads a saved model as the reference', async () => {
    show(trained(1));
    render(<ComparePanel />);
    const text = serialiseModelFile(DEFAULT_CONFIG, trained(7).checkpoint());
    fireEvent.change(screen.getByLabelText('Model file to compare against'), {
      target: { files: [new File([text], 'baseline.json')] },
    });
    await waitFor(() => expect(s().reference?.label).toBe('baseline.json'));
    expect(s().reference?.epoch).toBe(7);
    expect(s().reference?.losses.count).toBe(8);
  });

  it('lists each run’s mid-run setting changes', () => {
    const t = trained(2);
    t.setHyperparams({ ...DEFAULT_CONFIG.training, lr: 3 });
    t.trainEpoch();
    show(t);
    s().pinReference();
    show(trained(4));
    render(<ComparePanel />);
    expect(screen.getByText('Changed during the run')).toBeTruthy();
    const list = screen.getByRole('list', { name: 'Reference: changes during the run' });
    expect(list.textContent).toBe('Reference, epoch 2: Learning rate 0.03 → 3');
    expect(screen.getByText('This run: none.')).toBeTruthy();
  });
});
