// @vitest-environment jsdom
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { InsideTables, type TimelineAxis } from '../viz';
import { HistogramTimeline as Recorder } from '../worker';
import { InsideTrainingPanel } from './InsidePanel';

/** A two-layer timeline with three columns at record indices 0–2. */
function timeline() {
  const t = new Recorder(2);
  for (let k = 0; k < 3; k++) {
    t.record(k, [
      { W: Float32Array.from([0.1, -0.2]), gradW: Float32Array.from([1e-3 * (k + 1), 0]) },
      { W: Float32Array.from([0.5]), gradW: Float32Array.from([1e-2]) },
    ]);
  }
  return t.snapshot();
}

const axis: TimelineAxis = {
  layerName: (k) => ['Conv 1', 'Output'][k]!,
  at: (x) => `epoch ${x / 2}`,
  column: 'Epoch',
  cell: (x) => String(x / 2),
};

describe('Inside training panel with an MNIST axis', () => {
  it('names layers and labels columns as epochs, in charts and table', () => {
    const tl = timeline();
    render(
      <InsideTrainingPanel
        timeline={tl}
        axis={axis}
        normsHint="hint"
        table={
          <InsideTables
            timeline={tl}
            axis={axis}
            deadOf={(k) => (k === 0 ? '1 of 8 channels' : '–')}
          />
        }
      />,
    );
    expect(screen.getAllByText('Conv 1').length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole('button', { name: 'Table' }));
    expect(screen.getByRole('heading', { name: 'Per layer at epoch 1' })).toBeTruthy();
    expect(screen.getByText('1 of 8 channels')).toBeTruthy();
    // Gradient RMS over time: newest first, x as epochs.
    const rowHeads = screen.getAllByRole('rowheader').map((h) => h.textContent);
    expect(rowHeads).toEqual(['Conv 1', 'Output', '1', '0.5', '0']);
  });
});
