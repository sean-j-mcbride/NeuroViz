// @vitest-environment jsdom
import { render, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { smallSession } from '../test/fixtures';
import { formatPrecise } from './colour';
import { InsideTables } from './InsideTables';

describe('InsideTables', () => {
  const { timeline } = smallSession(10).snapshot();
  const T = timeline.epochs.length;

  it('summarises each layer at the latest epoch', () => {
    const { container } = render(<InsideTables timeline={timeline} />);
    const summary = container.querySelectorAll('table')[0]!;
    const rows = summary.querySelectorAll('tbody tr');
    expect(rows).toHaveLength(2);
    const cells = within(rows[1] as HTMLElement).getAllByRole('cell');
    expect(cells[0]!.textContent).toBe(formatPrecise(timeline.layers[1]!.weightRms[T - 1]!));
    expect(cells[2]!.textContent).toBe(formatPrecise(timeline.layers[1]!.gradRms[T - 1]!));
  });

  it('lists gradient RMS per recorded epoch, newest first, with the first ÷ last ratio', () => {
    const { container } = render(<InsideTables timeline={timeline} />);
    const rows = container.querySelectorAll('table')[1]!.querySelectorAll('tbody tr');
    expect(rows).toHaveLength(T);
    const newest = rows[0] as HTMLElement;
    expect(within(newest).getByRole('rowheader').textContent).toBe(String(timeline.epochs[T - 1]));
    const [l0, l1] = timeline.layers;
    const cells = within(newest).getAllByRole('cell');
    expect(cells.at(-1)!.textContent).toBe(
      formatPrecise(l0!.gradRms[T - 1]! / l1!.gradRms[T - 1]!),
    );
  });
});
