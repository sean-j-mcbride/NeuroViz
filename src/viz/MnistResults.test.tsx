// @vitest-environment jsdom
import { fireEvent, render, screen, within } from '@testing-library/react';
import { useState } from 'react';
import { describe, expect, it, vi } from 'vitest';
import type { MnistSet } from '../data';
import { type CellRef, ConfusionMatrix } from './ConfusionMatrix';
import { confusionCounts } from './mnistResults';
import { PredictionBars } from './PredictionBars';
import { WrongGallery } from './WrongGallery';

// Six test images: a 0 read right, two 1s (one taken for a 7), three 2s (one taken for a 7, one for a 1).
const test: MnistSet = {
  images: new Uint8Array(6 * 784),
  labels: Uint8Array.of(0, 1, 1, 2, 2, 2),
  source: Uint16Array.of(10, 11, 12, 13, 14, 15),
};
const evaluation = {
  predicted: Uint8Array.of(0, 7, 1, 2, 7, 1),
  confidence: Float32Array.of(0.9, 0.6, 0.8, 0.99, 0.95, 0.61),
};
const counts = confusionCounts(test.labels, evaluation.predicted);

function Harness({ onTry = () => {} }: { onTry?: (i: number) => void }) {
  const [filter, setFilter] = useState<CellRef | null>(null);
  return (
    <>
      <ConfusionMatrix counts={counts} rowShares={false} selected={filter} onSelect={setFilter} />
      <WrongGallery test={test} evaluation={evaluation} filter={filter} onTry={onTry} />
    </>
  );
}

const cell = (r: number, c: number) =>
  screen.getAllByRole('row')[r + 1]!.querySelectorAll('td')[c]!;

describe('confusion matrix and gallery', () => {
  it('writes every count in, true digit by row, prediction by column', () => {
    render(<Harness />);
    expect(cell(0, 0).textContent).toBe('1');
    expect(cell(2, 7).textContent).toBe('1');
    expect(cell(2, 2).textContent).toBe('1');
    expect(cell(3, 3).textContent).toBe('0');
    // Mistakes are buttons; correct cells and empty ones are not.
    expect(within(cell(2, 7)).getByRole('button').title).toMatch(
      /True 2, predicted 7: 1 image \(33\.3 % of the 2s\)/,
    );
    expect(within(cell(0, 0)).queryByRole('button')).toBeNull();
  });

  it('shows row shares as percentages', () => {
    render(<ConfusionMatrix counts={counts} rowShares selected={null} onSelect={() => {}} />);
    expect(cell(1, 7).textContent).toBe('50');
    expect(cell(3, 3).textContent).toBe('–');
  });

  it('lists the mistakes most confident first, then filters to a clicked cell and back', () => {
    render(<Harness />);
    const captions = () =>
      screen.getAllByRole('listitem').map((li) => li.textContent?.replace(/\s+/g, ' '));
    expect(captions()).toEqual(['2 → 7 · 95 %', '2 → 1 · 61 %', '1 → 7 · 60 %']);
    fireEvent.click(within(cell(1, 7)).getByRole('button'));
    expect(captions()).toEqual(['1 → 7 · 60 %']);
    expect(within(cell(1, 7)).getByRole('button').getAttribute('aria-pressed')).toBe('true');
    fireEvent.click(within(cell(1, 7)).getByRole('button'));
    expect(captions()).toHaveLength(3);
  });

  it('names each image, and Try passes its index on', () => {
    const onTry = vi.fn();
    render(<Harness onTry={onTry} />);
    const first = screen.getAllByRole('button', { name: /taken for/ })[0]!;
    expect(first.title).toMatch(/A 2, taken for a 7 \(95 % sure\)\. Official test image 14/);
    fireEvent.click(first);
    expect(onTry).toHaveBeenCalledWith(4);
  });
});

describe('prediction bars', () => {
  it('give every probability in words and mark the prediction', () => {
    const probs = new Float32Array(10);
    probs.set([0.02, 0.9, 0.08], 3);
    render(<PredictionBars probs={probs} />);
    const items = screen.getAllByRole('listitem');
    expect(items).toHaveLength(10);
    expect(items[4]!.getAttribute('aria-label')).toBe('4: 90.0 %, the prediction');
    expect(items[3]!.getAttribute('aria-label')).toBe('3: 2.0 %');
    expect(items[0]!.getAttribute('aria-label')).toBe('0: 0 %');
  });

  it('show no values before anything is drawn', () => {
    render(<PredictionBars probs={null} />);
    expect(screen.getAllByRole('listitem')[0]!.getAttribute('aria-label')).toBe('0');
  });
});
