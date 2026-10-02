// @vitest-environment jsdom
import { fireEvent, render } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { smallSession } from '../test/fixtures';
import { DecisionBoundary } from './DecisionBoundary';

const session = smallSession(0);
const snapshot = session.snapshot();
const data = session.data();
const domain = snapshot.domain;

/** Client pixel of a data point, with the overlay laid out at 320 × 320 from (0, 0). */
function pixelOf(set: 'train' | 'test', i: number) {
  const x = data[set].x;
  const toPx = (v: number) => ((v + domain) / (2 * domain)) * 320;
  return { clientX: toPx(x[2 * i]!), clientY: 320 - toPx(x[2 * i + 1]!) };
}

function renderPickable(showTestData: boolean) {
  const onPick = vi.fn();
  const { container } = render(
    <DecisionBoundary
      snapshot={snapshot}
      data={data}
      showTestData={showTestData}
      onPick={onPick}
    />,
  );
  const overlay = container.querySelector('canvas.points') as HTMLCanvasElement;
  overlay.getBoundingClientRect = () => ({ left: 0, top: 0, width: 320, height: 320 }) as DOMRect;
  return { overlay, onPick };
}

describe('DecisionBoundary picking', () => {
  it('picks the nearest training point within reach', () => {
    const { overlay, onPick } = renderPickable(false);
    const p = pixelOf('train', 7);
    fireEvent.click(overlay, { clientX: p.clientX + 2, clientY: p.clientY - 1 });
    expect(onPick).toHaveBeenCalledWith({ set: 'train', index: 7 });
  });

  it('ignores clicks far from every point', () => {
    const { overlay, onPick } = renderPickable(false);
    fireEvent.click(overlay, { clientX: 1, clientY: 1 }); // corner: no data near ±6
    expect(onPick).not.toHaveBeenCalled();
  });

  it('test points are pickable only while shown', () => {
    const p = pixelOf('test', 2);
    const hidden = renderPickable(false);
    fireEvent.click(hidden.overlay, p);
    expect(hidden.onPick).not.toHaveBeenCalledWith({ set: 'test', index: 2 });
    const shown = renderPickable(true);
    fireEvent.click(shown.overlay, p);
    expect(shown.onPick).toHaveBeenCalledWith({ set: 'test', index: 2 });
  });

  it('without onPick, clicks do nothing and the plot is not marked pickable', () => {
    const { container } = render(
      <DecisionBoundary snapshot={snapshot} data={data} showTestData={false} />,
    );
    expect(container.querySelector('.decision-boundary.picking')).toBeNull();
  });
});
