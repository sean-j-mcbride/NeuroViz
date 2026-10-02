// @vitest-environment jsdom
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { smallSession } from '../test/fixtures';
import { formatPrecise } from './colour';
import { NetworkGraph } from './NetworkGraph';

const session = smallSession();
const snapshot = session.snapshot();
const probed = session.snapshot({ probe: { set: 'train', index: 3 } });
const probe = probed.probe!;

const tooltip = () => document.querySelector('.tooltip');

describe('NetworkGraph', () => {
  it('draws one tile per neuron and one edge per weight', () => {
    const { container } = render(<NetworkGraph snapshot={snapshot} />);
    expect(container.querySelectorAll('.neuron')).toHaveLength(2 + 3 + 1);
    expect(container.querySelectorAll('path.edge')).toHaveLength(2 * 3 + 3 * 1);
    expect(container.querySelectorAll('path.edge-hit')).toHaveLength(9);
  });

  it('hovering an edge shows its weight, gradient and history; leaving hides it', () => {
    const { container } = render(<NetworkGraph snapshot={snapshot} />);
    const hit = container.querySelectorAll('path.edge-hit')[1]!; // x1 → h1.2
    fireEvent.mouseMove(hit, { clientX: 100, clientY: 100 });
    const card = tooltip()!;
    expect(card.textContent).toContain('x1 → h1.2');
    expect(card.textContent).toContain(`Weight${formatPrecise(snapshot.weights[0]!.W[1]!)}`);
    expect(card.textContent).toContain(
      `∂L/∂w (all training data)${formatPrecise(snapshot.weights[0]!.gradW[1]!)}`,
    );
    expect(card.querySelector('.sparkline polyline')).not.toBeNull();
    fireEvent.mouseLeave(hit);
    expect(tooltip()).toBeNull();
  });

  it('labels the gradient ∂(L + L2)/∂w when the snapshot was taken with L2 on', () => {
    const { container } = render(<NetworkGraph snapshot={{ ...snapshot, l2: 0.01 }} />);
    fireEvent.mouseMove(container.querySelectorAll('path.edge-hit')[0]!);
    expect(tooltip()!.textContent).toContain('∂(L + L2)/∂w');
  });

  it('hovering a hidden neuron shows its bias and bias gradient (never including L2)', () => {
    const { container } = render(<NetworkGraph snapshot={{ ...snapshot, l2: 0.01 }} />);
    fireEvent.mouseMove(container.querySelectorAll('.neuron')[2]!); // h1.1
    const text = tooltip()!.textContent!;
    expect(text).toContain('h1.1 (tanh)');
    expect(text).toContain(`Bias${formatPrecise(snapshot.weights[0]!.b[0]!)}`);
    expect(text).toContain('∂L/∂b (all training data)');
  });

  describe('step-through', () => {
    const at = (stage: number) =>
      render(<NetworkGraph snapshot={probed} trace={{ probe, stage }} />).container;
    const badges = (c: HTMLElement) =>
      Array.from(c.querySelectorAll('.probe-value'), (b) => b.textContent);

    it('stage 0 lights only the input column and dims the rest', () => {
      const c = at(0);
      expect(badges(c)).toHaveLength(2);
      expect(c.querySelectorAll('.neuron.dim')).toHaveLength(4);
      expect(c.querySelectorAll('path.edge.dim')).toHaveLength(9);
      expect(c.querySelectorAll('.probe-marker')).toHaveLength(6); // the point, on every tile
    });

    it('the forward pass reaches the output with p, then the loss stage shows L', () => {
      expect(badges(at(2)).at(-1)).toBe(`p ${probe.p.toFixed(2)}`);
      expect(badges(at(3)).at(-1)).toMatch(/^L /);
    });

    it('the last backward stage shows a gradient on every neuron and no dimmed edges', () => {
      const c = at(5);
      expect(badges(c)).toHaveLength(6);
      expect(c.querySelectorAll('.neuron.dim')).toHaveLength(0);
      expect(c.querySelectorAll('path.edge.dim')).toHaveLength(0);
    });

    it('hover cards add the probe values', () => {
      const c = at(5);
      fireEvent.mouseMove(c.querySelectorAll('.neuron')[5]!); // output
      const text = tooltip()!.textContent!;
      expect(text).toContain(`p = σ(z)${formatPrecise(probe.p)}`);
      expect(text).toContain('∂L/∂z = p − y');
    });
  });

  it('renders custom column headers', () => {
    render(<NetworkGraph snapshot={snapshot} renderHeader={(c) => <span>header {c}</span>} />);
    expect(screen.getByText('header 1')).toBeTruthy();
  });
});
