// @vitest-environment jsdom
import { act, fireEvent, render, screen } from '@testing-library/react';
import { useState } from 'react';
import { describe, expect, it, vi } from 'vitest';
import { smallSession } from '../test/fixtures';
import type { ProbeTrace } from '../worker';
import { StepThroughBar } from './StepThroughBar';

const NAMES = ['Input', 'Hidden 1', 'Output'];
const probe = smallSession().probe({ set: 'train', index: 0 })!;

function Harness({
  trace,
  onClose = () => {},
}: {
  trace: ProbeTrace | null;
  onClose?: () => void;
}) {
  const [stage, setStage] = useState(0);
  return (
    <StepThroughBar
      probe={trace}
      columnNames={NAMES}
      stage={stage}
      onStage={setStage}
      onClose={onClose}
    />
  );
}

describe('StepThroughBar', () => {
  it('asks for a point until one is picked; Close closes', () => {
    const onClose = vi.fn();
    render(<Harness trace={null} onClose={onClose} />);
    expect(screen.getByText(/Click a data point/)).toBeTruthy();
    fireEvent.click(screen.getByText('Close'));
    expect(onClose).toHaveBeenCalledOnce();
  });

  it('steps forward and back through the named stages, stopping at both ends', () => {
    render(<Harness trace={probe} />);
    const prev = screen.getByLabelText('Previous stage') as HTMLButtonElement;
    const next = screen.getByLabelText('Next stage') as HTMLButtonElement;
    expect(screen.getByText('Forward pass · Input')).toBeTruthy();
    expect(prev.disabled).toBe(true);
    fireEvent.click(next);
    expect(screen.getByText('Forward pass · Hidden 1')).toBeTruthy();
    for (let i = 0; i < 10; i++) fireEvent.click(next);
    expect(screen.getByText('Backward pass · Input')).toBeTruthy();
    expect(next.disabled).toBe(true);
    fireEvent.click(prev);
    expect(screen.getByText('Backward pass · Hidden 1')).toBeTruthy();
  });

  it('Animate plays to the end and stops; pressing it again restarts from the beginning', () => {
    vi.useFakeTimers();
    try {
      render(<Harness trace={probe} />);
      fireEvent.click(screen.getByText('▶ Animate'));
      expect(screen.getByText('❚❚ Pause')).toBeTruthy();
      for (let i = 0; i < 6; i++) act(() => vi.advanceTimersByTime(700));
      expect(screen.getByText('Backward pass · Input')).toBeTruthy();
      expect(screen.getByText('▶ Animate')).toBeTruthy();
      fireEvent.click(screen.getByText('▶ Animate'));
      expect(screen.getByText('Forward pass · Input')).toBeTruthy();
      act(() => vi.advanceTimersByTime(700));
      expect(screen.getByText('Forward pass · Hidden 1')).toBeTruthy();
      fireEvent.click(screen.getByText('❚❚ Pause'));
      act(() => vi.advanceTimersByTime(2000));
      expect(screen.getByText('Forward pass · Hidden 1')).toBeTruthy();
    } finally {
      vi.useRealTimers();
    }
  });

  it('shows the point, label, prediction and loss', () => {
    render(<Harness trace={probe} />);
    const text = document.querySelector('.step-through-point')!.textContent!;
    expect(text).toContain(`p ${probe.p.toFixed(3)}`);
    expect(text).toContain(probe.label === 1 ? 'label blue' : 'label orange');
  });
});
