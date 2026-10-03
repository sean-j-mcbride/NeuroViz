// @vitest-environment jsdom
import { fireEvent, render, screen, within } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';
import { MNIST_DEFAULT_CONFIG, MNIST_TEMPLATES, useMnistStore } from '../state/mnistStore';
import { MnistArchitecture } from './MnistArchitecture';
import { TransportControls } from './TransportBar';

const initial = useMnistStore.getState();
const s = () => useMnistStore.getState();

beforeEach(() => useMnistStore.setState(initial, true));

const imageLayers = () => screen.getByRole('list', { name: 'Image layers' });
const button = (name: string) => screen.getByRole('button', { name });

describe('MNIST network builder', () => {
  it('starts as the MLP; "Small CNN" shows each layer’s output shape and the flattened size', () => {
    render(<MnistArchitecture />);
    expect(button('MLP').getAttribute('aria-pressed')).toBe('true');
    expect(screen.getByText('784 inputs (28 × 28 pixels)')).toBeTruthy();
    fireEvent.click(button('Small CNN'));
    expect(s().config.network).toEqual(MNIST_TEMPLATES.cnn);
    expect(button('Small CNN').getAttribute('aria-pressed')).toBe('true');
    const shapes = within(imageLayers())
      .getAllByText(/^→ /)
      .map((e) => e.textContent);
    expect(shapes).toEqual(['→ 8 × 24 × 24', '→ 8 × 12 × 12', '→ 16 × 8 × 8', '→ 16 × 4 × 4']);
    expect(screen.getByText('Flatten → 256 values')).toBeTruthy();
    expect(screen.getByText(/Dropout applies to the dense layers only/)).toBeTruthy();
  });

  it('updates shapes as rows change, and adds and removes rows', () => {
    render(<MnistArchitecture />);
    fireEvent.click(button('Small CNN'));
    fireEvent.change(screen.getByRole('combobox', { name: 'Padding of conv layer 1' }), {
      target: { value: 'same' },
    });
    expect(within(imageLayers()).getByText('→ 8 × 28 × 28')).toBeTruthy();
    fireEvent.change(screen.getByRole('combobox', { name: 'Filters in conv layer 3' }), {
      target: { value: '32' },
    });
    // 28 → pool 14 → 5×5 valid 10 → pool 5.
    expect(screen.getByText('Flatten → 800 values')).toBeTruthy();
    fireEvent.click(button('Remove conv layer 4'));
    expect(s().config.network.conv).toHaveLength(3);
    fireEvent.click(button('+ Pool'));
    fireEvent.click(button('+ Conv'));
    expect(s().config.network.conv.map((r) => r.kind)).toEqual([
      'conv',
      'pool',
      'conv',
      'pool',
      'conv',
    ]);
    expect(button('MLP').getAttribute('aria-pressed')).toBe('false');
    expect(button('Small CNN').getAttribute('aria-pressed')).toBe('false');
  });

  it('flags the row that doesn’t fit, says why, and stops training', () => {
    s().setRunning(true);
    render(<MnistArchitecture />);
    fireEvent.click(button('Small CNN'));
    expect(s().running).toBe(true);
    fireEvent.click(button('+ Conv')); // 3×3 same on 16 × 4 × 4 is fine
    fireEvent.change(screen.getByRole('combobox', { name: 'Kernel size of conv layer 5' }), {
      target: { value: '5' },
    });
    fireEvent.change(screen.getByRole('combobox', { name: 'Padding of conv layer 5' }), {
      target: { value: 'valid' },
    });
    expect(screen.getByText('a 5 × 5 kernel doesn’t fit a 4 × 4 input')).toBeTruthy();
    expect(screen.getByRole('alert').textContent).toMatch(/can’t take a 28 × 28 image/);
    expect(s().running).toBe(false);
    // Fixing it lets training start again.
    fireEvent.click(button('Remove conv layer 5'));
    expect(screen.queryByRole('alert')).toBeNull();
  });

  it('MLP returns to the default network', () => {
    render(<MnistArchitecture />);
    fireEvent.click(button('Small CNN'));
    fireEvent.click(button('MLP'));
    expect(s().config.network).toEqual(MNIST_DEFAULT_CONFIG.network);
  });
});

describe('transport controls when blocked', () => {
  it('disables Play and Step and says why', () => {
    render(
      <TransportControls
        running={false}
        onRunning={() => {}}
        onReset={() => {}}
        onStep={() => {}}
        stepUnit="one mini-batch"
        speedUnit="batches"
        readout={{ label: 'Epoch', value: '0' }}
        lr={0.001}
        batchSize={64}
        batchSizes={[64]}
        onTraining={() => {}}
        speed="max"
        onSpeed={() => {}}
        seed={1}
        onSeed={() => {}}
        blocked="Fix the network first"
      />,
    );
    const play = button('▶ Play') as HTMLButtonElement;
    expect(play.disabled).toBe(true);
    expect(play.title).toBe('Fix the network first');
    expect((button('⏭ Step') as HTMLButtonElement).disabled).toBe(true);
  });
});
