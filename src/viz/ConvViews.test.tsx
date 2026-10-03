// @vitest-environment jsdom
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import type { ConvFilters, ConvLayerSpec, FeatureMap } from '../worker';
import { FeatureMaps } from './FeatureMaps';
import { FilterViewer } from './FilterViewer';

const rows: ConvLayerSpec[] = [
  { kind: 'conv', filters: 2, kernel: 5, stride: 1, padding: 'valid', activation: 'relu' },
  { kind: 'pool', size: 2 },
  { kind: 'conv', filters: 3, kernel: 3, stride: 1, padding: 'valid', activation: 'relu' },
];

function ramp(n: number): Float32Array {
  return Float32Array.from({ length: n }, (_, i) => (i % 7) / 7);
}

/** Lets clicks map to canvas pixels 1:1 (jsdom has no layout). */
function sizeCanvas(canvas: HTMLCanvasElement) {
  vi.spyOn(canvas, 'getBoundingClientRect').mockReturnValue({
    left: 0,
    top: 0,
    width: canvas.width,
    height: canvas.height,
  } as DOMRect);
}

describe('FeatureMaps', () => {
  const maps: FeatureMap[] = [
    { row: 0, shape: [2, 24, 24], data: ramp(2 * 576) },
    { row: 1, shape: [2, 12, 12], data: ramp(2 * 144) },
    { row: 2, shape: [3, 10, 10], data: ramp(3 * 100) },
  ];

  it('click a pixel: outlines its receptive field on the input and earlier layers, and says where', () => {
    const { container } = render(<FeatureMaps rows={rows} maps={maps} input={ramp(784)} />);
    expect(screen.getByText(/Click any pixel of a feature map/)).toBeTruthy();
    const pool = screen.getByRole('img', {
      name: 'Pool 1: 2 maps of 12 by 12',
    }) as HTMLCanvasElement;
    sizeCanvas(pool);
    // 12×12 maps are drawn 4 px per pixel; channel 2's tile starts at 48 + 4 gap. Pixel (row 3, col 5):
    fireEvent.click(pool, { clientX: 52 + 5 * 4 + 1, clientY: 3 * 4 + 1 });
    expect(screen.getByText(/Pool 1, channel 2, at row 3, column 5/).textContent).toMatch(
      /computed from input rows 6–11, columns 10–15 \(6 × 6 pixels\)/,
    );
    // The field on the input (1 map) and on Conv 1 (2 channels); the picked pixel itself.
    expect(container.querySelectorAll('.map-field')).toHaveLength(3);
    expect(container.querySelectorAll('.map-picked')).toHaveLength(1);
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(container.querySelectorAll('.map-field')).toHaveLength(0);
    expect(screen.getByText(/Click any pixel of a feature map/)).toBeTruthy();
  });

  it('clicking the same pixel again clears it', () => {
    const { container } = render(<FeatureMaps rows={rows} maps={maps} input={ramp(784)} />);
    const conv2 = screen.getByRole('img', {
      name: 'Conv 2: 3 maps of 10 by 10',
    }) as HTMLCanvasElement;
    sizeCanvas(conv2);
    fireEvent.click(conv2, { clientX: 2, clientY: 2 });
    expect(container.querySelectorAll('.map-field')).toHaveLength(1 + 2 + 2);
    fireEvent.click(conv2, { clientX: 2, clientY: 2 });
    expect(container.querySelectorAll('.map-field')).toHaveLength(0);
  });

  it('has a table of every channel', () => {
    render(<FeatureMaps rows={rows} maps={maps} input={ramp(784)} />);
    fireEvent.click(screen.getByRole('button', { name: 'Table' }));
    expect(screen.getAllByRole('row')).toHaveLength(1 + 2 + 2 + 3);
    expect(screen.getByRole('rowheader', { name: 'Pool 1' })).toBeTruthy();
  });
});

describe('FilterViewer', () => {
  const filters: ConvFilters[] = [
    {
      row: 0,
      inChannels: 1,
      outChannels: 2,
      kernel: 5,
      stride: 1,
      padding: 'valid',
      W: ramp(50),
      b: ramp(2),
    },
    {
      row: 2,
      inChannels: 2,
      outChannels: 3,
      kernel: 3,
      stride: 1,
      padding: 'valid',
      W: ramp(54),
      b: ramp(3),
    },
  ];

  it('picks a layer, opens a filter’s kernel with its weights, and lists filters as a table', () => {
    render(<FilterViewer filters={filters} rows={rows} />);
    fireEvent.click(screen.getByRole('button', { name: 'Conv 2' }));
    expect(screen.getByText(/Rows: the 3 filters. Columns: the 2 input channels/)).toBeTruthy();
    const grid = screen.getByRole('img', {
      name: /3 filters of 3 by 3 weights over 2 input channels/,
    }) as HTMLCanvasElement;
    sizeCanvas(grid);
    // Tiles are 3 px + 1 gap: column 1 (channel 2), row 2 (filter 3).
    fireEvent.click(grid, { clientX: 4 + 1, clientY: 8 + 1 });
    expect(screen.getByRole('heading', { name: 'Conv 2, filter 3, input channel 2' })).toBeTruthy();
    expect(
      screen.getByRole('table', { name: 'Weights of filter 3' }).querySelectorAll('td'),
    ).toHaveLength(9);
    fireEvent.click(screen.getByRole('button', { name: 'Close' }));
    expect(screen.queryByRole('table', { name: 'Weights of filter 3' })).toBeNull();

    fireEvent.click(screen.getByRole('button', { name: 'Table' }));
    expect(screen.getAllByRole('row')).toHaveLength(1 + 3);
  });
});

describe('FeatureMaps: gradients', () => {
  const maps: FeatureMap[] = [{ row: 0, shape: [2, 24, 24], data: ramp(2 * 576) }];
  const gradMaps: FeatureMap[] = [
    { row: 0, shape: [2, 24, 24], data: ramp(2 * 576).map((v) => v - 0.5) },
  ];
  const rows1 = rows.slice(0, 1);

  it('switches to −∂L/∂ maps and a saliency input, with a target digit to choose', () => {
    const onGradTarget = vi.fn();
    render(
      <FeatureMaps
        rows={rows1}
        maps={maps}
        input={ramp(784)}
        gradMaps={gradMaps}
        inputGrad={ramp(784)}
        target={7}
        gradTarget={null}
        onGradTarget={onGradTarget}
      />,
    );
    expect(screen.queryByText('Input (saliency)')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Gradients' }));
    expect(screen.getByText('Input (saliency)')).toBeTruthy();
    expect(screen.getByText(/more sure this is a 7/)).toBeTruthy();
    const select = screen.getByRole('combobox', { name: 'Gradient for' }) as HTMLSelectElement;
    expect(select.value).toBe('');
    expect(screen.getByRole('option', { name: 'its answer (7)' })).toBeTruthy();
    fireEvent.change(select, { target: { value: '3' } });
    expect(onGradTarget).toHaveBeenLastCalledWith(3);
    fireEvent.change(select, { target: { value: '' } });
    expect(onGradTarget).toHaveBeenLastCalledWith(null);
  });

  it('has no Gradients view without gradient maps', () => {
    render(<FeatureMaps rows={rows1} maps={maps} input={ramp(784)} />);
    expect(screen.queryByRole('button', { name: 'Gradients' })).toBeNull();
  });
});

describe('FilterViewer: dead filters', () => {
  const filters: ConvFilters[] = [
    {
      row: 0,
      inChannels: 1,
      outChannels: 3,
      kernel: 3,
      stride: 1,
      padding: 'valid',
      W: ramp(27),
      b: ramp(3),
    },
  ];

  it('hatches dead filters and counts them in the table', () => {
    const { container } = render(
      <FilterViewer
        filters={filters}
        rows={rows}
        dead={[{ layer: 0, flags: Uint8Array.from([0, 1, 1]) }]}
      />,
    );
    expect(container.querySelectorAll('.tile-dead')).toHaveLength(2);
    fireEvent.click(screen.getByRole('button', { name: 'Table' }));
    const cells = screen
      .getAllByRole('row')
      .slice(1)
      .map((r) => r.lastElementChild!.textContent);
    expect(cells).toEqual(['–', 'yes', 'yes']);
  });
});
