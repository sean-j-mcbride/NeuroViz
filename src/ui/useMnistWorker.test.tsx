// @vitest-environment jsdom
import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { MnistSubset } from '../data';
import { useMnistStore } from '../state/mnistStore';
import { useAppStore } from '../state/store';
import type { ToWorker } from '../worker';
import { loadMnist } from './mnistLoader';
import { useMnistWorker } from './useMnistWorker';

vi.mock('./mnistLoader', () => ({ loadMnist: vi.fn() }));
const load = vi.mocked(loadMnist);

/** Messages posted to every Worker the hook creates. */
let posted: ToWorker[] = [];

class FakeWorker {
  onmessage: ((e: MessageEvent) => void) | null = null;
  postMessage(msg: ToWorker) {
    posted.push(msg);
  }
  terminate() {}
}

const subset = {
  train: { images: new Uint8Array(0), labels: new Uint8Array(0), source: new Uint16Array(0) },
  test: { images: new Uint8Array(0), labels: new Uint8Array(0), source: new Uint16Array(0) },
  checksum: 1,
} satisfies MnistSubset;

const initialMnist = useMnistStore.getState();
const initialApp = useAppStore.getState();

beforeEach(() => {
  posted = [];
  vi.stubGlobal('Worker', FakeWorker);
  useMnistStore.setState(initialMnist, true);
  useAppStore.setState(initialApp, true);
});

afterEach(() => {
  vi.unstubAllGlobals();
  load.mockReset();
});

describe('useMnistWorker', () => {
  it('a failed download can be retried', async () => {
    load.mockRejectedValueOnce(new Error('offline')).mockResolvedValueOnce(subset);
    renderHook(() => useMnistWorker());
    await waitFor(() => expect(useMnistStore.getState().data.status).toBe('error'));
    act(() => useMnistStore.getState().setData({ status: 'idle' })); // "Try again"
    await waitFor(() => expect(useMnistStore.getState().data.status).toBe('ready'));
    expect(load).toHaveBeenCalledTimes(2);
    expect(posted.some((m) => m.type === 'init')).toBe(true);
  });

  it('posts nothing that needs a session before the images have loaded', async () => {
    load.mockReturnValue(new Promise(() => {})); // still downloading
    const { result } = renderHook(() => useMnistWorker());
    await waitFor(() => expect(useMnistStore.getState().data.status).toBe('loading'));
    act(() => result.current.step());
    act(() => useMnistStore.getState().setDrawn(new Float32Array(784)));
    act(() => useMnistStore.getState().setTraining({ lr: 0.003 }));
    expect(posted.filter((m) => m.type !== 'pause')).toEqual([]);
  });

  it('once loaded, starts a session and steps it', async () => {
    load.mockResolvedValue(subset);
    const { result } = renderHook(() => useMnistWorker());
    await waitFor(() => expect(posted.some((m) => m.type === 'init')).toBe(true));
    act(() => result.current.step());
    expect(posted.map((m) => m.type)).toContain('step');
  });
});
