// Shared test setup. Node-environment tests skip all of this; files that start
// with `// @vitest-environment jsdom` get Testing Library cleanup plus stubs for
// the browser APIs jsdom lacks.
import { afterEach } from 'vitest';

if (typeof window !== 'undefined') {
  const { cleanup } = await import('@testing-library/react');
  afterEach(cleanup);

  // Light colour scheme.
  window.matchMedia ??= (query: string) =>
    ({
      matches: false,
      media: query,
      onchange: null,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
      dispatchEvent: () => false,
    }) as MediaQueryList;

  // Every observed element reports an 800 px content box.
  globalThis.ResizeObserver ??= class {
    constructor(private readonly callback: ResizeObserverCallback) {}
    observe(target: Element) {
      this.callback(
        [{ target, contentRect: { width: 800, height: 600 } } as unknown as ResizeObserverEntry],
        this as unknown as ResizeObserver,
      );
    }
    unobserve() {}
    disconnect() {}
  };

  // jsdom has no 2D canvas; the components skip drawing when getContext returns null.
  HTMLCanvasElement.prototype.getContext = (() =>
    null) as typeof HTMLCanvasElement.prototype.getContext;
}
