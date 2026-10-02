// Web Worker entry: wires the TrainingController to postMessage.
// Typed with a minimal local interface so src/worker stays free of DOM and
// WebWorker lib types (see tsconfig.engine.json).
import { TrainingController } from './controller';
import type { FromWorker, ToWorker } from './protocol';

interface MessagePortLike {
  onmessage: (() => void) | null;
  postMessage(value: null): void;
}

interface WorkerScope {
  onmessage: ((event: { data: ToWorker }) => void) | null;
  postMessage(message: FromWorker, transfer: ArrayBuffer[]): void;
  setTimeout(fn: () => void, ms: number): unknown;
  MessageChannel: new () => { port1: MessagePortLike; port2: MessagePortLike };
  performance: { now(): number };
}

const scope = globalThis as unknown as WorkerScope;

// Zero-delay yields go through a MessageChannel: nested setTimeout(0) is clamped to ≥ 4 ms.
const pending: (() => void)[] = [];
const channel = new scope.MessageChannel();
channel.port1.onmessage = () => pending.shift()?.();

const controller = new TrainingController(
  (message, transfer) => scope.postMessage(message, transfer),
  {
    now: () => scope.performance.now(),
    defer: (fn, ms) => {
      if (ms > 0) {
        scope.setTimeout(fn, ms);
      } else {
        pending.push(fn);
        channel.port2.postMessage(null);
      }
    },
  },
);

scope.onmessage = (event) => controller.handle(event.data);
