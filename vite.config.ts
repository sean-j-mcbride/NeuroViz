import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  // Relative asset paths so the static build works from a GitHub Pages subpath.
  base: './',
  plugins: [react()],
  // The training worker is a module worker; bundle it as ES so it can share chunks.
  worker: { format: 'es' },
  test: {
    // Node by default; component tests opt in with `// @vitest-environment jsdom`.
    environment: 'node',
    include: ['src/**/*.test.{ts,tsx}'],
    setupFiles: ['src/test/setup.ts'],
  },
});
