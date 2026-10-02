import js from '@eslint/js';
import reactHooks from 'eslint-plugin-react-hooks';
import reactRefresh from 'eslint-plugin-react-refresh';
import globals from 'globals';
import tseslint from 'typescript-eslint';

const engineBoundary = {
  patterns: [
    {
      group: ['react', 'react-dom', 'react/*', 'zustand', 'zustand/*'],
      message: 'engine/ and data/ must stay framework-free (they run in a Web Worker).',
    },
    {
      group: [
        '**/viz',
        '**/viz/*',
        '**/ui',
        '**/ui/*',
        '**/state',
        '**/state/*',
        '**/worker',
        '**/worker/*',
      ],
      message: 'engine/ and data/ must not depend on app layers.',
    },
  ],
};

export default tseslint.config(
  { ignores: ['dist', 'node_modules'] },
  {
    files: ['**/*.{ts,tsx}'],
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    languageOptions: { ecmaVersion: 2022, globals: globals.browser },
    plugins: { 'react-hooks': reactHooks, 'react-refresh': reactRefresh },
    rules: {
      ...reactHooks.configs.recommended.rules,
      'react-refresh/only-export-components': ['warn', { allowConstantExport: true }],
      '@typescript-eslint/no-explicit-any': 'error',
    },
  },
  {
    files: ['src/engine/**/*.ts', 'src/data/**/*.ts'],
    rules: { 'no-restricted-imports': ['error', engineBoundary] },
  },
);
