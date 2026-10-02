import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App.tsx';
import { applyHash } from './ui/useUrlSync';
import './styles.css';

const root = document.getElementById('root');
if (!root) throw new Error('Missing #root element');

// A share link sets up the run before the first render, so the worker starts with it.
applyHash(window.location.hash);

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
