import { useState } from 'react';
import { type Tab, useTabStore } from './state/tab';
import {
  ArchitectureBuilder,
  ComparePanel,
  DatasetPanel,
  InsidePanel,
  MnistPage,
  NoticeBar,
  OptimiserControls,
  OutputPanel,
  PresetNote,
  ProjectBar,
  TransportBar,
  useTrainingWorker,
  useUrlSync,
} from './ui';

const TABS: { id: Tab; name: string; title: string }[] = [
  { id: 'playground', name: 'Playground', title: 'Small networks on 2D toy data' },
  { id: 'mnist', name: 'MNIST', title: 'Handwritten digits: 784 inputs, 10 classes' },
];

function PlaygroundPage() {
  const { step, requestCheckpoint } = useTrainingWorker();
  return (
    <>
      <div className="app-header page-controls">
        <TransportBar onStep={step} />
        <OptimiserControls />
        <ProjectBar requestCheckpoint={requestCheckpoint} />
      </div>
      <PresetNote />
      <main className="playground">
        <DatasetPanel />
        <ArchitectureBuilder />
        <OutputPanel />
      </main>
      <ComparePanel />
      <InsidePanel />
    </>
  );
}

export function App() {
  useUrlSync();
  const tab = useTabStore((s) => s.tab);
  const setTab = useTabStore.getState().setTab;
  // The MNIST page (its worker and its ~2 MB of images) is created on first visit, then kept.
  const [mnistVisited, setMnistVisited] = useState(tab === 'mnist');
  if (tab === 'mnist' && !mnistVisited) setMnistVisited(true);

  return (
    <div className="app">
      <header className="app-header">
        <h1>NeuroViz</h1>
        <div className="tabs" role="tablist" aria-label="Workspace">
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              role="tab"
              id={`tab-${t.id}`}
              aria-selected={tab === t.id}
              aria-controls={`page-${t.id}`}
              title={t.title}
              onClick={() => setTab(t.id)}
            >
              {t.name}
            </button>
          ))}
        </div>
      </header>
      <NoticeBar />
      <div
        id="page-playground"
        role="tabpanel"
        aria-labelledby="tab-playground"
        hidden={tab !== 'playground'}
      >
        <PlaygroundPage />
      </div>
      {mnistVisited && (
        <div id="page-mnist" role="tabpanel" aria-labelledby="tab-mnist" hidden={tab !== 'mnist'}>
          <MnistPage />
        </div>
      )}
    </div>
  );
}
