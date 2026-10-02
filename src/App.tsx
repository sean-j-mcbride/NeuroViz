import {
  ArchitectureBuilder,
  DatasetPanel,
  OutputPanel,
  TransportBar,
  useTrainingLoop,
} from './ui';

export function App() {
  const { step } = useTrainingLoop();
  return (
    <div className="app">
      <header className="app-header">
        <h1>NeuroViz</h1>
        <TransportBar onStep={step} />
      </header>
      <main className="playground">
        <DatasetPanel />
        <ArchitectureBuilder />
        <OutputPanel />
      </main>
    </div>
  );
}
