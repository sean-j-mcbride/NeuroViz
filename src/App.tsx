import {
  ArchitectureBuilder,
  DatasetPanel,
  InsidePanel,
  OptimiserControls,
  OutputPanel,
  TransportBar,
  useTrainingWorker,
} from './ui';

export function App() {
  const { step } = useTrainingWorker();
  return (
    <div className="app">
      <header className="app-header">
        <h1>NeuroViz</h1>
        <TransportBar onStep={step} />
        <OptimiserControls />
      </header>
      <main className="playground">
        <DatasetPanel />
        <ArchitectureBuilder />
        <OutputPanel />
      </main>
      <InsidePanel />
    </div>
  );
}
