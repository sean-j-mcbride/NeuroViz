import {
  ArchitectureBuilder,
  ComparePanel,
  DatasetPanel,
  InsidePanel,
  NoticeBar,
  OptimiserControls,
  OutputPanel,
  PresetNote,
  ProjectBar,
  TransportBar,
  useTrainingWorker,
  useUrlSync,
} from './ui';

export function App() {
  const { step, requestCheckpoint } = useTrainingWorker();
  useUrlSync();
  return (
    <div className="app">
      <header className="app-header">
        <h1>NeuroViz</h1>
        <TransportBar onStep={step} />
        <OptimiserControls />
        <ProjectBar requestCheckpoint={requestCheckpoint} />
      </header>
      <NoticeBar />
      <PresetNote />
      <main className="playground">
        <DatasetPanel />
        <ArchitectureBuilder />
        <OutputPanel />
      </main>
      <ComparePanel />
      <InsidePanel />
    </div>
  );
}
