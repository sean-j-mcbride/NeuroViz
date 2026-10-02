import { PRESETS } from '../state/presets';
import { sameConfig, useAppStore } from '../state/store';

/** What the loaded preset shows, what to look for, and a one-click comparison with its fix. */
export function PresetNote() {
  const presetId = useAppStore((s) => s.presetId);
  const config = useAppStore((s) => s.config);
  const epoch = useAppStore((s) => s.snapshot?.epoch ?? 0);
  const { tryPresetFix, dismissPreset } = useAppStore.getState();

  const preset = PRESETS.find((p) => p.id === presetId);
  if (!preset) return null;
  const isPreset = sameConfig(config, preset.config);
  const isFix = sameConfig(config, preset.fix);
  if (!isPreset && !isFix) return null;

  return (
    <aside className="preset-note" aria-label={`Preset: ${preset.name}`}>
      <div className="preset-note-head">
        <h2>{isFix ? `${preset.name}: the fix` : preset.name}</h2>
        <button type="button" className="close" onClick={dismissPreset} aria-label="Hide note">
          ✕
        </button>
      </div>
      {isPreset ? (
        <>
          <p>{preset.summary}</p>
          <p>
            <strong>Look for:</strong> {preset.lookFor}
          </p>
          <p className="preset-note-try">
            <span>
              <strong>Then try:</strong> {preset.tryThis}
            </span>
            <button
              type="button"
              onClick={tryPresetFix}
              disabled={epoch === 0}
              title={
                epoch === 0
                  ? 'Train first, so there is a run to compare against'
                  : 'Pin this run as the reference, then start the fixed set-up'
              }
            >
              Pin this run and try the fix
            </button>
          </p>
        </>
      ) : (
        <p>
          {preset.tryThis} Press Play, then compare with the pinned run under Compare and on the
          loss curve.
        </p>
      )}
    </aside>
  );
}
