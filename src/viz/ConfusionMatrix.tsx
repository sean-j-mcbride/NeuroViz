import { useMemo } from 'react';
import { MNIST_CLASSES } from '../data';
import { lutIndex, sequentialLut } from './colour';
import { usePrefersDark } from './hooks';

export interface CellRef {
  label: number;
  predicted: number;
}

interface ConfusionMatrixProps {
  /** `[10 × 10]`, row = true label, column = predicted class. */
  counts: Uint32Array;
  /** Show each row as shares of that digit's images instead of counts. */
  rowShares: boolean;
  selected: CellRef | null;
  onSelect: (cell: CellRef | null) => void;
}

/**
 * True digit (rows) against prediction (columns), as a table with every
 * number written in. The diagonal holds the correct answers; colour marks the
 * mistakes, scaled to the most common one, so the confusions stand out.
 * Clicking a mistake shows those images in the gallery.
 */
export function ConfusionMatrix({ counts, rowShares, selected, onSelect }: ConfusionMatrixProps) {
  const dark = usePrefersDark();
  const lut = useMemo(() => sequentialLut(dark), [dark]);
  const totals = Array.from({ length: MNIST_CLASSES }, (_, r) =>
    counts.subarray(r * MNIST_CLASSES, (r + 1) * MNIST_CLASSES).reduce((a, b) => a + b, 0),
  );
  let maxMistake = 0;
  counts.forEach((n, i) => {
    if (i % (MNIST_CLASSES + 1) !== 0) maxMistake = Math.max(maxMistake, n);
  });

  const text = (n: number, r: number) =>
    rowShares ? (totals[r] ? `${Math.round((n / totals[r]!) * 100)}` : '–') : String(n);

  return (
    <table className="confusion">
      <caption className="sr-only">
        Test images by true digit (rows) and predicted digit (columns)
        {rowShares ? ', as a percentage of each row' : ''}
      </caption>
      <thead>
        <tr>
          <th scope="col" className="confusion-corner">
            <span>true ↓</span> <span>predicted →</span>
          </th>
          {Array.from({ length: MNIST_CLASSES }, (_, c) => (
            <th key={c} scope="col">
              {c}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {Array.from({ length: MNIST_CLASSES }, (_, r) => (
          <tr key={r}>
            <th scope="row">{r}</th>
            {Array.from({ length: MNIST_CLASSES }, (_, c) => {
              const n = counts[r * MNIST_CLASSES + c]!;
              const correct = r === c;
              const isSelected = selected?.label === r && selected.predicted === c;
              const t = !correct && maxMistake > 0 ? n / maxMistake : 0;
              const k = lutIndex(t * 2 - 1) * 3;
              const style =
                !correct && n > 0
                  ? {
                      background: `rgb(${lut[k]} ${lut[k + 1]} ${lut[k + 2]})`,
                      color: t > 0.45 ? (dark ? '#14161a' : '#ffffff') : undefined,
                    }
                  : undefined;
              const describe =
                `True ${r}, predicted ${c}: ${n} image${n === 1 ? '' : 's'}` +
                (totals[r] ? ` (${((n / totals[r]!) * 100).toFixed(1)} % of the ${r}s)` : '');
              return (
                <td
                  key={c}
                  className={`${correct ? 'correct' : ''} ${isSelected ? 'selected' : ''}`}
                  style={style}
                >
                  {correct || n === 0 ? (
                    <span title={describe}>{text(n, r)}</span>
                  ) : (
                    <button
                      type="button"
                      title={`${describe}. Show them in the gallery.`}
                      aria-pressed={isSelected}
                      onClick={() => onSelect(isSelected ? null : { label: r, predicted: c })}
                    >
                      {text(n, r)}
                    </button>
                  )}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
