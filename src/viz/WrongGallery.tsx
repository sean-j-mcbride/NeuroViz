import { MNIST_PIXELS, type MnistSet } from '../data';
import type { MnistEvaluation } from '../worker';
import { DigitImage } from './DigitImage';
import { confidentMistakes } from './mnistResults';

/** How many images the gallery shows. */
export const GALLERY_SIZE = 24;

interface WrongGalleryProps {
  test: MnistSet;
  evaluation: MnistEvaluation;
  /** Only this confusion-matrix cell; null = every mistake. */
  filter: { label: number; predicted: number } | null;
  onTry: (index: number) => void;
}

/** The test images the network gets wrong most confidently, each with its label and guess. */
export function WrongGallery({ test, evaluation, filter, onTry }: WrongGalleryProps) {
  const all = confidentMistakes(test.labels, evaluation.predicted, evaluation.confidence, filter);
  const shown = all.slice(0, GALLERY_SIZE);
  if (shown.length === 0) {
    return (
      <p className="hint">{filter ? 'No such mistakes now.' : 'No mistakes on the test set.'}</p>
    );
  }
  return (
    <>
      <p className="hint">
        {all.length > shown.length
          ? `The ${shown.length} most confident of ${all.length.toLocaleString('en-GB')} mistakes.`
          : `All ${all.length} mistake${all.length === 1 ? '' : 's'}, most confident first.`}
      </p>
      <ul className="gallery">
        {shown.map((i) => {
          const label = test.labels[i]!;
          const guess = evaluation.predicted[i]!;
          const conf = Math.round(evaluation.confidence[i]! * 100);
          const text = `A ${label}, taken for a ${guess} (${conf} % sure). Official test image ${test.source[i]}`;
          return (
            <li key={i}>
              <button type="button" onClick={() => onTry(i)} title={`${text}. Try it on the pad.`}>
                <DigitImage
                  pixels={test.images.subarray(i * MNIST_PIXELS, (i + 1) * MNIST_PIXELS)}
                  size={56}
                  label={text}
                />
                <span className="gallery-caption">
                  {label} → <strong>{guess}</strong>
                  <span className="muted"> · {conf} %</span>
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </>
  );
}
