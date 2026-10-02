import { type ModelFile, parseModelFile } from '../state/modelFile';

/** Offers `text` to the user as a file download. */
export function downloadText(filename: string, text: string, type = 'application/json'): void {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.append(a);
  a.click();
  a.remove();
  // Revoke after the click has been handled.
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

/** Reads and checks a saved model; throws `ModelFileError` with a readable reason. */
export async function readModelFile(file: File): Promise<ModelFile> {
  return parseModelFile(await file.text());
}

export function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
