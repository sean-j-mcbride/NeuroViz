import { useEffect } from 'react';
import { useAppStore } from '../state/store';

/** How long an info notice stays up; errors stay until dismissed. */
const INFO_MS = 5000;

/** One message for the user: an error (stays) or a confirmation (fades). */
export function NoticeBar() {
  const notice = useAppStore((s) => s.notice);
  const setNotice = useAppStore.getState().setNotice;

  useEffect(() => {
    if (notice?.kind !== 'info') return;
    const t = setTimeout(() => setNotice(null), INFO_MS);
    return () => clearTimeout(t);
  }, [notice, setNotice]);

  if (!notice) return null;
  return (
    <div className={`notice ${notice.kind}`} role={notice.kind === 'error' ? 'alert' : 'status'}>
      <span>{notice.text}</span>
      <button type="button" className="close" onClick={() => setNotice(null)} aria-label="Dismiss">
        ✕
      </button>
    </div>
  );
}
