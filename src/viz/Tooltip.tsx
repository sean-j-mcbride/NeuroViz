import { type ReactNode, useLayoutEffect, useRef, useState } from 'react';

interface TooltipProps {
  /** Pointer position in viewport coordinates. */
  x: number;
  y: number;
  title: string;
  /** Label / value pairs. */
  rows: [string, string][];
  children?: ReactNode;
}

const OFFSET = 14;

/** A floating card near the pointer, flipped to stay inside the viewport. */
export function Tooltip({ x, y, title, rows, children }: TooltipProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState({ left: x + OFFSET, top: y + OFFSET });

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const { width, height } = el.getBoundingClientRect();
    const left = x + OFFSET + width > window.innerWidth - 8 ? x - OFFSET - width : x + OFFSET;
    const top = y + OFFSET + height > window.innerHeight - 8 ? y - OFFSET - height : y + OFFSET;
    setPos({ left: Math.max(8, left), top: Math.max(8, top) });
  }, [x, y]);

  return (
    <div ref={ref} className="tooltip" role="tooltip" style={pos}>
      <div className="tooltip-title">{title}</div>
      <table>
        <tbody>
          {rows.map(([label, value]) => (
            <tr key={label}>
              <th>{label}</th>
              <td>{value}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {children}
    </div>
  );
}
