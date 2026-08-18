import { useState } from 'react';
import { useSubstrate } from './client/useSubstrate';
import { Nebula } from './Nebula';
import { Star } from './Star';

export type SubstrateHook = ReturnType<typeof useSubstrate>;

// The proved loop: navigate (Nebula) → focus (Star) → compose → dispatch →
// integrate (proposals) → return. Truth lives in the kernel state; both
// surfaces are views over it.
export function App() {
  const sub = useSubstrate();
  const [focus, setFocus] = useState<string | null>(null);

  // A failed fetch before anything loaded is a dead server; after that, errors
  // surface as a banner — a healthy workspace is never unmounted over a blip.
  if (sub.error && !sub.ws)
    return (
      <div className="hint">
        substrate server unreachable — start with: npm run dev
        <div className="meta">{sub.error}</div>
        <p>
          <button onClick={() => void sub.reload()}>retry</button>
        </p>
      </div>
    );
  if (!sub.ws || !sub.ctx) return <div className="hint">loading nebula…</div>;

  const banner = sub.error ?? sub.status;
  return (
    <>
      {banner && (
        <div className="banner">
          {banner}
          {sub.error ? (
            <button onClick={() => void sub.reload()}>retry</button>
          ) : (
            <button onClick={sub.dismissStatus}>dismiss</button>
          )}
        </div>
      )}
      {focus && sub.ws.state.chunks.has(focus) ? (
        <Star sub={sub} docId={focus} onFocusDoc={setFocus} onBack={() => setFocus(null)} />
      ) : (
        <Nebula sub={sub} onFocus={setFocus} />
      )}
    </>
  );
}
