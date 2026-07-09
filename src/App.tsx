import { useEffect, useState } from 'react';
import type { Chunk } from './kernel/chunk';
import { loadStore, type Store } from './kernel/store';
import { Nebula } from './Nebula';
import { Star } from './Star';

// The proved loop: navigate (Nebula) → focus (Star) → compose → dispatch →
// integrate → return (back to the Nebula, layout unchanged).
export function App() {
  const [store, setStore] = useState<Store | null>(null);
  const [focus, setFocus] = useState<string | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    fetch('/snapshot.json')
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((snap) => setStore(loadStore(snap)))
      .catch(() => setError(true));
  }, []);

  if (error) return <div className="hint">No snapshot found — run: npm run ingest -- &lt;folder&gt;</div>;
  if (!store) return <div className="hint">loading nebula…</div>;

  const commit = (...cs: Chunk[]) =>
    setStore((prev) => {
      const next = new Map(prev);
      for (const c of cs) next.set(c.id, c);
      return next;
    });

  return focus ? (
    <Star store={store} docId={focus} commit={commit} onFocusDoc={setFocus} onBack={() => setFocus(null)} />
  ) : (
    <Nebula store={store} onFocus={setFocus} />
  );
}
