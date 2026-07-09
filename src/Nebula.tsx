import { useMemo, useState } from 'react';
import type { Chunk } from './kernel/chunk';
import { search } from './kernel/search';
import type { Store } from './kernel/store';

const W = 1000;
const H = 700;

// Deterministic layout: clusters (top-level dirs) on a ring, docs in a
// phyllotaxis spiral inside each cluster. Same snapshot → same sky, so
// spatial memory holds. Search illuminates stars in place, never moves them.
// STUB: soft-chain reclustering (embeddings/provenance/tags), breathing
// animation, zooming into the nebula behind a focused star.
function layout(store: Store) {
  const docs = [...store.values()].filter((c) => c.kind === 'doc').sort((a, b) => a.id.localeCompare(b.id));
  const groups = new Map<string, Chunk[]>();
  for (const d of docs) {
    const key = d.id.includes('/') ? d.id.split('/')[0] : '·';
    (groups.get(key) ?? groups.set(key, []).get(key)!).push(d);
  }
  const keys = [...groups.keys()].sort();
  return keys.map((key, i) => {
    const a = (2 * Math.PI * i) / keys.length - Math.PI / 2;
    const R = keys.length === 1 ? 0 : 240;
    const gx = W / 2 + R * Math.cos(a);
    const gy = H / 2 + R * Math.sin(a);
    const stars = groups.get(key)!.map((doc, j) => {
      const b = j * 2.39996; // golden angle
      const r = 18 * Math.sqrt(j + 0.4);
      return { doc, x: gx + r * Math.cos(b), y: gy + r * Math.sin(b) };
    });
    return { key, gx, gy, stars };
  });
}

export function Nebula({ store, onFocus }: { store: Store; onFocus: (id: string) => void }) {
  const [query, setQuery] = useState('');
  const clusters = useMemo(() => layout(store), [store]);
  const hits = useMemo(() => search(store, query), [store, query]);
  const searching = query.trim() !== '';

  return (
    <div className="nebula">
      <header>
        <h1>substrate</h1>
        <input placeholder="search the workspace…" value={query} onChange={(e) => setQuery(e.target.value)} />
      </header>
      <svg viewBox={`0 0 ${W} ${H}`}>
        {clusters.map(({ key, gx, gy, stars }) => (
          <g key={key}>
            <text className="cluster-label" x={gx} y={gy - 18 * Math.sqrt(stars.length) - 16}>
              {key}
            </text>
            {stars.map(({ doc, x, y }) => {
              const lit = hits.has(doc.id);
              return (
                <g
                  key={doc.id}
                  className={`star ${lit ? 'lit' : ''} ${searching && !lit ? 'dim' : ''}`}
                  onClick={() => onFocus(doc.id)}
                >
                  <circle cx={x} cy={y} r={4 + Math.min(doc.children.length, 10) * 0.7} />
                  <text x={x} y={y + 18}>
                    {doc.id.split('/').pop()}
                  </text>
                </g>
              );
            })}
          </g>
        ))}
      </svg>
    </div>
  );
}
