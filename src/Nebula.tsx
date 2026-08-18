import { useMemo, useState } from 'react';
import type { SubstrateHook } from './App';
import { docList, labelOf, proposalsForDoc } from './client/helpers';
import { buildIndexes, provenanceKind, searchChunks } from './index/indexes';
import { childOccurrences, occurrencesOfChunk } from './kernel/state';
import type { ChunkId } from './kernel/types';

const W = 1000;
const H = 700;

// Deterministic layout: clusters (top-level dirs) on a ring, docs in a
// phyllotaxis spiral inside each cluster. Same corpus → same sky, so spatial
// memory holds. Search illuminates stars in place, never moves them.
function layout(docs: { id: ChunkId; cluster: string; label: string; size: number }[]) {
  const groups = new Map<string, typeof docs>();
  for (const d of [...docs].sort((a, b) => a.label.localeCompare(b.label))) {
    (groups.get(d.cluster) ?? groups.set(d.cluster, []).get(d.cluster)!).push(d);
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

export function Nebula({ sub, onFocus }: { sub: SubstrateHook; onFocus: (id: ChunkId) => void }) {
  const { state, bindings } = sub.ws!;
  const [query, setQuery] = useState('');
  const [lens, setLens] = useState<'none' | 'provenance'>('none');

  const indexes = useMemo(() => buildIndexes(state), [sub.version]); // eslint-disable-line react-hooks/exhaustive-deps
  const docs = useMemo(
    () =>
      docList(state, bindings).map((id) => {
        const bound = bindings.find((b) => b.docChunkId === id);
        const rel = bound?.relPath ?? '';
        return {
          id,
          cluster: rel.includes('/') ? rel.split('/')[0] : '·',
          label: (rel ? rel.split('/').pop()! : labelOf(state, bindings, id)).replace(/\.md$/, ''),
          size: childOccurrences(state, id).length,
        };
      }),
    [sub.version], // eslint-disable-line react-hooks/exhaustive-deps
  );
  const clusters = useMemo(() => layout(docs), [docs]);

  const searching = query.trim() !== '';
  const hits = useMemo(() => {
    if (!searching) return new Set<ChunkId>();
    const lit = new Set<ChunkId>();
    for (const id of searchChunks(state, indexes, query)) {
      lit.add(id);
      for (const occ of occurrencesOfChunk(state, id)) lit.add(occ.containerId); // illuminate the containing star
    }
    return lit;
  }, [query, indexes, searching, state]);

  const proposalCount = useMemo(() => {
    const counts = new Map<ChunkId, number>();
    for (const d of docs) counts.set(d.id, proposalsForDoc(state, d.id).length);
    return counts;
  }, [docs, state, sub.version]); // eslint-disable-line react-hooks/exhaustive-deps

  const agentTouched = useMemo(() => {
    const touched = new Set<ChunkId>();
    if (lens !== 'provenance') return touched;
    for (const d of docs) {
      for (const occ of childOccurrences(state, d.id)) {
        const kind = provenanceKind(state, occ.chunkId);
        if (kind === 'agent') touched.add(d.id);
      }
    }
    return touched;
  }, [lens, docs, state, sub.version]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="nebula">
      <header>
        <h1>substrate</h1>
        <input placeholder="search the workspace…" value={query} onChange={(e) => setQuery(e.target.value)} />
        <button className={lens === 'provenance' ? 'active' : ''} onClick={() => setLens(lens === 'provenance' ? 'none' : 'provenance')}>
          provenance lens
        </button>
        <button onClick={() => void sub.syncNow()} disabled={sub.busy}>
          {sub.busy ? 'syncing…' : 'sync files'}
        </button>
      </header>
      <svg viewBox={`0 0 ${W} ${H}`}>
        {/* Every light in the sky is a document. The nebula itself is made
            from the stars: each doc emits a halo, and turbulence smears the
            overlap into cloud — denser writing, denser nebula. */}
        <defs>
          <filter id="nebula" x="-80%" y="-80%" width="260%" height="260%">
            <feTurbulence type="fractalNoise" baseFrequency="0.012" numOctaves="3" seed="7" />
            <feDisplacementMap in="SourceGraphic" scale="120" />
            <feGaussianBlur stdDeviation="14" />
          </filter>
        </defs>
        {clusters.map(({ key, gx, gy, stars }) => (
          <g key={key}>
            <g className="cloud" filter="url(#nebula)">
              {stars.map(({ doc, x, y }) => (
                <circle key={doc.id} cx={x} cy={y} r={26 + Math.min(doc.size, 40) * 1.2} />
              ))}
            </g>
            <text className="cluster-label" x={gx} y={gy - 18 * Math.sqrt(stars.length) - 16}>
              {key}
            </text>
            {stars.map(({ doc, x, y }, j) => {
              const lit = hits.has(doc.id);
              const pending = proposalCount.get(doc.id) ?? 0;
              return (
                <g
                  key={doc.id}
                  className={`star ${lit ? 'lit' : ''} ${searching && !lit ? 'dim' : ''} ${agentTouched.has(doc.id) ? 'agentic' : ''}`}
                  onClick={() => onFocus(doc.id)}
                >
                  <circle cx={x} cy={y} r={4 + Math.min(doc.size, 10) * 0.7} style={{ animationDelay: `-${(j * 1.7) % 5}s` }} />
                  {pending > 0 && (
                    <text className="badge" x={x + 10} y={y - 8}>
                      {pending}
                    </text>
                  )}
                  <text x={x} y={y + 18}>
                    {doc.label}
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
