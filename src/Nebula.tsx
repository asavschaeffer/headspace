import { useEffect, useMemo, useState } from 'react';
import type { WorkspaceSession } from './App';
import { ancestorContainers, docList, labelOf, proposalsForDoc } from './client/helpers';
import {
  WORKSPACE_ROOT,
  ancestorDirectoryIds,
  containerLabel,
  workspaceChildren,
  type WorkspaceContainerId,
  type WorkspaceSourceNode,
} from './client/workspaceTree';
import { buildIndexes, provenanceKind, searchChunks } from './index/indexes';
import { childOccurrences } from './kernel/state';
import type { ChunkId } from './kernel/types';

const W = 1000;
const H = 650;
const GOLDEN_ANGLE = 2.399963;

interface SkyDocument {
  id: ChunkId;
  label: string;
  path?: string;
  size: number;
  status?: WorkspaceSourceNode['status'];
  source?: WorkspaceSourceNode;
}

const documentPoints = (docs: SkyDocument[]) =>
  [...docs]
    .sort((a, b) => a.label.localeCompare(b.label))
    .map((doc, index) => {
      const radius = 25 * Math.sqrt(index + 0.35);
      const angle = index * GOLDEN_ANGLE - Math.PI / 2;
      return { doc, x: W / 2 + radius * Math.cos(angle), y: H / 2 + radius * Math.sin(angle) };
    });

const portalPoints = (directories: WorkspaceSourceNode[]) =>
  directories.map((directory, index) => {
    const angle = (2 * Math.PI * index) / Math.max(directories.length, 1) - Math.PI / 2;
    const radius = directories.length === 1 ? 225 : 255;
    return {
      directory,
      x: W / 2 + radius * Math.cos(angle),
      y: H / 2 + radius * Math.sin(angle),
    };
  });

const issuePoints = (issues: WorkspaceSourceNode[]) =>
  issues.map((issue, index) => {
    const angle = index * GOLDEN_ANGLE + Math.PI / 2;
    const radius = 170 + 12 * Math.sqrt(index);
    return { issue, x: W / 2 + radius * Math.cos(angle), y: H / 2 + radius * Math.sin(angle) };
  });

const keyboardActivate = (event: React.KeyboardEvent<SVGGElement>, action: () => void) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    action();
  }
};

export function Nebula({
  session,
  containerId,
  onOpenContainer,
  onFocus,
}: {
  session: WorkspaceSession;
  containerId: WorkspaceContainerId;
  onOpenContainer: (id: WorkspaceContainerId) => void;
  onFocus: (id: ChunkId) => void;
}) {
  const { state, bindings, sources } = session.ws!;
  const [query, setQuery] = useState('');
  const [lens, setLens] = useState<'none' | 'provenance'>('none');
  const [selectedIssue, setSelectedIssue] = useState<string | null>(null);

  useEffect(() => setSelectedIssue(null), [containerId]);

  const indexes = useMemo(() => buildIndexes(state), [session.version]); // eslint-disable-line react-hooks/exhaustive-deps
  const children = useMemo(() => workspaceChildren(sources, containerId), [sources, containerId, session.version]); // eslint-disable-line react-hooks/exhaustive-deps
  const directories = children.filter((node) => node.kind === 'directory');
  const represented = children.filter((node) => node.docChunkId);
  const issues = children.filter((node) => node.kind !== 'directory' && !node.docChunkId);
  const representedEverywhere = useMemo(
    () => new Set(sources.flatMap((source) => (source.representation ? [source.representation.rootChunkId] : []))),
    [sources],
  );
  const generated =
    containerId === WORKSPACE_ROOT
      ? docList(state, bindings).filter((id) => !representedEverywhere.has(id))
      : [];
  const docs: SkyDocument[] = [
    ...represented.map((node) => ({
      id: node.docChunkId!,
      label: node.label,
      path: node.path,
      size: childOccurrences(state, node.docChunkId!).length,
      status: node.status,
      source: node,
    })),
    ...generated.map((id) => ({
      id,
      label: labelOf(state, bindings, id),
      size: childOccurrences(state, id).length,
    })),
  ];
  const stars = useMemo(() => documentPoints(docs), [docs]);
  const portals = useMemo(() => portalPoints(directories), [directories]);
  const issueBodies = useMemo(() => issuePoints(issues), [issues]);

  const searching = query.trim() !== '';
  const hits = useMemo(() => {
    if (!searching) return new Set<ChunkId>();
    const lit = new Set<ChunkId>();
    for (const id of searchChunks(state, indexes, query)) {
      lit.add(id);
      for (const container of ancestorContainers(state, id)) lit.add(container);
    }
    return lit;
  }, [query, indexes, searching, state]);
  const normalizedQuery = query.trim().toLowerCase();
  const litDirectoryIds = useMemo(() => ancestorDirectoryIds(sources, hits), [sources, hits]);

  const proposalCount = useMemo(() => {
    const counts = new Map<ChunkId, number>();
    for (const doc of docs) counts.set(doc.id, proposalsForDoc(state, doc.id).length);
    return counts;
  }, [docs, state, session.version]); // eslint-disable-line react-hooks/exhaustive-deps

  const agentTouched = useMemo(() => {
    const touched = new Set<ChunkId>();
    if (lens !== 'provenance') return touched;
    for (const doc of docs) {
      for (const occurrence of childOccurrences(state, doc.id)) {
        if (provenanceKind(state, occurrence.chunkId) === 'agent') touched.add(doc.id);
      }
    }
    return touched;
  }, [lens, docs, state, session.version]); // eslint-disable-line react-hooks/exhaustive-deps

  const selected = children.find((child) => child.sourceId === selectedIssue) ?? null;
  const currentLabel = containerLabel(sources, containerId, session.ws!.identity?.displayName ?? 'workspace');

  return (
    <main className="nebula">
      <div className="nebula-tools">
        <div>
          <h2>{currentLabel}</h2>
          <div className="meta">
            {directories.length} container(s) · {docs.length} document(s) · {issues.length} source issue(s)
          </div>
        </div>
        <input
          aria-label="Search the workspace"
          placeholder="search the workspace…"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
        <button
          className={lens === 'provenance' ? 'active' : ''}
          onClick={() => setLens(lens === 'provenance' ? 'none' : 'provenance')}
        >
          provenance lens
        </button>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} aria-label={`${currentLabel} spatial contents`}>
        <defs>
          <filter id="nebula-cloud" x="-80%" y="-80%" width="260%" height="260%">
            <feTurbulence type="fractalNoise" baseFrequency="0.012" numOctaves="3" seed="7" />
            <feDisplacementMap in="SourceGraphic" scale="120" />
            <feGaussianBlur stdDeviation="14" />
          </filter>
        </defs>
        <g className="cloud" filter="url(#nebula-cloud)">
          {stars.map(({ doc, x, y }) => (
            <circle key={doc.id} cx={x} cy={y} r={26 + Math.min(doc.size, 40) * 1.2} />
          ))}
        </g>

        {portals.map(({ directory, x, y }) => {
          const childCount = sources.filter((source) => source.parentSourceId === directory.sourceId).length;
          const lit =
            litDirectoryIds.has(directory.sourceId) ||
            (normalizedQuery !== '' && directory.label.toLowerCase().includes(normalizedQuery));
          return (
            <g
              key={directory.key}
              className={`portal ${lit ? 'lit' : ''} ${searching && !lit ? 'dim' : ''}`}
              role="button"
              tabIndex={0}
              aria-label={`Open ${directory.label} container`}
              onClick={() => onOpenContainer(directory.sourceId)}
              onKeyDown={(event) => keyboardActivate(event, () => onOpenContainer(directory.sourceId))}
            >
              <circle cx={x} cy={y} r={42} />
              <circle className="portal-core" cx={x} cy={y} r={7} />
              <text x={x} y={y + 62}>{directory.label}</text>
              <text className="portal-count" x={x} y={y + 78}>{childCount} immediate</text>
            </g>
          );
        })}

        {stars.map(({ doc, x, y }, index) => {
          const lit = hits.has(doc.id) || (normalizedQuery !== '' && doc.label.toLowerCase().includes(normalizedQuery));
          const pending = proposalCount.get(doc.id) ?? 0;
          const sourceNotice = Boolean(
            doc.source &&
              (doc.status === 'failed' || doc.status === 'missing' || doc.source.diagnostics.length > 0),
          );
          const sourceNoticeLabel =
            doc.status === 'failed'
              ? 'source refresh failed'
              : doc.status === 'missing'
                ? 'source missing from latest ingestion'
                : sourceNotice
                  ? 'source has an ingestion warning'
                  : null;
          return (
            <g
              key={doc.id}
              className={`star ${lit ? 'lit' : ''} ${searching && !lit ? 'dim' : ''} ${agentTouched.has(doc.id) ? 'agentic' : ''} ${sourceNotice ? 'source-warning' : ''}`}
              role="button"
              tabIndex={0}
              aria-label={`Open ${doc.label}${sourceNoticeLabel ? `; ${sourceNoticeLabel}` : ''}`}
              onClick={() => onFocus(doc.id)}
              onKeyDown={(event) => keyboardActivate(event, () => onFocus(doc.id))}
            >
              <title>{`${doc.path ?? doc.label}${sourceNoticeLabel ? ` — ${sourceNoticeLabel}` : ''}`}</title>
              <circle cx={x} cy={y} r={5 + Math.min(doc.size, 10) * 0.7} style={{ animationDelay: `-${(index * 1.7) % 5}s` }} />
              {pending > 0 && <text className="badge" x={x + 11} y={y - 9}>{pending}</text>}
              {sourceNotice && doc.source && (
                <g
                  className="source-failure-badge"
                  role="button"
                  tabIndex={0}
                  aria-label={`Inspect source status for ${doc.label}`}
                  onClick={(event) => {
                    event.stopPropagation();
                    setSelectedIssue(doc.source!.sourceId);
                  }}
                  onKeyDown={(event) => {
                    event.stopPropagation();
                    keyboardActivate(event, () => setSelectedIssue(doc.source!.sourceId));
                  }}
                >
                  <circle cx={x + 14} cy={y + 10} r={7} />
                  <text x={x + 14} y={y + 13}>!</text>
                </g>
              )}
              <text x={x} y={y + 20}>{doc.label}</text>
            </g>
          );
        })}

        {issueBodies.map(({ issue, x, y }) => {
          const lit = normalizedQuery !== '' && issue.label.toLowerCase().includes(normalizedQuery);
          return (
            <g
              key={issue.key}
              className={`source-node ${issue.status} ${lit ? 'lit' : ''}`}
              role="button"
              tabIndex={0}
              aria-label={`Inspect ${issue.label}, ${issue.status}`}
              onClick={() => setSelectedIssue(issue.sourceId)}
              onKeyDown={(event) => keyboardActivate(event, () => setSelectedIssue(issue.sourceId))}
            >
              <circle cx={x} cy={y} r={10} />
              <text className="source-mark" x={x} y={y + 4}>{issue.status === 'failed' ? '!' : '?'}</text>
              <text x={x} y={y + 27}>{issue.label}</text>
            </g>
          );
        })}

        {children.length === 0 && generated.length === 0 && (
          <text className="empty-sky" x={W / 2} y={H / 2}>this container is empty</text>
        )}
      </svg>

      {selected && (
        <SourceStatusPanel
          source={selected}
          busy={session.busy}
          onRetry={() => void session.ingestNow()}
          onClose={() => setSelectedIssue(null)}
        />
      )}
    </main>
  );
}

export function SourceStatusPanel({
  source,
  busy,
  onRetry,
  onClose,
}: {
  source: WorkspaceSourceNode;
  busy: boolean;
  onRetry: () => void;
  onClose: () => void;
}) {
  return (
    <aside className="source-status" aria-live="polite">
      <div className="source-status-heading">
        <div>
          <div className="meta">{source.status} source</div>
          <h3>{source.label}</h3>
        </div>
        <button onClick={onClose} aria-label="Close source details">×</button>
      </div>
      <dl>
        <dt>path</dt><dd>{source.path}</dd>
        <dt>media</dt><dd>{source.mediaType}</dd>
        <dt>size</dt><dd>{source.sizeBytes.toLocaleString()} bytes</dd>
        <dt>adapter</dt><dd>{source.adapterLabel ?? 'no adapter available'}</dd>
        <dt>link</dt><dd>{source.symlinkStatus}</dd>
      </dl>
      {source.diagnostics.length > 0 && (
        <ul>{source.diagnostics.map((message) => <li key={message}>{message}</li>)}</ul>
      )}
      <button onClick={onRetry} disabled={busy}>{busy ? 'ingesting…' : 'retry ingestion'}</button>
    </aside>
  );
}
