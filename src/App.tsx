import { useEffect, useState } from 'react';
import { useSubstrate } from './client/useSubstrate';
import {
  containerExists,
  containerForDocument,
  containerLabel,
  workspaceCrumbs,
  WORKSPACE_ROOT,
} from './client/workspaceTree';
import { Nebula } from './Nebula';
import { Star } from './Star';

export type SubstrateHook = ReturnType<typeof useSubstrate>;

// The proved loop: navigate (Nebula) → focus (Star) → compose → dispatch →
// integrate (proposals) → return. Truth lives in the kernel state; both
// surfaces are views over it.
export function App() {
  const sub = useSubstrate();
  const [focus, setFocus] = useState<string | null>(null);
  const [containerId, setContainerId] = useState(WORKSPACE_ROOT);

  useEffect(() => {
    if (sub.ws && !containerExists(sub.ws.sources, containerId)) {
      setContainerId(WORKSPACE_ROOT);
      setFocus(null);
    }
  }, [containerId, sub.version, sub.ws]);

  // A failed fetch before anything loaded is a dead server; after that, errors
  // surface as a banner — a healthy workspace is never unmounted over a blip.
  if (sub.error && !sub.ws)
    return (
      <div className="hint">
        Headspace host unreachable — start with: npm start
        <div className="meta">{sub.error}</div>
        <p>
          <button onClick={() => void sub.reload()}>retry</button>
        </p>
      </div>
    );
  if (!sub.ws || !sub.ctx) return <div className="hint">loading nebula…</div>;

  const banner = sub.error ?? sub.status;
  const identity = sub.ws.identity;
  const crumbs = workspaceCrumbs(sub.ws.sources, containerId, identity?.displayName ?? 'workspace');
  const counts = sub.ws.lastIngestion?.counts;
  const focusDoc = (docId: string) => {
    const home = containerForDocument(sub.ws!.sources, sub.ws!.bindings, docId);
    if (home !== null) setContainerId(home);
    setFocus(docId);
  };
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
      <header className="workspace-header">
        <div className="brand">headspace</div>
        <div className="workspace-identity">
          <span>{identity?.displayName ?? 'local workspace'}</span>
          <span className="meta" title={identity?.rootDisplayPath}>
            {identity?.rootDisplayPath ?? 'workspace root unavailable'}
          </span>
        </div>
        <nav className="breadcrumbs" aria-label="Workspace location">
          {crumbs.map((crumb, index) => (
            <span key={crumb.containerId}>
              {index > 0 && <span className="crumb-separator">/</span>}
              <button
                className={crumb.containerId === containerId && !focus ? 'active' : ''}
                onClick={() => {
                  setContainerId(crumb.containerId);
                  setFocus(null);
                }}
              >
                {crumb.label}
              </button>
            </span>
          ))}
        </nav>
        {counts && (
          <span className="ingestion-summary meta" title={`ingestion ${sub.ws.lastIngestion?.id}`}>
            {counts.failed > 0 ? `${counts.failed} failed` : counts.unsupported > 0 ? `${counts.unsupported} unsupported` : 'sources ready'}
          </span>
        )}
        {sub.pendingCount > 0 && (
          <span className="pending-summary meta">
            {sub.pendingCount} local change{sub.pendingCount === 1 ? '' : 's'} awaiting the durable host · keep this tab open
          </span>
        )}
        <button
          onClick={() => void sub.ingestNow()}
          disabled={sub.busy || sub.collaborating || sub.recoveringTruth || sub.truthUnknown}
        >
          {sub.truthUnknown
            ? 'truth unavailable'
            : sub.recoveringTruth
            ? 'reloading truth…'
            : sub.collaborating
              ? 'collaborator thinking…'
              : sub.busy
                ? 'ingesting…'
                : 'ingest sources'}
        </button>
      </header>
      {sub.truthUnknown ? (
        <main className="hint" aria-live="assertive">
          The server may have changed, but authoritative truth is not reachable yet. Editing and dispatch are paused.
          <p><button onClick={() => void sub.reload()}>retry authoritative reload</button></p>
        </main>
      ) : focus && sub.ws.state.chunks.has(focus) ? (
        <Star
          sub={sub}
          docId={focus}
          onFocusDoc={focusDoc}
          onBack={() => setFocus(null)}
          backLabel={containerLabel(sub.ws.sources, containerId, sub.ws.identity?.displayName ?? 'workspace')}
        />
      ) : (
        <Nebula
          sub={sub}
          containerId={containerId}
          onOpenContainer={setContainerId}
          onFocus={focusDoc}
        />
      )}
    </>
  );
}
