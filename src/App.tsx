import { useEffect, useState } from 'react';
import { useWorkspace } from './client/useWorkspace';
import {
  containerExists,
  containerForDocument,
  containerLabel,
  workspaceCrumbs,
  WORKSPACE_ROOT,
} from './client/workspaceTree';
import { Nebula } from './Nebula';
import { Star } from './Star';

export type WorkspaceSession = ReturnType<typeof useWorkspace>;

// The proved loop: navigate (Nebula) → focus (Star) → compose → dispatch →
// integrate (proposals) → return. Truth lives in the kernel state; both
// surfaces are views over it.
export function App() {
  const session = useWorkspace();
  const [focus, setFocus] = useState<string | null>(null);
  const [containerId, setContainerId] = useState(WORKSPACE_ROOT);

  useEffect(() => {
    if (session.ws && !containerExists(session.ws.sources, containerId)) {
      setContainerId(WORKSPACE_ROOT);
      setFocus(null);
    }
  }, [containerId, session.version, session.ws]);

  // A failed fetch before anything loaded is an unreachable host; after that, errors
  // surface as a banner — a healthy workspace is never unmounted over a blip.
  if (session.error && !session.ws)
    return (
      <div className="hint">
        Headspace host unreachable — start with: npm start
        <div className="meta">{session.error}</div>
        <p>
          <button onClick={() => void session.reload()}>retry</button>
        </p>
      </div>
    );
  if (!session.ws || !session.ctx) return <div className="hint">loading nebula…</div>;

  const banner = session.error ?? session.status;
  const identity = session.ws.identity;
  const crumbs = workspaceCrumbs(session.ws.sources, containerId, identity?.displayName ?? 'workspace');
  const counts = session.ws.lastIngestion?.counts;
  const focusDoc = (docId: string) => {
    const home = containerForDocument(session.ws!.sources, session.ws!.bindings, docId);
    if (home !== null) setContainerId(home);
    setFocus(docId);
  };
  return (
    <>
      {banner && (
        <div className="banner">
          {banner}
          {session.error ? (
            <button onClick={() => void session.reload()}>retry</button>
          ) : (
            <button onClick={session.dismissStatus}>dismiss</button>
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
          <span className="ingestion-summary meta" title={`ingestion ${session.ws.lastIngestion?.id}`}>
            {counts.failed > 0 ? `${counts.failed} failed` : counts.unsupported > 0 ? `${counts.unsupported} unsupported` : 'sources ready'}
          </span>
        )}
        {session.pendingCount > 0 && (
          <span className="pending-summary meta">
            {session.pendingCount} local change{session.pendingCount === 1 ? '' : 's'} awaiting the durable host · keep this tab open
          </span>
        )}
        <button
          onClick={() => void session.ingestNow()}
          disabled={session.busy || session.collaborating || session.recoveringTruth || session.truthUnknown}
        >
          {session.truthUnknown
            ? 'truth unavailable'
            : session.recoveringTruth
            ? 'reloading truth…'
            : session.collaborating
              ? 'collaborator thinking…'
              : session.busy
                ? 'ingesting…'
                : 'ingest sources'}
        </button>
      </header>
      {session.truthUnknown ? (
        <main className="hint" aria-live="assertive">
          The host may have changed, but authoritative truth is not reachable yet. Editing and dispatch are paused.
          <p><button onClick={() => void session.reload()}>retry authoritative reload</button></p>
        </main>
      ) : focus && session.ws.state.chunks.has(focus) ? (
        <Star
          session={session}
          docId={focus}
          onFocusDoc={focusDoc}
          onBack={() => setFocus(null)}
          backLabel={containerLabel(session.ws.sources, containerId, session.ws.identity?.displayName ?? 'workspace')}
        />
      ) : (
        <Nebula
          session={session}
          containerId={containerId}
          onOpenContainer={setContainerId}
          onFocus={focusDoc}
        />
      )}
    </>
  );
}
