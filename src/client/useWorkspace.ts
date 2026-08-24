// The browser runs the same kernel as the host. Commits apply locally first
// (the UI is never blocked on the network), then post to the log. Failure
// semantics matter here: a network error means the host never saw the commit
// — keep it and retry; only a 409 means truth diverged — drop local commits
// and reload, reporting what was dropped.

import { useEffect, useMemo, useRef, useState } from 'react';
import { deserializeState } from '../kernel/serialize';
import type { WorkspaceGraph } from '../kernel/state';
import type { TxCtx } from '../kernel/tx';
import type { Commit } from '../kernel/types';
import type {
  CollaboratorCapability,
  CompletionRequest,
  CompletionResult,
} from '../collaboration/types';
import type {
  IngestionAdapterCapability,
  IngestionCatalog,
  IngestionItemResult,
  RepresentationRecord,
  SourceObservation,
  SourceRecord,
} from '../host/ingestion';

export interface BindingInfo {
  docChunkId: string;
  relPath: string;
  sourceId?: string;
  observationId?: string;
  mediaType?: string;
  adapterId?: string;
  adapterVersion?: string;
}

export interface WorkspaceIdentity {
  id: string | null;
  displayName: string;
  rootDisplayPath: string;
}

export interface SourceItemView {
  source: SourceRecord;
  observation: SourceObservation;
  representation: RepresentationRecord | null;
  lastResult: IngestionItemResult | null;
  parentSourceId: string | null;
  name: string;
  isWorkspaceRoot: boolean;
  presence: 'present' | 'missing' | 'unknown';
}

export interface Workspace {
  state: WorkspaceGraph;
  bindings: BindingInfo[];
  identity: WorkspaceIdentity | null;
  adapters: IngestionAdapterCapability[];
  collaborators: CollaboratorCapability[];
  sources: SourceItemView[];
  lastIngestion: IngestionCatalog['lastRun'];
}

export const HUMAN_ACTOR = 'human:asa';

const RETRY_START_MS = 1000;
const RETRY_CAP_MS = 30000;

// State replacement (reload/ingest) must never detach an in-flight proposal
// from the state object whose exact context it captured. The barrier is kept
// framework-free so its concurrency contract can be tested directly.
export class DispatchBarrier {
  private count = 0;
  private waiters = new Set<() => void>();

  get active(): number {
    return this.count;
  }

  enter(): () => void {
    this.count++;
    let released = false;
    return () => {
      if (released) return;
      released = true;
      this.count--;
      if (this.count !== 0) return;
      const pending = [...this.waiters];
      this.waiters.clear();
      for (const resolve of pending) resolve();
    };
  }

  wait(): Promise<void> {
    if (this.count === 0) return Promise.resolve();
    return new Promise((resolve) => this.waiters.add(resolve));
  }
}

export class SingleFlightDrain {
  private active: Promise<void> | null = null;

  get busy(): boolean {
    return this.active !== null;
  }

  run(work: () => Promise<void>): Promise<void> {
    if (this.active) return this.active;
    const active = Promise.resolve()
      .then(work)
      .finally(() => {
        if (this.active === active) this.active = null;
      });
    this.active = active;
    return active;
  }
}

export class StateReplacementMutex {
  private tail: Promise<void> = Promise.resolve();

  run<T>(work: () => Promise<T>): Promise<T> {
    const ready = this.tail;
    let release!: () => void;
    this.tail = new Promise<void>((resolve) => { release = resolve; });
    return (async () => {
      await ready;
      try {
        return await work();
      } finally {
        release();
      }
    })();
  }
}

// A rejected optimistic state is not merely "recovering" while one request is
// active: it remains unsafe until an authoritative snapshot has actually been
// installed. Keep that latch independent from React rendering so every write
// policy sees it synchronously.
export class TruthQuarantine {
  private value = false;

  get unknown(): boolean {
    return this.value;
  }

  quarantine(): void {
    this.value = true;
  }

  failedReplacement(method: string): boolean {
    if (method.toUpperCase() === 'GET' || method.toUpperCase() === 'HEAD') return false;
    this.quarantine();
    return true;
  }

  restore(): void {
    this.value = false;
  }
}

export function preventPendingUnload(event: BeforeUnloadEvent): void {
  event.preventDefault();
  event.returnValue = '';
}

export function useWorkspace() {
  const [ws, setWs] = useState<Workspace | null>(null);
  const [version, setVersion] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null); // transient banner, never unmounts the UI
  const [busy, setBusy] = useState(false);
  const [collaborating, setCollaborating] = useState(false);
  const [recoveringTruth, setRecoveringTruth] = useState(false);
  const [truthUnknown, setTruthUnknown] = useState(false);
  const [pendingCount, setPendingCount] = useState(0);
  const queue = useRef<Commit[]>([]);
  const retryDelay = useRef(RETRY_START_MS);
  const retryTimer = useRef<number | null>(null);
  const dispatchBarrier = useRef<DispatchBarrier | null>(null);
  if (!dispatchBarrier.current) dispatchBarrier.current = new DispatchBarrier();
  const commitDrain = useRef<SingleFlightDrain | null>(null);
  if (!commitDrain.current) commitDrain.current = new SingleFlightDrain();
  const divergenceRecovery = useRef<Promise<void> | null>(null);
  const replacementMutex = useRef<StateReplacementMutex | null>(null);
  if (!replacementMutex.current) replacementMutex.current = new StateReplacementMutex();
  const replacementRequests = useRef(0);
  const truthQuarantine = useRef<TruthQuarantine | null>(null);
  if (!truthQuarantine.current) truthQuarantine.current = new TruthQuarantine();

  const withStateReplacement = <T,>(work: () => Promise<T>): Promise<T> => {
    // Increment before the first await: no dispatch or local commit can start
    // in the gap between requesting replacement and acquiring the mutex.
    replacementRequests.current++;
    return replacementMutex.current!.run(work).finally(() => {
      replacementRequests.current--;
    });
  };

  const fetchWorkspace = async (endpoint = '/api/state', method = 'GET') => {
    try {
      const r = await fetch(endpoint, { method });
      if (!r.ok) throw new Error(`${endpoint}: ${r.status}`);
      const j = await r.json();
      setWs({
        state: deserializeState(j.state),
        bindings: j.bindings ?? [],
        identity: j.workspace ?? null,
        adapters: j.adapters ?? [],
        collaborators: j.collaborators ?? [],
        sources: j.sources ?? [],
        lastIngestion: j.lastIngestion ?? null,
      });
      setVersion((v) => v + 1);
      setError(null);
      setStatus(null);
      truthQuarantine.current!.restore();
      setTruthUnknown(false);
      return j;
    } catch (e) {
      // A mutating endpoint may have committed before its response was lost or
      // malformed. The old browser snapshot is therefore unsafe until a plain
      // authoritative GET succeeds.
      if (truthQuarantine.current!.failedReplacement(method)) {
        setTruthUnknown(true);
        setStatus('host mutation outcome is unknown — reload authoritative truth before editing');
      }
      setError(String(e));
      return null;
    }
  };

  const load = (endpoint = '/api/state', method = 'GET') =>
    withStateReplacement(async () => {
      if (dispatchBarrier.current!.active > 0) {
        setStatus('waiting for the active collaborator before replacing workspace state…');
        await dispatchBarrier.current!.wait();
        setStatus(null);
      }
      // Replacing the state while commits wait in the retry queue would fork the
      // screen from what later reaches the host: drain first, or refuse.
      if (queue.current.length > 0) {
        await pump();
        // A 409 schedules authoritative recovery behind this mutex. Return so
        // it can acquire the gate; awaiting it here would deadlock.
        if (divergenceRecovery.current) return null;
        if (queue.current.length > 0) {
          setStatus(`cannot reload: ${queue.current.length} local change(s) not yet accepted by the host`);
          return null;
        }
      }
      if (divergenceRecovery.current) return null;
      return fetchWorkspace(endpoint, method);
    });

  useEffect(() => {
    void load();
    return () => {
      if (retryTimer.current != null) clearTimeout(retryTimer.current);
    };
  }, []);

  useEffect(() => {
    // Keep one stable listener so protection begins synchronously with the
    // queue mutation rather than waiting for React to render `pendingCount`.
    const guard = (event: BeforeUnloadEvent): void => {
      if (queue.current.length > 0 || dispatchBarrier.current!.active > 0) preventPendingUnload(event);
    };
    window.addEventListener('beforeunload', guard);
    return () => window.removeEventListener('beforeunload', guard);
  }, []);

  const scheduleRetry = (msg: string) => {
    setStatus(msg);
    if (retryTimer.current != null) return;
    const delay = retryDelay.current;
    retryDelay.current = Math.min(delay * 2, RETRY_CAP_MS);
    retryTimer.current = window.setTimeout(() => {
      retryTimer.current = null;
      void pump();
    }, delay);
  };

  const recoverFromDivergence = (initialDropped: number): Promise<void> => {
    if (divergenceRecovery.current) return divergenceRecovery.current;
    truthQuarantine.current!.quarantine();
    setTruthUnknown(true);
    setRecoveringTruth(true);
    setStatus('truth diverged — waiting to reload authoritative state…');
    const recovery = withStateReplacement(async () => {
        await dispatchBarrier.current!.wait();

        // Anything produced while the rejected post and an in-flight dispatch
        // overlapped was built on the same detached state and cannot be retried.
        let dropped = initialDropped + queue.current.length;
        queue.current = [];
        setPendingCount(0);
        if (retryTimer.current != null) {
          clearTimeout(retryTimer.current);
          retryTimer.current = null;
        }

        const loaded = await fetchWorkspace();
        dropped += queue.current.length;
        queue.current = [];
        setPendingCount(0);
        if (loaded) {
          setStatus(`truth diverged — reloaded; ${dropped} local change(s) could not be kept`);
        }
        setRecoveringTruth(false);
      });
    divergenceRecovery.current = recovery;
    void recovery.then(
      () => {
        if (divergenceRecovery.current === recovery) divergenceRecovery.current = null;
      },
      () => {
        if (divergenceRecovery.current === recovery) divergenceRecovery.current = null;
        setRecoveringTruth(false);
      },
    );
    return recovery;
  };

  const pump = (): Promise<void> =>
    commitDrain.current!.run(async () => {
      while (queue.current.length > 0) {
        const commit = queue.current[0];
        let r: Response;
        try {
          r = await fetch('/api/commits', {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify({ commits: [commit] }),
          });
        } catch {
          scheduleRetry('host unreachable — changes held locally, retrying…');
          return;
        }
        if (r.status === 409) {
          // Truth diverged. Local queued commits were built on a state the
          // host refused; drop them atomically with the reload — including
          // anything enqueued while the reload was in flight, which was built
          // on the state object the reload discards.
          const dropped = queue.current.length;
          queue.current = [];
          setPendingCount(0);
          // Reload outside this active drain. Calling load() here could make
          // the drain await itself if a provider finishes during divergence.
          void recoverFromDivergence(dropped);
          return;
        }
        if (!r.ok) {
          scheduleRetry(`host error ${r.status} — changes held locally, retrying…`);
          return;
        }
        queue.current.shift();
        setPendingCount(queue.current.length);
        retryDelay.current = RETRY_START_MS;
        setStatus(null);
      }
    });

  const ctx: TxCtx | null = useMemo(() => {
    if (!ws) return null;
    return {
      state: ws.state,
      actorId: HUMAN_ACTOR,
      // A writer waits for an already-running dispatch reader, so work under
      // that reader may finish and be drained. Divergence is different: the
      // host rejected this state, so even an existing dispatch is fenced.
      policy: () =>
        !truthQuarantine.current!.unknown &&
        divergenceRecovery.current === null &&
        replacementRequests.current === 0,
      // Enqueue while the commit is still only validated: if this threw, the
      // kernel would not fold it, and the screen would never show a change the
      // host was never told about.
      onCommit: (commit) => {
        queue.current.push(commit);
        setPendingCount(queue.current.length);
      },
      // Folded now, so the render reads the state the commit produced.
      afterCommit: () => {
        void pump();
        setVersion((v) => v + 1);
      },
    };
  }, [ws]); // eslint-disable-line react-hooks/exhaustive-deps

  const ingestNow = async () => {
    setBusy(true);
    try {
      // Never turn a recovery retry into a mutating ingestion request.
      if (truthQuarantine.current!.unknown) await load();
      else await load('/api/ingest', 'POST');
    } finally {
      setBusy(false);
    }
  };

  const complete = async (request: CompletionRequest): Promise<CompletionResult> => {
    const response = await fetch('/api/complete', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(request),
    });
    let body: unknown = null;
    try {
      body = await response.json();
    } catch {
      // The status still gives an actionable boundary error below.
    }
    if (!response.ok) {
      const message = body && typeof body === 'object' && typeof (body as { error?: unknown }).error === 'string'
        ? (body as { error: string }).error
        : `collaborator request failed: HTTP ${response.status}`;
      throw new Error(message);
    }
    if (!body || typeof body !== 'object' || typeof (body as { text?: unknown }).text !== 'string') {
      throw new Error('collaborator returned an invalid completion result');
    }
    return body as CompletionResult;
  };

  const runDispatch = async <T,>(work: (dispatchCtx: TxCtx) => Promise<T>): Promise<T> => {
    if (!ctx) throw new Error('workspace is not loaded');
    if (truthQuarantine.current!.unknown || divergenceRecovery.current || replacementRequests.current > 0) {
      throw new Error('workspace truth is reloading; retry dispatch after recovery');
    }
    const release = dispatchBarrier.current!.enter();
    let active = true;
    // This context belongs only to the dispatch that already crossed the
    // replacement fence. A normal reload waits for it; unrelated UI writes use
    // the stricter shared context above and are rejected while that reload is
    // pending. Divergence quarantines both contexts immediately.
    const dispatchCtx: TxCtx = {
      ...ctx,
      policy: () =>
        active &&
        !truthQuarantine.current!.unknown &&
        divergenceRecovery.current === null,
    };
    setCollaborating(true);
    try {
      return await work(dispatchCtx);
    } finally {
      active = false;
      release();
      setCollaborating(dispatchBarrier.current!.active > 0);
    }
  };

  return {
    ws,
    ctx,
    version,
    error,
    status,
    busy,
    collaborating,
    recoveringTruth,
    truthUnknown,
    pendingCount,
    ingestNow,
    complete,
    runDispatch,
    reload: () => load(),
    dismissStatus: () => setStatus(null),
  };
}
