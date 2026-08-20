// The browser runs the same kernel as the server. Commits apply locally first
// (the UI is never blocked on the network), then post to the log. Failure
// semantics matter here: a network error means the server never saw the commit
// — keep it and retry; only a 409 means truth diverged — drop local commits
// and reload, reporting what was dropped.

import { useEffect, useMemo, useRef, useState } from 'react';
import { deserializeState } from '../kernel/serialize';
import type { SubstrateState } from '../kernel/state';
import type { TxCtx } from '../kernel/tx';
import type { Commit } from '../kernel/types';

export interface BindingInfo {
  docChunkId: string;
  relPath: string;
}

export interface Workspace {
  state: SubstrateState;
  bindings: BindingInfo[];
}

export const HUMAN_ACTOR = 'human:asa';

const RETRY_START_MS = 1000;
const RETRY_CAP_MS = 30000;

export function useSubstrate() {
  const [ws, setWs] = useState<Workspace | null>(null);
  const [version, setVersion] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null); // transient banner, never unmounts the UI
  const [busy, setBusy] = useState(false);
  const queue = useRef<Commit[]>([]);
  const posting = useRef(false);
  const retryDelay = useRef(RETRY_START_MS);
  const retryTimer = useRef<number | null>(null);

  const load = async (endpoint = '/api/state', method = 'GET') => {
    // Replacing the state while commits wait in the retry queue would fork the
    // screen from what later reaches the server: drain first, or refuse.
    if (queue.current.length > 0) {
      await pump();
      if (queue.current.length > 0) {
        setStatus(`cannot reload: ${queue.current.length} local change(s) not yet accepted by the server`);
        return null;
      }
    }
    try {
      const r = await fetch(endpoint, { method });
      if (!r.ok) throw new Error(`${endpoint}: ${r.status}`);
      const j = await r.json();
      setWs({ state: deserializeState(j.state), bindings: j.bindings ?? [] });
      setVersion((v) => v + 1);
      setError(null);
      return j;
    } catch (e) {
      setError(String(e));
      return null;
    }
  };

  useEffect(() => {
    void load();
    return () => {
      if (retryTimer.current != null) clearTimeout(retryTimer.current);
    };
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

  const pump = async () => {
    if (posting.current) return;
    posting.current = true;
    try {
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
          scheduleRetry('server unreachable — changes held locally, retrying…');
          return;
        }
        if (r.status === 409) {
          // Truth diverged. Local queued commits were built on a state the
          // server refused; drop them atomically with the reload — including
          // anything enqueued while the reload was in flight, which was built
          // on the state object the reload discards.
          const dropped = queue.current.length;
          queue.current = [];
          await load();
          queue.current = [];
          setStatus(`truth diverged — reloaded; ${dropped} local change(s) could not be kept`);
          return;
        }
        if (!r.ok) {
          scheduleRetry(`server error ${r.status} — changes held locally, retrying…`);
          return;
        }
        queue.current.shift();
        retryDelay.current = RETRY_START_MS;
        setStatus(null);
      }
    } finally {
      posting.current = false;
    }
  };

  const ctx: TxCtx | null = useMemo(() => {
    if (!ws) return null;
    return {
      state: ws.state,
      actorId: HUMAN_ACTOR,
      // Enqueue while the commit is still only validated: if this threw, the
      // kernel would not fold it, and the screen would never show a change the
      // server was never told about.
      onCommit: (commit) => {
        queue.current.push(commit);
      },
      // Folded now, so the render reads the state the commit produced.
      afterCommit: () => {
        void pump();
        setVersion((v) => v + 1);
      },
    };
  }, [ws]); // eslint-disable-line react-hooks/exhaustive-deps

  const syncNow = async () => {
    setBusy(true);
    await load('/api/sync', 'POST');
    setBusy(false);
  };

  return { ws, ctx, version, error, status, busy, syncNow, reload: () => load(), dismissStatus: () => setStatus(null) };
}
