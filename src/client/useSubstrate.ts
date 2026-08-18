// The browser runs the same kernel as the server. Commits apply locally first
// (the UI is never blocked on the network), then post to the log; a refused
// commit means truth moved — reload it.

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

export function useSubstrate() {
  const [ws, setWs] = useState<Workspace | null>(null);
  const [version, setVersion] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const queue = useRef<Commit[]>([]);
  const posting = useRef(false);

  const load = async (endpoint = '/api/state', method = 'GET') => {
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
  }, []);

  const pump = async () => {
    if (posting.current) return;
    posting.current = true;
    while (queue.current.length > 0) {
      const commit = queue.current[0];
      try {
        const r = await fetch('/api/commits', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ commits: [commit] }),
        });
        if (!r.ok) {
          // Truth diverged; drop the local queue and refetch.
          queue.current = [];
          await load();
          break;
        }
        queue.current.shift();
      } catch {
        queue.current = [];
        await load();
        break;
      }
    }
    posting.current = false;
  };

  const ctx: TxCtx | null = useMemo(() => {
    if (!ws) return null;
    return {
      state: ws.state,
      actorId: HUMAN_ACTOR,
      onCommit: (commit) => {
        queue.current.push(commit);
        void pump();
        setVersion((v) => v + 1);
      },
    };
  }, [ws]);

  const syncNow = async () => {
    setBusy(true);
    await load('/api/sync', 'POST');
    setBusy(false);
  };

  return { ws, ctx, version, error, busy, syncNow, reload: () => load() };
}
