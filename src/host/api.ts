// Dev-server seam: the same kernel runs in the browser and here; the client
// applies commits locally and posts them, the server replays them against its
// own materialized state (same validation) and appends them to the log. A
// commit that fails to replay is refused; the client refetches truth.

import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import type { IncomingMessage, ServerResponse } from 'node:http';
import type { Plugin } from 'vite';
import { blobHashOf } from '../kernel/hash';
import { serializeState } from '../kernel/serialize';
import { foldCommit, validateCommit } from '../kernel/state';
import type { Commit } from '../kernel/types';
import { writeProjection } from './markdown';
import { openWorkspace, type WorkspaceStore } from './store-fs';
import { syncWorkspace, type SyncReport } from './sync';

export interface SubstrateServerOptions {
  root?: string;
  contentDirs?: string[];
  contentFiles?: string[];
}

export interface BindingInfo {
  docChunkId: string;
  relPath: string;
}

function readBindings(root: string): BindingInfo[] {
  const out: BindingInfo[] = [];
  const walk = (dir: string) => {
    let entries;
    try {
      entries = readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const e of entries) {
      const full = join(dir, e.name);
      if (e.isDirectory()) walk(full);
      else if (e.name.endsWith('.json')) {
        try {
          const sc = JSON.parse(readFileSync(full, 'utf8'));
          if (sc.docChunkId && sc.relPath) out.push({ docChunkId: sc.docChunkId, relPath: sc.relPath });
        } catch {
          /* unreadable sidecar is reported via reconcile, not here */
        }
      }
    }
  };
  walk(join(root, '.substrate', 'sidecars'));
  return out;
}

const json = (res: ServerResponse, status: number, body: unknown) => {
  res.statusCode = status;
  res.setHeader('content-type', 'application/json');
  res.end(JSON.stringify(body));
};

const readBody = (req: IncomingMessage): Promise<string> =>
  new Promise((resolve, reject) => {
    let data = '';
    req.on('data', (c) => (data += c));
    req.on('end', () => resolve(data));
    req.on('error', reject);
  });

export function substrateServer(opts: SubstrateServerOptions = {}): Plugin {
  let ws: WorkspaceStore | null = null;
  const root = opts.root ?? process.cwd();

  // Stale locks (dead pids) are taken over inside openWorkspace; a live holder
  // is a real second writer and must not be steamrolled.
  const ensure = async (): Promise<WorkspaceStore> => {
    if (!ws) {
      ws = await openWorkspace(root);
      if (ws.state.commitCount === 0) await syncWorkspace(ws, opts);
    }
    return ws;
  };

  // One request at a time touches the workspace: handlers await inside their
  // bodies, and interleaving applyCommit/appendCommit would tear the log.
  let chain: Promise<unknown> = Promise.resolve();
  const exclusive = <T>(fn: () => Promise<T>): Promise<T> => {
    const next = chain.then(fn, fn);
    chain = next.catch(() => undefined);
    return next;
  };

  return {
    name: 'substrate-server',
    configureServer(server) {
      server.httpServer?.on('close', () => ws?.close());
      server.middlewares.use((req, res, next) => {
        const url = (req.url ?? '').split('?')[0];
        if (!url.startsWith('/api/')) return next();
        void exclusive(async () => {
          try {
            const w = await ensure();
            if (url === '/api/state' && req.method === 'GET') {
              return json(res, 200, { state: serializeState(w.state), bindings: readBindings(root) });
            }
            if (url === '/api/commits' && req.method === 'POST') {
              const { commits } = JSON.parse(await readBody(req)) as { commits: Commit[] };
              // Boundary checks the kernel cannot do synchronously: a commit
              // must extend the current head (conflicting advances are refused,
              // not silently interleaved), and every carried blob must hash to
              // its claimed address (immutable history is not rewritable).
              for (const c of commits) {
                if ((c.parentIds[0] ?? null) !== w.state.head) {
                  return json(res, 409, { error: `stale parent: commit ${c.id} was built on out-of-date state`, head: w.state.head });
                }
                for (const b of c.facts.blobs ?? []) {
                  if (b.hash !== (await blobHashOf(b.mediaType, b.text))) {
                    return json(res, 409, { error: `blob hash mismatch: ${b.hash}`, head: w.state.head });
                  }
                }
                try {
                  validateCommit(w.state, c); // replay = validation; same kernel, same invariants
                } catch (e) {
                  return json(res, 409, { error: String(e), head: w.state.head });
                }
                // Durable before it is visible: if the append throws, the 500
                // below reports a commit the server neither kept nor applied,
                // and the client's queue still holds it.
                w.appendCommit(c);
                foldCommit(w.state, c);
                w.snapshotIfDue();
              }
              return json(res, 200, { ok: true, head: w.state.head });
            }
            if (url === '/api/sync' && req.method === 'POST') {
              const report: SyncReport = await syncWorkspace(w, opts);
              return json(res, 200, { report, state: serializeState(w.state), bindings: readBindings(root) });
            }
            if (url === '/api/project' && req.method === 'POST') {
              const { relPath } = JSON.parse(await readBody(req)) as { relPath: string };
              await writeProjection(w.ctxFor('driver:fs'), { workspaceRoot: root, relPath });
              return json(res, 200, { ok: true });
            }
            return json(res, 404, { error: `no such endpoint: ${url}` });
          } catch (e) {
            return json(res, 500, { error: String(e) });
          }
        });
      });
    },
  };
}
