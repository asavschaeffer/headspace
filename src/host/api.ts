// Dev-server seam: the same kernel runs in the browser and here; the client
// applies commits locally and posts them, the server replays them against its
// own materialized state (same validation) and appends them to the log. A
// commit that fails to replay is refused; the client refetches truth.

import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import type { IncomingMessage, ServerResponse } from 'node:http';
import type { Plugin } from 'vite';
import { serializeState } from '../kernel/serialize';
import { applyCommit } from '../kernel/state';
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

  const ensure = async (): Promise<WorkspaceStore> => {
    if (!ws) {
      ws = await openWorkspace(root, { force: true });
      if (ws.state.commitCount === 0) await syncWorkspace(ws, opts);
    }
    return ws;
  };

  return {
    name: 'substrate-server',
    configureServer(server) {
      server.httpServer?.on('close', () => ws?.close());
      server.middlewares.use(async (req, res, next) => {
        const url = (req.url ?? '').split('?')[0];
        if (!url.startsWith('/api/')) return next();
        try {
          const w = await ensure();
          if (url === '/api/state' && req.method === 'GET') {
            return json(res, 200, { state: serializeState(w.state), bindings: readBindings(root) });
          }
          if (url === '/api/commits' && req.method === 'POST') {
            const { commits } = JSON.parse(await readBody(req)) as { commits: Commit[] };
            try {
              for (const c of commits) {
                applyCommit(w.state, c); // replay = validation; same kernel, same invariants
                w.appendCommit(c);
              }
            } catch (e) {
              return json(res, 409, { error: String(e), head: w.state.head });
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
    },
  };
}
