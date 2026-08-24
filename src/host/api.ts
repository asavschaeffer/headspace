// Dev-server seam: the same kernel runs in the browser and here; the client
// applies commits locally and posts them, the server replays them against its
// own materialized state (same validation) and appends them to the log. A
// commit that fails to replay is refused; the client refetches truth.

import { readdirSync, readFileSync } from 'node:fs';
import { basename, join } from 'node:path';
import type { IncomingMessage, ServerResponse } from 'node:http';
import type { Plugin } from 'vite';
import { OFFLINE_COLLABORATOR } from '../collaboration/stub';
import { blobHashOf } from '../kernel/hash';
import { serializeState } from '../kernel/serialize';
import { foldCommit, validateCommit } from '../kernel/state';
import type { Commit } from '../kernel/types';
import {
  CollaboratorError,
  defaultCollaboratorAdapters,
  dispatchToCollaborator,
  type CollaboratorAdapter,
} from './collaborators';
import {
  ingestionAdapterCapabilities,
  readIngestionCatalog,
  type IngestionCatalog,
  type IngestionItemResult,
  type RepresentationRecord,
  type SourceObservation,
  type SourceRecord,
} from './ingestion';
import { ProjectionConflictError, writeProjection } from './markdown';
import { openWorkspace, type WorkspaceStore } from './store-fs';
import { syncWorkspace, type SyncReport } from './sync';

export interface SubstrateServerOptions {
  root?: string;
  contentDirs?: string[];
  contentFiles?: string[];
  collaborators?: CollaboratorAdapter[];
}

export interface BindingInfo {
  docChunkId: string;
  relPath: string;
  sourceId?: string;
  observationId?: string;
  mediaType?: string;
  adapterId?: string;
  adapterVersion?: string;
}

function readBindings(root: string): BindingInfo[] {
  const out = new Map<string, BindingInfo>();
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
          if (sc.docChunkId && sc.relPath) out.set(sc.docChunkId, { docChunkId: sc.docChunkId, relPath: sc.relPath });
        } catch {
          /* unreadable sidecar is reported via reconcile, not here */
        }
      }
    }
  };
  walk(join(root, '.substrate', 'sidecars'));
  const catalog = readIngestionCatalog(root);
  if (catalog) {
    for (const source of catalog.sources) {
      if (!source.currentRepresentationId) continue;
      const representation = catalog.representations.find((item) => item.id === source.currentRepresentationId);
      if (!representation) continue;
      out.set(representation.rootChunkId, {
        docChunkId: representation.rootChunkId,
        relPath: source.currentRelPath,
        sourceId: source.id,
        observationId: representation.observationId,
        mediaType: representation.mediaType,
        adapterId: representation.adapter.id,
        adapterVersion: representation.adapter.version,
      });
    }
  }
  return [...out.values()].sort((a, b) => a.relPath.localeCompare(b.relPath));
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

function sourceViews(catalog: IngestionCatalog | null): SourceItemView[] {
  if (!catalog) return [];
  const current = catalog.sources.flatMap((source) => {
    const observation = catalog.observations.find((item) => item.id === source.currentObservationId);
    if (!observation) return [];
    return [{ source, observation }];
  });
  const normalized = (path: string): string => path.replaceAll('\\', '/').replace(/^\/+|\/+$/g, '') === '.'
    ? ''
    : path.replaceAll('\\', '/').replace(/^\/+|\/+$/g, '');
  const directories = current.filter((item) => item.observation.kind === 'directory');
  return current.map(({ source, observation }) => {
    const representation = source.currentRepresentationId
      ? (catalog.representations.find((item) => item.id === source.currentRepresentationId) ?? null)
      : null;
    const lastResult = catalog.lastRun?.items.find((item) => item.observation.sourceId === source.id) ?? null;
    const path = normalized(observation.relPath);
    const parent = directories
      .filter((candidate) => {
        const directoryPath = normalized(candidate.observation.relPath);
        return directoryPath && directoryPath !== path && path.startsWith(`${directoryPath}/`);
      })
      .sort(
        (a, b) => normalized(b.observation.relPath).length - normalized(a.observation.relPath).length,
      )[0];
    return {
      source,
      observation,
      representation,
      lastResult,
      parentSourceId: parent?.source.id ?? null,
      name: path ? path.slice(path.lastIndexOf('/') + 1) : '.',
      isWorkspaceRoot: observation.kind === 'directory' && path === '',
      presence: catalog.lastRun === null ? 'unknown' : lastResult ? 'present' : 'missing',
    };
  });
}

export function workspacePayload(
  root: string,
  ws: WorkspaceStore,
  collaborators: CollaboratorAdapter[] = defaultCollaboratorAdapters(),
) {
  const catalog = readIngestionCatalog(root);
  return {
    state: serializeState(ws.state),
    bindings: readBindings(root),
    workspace: {
      id: catalog?.workspaceId ?? null,
      displayName: basename(root) || root,
      rootDisplayPath: root,
    },
    adapters: ingestionAdapterCapabilities(),
    collaborators: [OFFLINE_COLLABORATOR, ...collaborators.map((adapter) => adapter.capability)],
    sources: sourceViews(catalog),
    lastIngestion: catalog?.lastRun ?? null,
  };
}

const json = (res: ServerResponse, status: number, body: unknown) => {
  res.statusCode = status;
  res.setHeader('content-type', 'application/json');
  res.end(JSON.stringify(body));
};

class BodyTooLargeError extends Error {}

const readBody = (req: IncomingMessage, maxBytes = Number.POSITIVE_INFINITY): Promise<string> =>
  new Promise((resolve, reject) => {
    let data = '';
    let bytes = 0;
    let tooLarge = false;
    req.on('data', (c: Buffer | string) => {
      if (tooLarge) return;
      bytes += typeof c === 'string' ? Buffer.byteLength(c) : c.byteLength;
      if (bytes > maxBytes) {
        tooLarge = true;
        reject(new BodyTooLargeError(`request body exceeds ${maxBytes} bytes`));
        return;
      }
      data += c;
    });
    req.on('end', () => {
      if (!tooLarge) resolve(data);
    });
    req.on('error', reject);
  });

export interface SubstrateServerRuntime {
  plugin: Plugin;
  /** Open, validate, and initialize the single owned workspace before serving. */
  ready(): Promise<void>;
  /** Close the owned workspace store. */
  close(): void;
}

export function createSubstrateServer(opts: SubstrateServerOptions = {}): SubstrateServerRuntime {
  let ws: WorkspaceStore | null = null;
  const root = opts.root ?? process.cwd();
  const collaborators = opts.collaborators ?? defaultCollaboratorAdapters();

  // Stale locks (dead pids) are taken over inside openWorkspace; a live holder
  // is a real second writer and must not be steamrolled.
  const ensure = async (): Promise<WorkspaceStore> => {
    if (!ws) {
      const opened = await openWorkspace(root);
      try {
        if (opened.state.commitCount === 0) await syncWorkspace(opened, opts);
      } catch (error) {
        try {
          opened.close();
        } catch (closeError) {
          throw new AggregateError(
            [error, closeError],
            'workspace initialization failed and its store could not close cleanly',
          );
        }
        throw error;
      }
      ws = opened;
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

  const ready = async (): Promise<void> => {
    await exclusive(async () => {
      await ensure();
    });
  };

  const close = (): void => {
    if (!ws) return;
    ws.close();
    ws = null;
  };

  const plugin: Plugin = {
    name: 'substrate-server',
    configureServer(server) {
      server.httpServer?.on('close', close);
      server.middlewares.use((req, res, next) => {
        const url = (req.url ?? '').split('?')[0];
        if (!url.startsWith('/api/')) return next();
        // Provider latency is deliberately outside the workspace mutation
        // queue. Completion sees the exact bounded context sent by the client
        // and creates no truth; commits remain durable while a model thinks.
        if (url === '/api/complete' && req.method === 'POST') {
          void (async () => {
            try {
              let body: unknown;
              try {
                body = JSON.parse(await readBody(req, 64 * 1024));
              } catch (error) {
                if (error instanceof BodyTooLargeError) {
                  return json(res, 413, { code: 'completion-context-too-large', error: error.message });
                }
                return json(res, 400, { code: 'invalid-completion-request', error: 'Completion request is not valid JSON.' });
              }
              const result = await dispatchToCollaborator(collaborators, body);
              return json(res, 200, result);
            } catch (error) {
              if (error instanceof CollaboratorError) {
                return json(res, error.httpStatus, {
                  code: error.diagnostic.code,
                  error: error.diagnostic.message,
                  retryable: error.diagnostic.retryable,
                });
              }
              return json(res, 500, { code: 'collaborator-failure', error: 'Collaborator failed unexpectedly.' });
            }
          })();
          return;
        }
        void exclusive(async () => {
          try {
            const w = await ensure();
            if (url === '/api/state' && req.method === 'GET') {
              return json(res, 200, workspacePayload(root, w, collaborators));
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
            if ((url === '/api/sync' || url === '/api/ingest') && req.method === 'POST') {
              const report: SyncReport = await syncWorkspace(w, opts);
              return json(res, 200, { report, ...workspacePayload(root, w, collaborators) });
            }
            if (url === '/api/project' && req.method === 'POST') {
              const { relPath } = JSON.parse(await readBody(req)) as { relPath: string };
              try {
                await writeProjection(w.ctxFor('driver:fs'), { workspaceRoot: root, relPath });
              } catch (e) {
                if (e instanceof ProjectionConflictError) {
                  return json(res, 409, { code: 'projection-conflict', error: e.message });
                }
                throw e;
              }
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
  return { close, plugin, ready };
}

export function substrateServer(opts: SubstrateServerOptions = {}): Plugin {
  return createSubstrateServer(opts).plugin;
}
