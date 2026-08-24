import assert from 'node:assert';
import { mkdtempSync, rmSync } from 'node:fs';
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { AddressInfo } from 'node:net';
import type { CollaboratorAdapter } from '../src/host/collaborators';
import { substrateServer } from '../src/host/api';

const root = mkdtempSync(join(tmpdir(), 'headspace-collaborator-api-'));
let releaseProvider!: () => void;
let providerStarted!: () => void;
const started = new Promise<void>((resolve) => { providerStarted = resolve; });
const released = new Promise<void>((resolve) => { releaseProvider = resolve; });
const delayed: CollaboratorAdapter = {
  capability: {
    id: 'test.remote',
    version: '7',
    label: 'Delayed test collaborator',
    actorId: 'agent:test:remote',
    execution: 'remote',
    proposalOnly: true,
    availability: { status: 'ready' },
  },
  async complete() {
    providerStarted();
    await released;
    return {
      text: 'delayed result',
      collaboratorId: 'test.remote',
      collaboratorVersion: '7',
      actorId: 'agent:test:remote',
    };
  },
};

type Middleware = (req: IncomingMessage, res: ServerResponse, next: () => void) => void;
const middlewares: Middleware[] = [];
const server = createServer((req, res) => {
  let index = 0;
  const next = (): void => {
    const middleware = middlewares[index++];
    if (middleware) return middleware(req, res, next);
    res.statusCode = 404;
    res.end('not found');
  };
  next();
});

const plugin = substrateServer({ root, collaborators: [delayed] });
if (typeof plugin.configureServer !== 'function') throw new Error('configureServer hook missing');
(plugin.configureServer as unknown as (server: {
  httpServer: unknown;
  middlewares: { use(middleware: Middleware): void };
}) => unknown)({
  httpServer: server,
  middlewares: { use: (middleware: Middleware) => middlewares.push(middleware) },
});

try {
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const { port } = server.address() as AddressInfo;
  const base = `http://127.0.0.1:${port}`;
  const completion = fetch(`${base}/api/complete`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      collaboratorId: 'test.remote',
      instruction: 'wait',
      context: {
        items: [{ chunkId: 'chunk', revisionId: 'revision', role: 'focus', text: 'visible', dependencies: [] }],
        chars: 7,
        dropped: 0,
      },
    }),
  });
  await started;

  const stateRequest = fetch(`${base}/api/state`);
  const first = await Promise.race([
    stateRequest.then(() => 'state'),
    new Promise<'timeout'>((resolve) => setTimeout(() => resolve('timeout'), 1000)),
  ]);
  releaseProvider();
  assert.equal(first, 'state', 'model latency must not occupy the workspace mutation queue');
  const stateResponse = await stateRequest;
  assert.equal(stateResponse.status, 200);
  const state = await stateResponse.json() as { collaborators: Array<{ id: string; execution: string }> };
  assert.deepEqual(
    state.collaborators.map((capability) => capability.id),
    ['headspace.offline-deterministic', 'test.remote'],
  );
  assert.equal(state.collaborators[1].execution, 'remote');

  const completionResponse = await completion;
  assert.equal(completionResponse.status, 200);
  assert.equal((await completionResponse.json() as { text: string }).text, 'delayed result');

  const malformed = await fetch(`${base}/api/complete`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: '{',
  });
  assert.equal(malformed.status, 400);
  assert.equal((await malformed.json() as { code: string }).code, 'invalid-completion-request');

  const unknown = await fetch(`${base}/api/complete`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      collaboratorId: 'browser-chosen-provider',
      instruction: 'no',
      context: { items: [], chars: 0, dropped: 0 },
    }),
  });
  assert.equal(unknown.status, 404);
  assert.equal((await unknown.json() as { code: string }).code, 'unknown-collaborator');

  const oversized = await fetch(`${base}/api/complete`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ padding: 'x'.repeat(70_000) }),
  });
  assert.equal(oversized.status, 413);
  assert.equal((await oversized.json() as { code: string }).code, 'completion-context-too-large');
} finally {
  releaseProvider();
  await new Promise<void>((resolve) => server.close(() => resolve()));
  rmSync(root, { recursive: true, force: true });
}

console.log('collaborator API OK — bounded allowlisted dispatch runs outside durable workspace mutations');
