import assert from 'node:assert';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { CompletionRequest } from '../src/collaboration/types';
import { dispatchToLocalCollaborator, OFFLINE_COLLABORATOR } from '../src/collaboration/stub';
import {
  CollaboratorError,
  createOpenAIResponsesAdapter,
  dispatchToCollaborator,
  validateCompletionRequest,
} from '../src/host/collaborators';
import { openWorkspace, type WorkspaceStore } from '../src/host/store-fs';
import { generateProposal } from '../src/kernel/select';
import { createComposite } from '../src/kernel/tx';
import { MEDIA_MARKDOWN } from '../src/kernel/types';

const request: CompletionRequest = {
  collaboratorId: 'openai.responses',
  instruction: 'continue carefully',
  context: {
    items: [
      {
        chunkId: 'chunk_focus',
        revisionId: 'revision_focus',
        role: 'focus',
        text: '# Exact displayed context',
        dependencies: [],
      },
      {
        chunkId: 'chunk_child',
        revisionId: 'revision_child',
        role: 'child',
        text: 'A supporting detail.',
        dependencies: [],
      },
    ],
    chars: 999_999, // the host must recompute this untrusted field
    dropped: 2,
  },
};

const local = await dispatchToLocalCollaborator({ ...request, collaboratorId: OFFLINE_COLLABORATOR.id });
assert.equal(local.collaboratorId, OFFLINE_COLLABORATOR.id);
assert.equal(local.actorId, OFFLINE_COLLABORATOR.actorId);
assert.match(local.text, /^\(stub\)/);
await assert.rejects(
  () => dispatchToLocalCollaborator({ ...request, collaboratorId: 'local.unregistered' }),
  /No local collaborator implementation/,
);

// Configuration is explicit and capabilities are safe to serialize.
for (const [apiKey, model, expectedReady] of [
  [undefined, undefined, false],
  ['secret-key', undefined, false],
  [undefined, 'model-explicit', false],
  ['secret-key', 'model-explicit', true],
] as const) {
  let calls = 0;
  const adapter = createOpenAIResponsesAdapter({
    apiKey,
    model,
    fetchImpl: (async () => {
      calls++;
      throw new Error('network should not run in this matrix');
    }) as typeof fetch,
  });
  assert.equal(adapter.capability.availability.status === 'ready', expectedReady);
  assert.doesNotMatch(JSON.stringify(adapter.capability), /secret-key/);
  if (!expectedReady) {
    await assert.rejects(() => dispatchToCollaborator([adapter], request), CollaboratorError);
    assert.equal(calls, 0, 'an unavailable adapter never attempts provider I/O');
  }
}

const validated = validateCompletionRequest(request);
assert.equal(
  validated.context.chars,
  request.context.items.reduce((total, item) => total + item.text.length, 0),
  'the host recomputes the bounded context size',
);
await assert.rejects(
  async () => validateCompletionRequest({ ...request, context: { items: [{ ...request.context.items[0], role: 'system' }] } }),
  /invalid role/,
);
await assert.rejects(
  async () => validateCompletionRequest({
    ...request,
    context: { items: [{ ...request.context.items[0], text: 'x'.repeat(6001) }] },
  }),
  (error: unknown) => error instanceof CollaboratorError && error.httpStatus === 413,
);
await assert.rejects(
  () => dispatchToCollaborator([], request),
  (error: unknown) => error instanceof CollaboratorError && error.diagnostic.code === 'unknown-collaborator',
);

let capturedUrl = '';
let capturedInit: RequestInit | undefined;
const successAdapter = createOpenAIResponsesAdapter({
  apiKey: 'secret-key',
  model: 'model-explicit',
  fetchImpl: (async (input: string | URL | Request, init?: RequestInit) => {
    capturedUrl = String(input);
    capturedInit = init;
    return new Response(JSON.stringify({
      id: 'resp_exact',
      status: 'completed',
      model: 'model-resolved-2026-08-21',
      output: [
        { type: 'reasoning', content: [{ type: 'summary_text', text: 'ignored' }] },
        {
          type: 'message',
          role: 'assistant',
          content: [
            { type: 'output_text', text: 'First proposed paragraph.' },
            { type: 'output_text', text: 'Second proposed paragraph.' },
          ],
        },
      ],
    }), { status: 200, headers: { 'content-type': 'application/json' } });
  }) as typeof fetch,
});
const completed = await dispatchToCollaborator([successAdapter], request);
assert.equal(completed.text, 'First proposed paragraph.\n\nSecond proposed paragraph.');
assert.equal(completed.actorId, 'agent:openai:model-explicit');
assert.equal(completed.providerResponseId, 'resp_exact');
assert.equal(completed.model, 'model-resolved-2026-08-21');
assert.equal(capturedUrl, 'https://api.openai.com/v1/responses', 'the browser cannot choose a credential destination');
assert.equal(capturedInit?.method, 'POST');
assert.equal((capturedInit?.headers as Record<string, string>).authorization, 'Bearer secret-key');
const providerBody = JSON.parse(String(capturedInit?.body));
assert.equal(providerBody.model, 'model-explicit');
assert.equal(providerBody.store, false);
assert.equal(providerBody.max_output_tokens, 1200);
assert.equal(providerBody.tools, undefined);
const sent = JSON.parse(providerBody.input[0].content[0].text);
assert.equal(sent.instruction, request.instruction);
assert.deepEqual(sent.context, request.context.items);
assert.equal(sent.omittedContextItems, 2);

const directAdapter = createOpenAIResponsesAdapter({
  apiKey: 'key',
  model: 'model',
  fetchImpl: (async () => new Response(
    JSON.stringify({ id: 'resp_direct', status: 'completed', model: 'model-resolved', output_text: '  direct convenience text  ', output: [] }),
    { status: 200, headers: { 'content-type': 'application/json' } },
  )) as typeof fetch,
});
assert.equal((await directAdapter.complete(request)).text, 'direct convenience text');

for (const status of ['failed', 'incomplete', 'queued', 'in_progress', 'cancelled']) {
  const adapter = createOpenAIResponsesAdapter({
    apiKey: 'key',
    model: 'model',
    fetchImpl: (async () => new Response(
      JSON.stringify({ status, output: [{ type: 'message', role: 'assistant', content: [{ type: 'output_text', text: 'partial' }] }] }),
      { status: 200, headers: { 'content-type': 'application/json' } },
    )) as typeof fetch,
  });
  await assert.rejects(
    () => adapter.complete(request),
    (error: unknown) => error instanceof CollaboratorError && error.diagnostic.code === 'provider-invalid-response',
  );
}

const malformed = createOpenAIResponsesAdapter({
  apiKey: 'key',
  model: 'model',
  fetchImpl: (async () => new Response('{', { status: 200 })) as typeof fetch,
});
await assert.rejects(() => malformed.complete(request), /malformed JSON/);
const empty = createOpenAIResponsesAdapter({
  apiKey: 'key',
  model: 'model',
  fetchImpl: (async () => new Response(JSON.stringify({ status: 'completed', model: 'model', output: [] }), { status: 200 })) as typeof fetch,
});
await assert.rejects(() => empty.complete(request), /without usable text/);
const unidentifiedModel = createOpenAIResponsesAdapter({
  apiKey: 'key',
  model: 'configured-alias',
  fetchImpl: (async () => new Response(
    JSON.stringify({ status: 'completed', output_text: 'text without a model identity' }),
    { status: 200 },
  )) as typeof fetch,
});
await assert.rejects(() => unidentifiedModel.complete(request), /without a resolved model identity/);
const refused = createOpenAIResponsesAdapter({
  apiKey: 'key',
  model: 'model',
  fetchImpl: (async () => new Response('document text must not be echoed', { status: 429 })) as typeof fetch,
});
await assert.rejects(
  () => refused.complete(request),
  (error: unknown) => error instanceof CollaboratorError &&
    error.diagnostic.code === 'provider-rejected' &&
    !error.message.includes('document text'),
);
let rejectedBodyCancelled = false;
const streamingRefusal = createOpenAIResponsesAdapter({
  apiKey: 'key',
  model: 'model',
  fetchImpl: (async () => new Response(
    new ReadableStream<Uint8Array>({
      pull() {
        // Deliberately never close: cancellation is the only bounded outcome.
      },
      cancel() {
        rejectedBodyCancelled = true;
      },
    }),
    { status: 429 },
  )) as typeof fetch,
});
await assert.rejects(() => streamingRefusal.complete(request), /refused/);
assert.equal(rejectedBodyCancelled, true, 'a rejected provider response cannot keep streaming after failure');

const timeout = createOpenAIResponsesAdapter({
  apiKey: 'key',
  model: 'model',
  timeoutMs: 5,
  fetchImpl: ((_: string | URL | Request, init?: RequestInit) => new Promise<Response>((_resolve, reject) => {
    init?.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')));
  })) as typeof fetch,
});
await assert.rejects(
  () => timeout.complete(request),
  (error: unknown) => error instanceof CollaboratorError && error.diagnostic.code === 'collaborator-timeout',
);

const bodyTimeout = createOpenAIResponsesAdapter({
  apiKey: 'key',
  model: 'model',
  timeoutMs: 5,
  fetchImpl: (async (_input: string | URL | Request, init?: RequestInit) => {
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode('{"status":"completed","output":['));
        init?.signal?.addEventListener('abort', () => controller.error(new DOMException('aborted', 'AbortError')));
      },
    });
    return new Response(body, { status: 200, headers: { 'content-type': 'application/json' } });
  }) as typeof fetch,
});
await assert.rejects(
  () => bodyTimeout.complete(request),
  (error: unknown) => error instanceof CollaboratorError && error.diagnostic.code === 'collaborator-timeout',
);

const oversizedResponse = createOpenAIResponsesAdapter({
  apiKey: 'key',
  model: 'model',
  fetchImpl: (async () => new Response('x'.repeat(1024 * 1024 + 1), { status: 200 })) as typeof fetch,
});
await assert.rejects(
  () => oversizedResponse.complete(request),
  (error: unknown) => error instanceof CollaboratorError &&
    error.diagnostic.code === 'provider-invalid-response' &&
    /allowed size/.test(error.message),
);

// A real provider-backed adapter still enters truth only through a proposal, with exact model
// authorship and operation inputs surviving restart.
const root = mkdtempSync(join(tmpdir(), 'headspace-model-provider-'));
let ws: WorkspaceStore | null = null;
try {
  ws = await openWorkspace(root);
  const ctx = ws.ctxFor('human:dispatcher');
  const doc = await createComposite(ctx, {
    join: '\n\n',
    blocks: [{ text: '# Model provider', mediaType: MEDIA_MARKDOWN }],
  });
  const before = {
    head: ws.state.head,
    commits: ws.state.commitCount,
    proposals: ws.state.proposals.size,
    operations: ws.state.operations.size,
  };
  const failingAdapter = createOpenAIResponsesAdapter({
    apiKey: 'key',
    model: 'model-explicit',
    fetchImpl: (async () => new Response('no', { status: 503 })) as typeof fetch,
  });
  await assert.rejects(
    () => generateProposal(ctx, {
      focusChunkId: doc.chunkId,
      instruction: 'fail before truth',
      modelActorId: failingAdapter.capability.actorId,
      complete: async (context, instruction) => (await failingAdapter.complete({
        collaboratorId: failingAdapter.capability.id,
        context,
        instruction,
      })).text,
    }),
    /refused/,
  );
  assert.deepEqual(
    {
      head: ws.state.head,
      commits: ws.state.commitCount,
      proposals: ws.state.proposals.size,
      operations: ws.state.operations.size,
    },
    before,
  );

  const generated = await generateProposal(ctx, {
    focusChunkId: doc.chunkId,
    instruction: 'use the configured model',
    modelActorId: successAdapter.capability.actorId,
    complete: async (context, instruction) => {
      const result = await successAdapter.complete({
        collaboratorId: successAdapter.capability.id,
        context,
        instruction,
      });
      return {
        text: result.text,
        producer: {
          id: result.collaboratorId,
          version: result.collaboratorVersion,
          implementation: result.model,
          receiptId: result.providerResponseId,
        },
      };
    },
  });
  const proposal = ws.state.proposals.get(generated.proposalId)!;
  assert.equal(proposal.createdBy, 'agent:openai:model-explicit');
  assert.deepEqual(proposal.producer, {
    id: 'openai.responses',
    version: '1',
    implementation: 'model-resolved-2026-08-21',
    receiptId: 'resp_exact',
  });
  const inputs = ws.state.operations.get(proposal.operationId!)!.inputRevisionIds;
  ws.close();
  ws = await openWorkspace(root);
  const restarted = ws.state.proposals.get(generated.proposalId)!;
  assert.equal(restarted.createdBy, 'agent:openai:model-explicit');
  assert.deepEqual(restarted.producer, proposal.producer);
  assert.deepEqual(
    (ws.state.operations.get(restarted.operationId!)?.params as { producer?: unknown }).producer,
    proposal.producer,
  );
  assert.deepEqual(ws.state.operations.get(restarted.operationId!)?.inputRevisionIds, inputs);
} finally {
  ws?.close();
  rmSync(root, { recursive: true, force: true });
}

console.log('collaborator adapters OK — explicit configuration, bounded remote dispatch, inert failure, and durable model provenance');
