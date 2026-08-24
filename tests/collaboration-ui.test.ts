import assert from 'node:assert';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import type { SubstrateHook } from '../src/App';
import { Star } from '../src/Star';
import { OFFLINE_COLLABORATOR, stubCompleter } from '../src/collaboration/stub';
import { childOccurrences, currentRevision, emptyState } from '../src/kernel/state';
import { generateProposal } from '../src/kernel/select';
import { acceptProposal, createComposite, propose, rejectProposal, type TxCtx } from '../src/kernel/tx';
import { MEDIA_MARKDOWN } from '../src/kernel/types';

const state = emptyState();
const ctx: TxCtx = { state, actorId: 'human:author' };
const doc = await createComposite(ctx, {
  join: '\n\n',
  blocks: [{ text: '# Inspectable collaboration', mediaType: MEDIA_MARKDOWN }],
});
const generated = await generateProposal(ctx, {
  focusChunkId: doc.chunkId,
  instruction: 'continue this',
  complete: stubCompleter,
  modelActorId: OFFLINE_COLLABORATOR.actorId,
});
const sub = {
  ws: {
    state,
    bindings: [],
    identity: { id: 'workspace_test', displayName: 'test', rootDisplayPath: '/test' },
    adapters: [],
    collaborators: [
      OFFLINE_COLLABORATOR,
      {
        id: 'openai.responses',
        version: '1',
        label: 'OpenAI Responses',
        actorId: 'agent:openai',
        execution: 'remote',
        proposalOnly: true,
        availability: {
          status: 'unavailable',
          diagnostic: {
            code: 'collaborator-unavailable',
            phase: 'configure',
            message: 'OpenAI is not configured on the host.',
            retryable: false,
          },
        },
      },
    ],
    sources: [],
    lastIngestion: null,
  },
  ctx,
  version: 1,
  error: null,
  status: null,
  busy: false,
  ingestNow: async () => null,
  complete: async () => { throw new Error('not invoked during server rendering'); },
  syncNow: async () => null,
  reload: async () => null,
  dismissStatus: () => undefined,
} as unknown as SubstrateHook;

const render = () =>
  renderToStaticMarkup(
    createElement(Star, {
      sub,
      docId: doc.chunkId,
      onFocusDoc: () => undefined,
      onBack: () => undefined,
      backLabel: 'test',
    }),
  );

const openMarkup = render();
assert.match(openMarkup, /offline deterministic collaborator/);
assert.match(openMarkup, /agent:stub · local · proposal-only/);
assert.match(openMarkup, /OpenAI Responses/);
assert.match(openMarkup, /displayed context leaves this machine on dispatch/);
assert.match(openMarkup, /OpenAI is not configured on the host/);
assert.match(openMarkup, /bounded context/);
assert.match(openMarkup, /the document you are working in/);
assert.match(openMarkup, new RegExp(doc.chunkId));
assert.match(openMarkup, /exact input revision\(s\)/);
assert.match(openMarkup, new RegExp(doc.blockChunkIds[0]));
assert.match(openMarkup, /identity, inputs, basis, and targets/);
assert.match(openMarkup, /dispatcher/);
assert.match(openMarkup, /human:author/);
assert.match(openMarkup, /fresh/);
assert.match(openMarkup, /1\. add new block/);
assert.match(openMarkup, /2\. place block/);
assert.match(openMarkup, /containerId/, 'structural proposal changes are rendered rather than silently omitted');
assert.match(openMarkup, /tempId/, 'create structure is shown alongside its proposed text');
assert.match(openMarkup, /freshness/);
assert.match(openMarkup, /headspace\.offline-deterministic@1/);

await acceptProposal({ ...ctx, actorId: 'human:reviewer' }, { proposalId: generated.proposalId });
const rewriteChunkId = doc.blockChunkIds[0];
const rewriteBasis = currentRevision(state, rewriteChunkId).id;
const rewrite = propose(ctx, {
  kind: 'suggested-edit',
  basisRevisionIds: [rewriteBasis],
  freshnessRevisionIds: [rewriteBasis],
  targetChunkIds: [rewriteChunkId],
  payload: [{ op: 'revise', chunkId: rewriteChunkId, text: '# Revised collaboration' }],
  note: 'test an exact historical before',
});
const rewriteOpenMarkup = render();
assert.match(rewriteOpenMarkup, /recorded basis text/);
assert.match(rewriteOpenMarkup, /# Inspectable collaboration/);
assert.match(rewriteOpenMarkup, /# Revised collaboration/);
assert.match(rewriteOpenMarkup, /exact proposed operation/);
assert.match(rewriteOpenMarkup, /chunkId/);
await acceptProposal({ ...ctx, actorId: 'human:reviewer' }, { proposalId: rewrite.proposalId });
const rewriteHistoryMarkup = render();
assert.match(
  rewriteHistoryMarkup,
  /# Inspectable collaboration/,
  'accepted history renders the immutable proposal basis, not the new live head as its before text',
);
assert.match(rewriteHistoryMarkup, /# Revised collaboration/);

const occurrence = childOccurrences(state, doc.chunkId)[0];
const current = currentRevision(state, rewriteChunkId);
propose(ctx, {
  kind: 'source-update',
  basisRevisionIds: [current.id],
  targetChunkIds: [rewriteChunkId],
  payload: [{ op: 'repin', occurrenceId: occurrence.id, revisionId: current.id }],
});
propose(ctx, {
  kind: 'suggested-edit',
  basisRevisionIds: [current.id],
  targetChunkIds: [rewriteChunkId],
  payload: [{ op: 'sever', occurrenceId: occurrence.id }],
});
const structuralMarkup = render();
assert.match(structuralMarkup, /update watched quote/);
assert.match(structuralMarkup, /remove block appearance/);
assert.match(structuralMarkup, /occurrenceId/);
assert.match(structuralMarkup, /revisionId/);

const unwanted = await generateProposal(ctx, {
  focusChunkId: doc.chunkId,
  instruction: 'unwanted',
  complete: stubCompleter,
  modelActorId: OFFLINE_COLLABORATOR.actorId,
});
rejectProposal({ ...ctx, actorId: 'human:reviewer' }, { proposalId: unwanted.proposalId });
const historyMarkup = render();
assert.match(historyMarkup, /accepted/);
assert.match(historyMarkup, /rejected/);
assert.match(historyMarkup, /resolved/);
assert.match(historyMarkup, /human:reviewer/);
assert.match(historyMarkup, /open · 5 total proposal\(s\)/);

console.log('collaboration UI OK — named adapter, bounded context, complete inspector, and outcome history');
