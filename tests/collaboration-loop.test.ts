import assert from 'node:assert';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { proposalHistoryForDoc } from '../src/client/helpers';
import { OFFLINE_COLLABORATOR, stubCompleter } from '../src/collaboration/stub';
import { childOccurrences, currentRevision, renderChunk } from '../src/kernel/state';
import { generateProposal, select } from '../src/kernel/select';
import {
  acceptProposal,
  createComposite,
  moveOccurrence,
  placeOccurrence,
  redactRevision,
  rejectProposal,
  revise,
  severOccurrence,
  staleReason,
  transclude,
} from '../src/kernel/tx';
import { MEDIA_COMPOSITE, MEDIA_MARKDOWN } from '../src/kernel/types';
import { openWorkspace, type WorkspaceStore } from '../src/host/store-fs';

const root = mkdtempSync(join(tmpdir(), 'headspace-collaboration-loop-'));
let ws: WorkspaceStore | null = null;
try {
  ws = await openWorkspace(root);
  const human = ws.ctxFor('human:author');
  const doc = await createComposite(human, {
    join: '\n\n',
    blocks: [
      { text: '# Focus', mediaType: MEDIA_MARKDOWN },
      { text: 'The source paragraph.', mediaType: MEDIA_MARKDOWN },
    ],
  });

  // Provider failure occurs before proposal creation, so it leaves no partial
  // operation, proposal, or truth for the user to clean up.
  const beforeFailure = {
    head: ws.state.head,
    commits: ws.state.commitCount,
    operations: ws.state.operations.size,
    proposals: ws.state.proposals.size,
    chunks: ws.state.chunks.size,
  };
  await assert.rejects(
    generateProposal(human, {
      focusChunkId: doc.chunkId,
      instruction: 'fail visibly',
      complete: async () => {
        throw new Error('provider unavailable');
      },
      modelActorId: 'agent:test-provider',
    }),
    /provider unavailable/,
  );
  assert.deepEqual(
    {
      head: ws.state.head,
      commits: ws.state.commitCount,
      operations: ws.state.operations.size,
      proposals: ws.state.proposals.size,
      chunks: ws.state.chunks.size,
    },
    beforeFailure,
  );

  const generated = await generateProposal(human, {
    focusChunkId: doc.chunkId,
    instruction: 'continue this thought',
    complete: stubCompleter,
    modelActorId: OFFLINE_COLLABORATOR.actorId,
  });
  const proposal = ws.state.proposals.get(generated.proposalId)!;
  assert.equal(proposal.createdBy, 'agent:stub');
  const proposeOperation = ws.state.operations.get(proposal.operationId)!;
  assert.equal(proposeOperation.actorId, 'human:author', 'dispatcher and content author remain distinct');
  assert.deepEqual(
    proposeOperation.inputRevisionIds,
    generated.context.items.map((item) => item.revisionId),
    'the proposal operation records every revision exposed to the collaborator',
  );
  assert.ok(generated.context.items.every((item) => item.chunkId && item.revisionId && item.role));

  // A slow collaborator must never relabel a newer, unseen focus revision as
  // its basis. The resulting proposal remains inspectable but immediately
  // reports stale and cannot be integrated.
  const delayedFocusDoc = await createComposite(human, {
    join: '\n\n',
    blocks: [{ text: 'focus before dispatch', mediaType: MEDIA_MARKDOWN }],
  });
  const focusBeforeDispatch = currentRevision(ws.state, delayedFocusDoc.chunkId).id;
  const delayedFocus = await generateProposal(human, {
    focusChunkId: delayedFocusDoc.chunkId,
    instruction: 'answer slowly',
    complete: async () => {
      await revise(human, { chunkId: delayedFocusDoc.chunkId, text: 'focus edited while thinking' });
      return 'answer based on the old focus';
    },
    modelActorId: 'agent:slow',
  });
  const delayedFocusProposal = ws.state.proposals.get(delayedFocus.proposalId)!;
  assert.deepEqual(delayedFocusProposal.basisRevisionIds, [focusBeforeDispatch]);
  assert.equal(
    delayedFocusProposal.payload[0].op === 'create'
      ? delayedFocusProposal.payload[0].derivedFrom?.sourceRevisionId
      : null,
    focusBeforeDispatch,
  );
  assert.match(staleReason(ws.state, delayedFocusProposal) ?? '', /context chunk .* moved on/);

  // Every selected context revision participates in freshness, not only the
  // target. Editing a child while completion is in flight makes the proposal
  // stale even though the composite focus revision itself did not advance.
  const delayedContextDoc = await createComposite(human, {
    join: '\n\n',
    blocks: [
      { text: 'context focus', mediaType: MEDIA_MARKDOWN },
      { text: 'context child before dispatch', mediaType: MEDIA_MARKDOWN },
    ],
  });
  const childBeforeDispatch = currentRevision(ws.state, delayedContextDoc.blockChunkIds[1]).id;
  const delayedContext = await generateProposal(human, {
    focusChunkId: delayedContextDoc.chunkId,
    instruction: 'use the child context',
    complete: async () => {
      await revise(human, {
        chunkId: delayedContextDoc.blockChunkIds[1],
        text: 'child edited while thinking',
      });
      return 'answer based on the old child';
    },
    modelActorId: 'agent:slow',
  });
  const delayedContextProposal = ws.state.proposals.get(delayedContext.proposalId)!;
  assert.ok(delayedContextProposal.freshnessRevisionIds?.includes(childBeforeDispatch));
  assert.match(staleReason(ws.state, delayedContextProposal) ?? '', /context chunk .* moved on/);

  // Composite arrangement is content too, even though moving or severing an
  // occurrence deliberately does not mint a new composite revision.
  const reorderedDoc = await createComposite(human, {
    join: '|',
    blocks: [
      { text: 'A', mediaType: MEDIA_MARKDOWN },
      { text: 'B', mediaType: MEDIA_MARKDOWN },
    ],
  });
  const reorderOccurrences = childOccurrences(ws.state, reorderedDoc.chunkId);
  const delayedReorder = await generateProposal(human, {
    focusChunkId: reorderedDoc.chunkId,
    instruction: 'use this exact order',
    complete: async () => {
      moveOccurrence(human, { occurrenceId: reorderOccurrences[0].id, at: 'end' });
      return 'answer based on A then B';
    },
    modelActorId: 'agent:slow',
  });
  assert.match(
    staleReason(ws.state, ws.state.proposals.get(delayedReorder.proposalId)!) ?? '',
    /context structure/,
  );

  const severedDoc = await createComposite(human, {
    join: '|',
    blocks: [
      { text: 'keep', mediaType: MEDIA_MARKDOWN },
      { text: 'removed while thinking', mediaType: MEDIA_MARKDOWN },
    ],
  });
  const severedOccurrence = childOccurrences(ws.state, severedDoc.chunkId)[1];
  const delayedSever = await generateProposal(human, {
    focusChunkId: severedDoc.chunkId,
    instruction: 'use every visible part',
    complete: async () => {
      severOccurrence(human, { occurrenceId: severedOccurrence.id });
      return 'answer based on both parts';
    },
    modelActorId: 'agent:slow',
  });
  assert.match(
    staleReason(ws.state, ws.state.proposals.get(delayedSever.proposalId)!) ?? '',
    /context structure/,
  );

  const innerDoc = await createComposite(human, {
    join: '|',
    blocks: [{ text: 'nested old', mediaType: MEDIA_MARKDOWN }],
  });
  const outerDoc = await createComposite(human, {
    join: '|',
    blocks: [{ text: 'outer', mediaType: MEDIA_MARKDOWN }],
  });
  placeOccurrence(human, { containerId: outerDoc.chunkId, chunkId: innerDoc.chunkId });
  const nestedBeforeDispatch = currentRevision(ws.state, innerDoc.blockChunkIds[0]).id;
  const delayedNestedRevision = await generateProposal(human, {
    focusChunkId: outerDoc.chunkId,
    instruction: 'use the nested words',
    complete: async () => {
      await revise(human, { chunkId: innerDoc.blockChunkIds[0], text: 'nested new' });
      return 'answer based on nested old';
    },
    modelActorId: 'agent:slow',
  });
  const nestedProposal = ws.state.proposals.get(delayedNestedRevision.proposalId)!;
  assert.ok(
    ws.state.operations.get(nestedProposal.operationId)?.inputRevisionIds.includes(nestedBeforeDispatch),
    'every recursively rendered revision is an exact proposal input',
  );
  assert.match(staleReason(ws.state, nestedProposal) ?? '', /context chunk .* moved on/);

  const redactionDoc = await createComposite(human, {
    join: '\n',
    blocks: [{ text: 'visible before redaction', mediaType: MEDIA_MARKDOWN }],
  });
  const visibleRevision = currentRevision(ws.state, redactionDoc.blockChunkIds[0]).id;
  const delayedRedaction = await generateProposal(human, {
    focusChunkId: redactionDoc.blockChunkIds[0],
    instruction: 'use the visible text',
    complete: async () => {
      redactRevision(human, { revisionId: visibleRevision });
      return 'answer based on text that is now redacted';
    },
    modelActorId: 'agent:slow',
  });
  assert.match(
    staleReason(ws.state, ws.state.proposals.get(delayedRedaction.proposalId)!) ?? '',
    /revision .* visibility has changed/,
  );

  // Occurrence context must describe what the appearance actually renders,
  // not the continuing chunk's newer head.
  const pinnedSource = await createComposite(human, {
    join: '\n',
    blocks: [{ text: 'pinned old', mediaType: MEDIA_MARKDOWN }],
  });
  const pinnedFocus = await createComposite(human, {
    join: '|',
    blocks: [{ text: 'anchor', mediaType: MEDIA_MARKDOWN }],
  });
  const pinnedLeafRevision = currentRevision(ws.state, pinnedSource.blockChunkIds[0]).id;
  const pinnedLeafOccurrence = transclude(human, {
    containerId: pinnedFocus.chunkId,
    sourceChunkId: pinnedSource.blockChunkIds[0],
  });
  await revise(human, { chunkId: pinnedSource.blockChunkIds[0], text: 'current new' });
  const pinnedItems = select(ws.state, pinnedFocus.chunkId);
  const pinnedChild = pinnedItems.find((item) => item.occurrenceId === pinnedLeafOccurrence.occurrenceId)!;
  assert.equal(pinnedChild.revisionId, pinnedLeafRevision);
  assert.equal(pinnedChild.text, 'pinned old');
  assert.equal(pinnedChild.dependencies[0].followsCurrent, false);
  assert.doesNotMatch(pinnedItems.map((item) => item.text).join('\n'), /current new/);
  const pinnedGeneration = await generateProposal(human, {
    focusChunkId: pinnedFocus.chunkId,
    instruction: 'respect the pinned appearance',
    complete: stubCompleter,
    modelActorId: OFFLINE_COLLABORATOR.actorId,
  });
  const pinnedProposal = ws.state.proposals.get(pinnedGeneration.proposalId)!;
  assert.ok(ws.state.operations.get(pinnedProposal.operationId)?.inputRevisionIds.includes(pinnedLeafRevision));
  await revise(human, { chunkId: pinnedSource.blockChunkIds[0], text: 'current newer again' });
  assert.equal(staleReason(ws.state, pinnedProposal), null, 'a pinned input does not falsely follow the source head');

  const pinnedComposite = await createComposite(human, {
    join: '|',
    blocks: [
      { text: 'left', mediaType: MEDIA_MARKDOWN },
      { text: 'right', mediaType: MEDIA_MARKDOWN },
    ],
  });
  const compositeFocus = await createComposite(human, { join: '\n', blocks: [] });
  const pinnedCompositeRevision = currentRevision(ws.state, pinnedComposite.chunkId).id;
  const compositeOccurrence = transclude(human, {
    containerId: compositeFocus.chunkId,
    sourceChunkId: pinnedComposite.chunkId,
  });
  await revise(human, {
    chunkId: pinnedComposite.chunkId,
    text: JSON.stringify({ join: '/' }),
    mediaType: MEDIA_COMPOSITE,
  });
  assert.equal(renderChunk(ws.state, compositeFocus.chunkId), 'left|right');
  const compositeItem = select(ws.state, compositeFocus.chunkId)
    .find((item) => item.occurrenceId === compositeOccurrence.occurrenceId)!;
  assert.equal(compositeItem.revisionId, pinnedCompositeRevision);
  assert.equal(compositeItem.text, 'left|right');

  // The open proposal and its complete inspection inputs survive restart.
  ws.close();
  ws = await openWorkspace(root);
  assert.equal(ws.state.proposals.get(generated.proposalId)?.status, 'open');
  assert.deepEqual(
    ws.state.operations.get(ws.state.proposals.get(generated.proposalId)!.operationId)?.inputRevisionIds,
    proposeOperation.inputRevisionIds,
  );
  assert.match(staleReason(ws.state, ws.state.proposals.get(delayedFocus.proposalId)!) ?? '', /moved on/);
  assert.match(staleReason(ws.state, ws.state.proposals.get(delayedContext.proposalId)!) ?? '', /moved on/);
  assert.match(staleReason(ws.state, ws.state.proposals.get(delayedReorder.proposalId)!) ?? '', /context structure/);
  assert.match(staleReason(ws.state, ws.state.proposals.get(delayedSever.proposalId)!) ?? '', /context structure/);
  assert.match(staleReason(ws.state, ws.state.proposals.get(delayedNestedRevision.proposalId)!) ?? '', /moved on/);
  assert.match(staleReason(ws.state, ws.state.proposals.get(delayedRedaction.proposalId)!) ?? '', /visibility/);

  const accepted = await acceptProposal(ws.ctxFor('human:reviewer'), { proposalId: generated.proposalId });
  assert.ok(accepted.applied);
  const createdId = accepted.createdChunkIds[0];
  const createdRevision = currentRevision(ws.state, createdId);
  assert.equal(createdRevision.createdBy, 'agent:stub', 'the collaborator remains author after human acceptance');
  assert.equal(ws.state.proposals.get(generated.proposalId)?.resolution?.by, 'human:reviewer');
  assert.equal(
    ws.state.proposals.get(generated.proposalId)?.resolution?.operationId,
    accepted.commit.operation.id,
  );
  const derivation = [...ws.state.derivations.values()].find((candidate) => candidate.childChunkId === createdId)!;
  assert.equal(derivation.sourceRevisionId, proposal.basisRevisionIds[0]);
  assert.equal(derivation.via, 'generate');

  const rejected = await generateProposal(ws.ctxFor('human:author'), {
    focusChunkId: doc.chunkId,
    instruction: 'an unwanted alternative',
    complete: stubCompleter,
    modelActorId: OFFLINE_COLLABORATOR.actorId,
  });
  rejectProposal(ws.ctxFor('human:reviewer'), { proposalId: rejected.proposalId });

  const superseded = await generateProposal(ws.ctxFor('human:author'), {
    focusChunkId: doc.blockChunkIds[1],
    instruction: 'rewrite the paragraph',
    complete: stubCompleter,
    modelActorId: OFFLINE_COLLABORATOR.actorId,
  });
  await revise(ws.ctxFor('human:author'), {
    chunkId: doc.blockChunkIds[1],
    text: 'The human moved this paragraph on.',
    mediaType: MEDIA_MARKDOWN,
  });
  assert.equal(
    (await acceptProposal(ws.ctxFor('human:reviewer'), { proposalId: superseded.proposalId })).applied,
    false,
  );

  const leftOpen = await generateProposal(ws.ctxFor('human:author'), {
    focusChunkId: doc.chunkId,
    instruction: 'leave this for later',
    complete: stubCompleter,
    modelActorId: OFFLINE_COLLABORATOR.actorId,
  });

  ws.close();
  ws = await openWorkspace(root);
  const history = proposalHistoryForDoc(ws.state, doc.chunkId);
  assert.equal(history.find((candidate) => candidate.id === generated.proposalId)?.status, 'accepted');
  assert.equal(history.find((candidate) => candidate.id === rejected.proposalId)?.status, 'rejected');
  assert.equal(history.find((candidate) => candidate.id === superseded.proposalId)?.status, 'superseded');
  assert.equal(history.find((candidate) => candidate.id === leftOpen.proposalId)?.status, 'open');
  assert.equal(currentRevision(ws.state, createdId).createdBy, 'agent:stub');
  const acceptedAfterRestart = ws.state.proposals.get(generated.proposalId)!;
  const acceptedResolution = acceptedAfterRestart.resolution!;
  assert.equal(acceptedResolution.by, 'human:reviewer');
  assert.deepEqual(
    ws.state.operations.get(acceptedAfterRestart.operationId)?.inputRevisionIds,
    proposeOperation.inputRevisionIds,
  );
  const derivationAfterRestart = [...ws.state.derivations.values()].find(
    (candidate) => candidate.childChunkId === createdId,
  )!;
  assert.equal(derivationAfterRestart.via, 'generate');
  assert.equal(derivationAfterRestart.sourceRevisionId, proposal.basisRevisionIds[0]);
  assert.equal(derivationAfterRestart.operationId, acceptedResolution.operationId);
  assert.ok(ws.state.proposals.get(rejected.proposalId)?.resolution?.operationId);
  assert.ok(ws.state.proposals.get(superseded.proposalId)?.resolution?.operationId);
  assert.match(ws.state.proposals.get(superseded.proposalId)?.resolution?.reason ?? '', /moved on|basis/);
  assert.equal(ws.state.proposals.get(delayedFocus.proposalId)?.status, 'open');
  assert.match(staleReason(ws.state, ws.state.proposals.get(delayedFocus.proposalId)!) ?? '', /moved on/);

  console.log('collaboration loop OK — inspectable context, inert failure, durable review history, and provenance');
} finally {
  ws?.close();
  rmSync(root, { recursive: true, force: true });
}
