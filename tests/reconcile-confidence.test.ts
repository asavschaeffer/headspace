import assert from 'node:assert';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { currentRevision, emptyState } from '../src/kernel/state';
import type { TxCtx } from '../src/kernel/tx';
import { importMarkdownFile, reconcileMarkdownFile } from '../src/host/markdown';

const root = mkdtempSync(join(tmpdir(), 'headspace-reconcile-confidence-'));
try {
  const state = emptyState();
  const ctx: TxCtx = { state, actorId: 'driver:fs' };
  // Similarity intentionally samples the head and tail of blocks above its
  // exact-comparison cap. Change only the unseen middle so the sampled score is
  // maximally convincing while still being explicitly approximate evidence.
  const head = 'recognizable head '.repeat(100);
  const middle = 'a'.repeat(5000);
  const tail = ' recognizable tail'.repeat(100);
  const original = `${head}${middle}${tail}\n`;
  const changed = `${head}${'b'.repeat(5000)}${tail}\n`;
  const imported = await importMarkdownFile(ctx, {
    workspaceRoot: root,
    relPath: 'long.md',
    text: original,
  });
  const revisionBefore = currentRevision(state, imported.blockChunkIds[0]).id;

  const result = await reconcileMarkdownFile(ctx, {
    workspaceRoot: root,
    relPath: 'long.md',
    text: changed,
  });

  assert.equal(result.action, 'proposal', 'sampled identity evidence cannot enter the automatic fast path');
  assert.equal(currentRevision(state, imported.blockChunkIds[0]).id, revisionBefore, 'source identity remains untouched before review');
  const proposal = state.proposals.get(result.proposalId!)!;
  assert.match(proposal.note ?? '', /sampled-similarity/);
  assert.match(proposal.note ?? '', /confidence 1\.00/);
  assert.deepEqual(proposal.payload.map((change) => change.op), ['revise']);

  // Even an exactly-computed similarity score is not enough to transfer
  // identity automatically when more than one prior block is a viable match.
  const shared = 'Shared paragraph words and context variant ';
  const ambiguousOriginal = `${shared}alpha\n\n${shared}beta\n`;
  const ambiguousChanged = `${shared}gamma\n\n${shared}delta\n`;
  const ambiguousImport = await importMarkdownFile(ctx, {
    workspaceRoot: root,
    relPath: 'ambiguous.md',
    text: ambiguousOriginal,
  });
  const ambiguousRevisions = ambiguousImport.blockChunkIds.map((id) => currentRevision(state, id).id);
  const ambiguousResult = await reconcileMarkdownFile(ctx, {
    workspaceRoot: root,
    relPath: 'ambiguous.md',
    text: ambiguousChanged,
  });
  assert.equal(ambiguousResult.action, 'proposal');
  assert.deepEqual(
    ambiguousImport.blockChunkIds.map((id) => currentRevision(state, id).id),
    ambiguousRevisions,
    'ambiguous matches cannot revise either candidate before review',
  );
  assert.match(state.proposals.get(ambiguousResult.proposalId!)?.note ?? '', /ambiguous match/);

  console.log('reconciliation confidence OK — sampled and ambiguous identity evidence become proposals');
} finally {
  rmSync(root, { recursive: true, force: true });
}
