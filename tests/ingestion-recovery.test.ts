import assert from 'node:assert';
import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { childOccurrences, currentRevision, renderChunk, revisionText } from '../src/kernel/state';
import { acceptProposal, revise, severOccurrence } from '../src/kernel/tx';
import {
  ingestWorkspace,
  readIngestionCatalog,
  type IngestionCatalog,
  type IngestionItemResult,
} from '../src/host/ingestion';
import { sidecarPath } from '../src/host/markdown';
import { openWorkspace, type WorkspaceStore } from '../src/host/store-fs';

const roots: string[] = [];
const freshRoot = (prefix: string): string => {
  const root = mkdtempSync(join(tmpdir(), prefix));
  roots.push(root);
  return root;
};
const crash = (root: string): void => rmSync(join(root, '.headspace', 'lock'));
const one = (items: IngestionItemResult[]): IngestionItemResult => {
  assert.equal(items.length, 1);
  return items[0];
};

let active: WorkspaceStore | null = null;
try {
  // A catalog publish can fail after the kernel import is already durable.
  // The write-ahead intent survives, log replay rediscovers the exact import,
  // and retry binds it without manufacturing replacement chunks.
  {
    const root = freshRoot('headspace-ingestion-catalog-crash-');
    writeFileSync(join(root, 'note.txt'), 'v1');
    active = await openWorkspace(root);
    let captured: IngestionCatalog | null = null;
    await assert.rejects(
      ingestWorkspace(
        active,
        { contentFiles: ['note.txt'] },
        {
          catalogPublish: (temporaryPath, destinationPath) => {
            const attempted = JSON.parse(readFileSync(temporaryPath, 'utf8')) as IngestionCatalog;
            if (attempted.representations.length > 0 && attempted.pendingMaterializations.length === 0) {
              captured = attempted;
              throw new Error('injected catalog publish failure');
            }
            renameSync(temporaryPath, destinationPath);
          },
        },
      ),
      /injected catalog publish failure/,
    );
    assert.ok(captured, 'fault reached the completed representation publication');
    const attemptedCatalog = captured as IngestionCatalog;
    const attemptedRepresentation = attemptedCatalog.representations[0];
    const attemptedSource = attemptedCatalog.sources[0];
    assert.ok(attemptedRepresentation);
    assert.equal(attemptedSource.currentRepresentationId, attemptedRepresentation.id);
    const durableChunkIds = [...active.state.chunks.keys()].sort();
    assert.equal(durableChunkIds.length, 2, 'the composite and leaf reached the durable kernel log');
    const pendingCatalog = readIngestionCatalog(root)!;
    assert.equal(pendingCatalog.pendingMaterializations.length, 1, 'disk retains the write-ahead intent');
    const pending = pendingCatalog.pendingMaterializations[0];
    if (pending.operationKind !== 'import') throw new Error('expected a current-format import intent');
    const correlated = [...active.state.operations.values()].find(
      (operation) => (operation.params as Record<string, unknown> | undefined)?.ingestionToken === pending.token,
    );
    assert.ok(correlated);
    assert.equal(correlated.actorId, 'adapter:headspace.text.native@1.0.0');
    assert.deepEqual(
      {
        productIdentityHash: (correlated.params as Record<string, unknown>).productIdentityHash,
        relationship: (correlated.params as Record<string, unknown>).relationship,
        mediaType: (correlated.params as Record<string, unknown>).mediaType,
        normalizedTextHash: (correlated.params as Record<string, unknown>).normalizedTextHash,
        normalizedRenderedTextHash: (correlated.params as Record<string, unknown>).normalizedRenderedTextHash,
      },
      {
        productIdentityHash: pending.productIdentityHash,
        relationship: pending.relationship,
        mediaType: pending.mediaType,
        normalizedTextHash: pending.normalizedTextHash,
        normalizedRenderedTextHash: pending.normalizedRenderedTextHash,
      },
    );

    const malformedPendingCases: Array<{
      field: string;
      remove(catalog: Record<string, unknown>): void;
    }> = [
      {
        field: 'pendingMaterializations',
        remove: (catalog) => delete catalog.pendingMaterializations,
      },
      {
        field: 'identityKey',
        remove: (catalog) => delete (catalog.sources as Array<Record<string, unknown>>)[0].identityKey,
      },
      {
        field: 'identityKey',
        remove: (catalog) => delete (catalog.observations as Array<Record<string, unknown>>)[0].identityKey,
      },
      {
        field: 'productIdentityHash',
        remove: (catalog) =>
          delete (catalog.pendingMaterializations as Array<Record<string, unknown>>)[0].productIdentityHash,
      },
      {
        field: 'normalizedRenderedTextHash',
        remove: (catalog) =>
          delete (catalog.pendingMaterializations as Array<Record<string, unknown>>)[0].normalizedRenderedTextHash,
      },
    ];
    for (const malformedCase of malformedPendingCases) {
      const malformed = structuredClone(pendingCatalog) as unknown as Record<string, unknown>;
      malformedCase.remove(malformed);
      writeFileSync(join(root, '.headspace', 'ingestion.json'), `${JSON.stringify(malformed, null, 2)}\n`);
      assert.throws(() => readIngestionCatalog(root), new RegExp(malformedCase.field));
    }
    writeFileSync(join(root, '.headspace', 'ingestion.json'), `${JSON.stringify(pendingCatalog, null, 2)}\n`);

    crash(root);
    active = await openWorkspace(root);
    assert.deepEqual([...active.state.chunks.keys()].sort(), durableChunkIds, 'restart replays the durable import');
    const recovered = one((await ingestWorkspace(active, { contentFiles: ['note.txt'] })).items);
    assert.equal(recovered.observation.sourceId, attemptedSource.id, 'recovery preserves continuing source identity');
    assert.equal(recovered.representation?.rootChunkId, attemptedRepresentation.rootChunkId);
    assert.equal(renderChunk(active.state, attemptedRepresentation.rootChunkId), 'v1');
    assert.deepEqual([...active.state.chunks.keys()].sort(), durableChunkIds, 'recovery does not duplicate chunks');
    assert.equal(readIngestionCatalog(root)?.pendingMaterializations.length, 0);
    const recoveredCatalog = readIngestionCatalog(root)!;
    const missingOperationIds = structuredClone(recoveredCatalog) as unknown as {
      representations: Array<Record<string, unknown>>;
    };
    delete missingOperationIds.representations[0].operationIds;
    writeFileSync(join(root, '.headspace', 'ingestion.json'), `${JSON.stringify(missingOperationIds, null, 2)}\n`);
    assert.throws(() => readIngestionCatalog(root), /operationIds/);
    writeFileSync(join(root, '.headspace', 'ingestion.json'), `${JSON.stringify(recoveredCatalog, null, 2)}\n`);
    active.close();
    active = null;
  }

  // A durable kernel import cannot be rebound through a catalog intent whose
  // exact product identity was altered after the final publication failed.
  {
    const root = freshRoot('headspace-ingestion-import-correlation-');
    writeFileSync(join(root, 'note.txt'), 'correlated product');
    active = await openWorkspace(root);
    await assert.rejects(
      ingestWorkspace(
        active,
        { contentFiles: ['note.txt'] },
        {
          catalogPublish: (temporaryPath, destinationPath) => {
            const attempted = JSON.parse(readFileSync(temporaryPath, 'utf8')) as IngestionCatalog;
            if (attempted.representations.length > 0 && attempted.pendingMaterializations.length === 0) {
              throw new Error('injected correlation publish failure');
            }
            renameSync(temporaryPath, destinationPath);
          },
        },
      ),
      /injected correlation publish failure/,
    );
    const catalog = readIngestionCatalog(root)!;
    assert.equal(catalog.pendingMaterializations.length, 1);
    catalog.pendingMaterializations[0].productIdentityHash = '0'.repeat(64);
    writeFileSync(join(root, '.headspace', 'ingestion.json'), `${JSON.stringify(catalog, null, 2)}\n`);
    active.close();
    active = await openWorkspace(root);
    await assert.rejects(
      ingestWorkspace(active, { contentFiles: ['note.txt'] }),
      /does not match ingestion productIdentityHash/,
    );
    assert.equal(readIngestionCatalog(root)?.pendingMaterializations.length, 1, 'refused correlation remains inspectable');
    assert.equal(readIngestionCatalog(root)?.representations.length, 0);
    active.close();
    active = null;
  }

  // Correct correlated revisions are insufficient when the materialized root
  // no longer contains them. Recovery validates the actual occurrence graph
  // and refuses to bind a root whose child was removed after the failed
  // catalog publication.
  {
    const root = freshRoot('headspace-ingestion-import-structure-');
    writeFileSync(join(root, 'note.txt'), 'must remain contained');
    active = await openWorkspace(root);
    await assert.rejects(
      ingestWorkspace(
        active,
        { contentFiles: ['note.txt'] },
        {
          catalogPublish: (temporaryPath, destinationPath) => {
            const attempted = JSON.parse(readFileSync(temporaryPath, 'utf8')) as IngestionCatalog;
            if (attempted.representations.length > 0 && attempted.pendingMaterializations.length === 0) {
              throw new Error('injected structure publish failure');
            }
            renameSync(temporaryPath, destinationPath);
          },
        },
      ),
      /injected structure publish failure/,
    );
    const pending = readIngestionCatalog(root)!.pendingMaterializations[0];
    const operation = [...active.state.operations.values()].find(
      (candidate) => (candidate.params as Record<string, unknown> | undefined)?.ingestionToken === pending.token,
    );
    assert.ok(operation);
    const rootRevision = active.state.revisions.get(operation.outputRevisionIds[0])!;
    const child = childOccurrences(active.state, rootRevision.chunkId)[0];
    assert.ok(child);
    severOccurrence(active.ctxFor('human:tamper'), { occurrenceId: child.id });
    assert.equal(renderChunk(active.state, rootRevision.chunkId), '');
    active.close();
    active = await openWorkspace(root);
    await assert.rejects(
      ingestWorkspace(active, { contentFiles: ['note.txt'] }),
      /child occurrence structure does not match its outputs/,
    );
    assert.equal(readIngestionCatalog(root)?.pendingMaterializations.length, 1);
    assert.equal(readIngestionCatalog(root)?.representations.length, 0);
    active.close();
    active = null;
  }

  // The narrower kernel-commit/Markdown-sidecar gap is recoverable too. A
  // deliberately blocked sidecar publication leaves the intent pending; after
  // restart, Headspace reconstructs the manifest from the correlated import.
  {
    const root = freshRoot('headspace-ingestion-sidecar-crash-');
    writeFileSync(join(root, 'note.md'), '# Durable Markdown\n\nSecond block\n');
    active = await openWorkspace(root);
    const recoveredSidecar = sidecarPath(root, 'note.md');
    const failed = one(
      (
        await ingestWorkspace(active, { contentFiles: ['note.md'] }, {
          sidecarPublish: () => {
            throw new Error('injected sidecar publish failure');
          },
        })
      ).items,
    );
    assert.equal(failed.status, 'failed');
    assert.match(failed.diagnostics[0].message, /injected sidecar publish failure/);
    const durableChunkIds = [...active.state.chunks.keys()].sort();
    assert.ok(durableChunkIds.length >= 2);
    assert.equal(readIngestionCatalog(root)?.pendingMaterializations.length, 1);

    crash(root);
    active = await openWorkspace(root);
    const recovered = one((await ingestWorkspace(active, { contentFiles: ['note.md'] })).items);
    assert.ok(recovered.representation);
    assert.equal(renderChunk(active.state, recovered.representation!.rootChunkId), '# Durable Markdown\n\nSecond block');
    assert.deepEqual([...active.state.chunks.keys()].sort(), durableChunkIds);
    assert.equal((JSON.parse(readFileSync(recoveredSidecar, 'utf8')) as { docChunkId: string }).docChunkId, recovered.representation!.rootChunkId);
    assert.equal(readIngestionCatalog(root)?.pendingMaterializations.length, 0);
    assert.equal(readIngestionCatalog(root)?.sources.length, 1);
    active.close();
    active = null;
  }

  // A newer external observation invalidates every older proposal for that
  // source. Stale v2 can never be accepted after the file has advanced to v3.
  {
    const root = freshRoot('headspace-ingestion-proposal-freshness-');
    const sourcePath = join(root, 'note.txt');
    writeFileSync(sourcePath, 'v1');
    active = await openWorkspace(root);
    const first = one((await ingestWorkspace(active, { contentFiles: ['note.txt'] })).items);
    const leafId = first.representation!.contentChunkIds[0];
    await revise(active.ctxFor('human:test'), { chunkId: leafId, text: 'human', mediaType: 'text/plain' });

    writeFileSync(sourcePath, 'v2');
    const v2 = one((await ingestWorkspace(active, { contentFiles: ['note.txt'] })).items);
    assert.equal(v2.status, 'proposal');
    const p2 = v2.proposalId!;
    assert.equal(active.state.proposals.get(p2)?.status, 'open');

    writeFileSync(sourcePath, Buffer.from([0xc3, 0x28]));
    const unreadable = one((await ingestWorkspace(active, { contentFiles: ['note.txt'] })).items);
    assert.equal(unreadable.status, 'failed');
    assert.equal(active.state.proposals.get(p2)?.status, 'superseded');
    assert.ok(active.state.proposals.get(p2)?.resolution);
    assert.equal(revisionText(active.state, active.state.chunks.get(leafId)!.currentRevisionId), 'human');
    await assert.rejects(acceptProposal(active.ctxFor('human:test'), { proposalId: p2 }), /is superseded/);

    writeFileSync(sourcePath, 'v3');
    const v3 = one((await ingestWorkspace(active, { contentFiles: ['note.txt'] })).items);
    assert.equal(v3.status, 'proposal');
    const p3 = v3.proposalId!;
    assert.notEqual(p3, p2);
    assert.equal(active.state.proposals.get(p2)?.status, 'superseded');
    assert.equal(active.state.proposals.get(p3)?.status, 'open');
    assert.equal(
      active.state.proposals.get(p3)?.payload.find((change) => change.op === 'revise')?.text,
      'v3',
    );
    assert.equal(
      [...active.state.proposals.values()].filter(
        (proposal) => proposal.status === 'open' && proposal.targetChunkIds.includes(leafId),
      ).length,
      1,
    );
    assert.equal(revisionText(active.state, active.state.chunks.get(leafId)!.currentRevisionId), 'human');
    active.close();
    active = null;
  }

  // Re-ingesting the same plain-text observation after another human edit must
  // rebase the proposal instead of returning a permanently stale standing one.
  {
    const root = freshRoot('headspace-ingestion-text-proposal-basis-');
    const sourcePath = join(root, 'note.txt');
    writeFileSync(sourcePath, 'alpha beta gamma');
    active = await openWorkspace(root);
    const first = one((await ingestWorkspace(active, { contentFiles: ['note.txt'] })).items);
    const leafId = first.representation!.contentChunkIds[0];
    await revise(active.ctxFor('human:test'), { chunkId: leafId, text: 'alpha beta human one', mediaType: 'text/plain' });
    writeFileSync(sourcePath, 'alpha beta file change');
    const firstProposal = one((await ingestWorkspace(active, { contentFiles: ['note.txt'] })).items);
    assert.equal(firstProposal.status, 'proposal');

    await revise(active.ctxFor('human:test'), { chunkId: leafId, text: 'alpha beta human two', mediaType: 'text/plain' });
    const secondHumanHead = currentRevision(active.state, leafId).id;
    const rebased = one((await ingestWorkspace(active, { contentFiles: ['note.txt'] })).items);
    assert.equal(rebased.observation.id, firstProposal.observation.id, 'external observation stayed identical');
    assert.notEqual(rebased.proposalId, firstProposal.proposalId);
    assert.equal(active.state.proposals.get(firstProposal.proposalId!)?.status, 'superseded');
    assert.deepEqual(active.state.proposals.get(rebased.proposalId!)?.basisRevisionIds, [secondHumanHead]);
    active.close();
    active = null;
  }

  // Markdown proposal reuse includes the complete current projection basis.
  // A leaf advancing H1 -> H2 therefore produces a fresh proposal even when
  // the file observation and proposed product did not change.
  {
    const root = freshRoot('headspace-ingestion-markdown-proposal-basis-');
    const sourcePath = join(root, 'note.md');
    writeFileSync(sourcePath, 'alpha beta gamma\n');
    active = await openWorkspace(root);
    const first = one((await ingestWorkspace(active, { contentFiles: ['note.md'] })).items);
    const leafId = first.representation!.contentChunkIds[0];
    await revise(active.ctxFor('human:test'), { chunkId: leafId, text: 'alpha beta human one', mediaType: 'text/markdown' });
    writeFileSync(sourcePath, 'alpha beta file change\n');
    const firstProposal = one((await ingestWorkspace(active, { contentFiles: ['note.md'] })).items);
    assert.equal(firstProposal.status, 'proposal');

    await revise(active.ctxFor('human:test'), { chunkId: leafId, text: 'alpha beta human two', mediaType: 'text/markdown' });
    const secondHumanHead = currentRevision(active.state, leafId).id;
    const rebased = one((await ingestWorkspace(active, { contentFiles: ['note.md'] })).items);
    assert.equal(rebased.observation.id, firstProposal.observation.id, 'external observation stayed identical');
    assert.notEqual(rebased.proposalId, firstProposal.proposalId);
    assert.equal(active.state.proposals.get(firstProposal.proposalId!)?.status, 'superseded');
    assert.ok(active.state.proposals.get(rebased.proposalId!)?.basisRevisionIds.includes(secondHumanHead));
    active.close();
    active = null;
  }

  // Lexical escapes are observable refusals, never silent omissions.
  {
    const envelope = freshRoot('headspace-ingestion-path-escape-');
    const root = join(envelope, 'workspace');
    mkdirSync(root);
    writeFileSync(join(envelope, 'outside.txt'), 'outside');
    active = await openWorkspace(root);
    const report = await ingestWorkspace(active, { contentFiles: ['../outside.txt'] });
    assert.equal(report.items.length, 0);
    assert.equal(report.diagnostics[0]?.code, 'source.path-escape');
    assert.equal(active.state.chunks.size, 0);
    active.close();
    active = null;
  }

  // Windows path spelling is case-insensitive. Different casing must resolve
  // to the same durable source and representation instead of duplicating it.
  if (process.platform === 'win32') {
    const root = freshRoot('headspace-ingestion-windows-alias-');
    writeFileSync(join(root, 'note.txt'), 'same file');
    active = await openWorkspace(root);
    const lower = one((await ingestWorkspace(active, { contentFiles: ['note.txt'] })).items);
    const chunks = [...active.state.chunks.keys()].sort();
    const upper = one((await ingestWorkspace(active, { contentFiles: ['NOTE.TXT'] })).items);
    assert.equal(upper.observation.sourceId, lower.observation.sourceId);
    assert.equal(upper.representation?.rootChunkId, lower.representation?.rootChunkId);
    assert.deepEqual([...active.state.chunks.keys()].sort(), chunks);
    assert.equal(readIngestionCatalog(root)?.sources.length, 1);
    active.close();
    active = null;
  }

  console.log('ingestion recovery OK — write-ahead identity, stale proposal supersession, and path confinement');
} finally {
  active?.close();
  for (const root of roots) rmSync(root, { recursive: true, force: true });
}
