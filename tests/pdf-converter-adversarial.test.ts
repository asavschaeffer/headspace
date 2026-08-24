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
import { currentRevision, revisionText } from '../src/kernel/state';
import { revise } from '../src/kernel/tx';
import { MEDIA_MARKDOWN } from '../src/kernel/types';
import {
  ingestWorkspace,
  ingestionAdapterCapabilities,
  ingestionCatalogPath,
  readIngestionCatalog,
  type IngestionCatalog,
  type IngestionItemResult,
  type IngestionRunReport,
  type IngestionRuntime,
} from '../src/host/ingestion';
import { workspacePayload } from '../src/host/api';
import { WORKSPACE_ROOT, workspaceChildren } from '../src/client/workspaceTree';
import { openWorkspace, type WorkspaceStore } from '../src/host/store-fs';

const roots: string[] = [];
const freshRoot = (prefix: string): string => {
  const root = mkdtempSync(join(tmpdir(), prefix));
  roots.push(root);
  return root;
};
const byPath = (report: IngestionRunReport, relPath: string): IngestionItemResult => {
  const item = report.items.find((candidate) => candidate.observation.relPath === relPath);
  assert.ok(item, `missing ingestion result for ${relPath}`);
  return item;
};
const response = (text: string, warnings: string[] = []): Response =>
  new Response(JSON.stringify({ mediaType: MEDIA_MARKDOWN, text, warnings }), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });
const configured = (
  fetch: typeof globalThis.fetch,
  overrides: Partial<NonNullable<IngestionRuntime['pdfConverter']>> = {},
): IngestionRuntime => ({
  environment: {},
  fetch,
  pdfConverter: {
    url: 'https://pdf.example.test/convert',
    bearerToken: 'test-only-secret',
    serviceIdentity: 'tenant-a',
    implementationVersion: 'converter-2026.08',
    timeoutMs: 100,
    maxResponseBytes: 16 * 1024,
    ...overrides,
  },
});

let active: WorkspaceStore | null = null;
try {
  // Provider identity is public capability/provenance, while endpoint and
  // credentials remain host configuration. Credentials cannot cross cleartext
  // transport except to an explicit loopback host.
  {
    const neverFetch: typeof globalThis.fetch = async () => {
      throw new Error('capability inspection must not fetch');
    };
    const tenantA = ingestionAdapterCapabilities(configured(neverFetch))[2];
    const tenantB = ingestionAdapterCapabilities(
      configured(neverFetch, { serviceIdentity: 'tenant-b', implementationVersion: 'converter-9' }),
    )[2];
    assert.deepEqual(tenantA.provider, {
      identity: 'tenant-a',
      implementationVersion: 'converter-2026.08',
    });
    assert.deepEqual(tenantB.provider, { identity: 'tenant-b', implementationVersion: 'converter-9' });
    assert.notDeepEqual(tenantA.provider, tenantB.provider);

    const insecure = ingestionAdapterCapabilities(
      configured(neverFetch, { url: 'http://converter.example.test/convert' }),
    )[2];
    assert.equal(insecure.availability.status, 'unavailable');
    assert.equal(
      insecure.availability.status === 'unavailable' && insecure.availability.diagnostic.code,
      'adapter.pdf-converter-insecure-token-transport',
    );
    assert.deepEqual(insecure.provider, tenantA.provider);
    const loopback = ingestionAdapterCapabilities(
      configured(neverFetch, { url: 'http://127.0.0.1:4319/convert' }),
    )[2];
    assert.deepEqual(loopback.availability, { status: 'ready' });
    const embedded = ingestionAdapterCapabilities(
      configured(neverFetch, { url: 'https://user:supersecret@converter.example.test/convert' }),
    )[2];
    assert.equal(embedded.availability.status, 'unavailable');
    assert.equal(
      embedded.availability.status === 'unavailable' && embedded.availability.diagnostic.code,
      'adapter.pdf-converter-embedded-credentials',
    );
    assert.equal(JSON.stringify(embedded).includes('supersecret'), false);
    const fragmented = ingestionAdapterCapabilities(
      configured(neverFetch, { url: 'https://converter.example.test/convert#secret-fragment' }),
    )[2];
    assert.equal(fragmented.availability.status, 'unavailable');
  }

  // Credentials embedded in an endpoint are refused before fetch and can
  // never leak through the durable catalog or its UI/API projection.
  {
    const root = freshRoot('headspace-pdf-embedded-credential-');
    writeFileSync(join(root, 'credential.pdf'), '%PDF-credential');
    let calls = 0;
    active = await openWorkspace(root);
    const report = await ingestWorkspace(
      active,
      { contentFiles: ['credential.pdf'] },
      configured(async () => {
        calls++;
        throw new Error('must not fetch');
      }, { url: 'https://user:catalog-secret@converter.example.test/convert' }),
    );
    assert.equal(calls, 0);
    assert.equal(byPath(report, 'credential.pdf').status, 'unsupported');
    const catalogText = readFileSync(ingestionCatalogPath(root), 'utf8');
    assert.equal(catalogText.includes('catalog-secret'), false);
    assert.equal(catalogText.includes('user@'), false);
    active.close();
    active = null;
  }

  // Changing a configured converter tenant is an adapter change even if the
  // source bytes and produced Markdown are identical.
  {
    const root = freshRoot('headspace-pdf-provider-identity-');
    writeFileSync(join(root, 'same.pdf'), '%PDF-same');
    let calls = 0;
    const fetch: typeof globalThis.fetch = async () => {
      calls++;
      return response('# Same product\n', ['stable warning']);
    };
    active = await openWorkspace(root);
    const first = byPath(await ingestWorkspace(active, { contentFiles: ['same.pdf'] }, configured(fetch)), 'same.pdf');
    const second = byPath(
      await ingestWorkspace(
        active,
        { contentFiles: ['same.pdf'] },
        configured(fetch, { serviceIdentity: 'tenant-b', implementationVersion: 'converter-9' }),
      ),
      'same.pdf',
    );
    assert.equal(first.status, 'imported');
    assert.equal(second.status, 'updated', 'a tenant switch cannot silently reuse the old representation');
    assert.equal(calls, 2);
    assert.equal(second.observation.sourceId, first.observation.sourceId);
    assert.equal(second.observation.id, first.observation.id);
    assert.equal(second.representation?.rootChunkId, first.representation?.rootChunkId);
    assert.notEqual(second.representation?.id, first.representation?.id);
    assert.deepEqual(second.representation?.adapter.provider, {
      identity: 'tenant-b',
      implementationVersion: 'converter-9',
    });
    const apiSource = workspacePayload(root, active).sources.find(
      (source) => source.observation.relPath === 'same.pdf',
    );
    assert.deepEqual(apiSource?.representation?.adapter.provider, {
      identity: 'tenant-b',
      implementationVersion: 'converter-9',
    });
    const catalogText = readFileSync(ingestionCatalogPath(root), 'utf8');
    assert.match(catalogText, /tenant-b/);
    assert.equal(catalogText.includes('pdf.example.test'), false);
    assert.equal(catalogText.includes('test-only-secret'), false);
    active.close();
    active = await openWorkspace(root);
    await ingestWorkspace(active, { contentFiles: ['same.pdf'] }, { environment: {} });
    const restartedPayload = workspacePayload(root, active);
    const restartedNode = workspaceChildren(restartedPayload.sources, WORKSPACE_ROOT).find(
      (node) => node.path === 'same.pdf',
    );
    assert.ok(restartedNode);
    assert.equal(
      restartedNode.adapterLabel,
      'headspace.pdf-to-markdown.http@1.0.0 via tenant-b@converter-9',
    );
    assert.ok(restartedNode.diagnostics.includes('stable warning'));
    assert.ok(restartedNode.diagnostics.some((message) => /HEADSPACE_PDF_CONVERTER_URL/.test(message)));
    assert.equal(calls, 2, 'offline restart reuses durable conversion without network access');
    active.close();
    active = null;
  }

  // Timeout is bounded, aborts the request, and opens a run-scoped circuit so
  // later PDFs do not each consume another deadline. Native work still runs.
  {
    const root = freshRoot('headspace-pdf-timeout-circuit-');
    writeFileSync(join(root, 'a.pdf'), '%PDF-a');
    writeFileSync(join(root, 'b.pdf'), '%PDF-b');
    writeFileSync(join(root, 'z.txt'), 'offline still works');
    let calls = 0;
    let aborted = false;
    const hungFetch: typeof globalThis.fetch = async (_input, init) => {
      calls++;
      return await new Promise<Response>((_resolve, reject) => {
        const signal = init?.signal;
        assert.ok(signal);
        signal.addEventListener(
          'abort',
          () => {
            aborted = true;
            reject(new Error('request aborted'));
          },
          { once: true },
        );
      });
    };
    active = await openWorkspace(root);
    const started = Date.now();
    const report = await ingestWorkspace(
      active,
      { contentDirs: ['.'] },
      configured(hungFetch, { timeoutMs: 20 }),
    );
    assert.ok(Date.now() - started < 1_000, 'a hung provider is bounded by one short deadline');
    assert.equal(byPath(report, 'a.pdf').diagnostics[0].code, 'adapter.pdf-converter-timeout');
    assert.equal(byPath(report, 'b.pdf').diagnostics[0].code, 'adapter.pdf-converter-circuit-open');
    assert.equal(byPath(report, 'z.txt').status, 'imported');
    assert.equal(calls, 1);
    assert.equal(aborted, true);
    active.close();
    active = null;
  }

  // The deadline also bounds a response that delivered headers and a valid
  // JSON prefix but then stalled mid-body. No partial product reaches the
  // kernel, and the next PDF observes the run circuit.
  {
    const root = freshRoot('headspace-pdf-response-stall-');
    writeFileSync(join(root, 'a.pdf'), '%PDF-stalled-a');
    writeFileSync(join(root, 'b.pdf'), '%PDF-stalled-b');
    let calls = 0;
    let streamAborted = false;
    const stalledFetch: typeof globalThis.fetch = async (_input, init) => {
      calls++;
      const signal = init?.signal;
      assert.ok(signal);
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(new TextEncoder().encode('{"mediaType":"text/markdown","text":"partial'));
          signal.addEventListener(
            'abort',
            () => {
              streamAborted = true;
              controller.error(new Error('response stream aborted'));
            },
            { once: true },
          );
        },
      });
      return new Response(body, { status: 200, headers: { 'content-type': 'application/json' } });
    };
    active = await openWorkspace(root);
    const report = await ingestWorkspace(
      active,
      { contentDirs: ['.'] },
      configured(stalledFetch, { timeoutMs: 20 }),
    );
    const first = byPath(report, 'a.pdf');
    const second = byPath(report, 'b.pdf');
    assert.equal(first.status, 'failed');
    assert.equal(first.diagnostics[0].code, 'adapter.pdf-converter-timeout');
    assert.equal(first.representation, null);
    assert.equal(second.diagnostics[0].code, 'adapter.pdf-converter-circuit-open');
    assert.equal(calls, 1);
    assert.equal(streamAborted, true);
    assert.equal(active.state.chunks.size, 0);
    assert.equal(active.state.proposals.size, 0);
    assert.equal(readIngestionCatalog(root)?.representations.length, 0);
    assert.equal(readIngestionCatalog(root)?.pendingMaterializations.length, 0);
    active.close();
    active = null;
  }

  // Successful deadlines clean up their timer; retryable provider and network
  // failures trip the same one-run circuit, while the byte cap is a visible
  // non-retryable failure.
  {
    const successRoot = freshRoot('headspace-pdf-deadline-cleanup-');
    writeFileSync(join(successRoot, 'ok.pdf'), '%PDF-ok');
    let successfulSignal: AbortSignal | null = null;
    const fastFetch: typeof globalThis.fetch = async (_input, init) => {
      successfulSignal = init?.signal ?? null;
      return response('# Fast\n');
    };
    active = await openWorkspace(successRoot);
    await ingestWorkspace(active, { contentFiles: ['ok.pdf'] }, configured(fastFetch, { timeoutMs: 20 }));
    await new Promise((resolve) => setTimeout(resolve, 40));
    assert.ok(successfulSignal);
    assert.equal((successfulSignal as AbortSignal).aborted, false, 'successful request timer was cleared');
    active.close();
    active = null;

    for (const failure of ['provider', 'network'] as const) {
      const root = freshRoot(`headspace-pdf-${failure}-circuit-`);
      writeFileSync(join(root, 'a.pdf'), '%PDF-a');
      writeFileSync(join(root, 'b.pdf'), '%PDF-b');
      let calls = 0;
      const failingFetch: typeof globalThis.fetch = async () => {
        calls++;
        if (failure === 'provider') return new Response(null, { status: 503 });
        throw new Error('connection refused at https://pdf.example.test/convert with Bearer test-only-secret');
      };
      active = await openWorkspace(root);
      const report = await ingestWorkspace(active, { contentDirs: ['.'] }, configured(failingFetch));
      assert.equal(
        byPath(report, 'a.pdf').diagnostics[0].code,
        failure === 'provider' ? 'adapter.pdf-converter-provider-error' : 'adapter.pdf-converter-fetch-failed',
      );
      assert.equal(byPath(report, 'b.pdf').diagnostics[0].code, 'adapter.pdf-converter-circuit-open');
      assert.equal(calls, 1);
      if (failure === 'network') {
        const catalogText = readFileSync(ingestionCatalogPath(root), 'utf8');
        assert.equal(catalogText.includes('pdf.example.test'), false, 'transport diagnostics do not persist endpoint URLs');
        assert.equal(catalogText.includes('test-only-secret'), false, 'transport diagnostics do not persist credentials');
      }
      active.close();
      active = null;
    }

    const capRoot = freshRoot('headspace-pdf-response-cap-');
    writeFileSync(join(capRoot, 'large.pdf'), '%PDF-large');
    active = await openWorkspace(capRoot);
    const capped = await ingestWorkspace(
      active,
      { contentFiles: ['large.pdf'] },
      configured(async () => response('x'.repeat(2_000)), { maxResponseBytes: 64 }),
    );
    assert.equal(byPath(capped, 'large.pdf').status, 'failed');
    assert.equal(byPath(capped, 'large.pdf').diagnostics[0].code, 'adapter.pdf-converter-response-too-large');
    active.close();
    active = null;
  }

  // A derived clean update writes its exact intent before revise. If catalog
  // publication fails after the kernel commit, restart binds that operation
  // and its warnings without converter access or a duplicate revision.
  {
    const root = freshRoot('headspace-pdf-revise-recovery-');
    const sourcePath = join(root, 'recover.pdf');
    writeFileSync(sourcePath, '%PDF-v1');
    let calls = 0;
    const fetch: typeof globalThis.fetch = async (_input, init) => {
      calls++;
      const marker = Buffer.from(await new Response(init?.body).arrayBuffer()).toString('utf8');
      if (marker === '%PDF-v1') return response('# Version one\n', ['v1 warning']);
      if (marker === '%PDF-v2') return response('# Version two\n', ['v2 exact warning']);
      return response('# Version three\n', ['v3 warning']);
    };
    active = await openWorkspace(root);
    const first = byPath(await ingestWorkspace(active, { contentFiles: ['recover.pdf'] }, configured(fetch)), 'recover.pdf');
    const leafId = first.representation!.contentChunkIds[0];
    writeFileSync(sourcePath, '%PDF-v2');
    let injected = false;
    await assert.rejects(
      ingestWorkspace(
        active,
        { contentFiles: ['recover.pdf'] },
        {
          ...configured(fetch),
          catalogPublish: (temporaryPath, destinationPath) => {
            const attempted = JSON.parse(readFileSync(temporaryPath, 'utf8')) as IngestionCatalog;
            const source = attempted.sources.find((candidate) => candidate.currentRelPath === 'recover.pdf');
            const bound = attempted.representations.find((candidate) => candidate.id === source?.currentRepresentationId);
            if (
              !injected &&
              attempted.pendingMaterializations.length === 0 &&
              bound?.observationId === source?.currentObservationId &&
              bound?.warnings.some((warning) => warning.message === 'v2 exact warning') === true
            ) {
              injected = true;
              throw new Error('injected derived catalog publish failure');
            }
            renameSync(temporaryPath, destinationPath);
          },
        },
      ),
      /injected derived catalog publish failure/,
    );
    assert.equal(injected, true);
    const converterRevisionId = currentRevision(active.state, leafId).id;
    assert.equal(revisionText(active.state, converterRevisionId), '# Version two\n');
    const pendingCatalog = readIngestionCatalog(root)!;
    assert.equal(pendingCatalog.pendingMaterializations.length, 1);
    assert.equal(pendingCatalog.pendingMaterializations[0].operationKind, 'revise');
    assert.equal(pendingCatalog.pendingMaterializations[0].warnings[0].message, 'v2 exact warning');
    assert.equal(pendingCatalog.pendingMaterializations[0].priorRepresentationId, first.representation?.id);
    assert.deepEqual(
      pendingCatalog.pendingMaterializations[0].priorOutputRevisionIds,
      first.representation?.outputRevisionIds,
    );
    const correlatedOperation = [...active.state.operations.values()].find(
      (operation) =>
        (operation.params as Record<string, unknown> | undefined)?.ingestionToken ===
        pendingCatalog.pendingMaterializations[0].token,
    );
    assert.ok(correlatedOperation);
    assert.equal(
      (correlatedOperation.params as Record<string, unknown>).productIdentityHash,
      pendingCatalog.pendingMaterializations[0].productIdentityHash,
    );
    // Exercise the migration fallback too: old pending records lack the new
    // embedded provenance but still point at their durable prior binding.
    delete pendingCatalog.pendingMaterializations[0].priorRepresentationId;
    delete pendingCatalog.pendingMaterializations[0].priorOutputRevisionIds;
    writeFileSync(ingestionCatalogPath(root), `${JSON.stringify(pendingCatalog, null, 2)}\n`);

    const human = await revise(active.ctxFor('human:after-converter'), {
      chunkId: leafId,
      text: '# Human after converter\n',
      mediaType: MEDIA_MARKDOWN,
    });
    const humanRevisionId = human.revisionId;
    const humanOperationId = human.commit.operation.id;
    const revisionCount = active.state.revisions.size;
    active.close();
    active = null;

    active = await openWorkspace(root);
    const recovered = byPath(
      await ingestWorkspace(active, { contentFiles: ['recover.pdf'] }, { environment: {} }),
      'recover.pdf',
    );
    assert.equal(recovered.status, 'unchanged');
    assert.equal(currentRevision(active.state, leafId).id, humanRevisionId);
    assert.equal(revisionText(active.state, humanRevisionId), '# Human after converter\n');
    assert.equal(active.state.revisions.size, revisionCount, 'recovery did not manufacture another revision');
    assert.equal(recovered.representation?.observationId, recovered.observation.id);
    assert.equal(recovered.representation?.warnings[0].message, 'v2 exact warning');
    assert.ok(recovered.representation?.outputRevisionIds.includes(converterRevisionId));
    assert.equal(recovered.representation?.outputRevisionIds.includes(humanRevisionId), false);
    assert.ok(recovered.representation?.operationIds.includes(correlatedOperation.id));
    assert.equal(recovered.representation?.operationIds.includes(humanOperationId), false);
    assert.deepEqual(recovered.representation?.adapter.provider, {
      identity: 'tenant-a',
      implementationVersion: 'converter-2026.08',
    });
    assert.equal(readIngestionCatalog(root)?.pendingMaterializations.length, 0);
    assert.equal(calls, 2, 'offline recovery did not invoke the converter');

    writeFileSync(sourcePath, '%PDF-v3');
    const laterChange = byPath(
      await ingestWorkspace(active, { contentFiles: ['recover.pdf'] }, configured(fetch)),
      'recover.pdf',
    );
    assert.equal(laterChange.status, 'proposal');
    assert.equal(
      active.state.proposals.get(laterChange.proposalId!)?.payload.find((change) => change.op === 'revise')?.text,
      '# Version three\n',
    );
    assert.equal(currentRevision(active.state, leafId).id, humanRevisionId);
    assert.equal(revisionText(active.state, humanRevisionId), '# Human after converter\n');
    assert.equal(calls, 3);
    active.close();
    active = null;
  }

  // A conflicting retry is keyed by exact converter product identity. Changed
  // output/warnings supersede the standing proposal and persist structured
  // producer provenance through log replay.
  {
    const root = freshRoot('headspace-pdf-proposal-product-');
    const sourcePath = join(root, 'conflict.pdf');
    writeFileSync(sourcePath, '%PDF-base');
    let mode: 'base' | 'candidate-a' | 'candidate-b' = 'base';
    const fetch: typeof globalThis.fetch = async () =>
      mode === 'base'
        ? response('# Base\n')
        : mode === 'candidate-a'
          ? response('# Candidate A\n', ['warning A'])
          : response('# Candidate B\n', ['warning B']);
    active = await openWorkspace(root);
    const first = byPath(await ingestWorkspace(active, { contentFiles: ['conflict.pdf'] }, configured(fetch)), 'conflict.pdf');
    const leafId = first.representation!.contentChunkIds[0];
    await revise(active.ctxFor('human:test'), {
      chunkId: leafId,
      text: '# Human-owned edit\n',
      mediaType: MEDIA_MARKDOWN,
    });
    writeFileSync(sourcePath, '%PDF-new-observation');
    mode = 'candidate-a';
    const candidateA = byPath(
      await ingestWorkspace(active, { contentFiles: ['conflict.pdf'] }, configured(fetch)),
      'conflict.pdf',
    );
    assert.equal(candidateA.status, 'proposal');
    assert.equal(candidateA.diagnostics[0].message, 'warning A');
    const proposalA = active.state.proposals.get(candidateA.proposalId!)!;
    assert.equal(proposalA.status, 'open');

    await revise(active.ctxFor('human:test'), {
      chunkId: leafId,
      text: '# Human-owned edit two\n',
      mediaType: MEDIA_MARKDOWN,
    });
    const secondHumanHead = currentRevision(active.state, leafId).id;
    const candidateARebased = byPath(
      await ingestWorkspace(active, { contentFiles: ['conflict.pdf'] }, configured(fetch)),
      'conflict.pdf',
    );
    assert.equal(candidateARebased.status, 'proposal');
    assert.notEqual(candidateARebased.proposalId, candidateA.proposalId);
    assert.equal(active.state.proposals.get(candidateA.proposalId!)?.status, 'superseded');
    assert.deepEqual(active.state.proposals.get(candidateARebased.proposalId!)?.basisRevisionIds, [secondHumanHead]);

    mode = 'candidate-b';
    const candidateB = byPath(
      await ingestWorkspace(active, { contentFiles: ['conflict.pdf'] }, configured(fetch)),
      'conflict.pdf',
    );
    assert.equal(candidateB.status, 'proposal');
    assert.equal(candidateB.observation.id, candidateA.observation.id);
    assert.equal(candidateB.diagnostics[0].message, 'warning B');
    assert.notEqual(candidateB.proposalId, candidateA.proposalId);
    assert.equal(active.state.proposals.get(candidateARebased.proposalId!)?.status, 'superseded');
    const proposalB = active.state.proposals.get(candidateB.proposalId!)!;
    assert.equal(proposalB.status, 'open');
    assert.equal(
      proposalB.payload.find((change) => change.op === 'revise')?.text,
      '# Candidate B\n',
    );
    assert.match(proposalB.note ?? '', /warning B/);
    assert.equal((proposalB.note ?? '').includes('warning A'), false);
    assert.deepEqual(proposalB.producer, {
      id: 'tenant-a',
      version: 'converter-2026.08',
      implementation: 'headspace.pdf-to-markdown.http@1.0.0',
      receiptId: proposalB.producer?.receiptId,
    });
    assert.match(proposalB.producer?.receiptId ?? '', /^[a-f0-9]{64}$/);
    const proposeOperation = active.state.operations.get(proposalB.operationId!)!;
    const params = proposeOperation.params as Record<string, unknown>;
    assert.equal(params.sourceId, candidateB.observation.sourceId);
    assert.equal(params.observationId, candidateB.observation.id);
    assert.equal(params.productIdentityHash, proposalB.producer?.receiptId);
    assert.deepEqual(params.adapter, candidateB.adapter);
    assert.deepEqual(params.warnings, [
      {
        code: 'adapter.pdf-converter-warning',
        severity: 'warning',
        phase: 'adapt',
        message: 'warning B',
      },
    ]);
    assert.equal(params.kind, 'reconciliation');
    assert.deepEqual(params.producer, proposalB.producer);
    active.close();
    active = null;

    active = await openWorkspace(root);
    const restarted = active.state.proposals.get(candidateB.proposalId!)!;
    assert.deepEqual(restarted.producer, proposalB.producer);
    assert.deepEqual(active.state.operations.get(restarted.operationId!)?.params, params);
    assert.equal(active.state.proposals.get(candidateA.proposalId!)?.status, 'superseded');
    active.close();
    active = null;
  }

  console.log('PDF converter adversarial cases OK — bounded remote work, exact recovery, and durable provenance');
} finally {
  active?.close();
  for (const root of roots) rmSync(root, { recursive: true, force: true });
}
