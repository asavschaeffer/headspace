import assert from 'node:assert';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { currentRevision, renderChunk } from '../src/kernel/state';
import { MEDIA_MARKDOWN } from '../src/kernel/types';
import {
  ingestWorkspace,
  ingestionAdapterCapabilities,
  ingestionCatalogPath,
  readIngestionCatalog,
  type IngestionItemResult,
  type IngestionRuntime,
} from '../src/host/ingestion';
import { sidecarPath } from '../src/host/markdown';
import { openWorkspace, type WorkspaceStore } from '../src/host/store-fs';

const envelope = mkdtempSync(join(tmpdir(), 'headspace-pdf-converter-'));
const root = join(envelope, 'workspace');
const docs = join(root, 'docs');
mkdirSync(docs, { recursive: true });

const pdfs = {
  good: Buffer.from('%PDF-good'),
  providerError: Buffer.from('%PDF-provider-error'),
  invalidUtf8: Buffer.from('%PDF-invalid-utf8'),
  wrongMedia: Buffer.from('%PDF-wrong-media'),
};
writeFileSync(join(docs, 'good.pdf'), pdfs.good);
writeFileSync(join(docs, 'provider-error.pdf'), pdfs.providerError);
writeFileSync(join(docs, 'invalid-utf8.pdf'), pdfs.invalidUtf8);
writeFileSync(join(docs, 'wrong-media.pdf'), pdfs.wrongMedia);
writeFileSync(join(docs, 'offline.txt'), 'native ingestion survives converter failures');

const byPath = (items: IngestionItemResult[], relPath: string): IngestionItemResult => {
  const item = items.find((candidate) => candidate.observation.relPath === relPath);
  assert.ok(item, `missing ingestion result for ${relPath}`);
  return item;
};

const offlineRuntime: IngestionRuntime = { environment: {} };
const calls: Array<{ url: string; method?: string; authorization?: string; body: Buffer }> = [];
const converterFetch: typeof globalThis.fetch = async (input, init) => {
  const body = Buffer.from(await new Response(init?.body).arrayBuffer());
  const headers = new Headers(init?.headers);
  calls.push({
    url: String(input),
    method: init?.method,
    authorization: headers.get('authorization') ?? undefined,
    body,
  });
  const marker = body.toString('utf8');
  if (marker === pdfs.providerError.toString('utf8')) return new Response(null, { status: 503 });
  if (marker === pdfs.invalidUtf8.toString('utf8')) {
    return new Response(Uint8Array.from([0xc3, 0x28]).buffer, {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });
  }
  if (marker === pdfs.wrongMedia.toString('utf8')) {
    return new Response('# Markdown without the declared envelope', {
      status: 200,
      headers: { 'content-type': 'text/markdown' },
    });
  }
  assert.equal(marker, pdfs.good.toString('utf8'));
  return new Response(
    JSON.stringify({
      mediaType: MEDIA_MARKDOWN,
      text: '# Converted PDF\n\nDerived Markdown body.\n',
      warnings: ['Page 2 contained low-confidence OCR'],
    }),
    { status: 200, headers: { 'content-type': 'application/json; charset=utf-8' } },
  );
};

const configuredRuntime: IngestionRuntime = {
  environment: {},
  fetch: converterFetch,
  pdfConverter: {
    url: 'https://converter.invalid/v1/pdf-to-markdown',
    bearerToken: 'host-secret-token',
    serviceIdentity: 'test-converter-tenant',
    implementationVersion: '2026.08',
  },
};

let ws: WorkspaceStore | null = null;
try {
  const offlineCapability = ingestionAdapterCapabilities(offlineRuntime).find(
    (capability) => capability.id === 'headspace.pdf-to-markdown.http',
  );
  assert.ok(offlineCapability, 'PDF capability remains discoverable while unavailable');
  assert.equal(offlineCapability.availability.status, 'unavailable');
  assert.equal(
    offlineCapability.availability.status === 'unavailable' && offlineCapability.availability.diagnostic.code,
    'adapter.pdf-converter-unconfigured',
  );
  assert.deepEqual(offlineCapability.outputs, [
    { relationship: 'derived', mediaType: MEDIA_MARKDOWN, writeback: 'none' },
  ]);

  ws = await openWorkspace(root);
  const offline = await ingestWorkspace(ws, { contentFiles: ['docs/good.pdf'] }, offlineRuntime);
  const unavailable = byPath(offline.items, 'docs/good.pdf');
  assert.equal(unavailable.status, 'unsupported');
  assert.equal(unavailable.adapter?.id, 'headspace.pdf-to-markdown.http');
  assert.equal(unavailable.diagnostics[0].code, 'adapter.pdf-converter-unconfigured');
  assert.equal(calls.length, 0, 'an unavailable adapter never attempts network access');
  const stableSourceId = unavailable.observation.sourceId;

  const configuredCapability = ingestionAdapterCapabilities(configuredRuntime).find(
    (capability) => capability.id === 'headspace.pdf-to-markdown.http',
  )!;
  assert.deepEqual(configuredCapability.availability, { status: 'ready' });
  assert.deepEqual(configuredCapability.provider, {
    identity: 'test-converter-tenant',
    implementationVersion: '2026.08',
  });

  const converted = await ingestWorkspace(
    ws,
    { contentFiles: ['docs/good.pdf', 'docs/invalid-utf8.pdf', 'docs/wrong-media.pdf', 'docs/offline.txt'] },
    configuredRuntime,
  );
  const good = byPath(converted.items, 'docs/good.pdf');
  assert.equal(good.status, 'imported');
  assert.equal(good.observation.sourceId, stableSourceId, 'adapter availability does not replace source identity');
  assert.equal(good.adapter?.id, 'headspace.pdf-to-markdown.http');
  assert.ok(good.representation);
  assert.equal(good.representation.relationship, 'derived');
  assert.equal(good.representation.mediaType, MEDIA_MARKDOWN);
  assert.equal(good.representation.sourceId, good.observation.sourceId);
  assert.equal(good.representation.observationId, good.observation.id);
  assert.deepEqual(good.representation.adapter, {
    id: 'headspace.pdf-to-markdown.http',
    version: '1.0.0',
    provider: { identity: 'test-converter-tenant', implementationVersion: '2026.08' },
  });
  assert.equal(good.representation.warnings[0].code, 'adapter.pdf-converter-warning');
  assert.match(good.representation.warnings[0].message, /low-confidence OCR/);
  assert.equal(renderChunk(ws.state, good.representation.rootChunkId), '# Converted PDF\n\nDerived Markdown body.\n');
  assert.equal(good.representation.contentChunkIds.length, 1);
  const derivedLeaf = currentRevision(ws.state, good.representation.contentChunkIds[0]);
  assert.equal(derivedLeaf.mediaType, MEDIA_MARKDOWN);
  assert.equal(derivedLeaf.createdBy, 'adapter:headspace.pdf-to-markdown.http@1.0.0');
  assert.equal(good.representation.operationIds.length, 1);
  const operation = ws.state.operations.get(good.representation.operationIds[0])!;
  assert.equal((operation.params as Record<string, unknown>).sourceId, good.observation.sourceId);
  assert.equal((operation.params as Record<string, unknown>).observationId, good.observation.id);
  assert.equal(existsSync(sidecarPath(root, 'docs/good.pdf')), false, 'derived Markdown never receives a native file sidecar');
  assert.deepEqual(readFileSync(join(docs, 'good.pdf')), pdfs.good, 'conversion never writes Markdown over the PDF');

  const native = byPath(converted.items, 'docs/offline.txt');
  assert.equal(native.status, 'imported', 'offline adapters continue after converter failures');
  assert.equal(native.adapter?.id, 'headspace.text.native');

  const providerRun = await ingestWorkspace(ws, { contentFiles: ['docs/provider-error.pdf'] }, configuredRuntime);
  const providerError = byPath(providerRun.items, 'docs/provider-error.pdf');
  assert.equal(providerError.status, 'failed');
  assert.match(providerError.diagnostics[0].message, /HTTP 503/);
  const invalidUtf8 = byPath(converted.items, 'docs/invalid-utf8.pdf');
  assert.equal(invalidUtf8.status, 'failed');
  assert.match(invalidUtf8.diagnostics[0].message, /not valid UTF-8/);
  const wrongMedia = byPath(converted.items, 'docs/wrong-media.pdf');
  assert.equal(wrongMedia.status, 'failed');
  assert.match(wrongMedia.diagnostics[0].message, /unsupported content type text\/markdown/);
  assert.equal(calls.length, 4);
  for (const call of calls) {
    assert.equal(call.url, 'https://converter.invalid/v1/pdf-to-markdown');
    assert.equal(call.method, 'POST');
    assert.equal(call.authorization, 'Bearer host-secret-token');
  }

  const catalogBeforeRestart = readIngestionCatalog(root)!;
  const durable = catalogBeforeRestart.representations.find((record) => record.id === good.representation!.id)!;
  assert.equal(durable.sourceId, good.observation.sourceId);
  assert.equal(durable.observationId, good.observation.id);
  assert.equal(durable.warnings[0].message, 'Page 2 contained low-confidence OCR');
  const durableCatalogText = readFileSync(ingestionCatalogPath(root), 'utf8');
  assert.equal(durableCatalogText.includes('host-secret-token'), false, 'host credentials are never persisted in the catalog');
  assert.equal(durableCatalogText.includes('converter.invalid'), false, 'the converter endpoint is never persisted in the catalog');
  assert.match(durableCatalogText, /test-converter-tenant/);
  assert.match(durableCatalogText, /2026\.08/);

  const rootChunkId = good.representation.rootChunkId;
  ws.close();
  ws = await openWorkspace(root);
  const reopened = await ingestWorkspace(ws, { contentDirs: ['docs'] }, offlineRuntime);
  const offlineDerived = byPath(reopened.items, 'docs/good.pdf');
  assert.equal(offlineDerived.status, 'unchanged', 'a durable derived representation remains usable while its provider is offline');
  assert.equal(offlineDerived.representation?.id, good.representation.id);
  assert.equal(offlineDerived.representation?.rootChunkId, rootChunkId);
  assert.equal(offlineDerived.representation?.warnings[0].message, 'Page 2 contained low-confidence OCR');
  assert.equal(offlineDerived.diagnostics[0].code, 'adapter.pdf-converter-unconfigured');
  assert.equal(calls.length, 4, 'restart reuse does not contact the converter');

  console.log('PDF converter tenant OK — derived Markdown is durable, non-projecting, and failure-isolated');
} finally {
  ws?.close();
  rmSync(envelope, { recursive: true, force: true });
}
