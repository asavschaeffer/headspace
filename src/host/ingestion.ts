// Observable ingestion seam: confined source observation, adapter selection,
// representation materialization, and durable external-source provenance.
// Parsers produce data; the host owns kernel transactions and bindings.

import { createHash, randomUUID } from 'node:crypto';
import {
  existsSync,
  lstatSync,
  readFileSync,
  readdirSync,
  readlinkSync,
  realpathSync,
} from 'node:fs';
import { extname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { METHOD_BLOCKS, decomposeText } from '../kernel/decompose';
import { childOccurrences, currentRevision, renderChunk, revisionText } from '../kernel/state';
import { createComposite, propose, revise, supersedeProposal, type TxCtx } from '../kernel/tx';
import { MEDIA_COMPOSITE, MEDIA_MARKDOWN, MEDIA_TEXT, type Operation } from '../kernel/types';
import { atomicWriteText, type AtomicPublish } from './atomic-file';
import { importMarkdownFile, reconcileMarkdownFile, recoverMarkdownImport, sidecarPath } from './markdown';
import type { WorkspaceStore } from './store-fs';
import { HEADSPACE_DATA_DIRNAME, workspaceDataPaths } from './workspace-data';

const CATALOG_SCHEMA_VERSION = 1;
const SKIP = new Set(['node_modules', '.git', HEADSPACE_DATA_DIRNAME, 'dist', 'public']);

export type SourceId = string;
export type ObservationId = string;
export type RepresentationId = string;
export type SourceKind = 'file' | 'directory' | 'symlink' | 'other';
export type IngestionStatus = 'imported' | 'updated' | 'unchanged' | 'proposal' | 'unsupported' | 'failed';

export interface SourceFingerprint {
  algorithm: 'sha256';
  value: string;
  basis: 'file-bytes' | 'directory-entries' | 'symlink-target' | 'metadata';
}

export interface SourceObservation {
  id: ObservationId;
  sourceId: SourceId;
  identityKey: string;
  kind: SourceKind;
  relPath: string;
  mediaType: string;
  sizeBytes: number;
  fingerprint: SourceFingerprint;
  symlink: {
    status: 'not-symlink' | 'unfollowed-inside-root' | 'unfollowed-outside-root' | 'unresolved';
  };
  observedAt: string;
}

export interface IngestionDiagnostic {
  code: string;
  severity: 'info' | 'warning' | 'error';
  phase: 'observe' | 'select' | 'read' | 'adapt' | 'materialize' | 'bind';
  message: string;
  retryable?: boolean;
}

export interface AdapterRef {
  id: string;
  version: string;
  provider?: {
    identity: string;
    implementationVersion: string;
  };
}

export interface IngestionAdapterCapability extends AdapterRef {
  acceptedKinds: SourceKind[];
  acceptedMediaTypes: string[];
  outputs: Array<{
    relationship: 'native' | 'derived';
    mediaType: string;
    writeback: 'round-trip' | 'none';
  }>;
  availability: { status: 'ready' } | { status: 'unavailable'; diagnostic: IngestionDiagnostic };
}

export interface AdapterInput {
  observation: SourceObservation;
  readBytes(): Promise<Uint8Array>;
}

export interface AdapterProduct {
  relationship: 'native' | 'derived';
  mediaType: string;
  text: string;
  warnings: IngestionDiagnostic[];
}

export interface IngestionAdapter {
  capability: IngestionAdapterCapability;
  ingest(input: AdapterInput): Promise<AdapterProduct>;
}

const adapterRefOf = (capability: IngestionAdapterCapability): AdapterRef => ({
  id: capability.id,
  version: capability.version,
  ...(capability.provider ? { provider: { ...capability.provider } } : {}),
});

const sameAdapterRef = (a: AdapterRef, b: AdapterRef): boolean =>
  a.id === b.id &&
  a.version === b.version &&
  a.provider?.identity === b.provider?.identity &&
  a.provider?.implementationVersion === b.provider?.implementationVersion;

export interface RepresentationRecord {
  id: RepresentationId;
  sourceId: SourceId;
  observationId: ObservationId;
  relationship: 'native' | 'derived';
  mediaType: string;
  adapter: AdapterRef;
  rootChunkId: string;
  contentChunkIds: string[];
  outputRevisionIds: string[];
  operationIds: string[];
  warnings: IngestionDiagnostic[];
  createdAt: string;
}

export interface SourceRecord {
  id: SourceId;
  identityKey: string;
  currentObservationId: ObservationId;
  currentRelPath: string;
  currentRepresentationId?: RepresentationId;
}

interface PendingMaterializationBase {
  token: string;
  sourceId: SourceId;
  observationId: ObservationId;
  relPath: string;
  adapter: AdapterRef;
  relationship: 'native' | 'derived';
  mediaType: string;
  normalizedTextHash: string;
  productIdentityHash: string;
  warnings: IngestionDiagnostic[];
  startedAt: string;
}

export interface PendingImportMaterialization extends PendingMaterializationBase {
  operationKind: 'import';
  normalizedRenderedTextHash: string;
}

export interface PendingRevisionMaterialization extends PendingMaterializationBase {
  operationKind: 'revise';
  rootChunkId: string;
  contentChunkIds: string[];
  targetChunkId: string;
  basisRevisionId: string;
  priorRepresentationId: RepresentationId;
  priorOutputRevisionIds: string[];
}

export type PendingMaterialization = PendingImportMaterialization | PendingRevisionMaterialization;

export interface IngestionItemResult {
  status: IngestionStatus;
  observation: SourceObservation;
  adapter: AdapterRef | null;
  diagnostics: IngestionDiagnostic[];
  representation: RepresentationRecord | null;
  proposalId?: string;
}

export interface IngestionRunReport {
  id: string;
  startedAt: string;
  finishedAt: string;
  items: IngestionItemResult[];
  counts: Record<IngestionStatus, number>;
  diagnostics: IngestionDiagnostic[];
}

export interface IngestionCatalog {
  schemaVersion: typeof CATALOG_SCHEMA_VERSION;
  workspaceId: string;
  sources: SourceRecord[];
  observations: SourceObservation[];
  representations: RepresentationRecord[];
  pendingMaterializations: PendingMaterialization[];
  lastRun: IngestionRunReport | null;
}

export interface IngestionRuntime {
  catalogPublish?: AtomicPublish;
  sidecarPublish?: AtomicPublish;
  fetch?: typeof globalThis.fetch;
  environment?: Readonly<Record<string, string | undefined>>;
  pdfConverter?: {
    url?: string;
    bearerToken?: string;
    serviceIdentity?: string;
    implementationVersion?: string;
    timeoutMs?: number;
    maxResponseBytes?: number;
  };
}

interface CandidateObservation {
  kind: SourceKind;
  identityKey: string;
  relPath: string;
  mediaType: string;
  sizeBytes: number;
  fingerprint: SourceFingerprint;
  symlink: SourceObservation['symlink'];
}

interface RegisteredObservation {
  observation: SourceObservation;
  source: SourceRecord;
  newSource: boolean;
  newObservation: boolean;
}

const sha256 = (bytes: Uint8Array | string): string => createHash('sha256').update(bytes).digest('hex');
const normalizedRel = (root: string, abs: string): string => relative(root, abs).replaceAll('\\', '/') || '.';
const normalizedIdentityKey = (relPath: string): string =>
  process.platform === 'win32' ? relPath.toLowerCase() : relPath;
const identityKeyForRealPath = (realRoot: string, realPath: string): string =>
  normalizedIdentityKey(normalizedRel(realRoot, realPath));

function isWithin(root: string, candidate: string): boolean {
  const rel = relative(root, candidate);
  return rel === '' || (rel !== '..' && !rel.startsWith(`..${sep}`) && !isAbsolute(rel));
}

function mediaTypeOf(relPath: string): string {
  switch (extname(relPath).toLowerCase()) {
    case '.md':
    case '.markdown':
      return MEDIA_MARKDOWN;
    case '.txt':
    case '.text':
      return MEDIA_TEXT;
    case '.pdf':
      return 'application/pdf';
    case '.json':
      return 'application/json';
    case '.ts':
      return 'text/typescript';
    case '.tsx':
      return 'text/tsx';
    case '.js':
      return 'text/javascript';
    case '.py':
      return 'text/x-python';
    case '.png':
      return 'image/png';
    case '.jpg':
    case '.jpeg':
      return 'image/jpeg';
    default:
      return 'application/octet-stream';
  }
}

const diagnostic = (
  code: string,
  phase: IngestionDiagnostic['phase'],
  message: string,
  severity: IngestionDiagnostic['severity'] = 'error',
): IngestionDiagnostic => ({ code, phase, message, severity });

function observeConfiguredSources(
  workspaceRoot: string,
  opts: { contentDirs?: string[]; contentFiles?: string[] },
): { candidates: CandidateObservation[]; diagnostics: IngestionDiagnostic[] } {
  const root = resolve(workspaceRoot);
  const realRoot = realpathSync(root);
  const candidates = new Map<string, CandidateObservation>();
  const diagnostics: IngestionDiagnostic[] = [];

  const addSymlink = (abs: string): void => {
    const relPath = normalizedRel(root, abs);
    const identityKey = normalizedIdentityKey(relPath);
    let target = '';
    let status: SourceObservation['symlink']['status'] = 'unresolved';
    try {
      target = readlinkSync(abs);
      const resolvedTarget = realpathSync(abs);
      status = isWithin(realRoot, resolvedTarget) ? 'unfollowed-inside-root' : 'unfollowed-outside-root';
    } catch {
      // A broken or racing link remains visible but is never followed.
    }
    candidates.set(identityKey, {
      kind: 'symlink',
      identityKey,
      relPath,
      mediaType: 'inode/symlink',
      sizeBytes: lstatSync(abs).size,
      fingerprint: { algorithm: 'sha256', value: sha256(target), basis: 'symlink-target' },
      symlink: { status },
    });
  };

  const addRegularFile = (abs: string): void => {
    const relPath = normalizedRel(root, abs);
    const real = realpathSync(abs);
    if (!isWithin(root, abs) || !isWithin(realRoot, real)) {
      diagnostics.push(diagnostic('source.outside-root', 'observe', `refusing source outside workspace: ${relPath}`));
      return;
    }
    const stat = lstatSync(abs);
    const bytes = readFileSync(real);
    const identityKey = identityKeyForRealPath(realRoot, real);
    candidates.set(identityKey, {
      kind: 'file',
      identityKey,
      relPath,
      mediaType: mediaTypeOf(relPath),
      sizeBytes: stat.size,
      fingerprint: { algorithm: 'sha256', value: sha256(bytes), basis: 'file-bytes' },
      symlink: { status: 'not-symlink' },
    });
  };

  const walkDirectory = (abs: string): void => {
    const real = realpathSync(abs);
    if (!isWithin(root, abs) || !isWithin(realRoot, real)) {
      diagnostics.push(diagnostic('source.outside-root', 'observe', `refusing directory outside workspace: ${normalizedRel(root, abs)}`));
      return;
    }
    const entries = readdirSync(abs, { withFileTypes: true })
      .filter((entry) => !SKIP.has(entry.name) && !entry.name.startsWith('.'))
      .sort((a, b) => a.name.localeCompare(b.name));
    const relPath = normalizedRel(root, abs);
    const identityKey = identityKeyForRealPath(realRoot, real);
    const entryShape = entries.map((entry) => {
      const kind = entry.isSymbolicLink() ? 'symlink' : entry.isDirectory() ? 'directory' : entry.isFile() ? 'file' : 'other';
      return `${entry.name}\0${kind}`;
    });
    candidates.set(identityKey, {
      kind: 'directory',
      identityKey,
      relPath,
      mediaType: 'inode/directory',
      sizeBytes: lstatSync(abs).size,
      fingerprint: { algorithm: 'sha256', value: sha256(entryShape.join('\0')), basis: 'directory-entries' },
      symlink: { status: 'not-symlink' },
    });
    for (const entry of entries) {
      const full = join(abs, entry.name);
      try {
        if (entry.isSymbolicLink()) addSymlink(full);
        else if (entry.isDirectory()) walkDirectory(full);
        else if (entry.isFile()) addRegularFile(full);
        else {
          const otherRel = normalizedRel(root, full);
          const identityKey = normalizedIdentityKey(otherRel);
          const stat = lstatSync(full);
          candidates.set(identityKey, {
            kind: 'other',
            identityKey,
            relPath: otherRel,
            mediaType: 'application/octet-stream',
            sizeBytes: stat.size,
            fingerprint: {
              algorithm: 'sha256',
              value: sha256(`${stat.mode}\0${stat.size}\0${stat.mtimeMs}`),
              basis: 'metadata',
            },
            symlink: { status: 'not-symlink' },
          });
        }
      } catch (e) {
        diagnostics.push(
          diagnostic('source.observe-failed', 'observe', `${normalizedRel(root, full)}: ${(e as Error).message}`),
        );
      }
    }
  };

  const requestedPath = (requested: string, expected: 'directory' | 'file'): void => {
    const abs = resolve(root, requested);
    if (!isWithin(root, abs)) {
      diagnostics.push(diagnostic('source.path-escape', 'observe', `requested path escapes workspace: ${requested}`));
      return;
    }
    try {
      const stat = lstatSync(abs);
      if (stat.isSymbolicLink()) {
        addSymlink(abs);
        diagnostics.push(diagnostic('source.unfollowed-link', 'observe', `did not follow requested link: ${requested}`, 'warning'));
        return;
      }
      const real = realpathSync(abs);
      if (!isWithin(realRoot, real)) {
        diagnostics.push(diagnostic('source.outside-root', 'observe', `requested path resolves outside workspace: ${requested}`));
        return;
      }
      if (expected === 'directory' && stat.isDirectory()) walkDirectory(abs);
      else if (expected === 'file' && stat.isFile()) addRegularFile(abs);
      else diagnostics.push(diagnostic('source.kind-mismatch', 'observe', `requested ${expected} has another kind: ${requested}`));
    } catch (e) {
      diagnostics.push(diagnostic('source.missing', 'observe', `${requested}: ${(e as Error).message}`, 'warning'));
    }
  };

  for (const dir of opts.contentDirs ?? []) requestedPath(dir, 'directory');
  for (const file of opts.contentFiles ?? []) requestedPath(file, 'file');
  return { candidates: [...candidates.values()].sort((a, b) => a.relPath.localeCompare(b.relPath)), diagnostics };
}

export function ingestionCatalogPath(workspaceRoot: string): string {
  return workspaceDataPaths(workspaceRoot).ingestionCatalogPath;
}

const emptyCatalog = (): IngestionCatalog => ({
  schemaVersion: CATALOG_SCHEMA_VERSION,
  workspaceId: `workspace_${randomUUID()}`,
  sources: [],
  observations: [],
  representations: [],
  pendingMaterializations: [],
  lastRun: null,
});

type JsonObject = Record<string, unknown>;

const isJsonObject = (value: unknown): value is JsonObject =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

function requireCatalog(condition: unknown, path: string): asserts condition {
  if (!condition) throw new Error(`invalid ingestion catalog: ${path}`);
}

function requireStringArray(value: unknown, path: string): asserts value is string[] {
  requireCatalog(Array.isArray(value) && value.every((item) => typeof item === 'string'), `${path} must be a string array`);
}

function requireAdapterRef(value: unknown, path: string): asserts value is AdapterRef {
  requireCatalog(isJsonObject(value), `${path} must be an object`);
  requireCatalog(typeof value.id === 'string', `${path}.id must be a string`);
  requireCatalog(typeof value.version === 'string', `${path}.version must be a string`);
  if (value.provider !== undefined) {
    requireCatalog(isJsonObject(value.provider), `${path}.provider must be an object`);
    requireCatalog(typeof value.provider.identity === 'string', `${path}.provider.identity must be a string`);
    requireCatalog(
      typeof value.provider.implementationVersion === 'string',
      `${path}.provider.implementationVersion must be a string`,
    );
  }
}

function requireDiagnostic(value: unknown, path: string): asserts value is IngestionDiagnostic {
  requireCatalog(isJsonObject(value), `${path} must be an object`);
  requireCatalog(typeof value.code === 'string', `${path}.code must be a string`);
  requireCatalog(
    value.severity === 'info' || value.severity === 'warning' || value.severity === 'error',
    `${path}.severity is unsupported`,
  );
  requireCatalog(
    value.phase === 'observe' ||
      value.phase === 'select' ||
      value.phase === 'read' ||
      value.phase === 'adapt' ||
      value.phase === 'materialize' ||
      value.phase === 'bind',
    `${path}.phase is unsupported`,
  );
  requireCatalog(typeof value.message === 'string', `${path}.message must be a string`);
  requireCatalog(value.retryable === undefined || typeof value.retryable === 'boolean', `${path}.retryable must be boolean`);
}

function requireDiagnostics(value: unknown, path: string): asserts value is IngestionDiagnostic[] {
  requireCatalog(Array.isArray(value), `${path} must be an array`);
  value.forEach((diagnostic, index) => requireDiagnostic(diagnostic, `${path}[${index}]`));
}

function requireSource(value: unknown, path: string): asserts value is SourceRecord {
  requireCatalog(isJsonObject(value), `${path} must be an object`);
  for (const field of ['id', 'identityKey', 'currentObservationId', 'currentRelPath'] as const) {
    requireCatalog(typeof value[field] === 'string', `${path}.${field} must be a string`);
  }
  requireCatalog(
    value.currentRepresentationId === undefined || typeof value.currentRepresentationId === 'string',
    `${path}.currentRepresentationId must be a string`,
  );
}

function requireObservation(value: unknown, path: string): asserts value is SourceObservation {
  requireCatalog(isJsonObject(value), `${path} must be an object`);
  for (const field of ['id', 'sourceId', 'identityKey', 'relPath', 'mediaType', 'observedAt'] as const) {
    requireCatalog(typeof value[field] === 'string', `${path}.${field} must be a string`);
  }
  requireCatalog(
    value.kind === 'file' || value.kind === 'directory' || value.kind === 'symlink' || value.kind === 'other',
    `${path}.kind is unsupported`,
  );
  requireCatalog(typeof value.sizeBytes === 'number' && Number.isFinite(value.sizeBytes), `${path}.sizeBytes must be a number`);
  requireCatalog(isJsonObject(value.fingerprint), `${path}.fingerprint must be an object`);
  requireCatalog(value.fingerprint.algorithm === 'sha256', `${path}.fingerprint.algorithm is unsupported`);
  requireCatalog(typeof value.fingerprint.value === 'string', `${path}.fingerprint.value must be a string`);
  requireCatalog(
    value.fingerprint.basis === 'file-bytes' ||
      value.fingerprint.basis === 'directory-entries' ||
      value.fingerprint.basis === 'symlink-target' ||
      value.fingerprint.basis === 'metadata',
    `${path}.fingerprint.basis is unsupported`,
  );
  requireCatalog(isJsonObject(value.symlink), `${path}.symlink must be an object`);
  requireCatalog(
    value.symlink.status === 'not-symlink' ||
      value.symlink.status === 'unfollowed-inside-root' ||
      value.symlink.status === 'unfollowed-outside-root' ||
      value.symlink.status === 'unresolved',
    `${path}.symlink.status is unsupported`,
  );
}

function requireRepresentation(value: unknown, path: string): asserts value is RepresentationRecord {
  requireCatalog(isJsonObject(value), `${path} must be an object`);
  for (const field of ['id', 'sourceId', 'observationId', 'mediaType', 'rootChunkId', 'createdAt'] as const) {
    requireCatalog(typeof value[field] === 'string', `${path}.${field} must be a string`);
  }
  requireCatalog(value.relationship === 'native' || value.relationship === 'derived', `${path}.relationship is unsupported`);
  requireAdapterRef(value.adapter, `${path}.adapter`);
  requireStringArray(value.contentChunkIds, `${path}.contentChunkIds`);
  requireStringArray(value.outputRevisionIds, `${path}.outputRevisionIds`);
  requireStringArray(value.operationIds, `${path}.operationIds`);
  requireDiagnostics(value.warnings, `${path}.warnings`);
}

function requirePendingMaterialization(value: unknown, path: string): asserts value is PendingMaterialization {
  requireCatalog(isJsonObject(value), `${path} must be an object`);
  for (const field of [
    'token',
    'sourceId',
    'observationId',
    'relPath',
    'mediaType',
    'normalizedTextHash',
    'productIdentityHash',
    'startedAt',
  ] as const) {
    requireCatalog(typeof value[field] === 'string', `${path}.${field} must be a string`);
  }
  requireCatalog(value.relationship === 'native' || value.relationship === 'derived', `${path}.relationship is unsupported`);
  requireAdapterRef(value.adapter, `${path}.adapter`);
  requireDiagnostics(value.warnings, `${path}.warnings`);
  if (value.operationKind === 'import') {
    requireCatalog(
      typeof value.normalizedRenderedTextHash === 'string',
      `${path}.normalizedRenderedTextHash must be a string for import`,
    );
    return;
  }
  requireCatalog(value.operationKind === 'revise', `${path}.operationKind is unsupported`);
  for (const field of [
    'rootChunkId',
    'targetChunkId',
    'basisRevisionId',
    'priorRepresentationId',
  ] as const) {
    requireCatalog(typeof value[field] === 'string', `${path}.${field} must be a string for revise`);
  }
  requireStringArray(value.contentChunkIds, `${path}.contentChunkIds`);
  requireStringArray(value.priorOutputRevisionIds, `${path}.priorOutputRevisionIds`);
}

const INGESTION_STATUSES: readonly IngestionStatus[] = [
  'imported',
  'updated',
  'unchanged',
  'proposal',
  'unsupported',
  'failed',
];

function requireRunItem(value: unknown, path: string): asserts value is IngestionItemResult {
  requireCatalog(isJsonObject(value), `${path} must be an object`);
  requireCatalog(INGESTION_STATUSES.includes(value.status as IngestionStatus), `${path}.status is unsupported`);
  requireObservation(value.observation, `${path}.observation`);
  if (value.adapter !== null) requireAdapterRef(value.adapter, `${path}.adapter`);
  requireDiagnostics(value.diagnostics, `${path}.diagnostics`);
  if (value.representation !== null) requireRepresentation(value.representation, `${path}.representation`);
  requireCatalog(value.proposalId === undefined || typeof value.proposalId === 'string', `${path}.proposalId must be a string`);
}

function requireRun(value: unknown, path: string): asserts value is IngestionRunReport {
  requireCatalog(isJsonObject(value), `${path} must be an object`);
  for (const field of ['id', 'startedAt', 'finishedAt'] as const) {
    requireCatalog(typeof value[field] === 'string', `${path}.${field} must be a string`);
  }
  requireCatalog(Array.isArray(value.items), `${path}.items must be an array`);
  value.items.forEach((item, index) => requireRunItem(item, `${path}.items[${index}]`));
  requireCatalog(isJsonObject(value.counts), `${path}.counts must be an object`);
  for (const status of INGESTION_STATUSES) {
    requireCatalog(typeof value.counts[status] === 'number', `${path}.counts.${status} must be a number`);
  }
  requireDiagnostics(value.diagnostics, `${path}.diagnostics`);
}

function requireIngestionCatalog(value: unknown): asserts value is IngestionCatalog {
  requireCatalog(isJsonObject(value), 'root must be an object');
  if (value.schemaVersion !== CATALOG_SCHEMA_VERSION) {
    throw new Error(`unsupported ingestion catalog schema: ${String(value.schemaVersion)}`);
  }
  requireCatalog(typeof value.workspaceId === 'string', 'workspaceId must be a string');
  requireCatalog(Array.isArray(value.sources), 'sources must be an array');
  value.sources.forEach((source, index) => requireSource(source, `sources[${index}]`));
  requireCatalog(Array.isArray(value.observations), 'observations must be an array');
  value.observations.forEach((observation, index) => requireObservation(observation, `observations[${index}]`));
  requireCatalog(Array.isArray(value.representations), 'representations must be an array');
  value.representations.forEach((representation, index) =>
    requireRepresentation(representation, `representations[${index}]`),
  );
  requireCatalog(Array.isArray(value.pendingMaterializations), 'pendingMaterializations must be an array');
  value.pendingMaterializations.forEach((pending, index) =>
    requirePendingMaterialization(pending, `pendingMaterializations[${index}]`),
  );
  requireCatalog(value.lastRun === null || isJsonObject(value.lastRun), 'lastRun must be null or an object');
  if (value.lastRun !== null) requireRun(value.lastRun, 'lastRun');
}

export function readIngestionCatalog(workspaceRoot: string): IngestionCatalog | null {
  const path = ingestionCatalogPath(workspaceRoot);
  if (!existsSync(path)) return null;
  const parsed: unknown = JSON.parse(readFileSync(path, 'utf8'));
  requireIngestionCatalog(parsed);
  return parsed;
}

export function writeIngestionCatalog(
  workspaceRoot: string,
  catalog: IngestionCatalog,
  publish?: AtomicPublish,
): void {
  requireIngestionCatalog(catalog);
  atomicWriteText(ingestionCatalogPath(workspaceRoot), `${JSON.stringify(catalog, null, 2)}\n`, publish);
}

function registerObservation(catalog: IngestionCatalog, candidate: CandidateObservation, now: string): RegisteredObservation {
  let source = catalog.sources.find((item) => item.identityKey === candidate.identityKey);
  const newSource = !source;
  if (!source) {
    source = {
      id: `source_${randomUUID()}`,
      identityKey: candidate.identityKey,
      currentObservationId: '',
      currentRelPath: candidate.relPath,
    };
    catalog.sources.push(source);
  }
  const previous = catalog.observations.find((item) => item.id === source!.currentObservationId);
  const unchanged =
    previous?.kind === candidate.kind &&
    previous.mediaType === candidate.mediaType &&
    previous.fingerprint.value === candidate.fingerprint.value &&
    previous.fingerprint.basis === candidate.fingerprint.basis &&
    previous.symlink.status === candidate.symlink.status;
  const observation: SourceObservation = unchanged
    ? previous!
    : {
        ...candidate,
        id: `observation_${randomUUID()}`,
        sourceId: source.id,
        observedAt: now,
      };
  if (!unchanged) catalog.observations.push(observation);
  source.currentObservationId = observation.id;
  source.currentRelPath = candidate.relPath;
  return { observation, source, newSource, newObservation: !unchanged };
}

function decodeUtf8(bytes: Uint8Array): string {
  return new TextDecoder('utf-8', { fatal: true }).decode(bytes);
}

const nativeAdapter = (
  id: string,
  mediaType: string,
  writeback: 'round-trip' | 'none',
): IngestionAdapter => ({
  capability: {
    id,
    version: '1.0.0',
    acceptedKinds: ['file'],
    acceptedMediaTypes: [mediaType],
    outputs: [{ relationship: 'native', mediaType, writeback }],
    availability: { status: 'ready' },
  },
  ingest: async ({ readBytes }) => ({
    relationship: 'native',
    mediaType,
    text: decodeUtf8(await readBytes()),
    warnings: [],
  }),
});

const NATIVE_ADAPTERS: IngestionAdapter[] = [
  nativeAdapter('headspace.markdown.native', MEDIA_MARKDOWN, 'round-trip'),
  // Plain text is natively represented, but write-back is not advertised until
  // it has the same guarded projection manifest as Markdown.
  nativeAdapter('headspace.text.native', MEDIA_TEXT, 'none'),
];

const PDF_CONVERTER_ID = 'headspace.pdf-to-markdown.http';
const PDF_CONVERTER_VERSION = '1.0.0';
const DEFAULT_PDF_CONVERTER_TIMEOUT_MS = 30_000;
const DEFAULT_PDF_CONVERTER_MAX_RESPONSE_BYTES = 8 * 1024 * 1024;

class AdapterIngestionError extends Error {
  override name = 'AdapterIngestionError';

  constructor(
    readonly diagnosticCode: string,
    message: string,
    readonly retryable: boolean,
  ) {
    super(message);
  }
}

const timeoutError = (timeoutMs: number): AdapterIngestionError =>
  new AdapterIngestionError(
    'adapter.pdf-converter-timeout',
    `PDF converter timed out after ${timeoutMs}ms`,
    true,
  );

async function withConverterDeadline<T>(timeoutMs: number, run: (signal: AbortSignal) => Promise<T>): Promise<T> {
  const controller = new AbortController();
  let timeout: ReturnType<typeof setTimeout> | undefined;
  const deadline = new Promise<never>((_, reject) => {
    timeout = setTimeout(() => {
      controller.abort();
      reject(timeoutError(timeoutMs));
    }, timeoutMs);
  });
  try {
    return await Promise.race([run(controller.signal), deadline]);
  } finally {
    if (timeout) clearTimeout(timeout);
  }
}

async function readCappedResponseBytes(response: Response, maxBytes: number): Promise<Uint8Array> {
  const declaredLength = Number(response.headers.get('content-length'));
  if (Number.isFinite(declaredLength) && declaredLength > maxBytes) {
    try {
      await response.body?.cancel();
    } catch {
      // The cap failure is the useful diagnostic; cancellation is best effort.
    }
    throw new AdapterIngestionError(
      'adapter.pdf-converter-response-too-large',
      `PDF converter response exceeds ${maxBytes} bytes`,
      false,
    );
  }
  if (!response.body) return new Uint8Array();
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      total += value.byteLength;
      if (total > maxBytes) {
        try {
          await reader.cancel();
        } catch {
          // The cap failure is the useful diagnostic; cancellation is best effort.
        }
        throw new AdapterIngestionError(
          'adapter.pdf-converter-response-too-large',
          `PDF converter response exceeds ${maxBytes} bytes`,
          false,
        );
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  const bytes = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

interface PdfConverterResponse {
  mediaType: typeof MEDIA_MARKDOWN;
  text: string;
  warnings?: string[];
}

function pdfConverterAdapter(runtime: IngestionRuntime): IngestionAdapter {
  const environment = runtime.environment ?? process.env;
  const configuredUrl = runtime.pdfConverter?.url ?? environment.HEADSPACE_PDF_CONVERTER_URL;
  const bearerToken = runtime.pdfConverter?.bearerToken ?? environment.HEADSPACE_PDF_CONVERTER_TOKEN;
  const configuredServiceIdentity =
    runtime.pdfConverter?.serviceIdentity ?? environment.HEADSPACE_PDF_CONVERTER_SERVICE_IDENTITY;
  const configuredImplementationVersion =
    runtime.pdfConverter?.implementationVersion ?? environment.HEADSPACE_PDF_CONVERTER_IMPLEMENTATION_VERSION;
  const configuredProvider =
    configuredServiceIdentity?.trim() && configuredImplementationVersion?.trim()
      ? {
          identity: configuredServiceIdentity.trim(),
          implementationVersion: configuredImplementationVersion.trim(),
        }
      : undefined;
  const unavailable = (
    code: string,
    message: string,
    provider: AdapterRef['provider'] = configuredProvider,
  ): IngestionAdapter => ({
    capability: {
      id: PDF_CONVERTER_ID,
      version: PDF_CONVERTER_VERSION,
      ...(provider ? { provider } : {}),
      acceptedKinds: ['file'],
      acceptedMediaTypes: ['application/pdf'],
      outputs: [{ relationship: 'derived', mediaType: MEDIA_MARKDOWN, writeback: 'none' }],
      availability: { status: 'unavailable', diagnostic: diagnostic(code, 'select', message, 'warning') },
    },
    ingest: async () => {
      throw new Error(message);
    },
  });

  if (!configuredUrl?.trim()) {
    return unavailable(
      'adapter.pdf-converter-unconfigured',
      'no adapter available: No available adapter accepts application/pdf until HEADSPACE_PDF_CONVERTER_URL is configured',
    );
  }

  let url: URL;
  try {
    url = new URL(configuredUrl);
  } catch {
    return unavailable('adapter.pdf-converter-invalid-url', 'PDF conversion is unavailable: converter URL is invalid');
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    return unavailable(
      'adapter.pdf-converter-invalid-url',
      'PDF conversion is unavailable: converter URL must use HTTP or HTTPS',
    );
  }
  if (url.username || url.password) {
    return unavailable(
      'adapter.pdf-converter-embedded-credentials',
      'PDF conversion is unavailable: do not embed credentials in the converter URL; use the bearer-token setting',
    );
  }
  if (url.hash) {
    return unavailable(
      'adapter.pdf-converter-invalid-url',
      'PDF conversion is unavailable: converter URL must not include a fragment',
    );
  }
  if (!configuredServiceIdentity?.trim()) {
    return unavailable(
      'adapter.pdf-converter-identity-unconfigured',
      'PDF conversion is unavailable: configure a non-secret converter service identity',
    );
  }
  if (!configuredImplementationVersion?.trim()) {
    return unavailable(
      'adapter.pdf-converter-version-unconfigured',
      'PDF conversion is unavailable: configure the converter implementation version',
    );
  }
  const provider = configuredProvider!;
  const tokenUsesSecureTransport =
    url.protocol === 'https:' ||
    new Set(['localhost', '127.0.0.1', '::1', '[::1]']).has(url.hostname.toLowerCase());
  if (bearerToken && !tokenUsesSecureTransport) {
    return unavailable(
      'adapter.pdf-converter-insecure-token-transport',
      'PDF conversion is unavailable: bearer credentials require HTTPS except for an explicit loopback host',
      provider,
    );
  }
  const fetchImpl = runtime.fetch ?? (typeof globalThis.fetch === 'function' ? globalThis.fetch.bind(globalThis) : undefined);
  if (!fetchImpl) {
    return unavailable('adapter.pdf-converter-fetch-unavailable', 'PDF conversion is unavailable: this host has no fetch implementation');
  }
  const configuredTimeout = runtime.pdfConverter?.timeoutMs ?? environment.HEADSPACE_PDF_CONVERTER_TIMEOUT_MS;
  const configuredMaxBytes =
    runtime.pdfConverter?.maxResponseBytes ?? environment.HEADSPACE_PDF_CONVERTER_MAX_RESPONSE_BYTES;
  const timeoutMs = configuredTimeout === undefined ? DEFAULT_PDF_CONVERTER_TIMEOUT_MS : Number(configuredTimeout);
  const maxResponseBytes =
    configuredMaxBytes === undefined ? DEFAULT_PDF_CONVERTER_MAX_RESPONSE_BYTES : Number(configuredMaxBytes);
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs <= 0) {
    return unavailable(
      'adapter.pdf-converter-invalid-timeout',
      'PDF conversion is unavailable: converter timeout must be a positive integer',
    );
  }
  if (!Number.isSafeInteger(maxResponseBytes) || maxResponseBytes <= 0) {
    return unavailable(
      'adapter.pdf-converter-invalid-response-cap',
      'PDF conversion is unavailable: converter response cap must be a positive integer',
    );
  }
  let circuitFailure: AdapterIngestionError | null = null;

  return {
    capability: {
      id: PDF_CONVERTER_ID,
      version: PDF_CONVERTER_VERSION,
      provider,
      acceptedKinds: ['file'],
      acceptedMediaTypes: ['application/pdf'],
      outputs: [{ relationship: 'derived', mediaType: MEDIA_MARKDOWN, writeback: 'none' }],
      availability: { status: 'ready' },
    },
    // Minimal HTTP contract: POST the raw PDF as application/pdf and receive a
    // UTF-8 application/json body shaped as
    // { mediaType: "text/markdown", text: string, warnings?: string[] }.
    ingest: async ({ readBytes }) => {
      if (circuitFailure) {
        throw new AdapterIngestionError(
          'adapter.pdf-converter-circuit-open',
          `PDF converter skipped after ${circuitFailure.message}`,
          true,
        );
      }
      const pdfBytes = await readBytes();
      const headers: Record<string, string> = {
        accept: 'application/json',
        'content-type': 'application/pdf',
      };
      if (bearerToken) headers.authorization = `Bearer ${bearerToken}`;
      try {
        return await withConverterDeadline(timeoutMs, async (signal) => {
          let response: Response;
          try {
            response = await fetchImpl(url, {
            method: 'POST',
            headers,
            body: Uint8Array.from(pdfBytes).buffer,
            redirect: 'error',
            signal,
            });
          } catch (e) {
            if (signal.aborted) throw timeoutError(timeoutMs);
            throw new AdapterIngestionError(
              'adapter.pdf-converter-fetch-failed',
              'PDF converter request failed before a response was received',
              true,
            );
          }
          if (!response.ok) {
          try {
            await response.body?.cancel();
          } catch {
            // The provider status is the useful diagnostic.
          }
          throw new AdapterIngestionError(
            'adapter.pdf-converter-provider-error',
            `PDF converter returned HTTP ${response.status}`,
            response.status >= 500 || response.status === 429,
          );
          }
          const responseType = response.headers.get('content-type')?.split(';', 1)[0].trim().toLowerCase();
          if (responseType !== 'application/json') {
          try {
            await response.body?.cancel();
          } catch {
            // The media-type failure is the useful diagnostic.
          }
          throw new Error(`PDF converter returned unsupported content type ${responseType ?? '(missing)'}`);
          }
          let responseBytes: Uint8Array;
          try {
            responseBytes = await readCappedResponseBytes(response, maxResponseBytes);
          } catch (e) {
            if (e instanceof AdapterIngestionError) throw e;
            if (signal.aborted) throw timeoutError(timeoutMs);
            throw new AdapterIngestionError(
              'adapter.pdf-converter-fetch-failed',
              'PDF converter response stream failed before a complete bounded response was received',
              true,
            );
          }
          let decoded: string;
          try {
            decoded = decodeUtf8(responseBytes);
          } catch (e) {
            throw new Error(`PDF converter response is not valid UTF-8: ${(e as Error).message}`);
          }
          let payload: unknown;
          try {
            payload = JSON.parse(decoded);
          } catch (e) {
            throw new Error(`PDF converter response is not valid JSON: ${(e as Error).message}`);
          }
          if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
            throw new Error('PDF converter response must be a JSON object');
          }
          const candidate = payload as Partial<PdfConverterResponse>;
          if (candidate.mediaType !== MEDIA_MARKDOWN) {
            throw new Error(`PDF converter returned undeclared media type ${String(candidate.mediaType)}`);
          }
          if (typeof candidate.text !== 'string') throw new Error('PDF converter response lacks Markdown text');
          if (candidate.warnings !== undefined && !Array.isArray(candidate.warnings)) {
            throw new Error('PDF converter warnings must be an array of strings');
          }
          if (candidate.warnings?.some((warning) => typeof warning !== 'string')) {
            throw new Error('PDF converter warnings must be an array of strings');
          }
          return {
            relationship: 'derived',
            mediaType: MEDIA_MARKDOWN,
            text: candidate.text,
            warnings: (candidate.warnings ?? []).map((warning) =>
              diagnostic('adapter.pdf-converter-warning', 'adapt', warning, 'warning'),
            ),
          };
        });
      } catch (e) {
        if (e instanceof AdapterIngestionError && e.retryable) {
          circuitFailure = e;
        }
        throw e;
      }
    },
  };
}

function adaptersFor(runtime: IngestionRuntime): IngestionAdapter[] {
  return [...NATIVE_ADAPTERS, pdfConverterAdapter(runtime)];
}

export function ingestionAdapterCapabilities(runtime: IngestionRuntime = {}): IngestionAdapterCapability[] {
  return adaptersFor(runtime).map((adapter) => structuredClone(adapter.capability));
}

function confinedReadBytes(workspaceRoot: string, observation: SourceObservation): Uint8Array {
  const root = resolve(workspaceRoot);
  const realRoot = realpathSync(root);
  const abs = resolve(root, observation.relPath);
  if (!isWithin(root, abs)) throw new Error(`source path escaped after observation: ${observation.relPath}`);
  const stat = lstatSync(abs);
  if (!stat.isFile() || stat.isSymbolicLink()) throw new Error(`source kind changed after observation: ${observation.relPath}`);
  const real = realpathSync(abs);
  if (!isWithin(realRoot, real)) throw new Error(`source resolved outside workspace after observation: ${observation.relPath}`);
  const bytes = readFileSync(real);
  if (sha256(bytes) !== observation.fingerprint.value) throw new Error(`source changed during ingestion: ${observation.relPath}`);
  return bytes;
}

function currentRepresentation(catalog: IngestionCatalog, source: SourceRecord): RepresentationRecord | undefined {
  return source.currentRepresentationId
    ? catalog.representations.find((representation) => representation.id === source.currentRepresentationId)
    : undefined;
}

function representationSnapshot(
  ws: WorkspaceStore,
  rootChunkId: string,
  contentChunkIds: string[],
): { outputRevisionIds: string[]; operationIds: string[] } {
  const revisions = [rootChunkId, ...contentChunkIds].map((chunkId) => currentRevision(ws.state, chunkId));
  return {
    outputRevisionIds: revisions.map((revision) => revision.id),
    operationIds: [...new Set(revisions.map((revision) => revision.operationId))],
  };
}

function captureRepresentation(
  ws: WorkspaceStore,
  catalog: IngestionCatalog,
  source: SourceRecord,
  observation: SourceObservation,
  adapter: AdapterRef,
  product: AdapterProduct,
  rootChunkId: string,
  contentChunkIds: string[],
  now: string,
  provenance = representationSnapshot(ws, rootChunkId, contentChunkIds),
): RepresentationRecord {
  const representation: RepresentationRecord = {
    id: `representation_${randomUUID()}`,
    sourceId: source.id,
    observationId: observation.id,
    relationship: product.relationship,
    mediaType: product.mediaType,
    adapter,
    rootChunkId,
    contentChunkIds,
    outputRevisionIds: provenance.outputRevisionIds,
    operationIds: provenance.operationIds,
    warnings: product.warnings,
    createdAt: now,
  };
  catalog.representations.push(representation);
  source.currentRepresentationId = representation.id;
  return representation;
}

const normalizedTextHash = (text: string): string => sha256(text.replace(/\r\n?/g, '\n'));

function importRenderedText(product: AdapterProduct): string {
  if (product.relationship === 'native' && product.mediaType === MEDIA_MARKDOWN) {
    const text = product.text.replace(/\r\n/g, '\n');
    return decomposeText(text, METHOD_BLOCKS).map((span) => text.slice(span.start, span.end)).join('\n\n');
  }
  return product.text;
}

function productIdentityHash(adapter: AdapterRef, product: AdapterProduct): string {
  return sha256(
    JSON.stringify({
      adapter,
      relationship: product.relationship,
      mediaType: product.mediaType,
      normalizedTextHash: normalizedTextHash(product.text),
      warnings: product.warnings.map((warning) => ({
        code: warning.code,
        severity: warning.severity,
        phase: warning.phase,
        message: warning.message,
        retryable: warning.retryable ?? null,
      })),
    }),
  );
}

function operationParams(operation: Operation): Record<string, unknown> | null {
  return operation.params && typeof operation.params === 'object' && !Array.isArray(operation.params)
    ? (operation.params as Record<string, unknown>)
    : null;
}

function operationForPending(ws: WorkspaceStore, pending: PendingMaterialization): Operation | undefined {
  const matches = [...ws.state.operations.values()].filter(
    (operation) => operationParams(operation)?.ingestionToken === pending.token,
  );
  if (matches.length > 1) throw new Error(`multiple kernel operations claim ingestion token ${pending.token}`);
  const operation = matches[0];
  if (!operation) return undefined;
  const params = operationParams(operation)!;
  if (params.sourceId !== pending.sourceId || params.observationId !== pending.observationId) {
    throw new Error(`kernel operation ${operation.id} does not match its ingestion intent`);
  }
  if (operation.actorId !== `adapter:${pending.adapter.id}@${pending.adapter.version}`) {
    throw new Error(`kernel operation ${operation.id} actor does not match its ingestion adapter`);
  }
  if (pending.operationKind === 'revise' && params.productIdentityHash !== pending.productIdentityHash) {
    throw new Error(`kernel operation ${operation.id} does not match its ingestion product identity`);
  }
  if (pending.operationKind === 'import') {
    const expected = {
      productIdentityHash: pending.productIdentityHash,
      relationship: pending.relationship,
      mediaType: pending.mediaType,
      normalizedTextHash: pending.normalizedTextHash,
      normalizedRenderedTextHash: pending.normalizedRenderedTextHash,
    };
    for (const [key, value] of Object.entries(expected)) {
      if (params[key] !== value) {
        throw new Error(`kernel operation ${operation.id} does not match ingestion ${key}`);
      }
    }
  }
  return operation;
}

function importOperationParams(pending: PendingImportMaterialization): Record<string, unknown> {
  return {
    ingestionToken: pending.token,
    sourceId: pending.sourceId,
    observationId: pending.observationId,
    productIdentityHash: pending.productIdentityHash,
    relationship: pending.relationship,
    mediaType: pending.mediaType,
    normalizedTextHash: pending.normalizedTextHash,
    normalizedRenderedTextHash: pending.normalizedRenderedTextHash,
  };
}

function pendingFor(
  catalog: IngestionCatalog,
  source: SourceRecord,
  observation: SourceObservation,
  adapter: AdapterRef,
  product: AdapterProduct,
  now: string,
): PendingImportMaterialization {
  const existing = catalog.pendingMaterializations.find(
    (pending): pending is PendingImportMaterialization =>
      pending.operationKind === 'import' &&
      pending.sourceId === source.id &&
      pending.observationId === observation.id &&
      sameAdapterRef(pending.adapter, adapter) &&
      pending.productIdentityHash === productIdentityHash(adapter, product),
  );
  if (existing) return existing;
  const pending: PendingImportMaterialization = {
    token: `materialization_${randomUUID()}`,
    operationKind: 'import',
    sourceId: source.id,
    observationId: observation.id,
    relPath: observation.relPath,
    adapter,
    relationship: product.relationship,
    mediaType: product.mediaType,
    normalizedTextHash: normalizedTextHash(product.text),
    normalizedRenderedTextHash: normalizedTextHash(importRenderedText(product)),
    productIdentityHash: productIdentityHash(adapter, product),
    warnings: product.warnings,
    startedAt: now,
  };
  catalog.pendingMaterializations.push(pending);
  return pending;
}

function pendingRevisionFor(
  catalog: IngestionCatalog,
  source: SourceRecord,
  observation: SourceObservation,
  adapter: AdapterRef,
  product: AdapterProduct,
  representation: RepresentationRecord,
  targetChunkId: string,
  basisRevisionId: string,
  now: string,
): PendingRevisionMaterialization {
  const identityHash = productIdentityHash(adapter, product);
  const existing = catalog.pendingMaterializations.find(
    (pending): pending is PendingRevisionMaterialization =>
      pending.operationKind === 'revise' &&
      pending.sourceId === source.id &&
      pending.observationId === observation.id &&
      sameAdapterRef(pending.adapter, adapter) &&
      pending.productIdentityHash === identityHash &&
      pending.targetChunkId === targetChunkId &&
      pending.basisRevisionId === basisRevisionId,
  );
  if (existing) return existing;
  const pending: PendingRevisionMaterialization = {
    token: `materialization_${randomUUID()}`,
    operationKind: 'revise',
    sourceId: source.id,
    observationId: observation.id,
    relPath: observation.relPath,
    adapter,
    relationship: product.relationship,
    mediaType: product.mediaType,
    normalizedTextHash: normalizedTextHash(product.text),
    productIdentityHash: identityHash,
    warnings: product.warnings,
    startedAt: now,
    rootChunkId: representation.rootChunkId,
    contentChunkIds: [...representation.contentChunkIds],
    targetChunkId,
    basisRevisionId,
    priorRepresentationId: representation.id,
    priorOutputRevisionIds: [...representation.outputRevisionIds],
  };
  catalog.pendingMaterializations.push(pending);
  return pending;
}

function clearPending(catalog: IngestionCatalog, token: string): void {
  catalog.pendingMaterializations = catalog.pendingMaterializations.filter((pending) => pending.token !== token);
}

function recoveredRevisionProvenance(
  ws: WorkspaceStore,
  catalog: IngestionCatalog,
  pending: PendingRevisionMaterialization,
  correlatedRevisionId: string,
): { outputRevisionIds: string[]; operationIds: string[] } {
  const priorRepresentation = catalog.representations.find(
    (candidate) => candidate.id === pending.priorRepresentationId,
  );
  if (!priorRepresentation) {
    throw new Error(`ingestion intent ${pending.token} lost its prior representation provenance`);
  }
  const priorOutputRevisionIds = pending.priorOutputRevisionIds;
  const expectedChunkIds = [pending.rootChunkId, ...pending.contentChunkIds];
  if (
    priorRepresentation.sourceId !== pending.sourceId ||
    priorRepresentation.rootChunkId !== pending.rootChunkId ||
    priorRepresentation.contentChunkIds.length !== pending.contentChunkIds.length ||
    priorRepresentation.contentChunkIds.some((chunkId, index) => chunkId !== pending.contentChunkIds[index]) ||
    priorRepresentation.outputRevisionIds.length !== priorOutputRevisionIds.length ||
    priorRepresentation.outputRevisionIds.some((revisionId, index) => revisionId !== priorOutputRevisionIds[index]) ||
    new Set(expectedChunkIds).size !== expectedChunkIds.length ||
    priorOutputRevisionIds.length !== expectedChunkIds.length
  ) {
    throw new Error(`ingestion intent ${pending.token} has an invalid prior representation shape`);
  }
  const priorByChunk = new Map<string, string>();
  for (const revisionId of priorOutputRevisionIds) {
    const revision = ws.state.revisions.get(revisionId);
    if (!revision || !expectedChunkIds.includes(revision.chunkId) || priorByChunk.has(revision.chunkId)) {
      throw new Error(`ingestion intent ${pending.token} has invalid historical revision ${revisionId}`);
    }
    priorByChunk.set(revision.chunkId, revisionId);
  }
  if (priorByChunk.get(pending.targetChunkId) !== pending.basisRevisionId) {
    throw new Error(`ingestion intent ${pending.token} prior representation does not contain its revise basis`);
  }
  const outputRevisionIds = expectedChunkIds.map((chunkId) =>
    chunkId === pending.targetChunkId ? correlatedRevisionId : priorByChunk.get(chunkId)!,
  );
  const operationIds = [...new Set(outputRevisionIds.map((revisionId) => {
    const revision = ws.state.revisions.get(revisionId);
    if (!revision) throw new Error(`ingestion intent ${pending.token} lost historical revision ${revisionId}`);
    return revision.operationId;
  }))];
  return { outputRevisionIds, operationIds };
}

async function recoverPendingMaterializations(
  ws: WorkspaceStore,
  catalog: IngestionCatalog,
  persist: () => void,
): Promise<void> {
  for (const pending of [...catalog.pendingMaterializations]) {
    const operation = operationForPending(ws, pending);
    // The intent became durable before the kernel commit. No operation means
    // the attempted materialization never became truth and is safe to retry.
    if (!operation) {
      clearPending(catalog, pending.token);
      persist();
      continue;
    }
    const operationKind = pending.operationKind;
    if (operation.kind !== operationKind) {
      throw new Error(`ingestion operation ${operation.id} is ${operation.kind}, expected ${operationKind}`);
    }
    const source = catalog.sources.find((candidate) => candidate.id === pending.sourceId);
    const observation = catalog.observations.find((candidate) => candidate.id === pending.observationId);
    if (!source || !observation) throw new Error(`ingestion intent ${pending.token} lost its source observation`);

    if (operationKind === 'revise') {
      if (operation.outputRevisionIds.length !== 1) {
        throw new Error(`ingestion operation ${operation.id} has an invalid revise intent shape`);
      }
      if (
        operation.actorId !== `adapter:${pending.adapter.id}@${pending.adapter.version}` ||
        operation.inputRevisionIds.length !== 1 ||
        operation.inputRevisionIds[0] !== pending.basisRevisionId
      ) {
        throw new Error(`ingestion operation ${operation.id} does not match its correlated revise intent`);
      }
      const revision = ws.state.revisions.get(operation.outputRevisionIds[0]);
      if (
        !revision ||
        revision.chunkId !== pending.targetChunkId ||
        revision.mediaType !== pending.mediaType ||
        normalizedTextHash(revisionText(ws.state, revision.id)) !== pending.normalizedTextHash ||
        !pending.contentChunkIds.includes(pending.targetChunkId)
      ) {
        throw new Error(`ingestion operation ${operation.id} output does not match its correlated revise intent`);
      }
      const provenance = recoveredRevisionProvenance(ws, catalog, pending, revision.id);
      const alreadyRecovered = catalog.representations.find(
        (representation) =>
          representation.sourceId === pending.sourceId &&
          representation.observationId === pending.observationId &&
          representation.relationship === pending.relationship &&
          representation.mediaType === pending.mediaType &&
          sameAdapterRef(representation.adapter, pending.adapter) &&
          representation.operationIds.includes(operation.id) &&
          representation.outputRevisionIds.length === provenance.outputRevisionIds.length &&
          representation.outputRevisionIds.every((revisionId, index) => revisionId === provenance.outputRevisionIds[index]),
      );
      if (alreadyRecovered) source.currentRepresentationId = alreadyRecovered.id;
      else {
        captureRepresentation(
          ws,
          catalog,
          source,
          observation,
          pending.adapter,
          {
            relationship: pending.relationship,
            mediaType: pending.mediaType,
            text: '',
            warnings: pending.warnings,
          },
          pending.rootChunkId,
          pending.contentChunkIds,
          new Date().toISOString(),
          provenance,
        );
      }
      clearPending(catalog, pending.token);
      persist();
      continue;
    }

    if (operation.outputRevisionIds.length < 1) {
      throw new Error(`ingestion operation ${operation.id} has an invalid composite import shape`);
    }
    const revisions = operation.outputRevisionIds.map((id) => {
      const revision = ws.state.revisions.get(id);
      if (!revision) throw new Error(`ingestion operation ${operation.id} lost output revision ${id}`);
      return revision;
    });
    if (revisions[0].mediaType !== MEDIA_COMPOSITE) {
      throw new Error(`ingestion operation ${operation.id} did not produce a composite root`);
    }
    if (
      operation.inputRevisionIds.length !== 0 ||
      revisions.some(
        (revision) =>
          revision.operationId !== operation.id ||
          revision.createdBy !== `adapter:${pending.adapter.id}@${pending.adapter.version}`,
      )
    ) {
      throw new Error(`ingestion operation ${operation.id} revisions do not match their adapter import`);
    }
    const rootChunkId = revisions[0].chunkId;
    const contentChunkIds = revisions.slice(1).map((revision) => revision.chunkId);
    const params = operationParams(operation)!;
    if (params.blocks !== contentChunkIds.length) {
      throw new Error(`ingestion operation ${operation.id} block count does not match its outputs`);
    }
    let expectedJoin: string;
    let expectedContentCount: number | null;
    if (pending.relationship === 'native' && pending.mediaType === MEDIA_MARKDOWN) {
      expectedJoin = '\n\n';
      expectedContentCount = null;
    } else if (
      (pending.relationship === 'native' && pending.mediaType === MEDIA_TEXT) ||
      (pending.relationship === 'derived' && pending.mediaType === MEDIA_MARKDOWN)
    ) {
      expectedJoin = '';
      expectedContentCount = 1;
    } else {
      throw new Error(
        `ingestion operation ${operation.id} has unsupported ${pending.relationship} ${pending.mediaType} import provenance`,
      );
    }
    if (
      (expectedContentCount !== null && contentChunkIds.length !== expectedContentCount) ||
      revisions.slice(1).some((revision) => revision.mediaType !== pending.mediaType)
    ) {
      throw new Error(`ingestion operation ${operation.id} content does not match its pending media type`);
    }
    let joinValue: unknown;
    try {
      joinValue = (JSON.parse(revisionText(ws.state, revisions[0].id)) as { join?: unknown }).join;
    } catch {
      throw new Error(`ingestion operation ${operation.id} has an unreadable composite root`);
    }
    if (joinValue !== expectedJoin) {
      throw new Error(`ingestion operation ${operation.id} composite join does not match its pending relationship`);
    }
    const children = childOccurrences(ws.state, rootChunkId);
    if (
      currentRevision(ws.state, rootChunkId).id !== revisions[0].id ||
      children.length !== contentChunkIds.length ||
      children.some(
        (occurrence, index) =>
          occurrence.chunkId !== contentChunkIds[index] ||
          occurrence.mode !== 'contain' ||
          occurrence.pin !== 'current' ||
          occurrence.watch ||
          currentRevision(ws.state, occurrence.chunkId).id !== revisions[index + 1].id,
      )
    ) {
      throw new Error(`ingestion operation ${operation.id} child occurrence structure does not match its outputs`);
    }
    const historicalRenderedText = revisions.slice(1).map((revision) => revisionText(ws.state, revision.id)).join(expectedJoin);
    const renderedText = renderChunk(ws.state, rootChunkId);
    if (normalizedTextHash(renderedText) !== normalizedTextHash(historicalRenderedText)) {
      throw new Error(`ingestion operation ${operation.id} actual composite rendering does not match its outputs`);
    }
    const renderedHash = normalizedTextHash(renderedText);
    if (renderedHash !== pending.normalizedRenderedTextHash) {
      throw new Error(`ingestion operation ${operation.id} rendered output does not match its ingestion intent`);
    }
    if (pending.relationship === 'native' && pending.mediaType === MEDIA_MARKDOWN) {
      await recoverMarkdownImport(ws.state, {
        workspaceRoot: ws.root,
        relPath: pending.relPath,
        docChunkId: rootChunkId,
        lastImportedFileHash: pending.normalizedTextHash,
      });
    }
    const alreadyRecovered = catalog.representations.find(
      (representation) =>
        representation.sourceId === pending.sourceId &&
        representation.observationId === pending.observationId &&
        representation.operationIds.includes(operation.id),
    );
    if (alreadyRecovered) source.currentRepresentationId = alreadyRecovered.id;
    else {
      captureRepresentation(
        ws,
        catalog,
        source,
        observation,
        pending.adapter,
        {
          relationship: pending.relationship,
          mediaType: pending.mediaType,
          text: '',
          warnings: pending.warnings,
        },
        rootChunkId,
        contentChunkIds,
        new Date().toISOString(),
        { outputRevisionIds: operation.outputRevisionIds, operationIds: [operation.id] },
      );
    }
    clearPending(catalog, pending.token);
    persist();
  }
}

function markdownDocChunkId(workspaceRoot: string, relPath: string): string {
  const parsed = JSON.parse(readFileSync(sidecarPath(workspaceRoot, relPath), 'utf8')) as { docChunkId?: unknown };
  if (typeof parsed.docChunkId !== 'string') throw new Error(`Markdown sidecar lacks docChunkId: ${relPath}`);
  return parsed.docChunkId;
}

function openSourceProposals(ws: WorkspaceStore, rootChunkId: string, sourceId: string) {
  const prefix = `[source:${sourceId}:`;
  return [...ws.state.proposals.values()].filter(
    (proposal) =>
      proposal.status === 'open' &&
      proposal.kind === 'reconciliation' &&
      proposal.targetChunkIds.includes(rootChunkId) &&
      proposal.note?.startsWith(prefix),
  );
}

const sourceObservationMarker = (sourceId: string, observationId: string): string =>
  `[source:${sourceId}:${observationId}]`;

const sourceProductMarker = (
  sourceId: string,
  observationId: string,
  adapter: AdapterRef,
  product: AdapterProduct,
  basisRevisionId: string,
): string =>
  `${sourceObservationMarker(sourceId, observationId)}[product:${productIdentityHash(adapter, product)}]` +
  `[basis:${basisRevisionId}]`;

function invalidateOlderSourceProposals(
  ctx: TxCtx,
  representation: RepresentationRecord,
  source: SourceRecord,
  observation: SourceObservation,
): void {
  const currentMarker = sourceObservationMarker(source.id, observation.id);
  const targets = new Set([representation.rootChunkId, ...representation.contentChunkIds]);
  for (const proposal of ctx.state.proposals.values()) {
    if (
      proposal.status !== 'open' ||
      proposal.kind !== 'reconciliation' ||
      !proposal.targetChunkIds.some((target) => targets.has(target))
    ) {
      continue;
    }
    const note = proposal.note ?? '';
    const ingestionOwned = note.startsWith('[source:') || note.startsWith('[file:');
    if (!ingestionOwned || note.startsWith(currentMarker)) continue;
    supersedeProposal(ctx, {
      proposalId: proposal.id,
      reason: `${observation.relPath} advanced to observation ${observation.id}`,
    });
  }
}

async function materializeDerivedMarkdown(opts: {
  ws: WorkspaceStore;
  catalog: IngestionCatalog;
  persist(): void;
  ctx: TxCtx;
  source: SourceRecord;
  observation: SourceObservation;
  adapter: AdapterRef;
  product: AdapterProduct;
  prior?: RepresentationRecord;
}): Promise<IngestionItemResult> {
  const { ws, catalog, persist, ctx, source, observation, adapter, product, prior } = opts;
  if (product.relationship !== 'derived' || product.mediaType !== MEDIA_MARKDOWN) {
    throw new Error(`unsupported derived representation: ${product.relationship} ${product.mediaType}`);
  }

  if (!prior || !ws.state.chunks.has(prior.rootChunkId)) {
    const pending = pendingFor(catalog, source, observation, adapter, product, new Date().toISOString());
    persist();
    const imported = await createComposite(ctx, {
      join: '',
      blocks: [{ text: product.text, mediaType: MEDIA_MARKDOWN }],
      opKind: 'import',
      operationParams: importOperationParams(pending),
    });
    const representation = captureRepresentation(
      ws,
      catalog,
      source,
      observation,
      adapter,
      product,
      imported.chunkId,
      imported.blockChunkIds,
      new Date().toISOString(),
      { outputRevisionIds: imported.commit.operation.outputRevisionIds, operationIds: [imported.commit.operation.id] },
    );
    clearPending(catalog, pending.token);
    persist();
    return { status: 'imported', observation, adapter, diagnostics: product.warnings, representation };
  }

  if (prior.relationship !== 'derived' || prior.mediaType !== MEDIA_MARKDOWN || prior.contentChunkIds.length !== 1) {
    throw new Error('derived Markdown representation shape is no longer one Markdown content chunk');
  }
  const contentChunkId = prior.contentChunkIds[0];
  if (!ws.state.chunks.has(contentChunkId)) throw new Error('derived Markdown representation lost its content chunk');
  const head = currentRevision(ws.state, contentChunkId);
  const priorRevision = prior.outputRevisionIds
    .map((revisionId) => ws.state.revisions.get(revisionId))
    .find((revision) => revision?.chunkId === contentChunkId);
  if (!priorRevision) throw new Error('derived Markdown representation lost its source revision');

  const sourceProposals = openSourceProposals(ws, contentChunkId, source.id);
  if (revisionText(ws.state, head.id) === product.text) {
    for (const proposal of sourceProposals) {
      supersedeProposal(ctx, {
        proposalId: proposal.id,
        reason: `${observation.relPath} now matches the derived representation`,
      });
    }
    const representation = captureRepresentation(
      ws,
      catalog,
      source,
      observation,
      adapter,
      product,
      prior.rootChunkId,
      prior.contentChunkIds,
      new Date().toISOString(),
    );
    persist();
    return { status: 'updated', observation, adapter, diagnostics: product.warnings, representation };
  }

  if (head.id !== priorRevision.id) {
    const marker = sourceProductMarker(source.id, observation.id, adapter, product, head.id);
    const existing = sourceProposals.find((proposal) => proposal.note?.startsWith(marker));
    for (const proposal of sourceProposals) {
      if (proposal.id === existing?.id) continue;
      supersedeProposal(ctx, {
        proposalId: proposal.id,
        reason: `${observation.relPath} changed again before the earlier derived reconciliation was resolved`,
      });
    }
    const identityHash = productIdentityHash(adapter, product);
    const proposalWarnings = product.warnings.map((warning) => ({
      code: warning.code,
      severity: warning.severity,
      phase: warning.phase,
      message: warning.message,
      ...(warning.retryable === undefined ? {} : { retryable: warning.retryable }),
    }));
    const proposalId =
      existing?.id ??
      propose(ctx, {
        kind: 'reconciliation',
        basisRevisionIds: [head.id],
        targetChunkIds: [contentChunkId],
        payload: [{ op: 'revise', chunkId: contentChunkId, text: product.text, mediaType: MEDIA_MARKDOWN }],
        note:
          `${marker} ${observation.relPath} produced new derived Markdown after its Headspace representation changed; ` +
          `review before replacing it; converter warnings=${JSON.stringify(proposalWarnings)}`,
        inputRevisionIds: [head.id],
        producer: adapter.provider
          ? {
              id: adapter.provider.identity,
              version: adapter.provider.implementationVersion,
              implementation: `${adapter.id}@${adapter.version}`,
              receiptId: identityHash,
            }
          : { id: adapter.id, version: adapter.version, receiptId: identityHash },
        operationParams: {
          sourceId: source.id,
          observationId: observation.id,
          adapter,
          relationship: product.relationship,
          mediaType: product.mediaType,
          normalizedTextHash: normalizedTextHash(product.text),
          productIdentityHash: identityHash,
          warnings: proposalWarnings,
        },
      }).proposalId;
    return {
      status: 'proposal',
      observation,
      adapter,
      diagnostics: [
        ...product.warnings,
        diagnostic('source.review-required', 'materialize', 'Both source and derived representation changed', 'warning'),
      ],
      representation: prior,
      proposalId,
    };
  }

  const pending = pendingRevisionFor(
    catalog,
    source,
    observation,
    adapter,
    product,
    prior,
    contentChunkId,
    head.id,
    new Date().toISOString(),
  );
  persist();
  await revise(ctx, {
    chunkId: contentChunkId,
    text: product.text,
    mediaType: MEDIA_MARKDOWN,
    operationParams: {
      ingestionToken: pending.token,
      sourceId: source.id,
      observationId: observation.id,
      productIdentityHash: pending.productIdentityHash,
    },
  });
  const representation = captureRepresentation(
    ws,
    catalog,
    source,
    observation,
    adapter,
    product,
    prior.rootChunkId,
    prior.contentChunkIds,
    new Date().toISOString(),
  );
  for (const proposal of sourceProposals) {
    supersedeProposal(ctx, {
      proposalId: proposal.id,
      reason: `${observation.relPath} advanced cleanly to a newer derived observation`,
    });
  }
  clearPending(catalog, pending.token);
  persist();
  return { status: 'updated', observation, adapter, diagnostics: product.warnings, representation };
}

class IngestionCatalogPersistenceError extends Error {
  override name = 'IngestionCatalogPersistenceError';
}

export async function ingestWorkspace(
  ws: WorkspaceStore,
  opts: { contentDirs?: string[]; contentFiles?: string[] },
  runtime: IngestionRuntime = {},
): Promise<IngestionRunReport> {
  const startedAt = new Date().toISOString();
  const catalog = readIngestionCatalog(ws.root) ?? emptyCatalog();
  const persist = (): void => {
    try {
      writeIngestionCatalog(ws.root, catalog, runtime.catalogPublish);
    } catch (e) {
      throw new IngestionCatalogPersistenceError(`ingestion catalog publication failed: ${(e as Error).message}`);
    }
  };
  await recoverPendingMaterializations(ws, catalog, persist);
  const observed = observeConfiguredSources(ws.root, opts);
  const adapters = adaptersFor(runtime);
  const items: IngestionItemResult[] = [];

  for (const candidate of observed.candidates) {
    const registered = registerObservation(catalog, candidate, startedAt);
    const { observation, source } = registered;
    // Source and immutable observation identity become durable before any
    // adapter is allowed to materialize kernel facts.
    persist();
    const boundRepresentation = currentRepresentation(catalog, source);
    if (boundRepresentation) {
      // Proposal freshness includes external truth, not only the kernel head.
      // This runs before adapter selection so an unreadable/unsupported newer
      // observation still makes an older file proposal unacceptably stale.
      invalidateOlderSourceProposals(
        ws.ctxFor('adapter:ingestion'),
        boundRepresentation,
        source,
        observation,
      );
    }

    if (observation.kind === 'directory') {
      items.push({
        status: registered.newObservation ? 'imported' : 'unchanged',
        observation,
        adapter: null,
        diagnostics: [],
        representation: null,
      });
      continue;
    }
    if (observation.kind === 'symlink') {
      items.push({
        status: 'unsupported',
        observation,
        adapter: null,
        diagnostics: [diagnostic('source.unfollowed-link', 'select', `Headspace will not follow ${observation.relPath}`, 'warning')],
        representation: null,
      });
      continue;
    }

    const matchingAdapters = adapters.filter(
      (candidateAdapter) =>
        candidateAdapter.capability.acceptedKinds.includes(observation.kind) &&
        candidateAdapter.capability.acceptedMediaTypes.includes(observation.mediaType),
    );
    const adapter = matchingAdapters.find((candidateAdapter) => candidateAdapter.capability.availability.status === 'ready');
    const prior = currentRepresentation(catalog, source);
    if (!adapter) {
      const knownAdapter = matchingAdapters.find(
        (candidateAdapter) =>
          candidateAdapter.capability.id === prior?.adapter.id && candidateAdapter.capability.version === prior.adapter.version,
      );
      if (
        knownAdapter &&
        !registered.newObservation &&
        prior?.observationId === observation.id &&
        ws.state.chunks.has(prior.rootChunkId)
      ) {
        const availabilityDiagnostics =
          knownAdapter.capability.availability.status === 'unavailable'
            ? [knownAdapter.capability.availability.diagnostic]
            : [];
        items.push({
          status: 'unchanged',
          observation,
          adapter: prior.adapter,
          diagnostics: availabilityDiagnostics,
          representation: prior,
        });
        continue;
      }
      const unavailable = matchingAdapters.find(
        (candidateAdapter) => candidateAdapter.capability.availability.status === 'unavailable',
      );
      items.push({
        status: 'unsupported',
        observation,
        adapter: unavailable ? adapterRefOf(unavailable.capability) : null,
        diagnostics:
          unavailable?.capability.availability.status === 'unavailable'
            ? [unavailable.capability.availability.diagnostic]
            : [
                diagnostic(
                  'adapter.unsupported-media-type',
                  'select',
                  `No available adapter accepts ${observation.mediaType}`,
                  'warning',
                ),
              ],
        representation: prior ?? null,
      });
      continue;
    }

    const adapterRef = adapterRefOf(adapter.capability);
    if (
      !registered.newObservation &&
      prior?.observationId === observation.id &&
      sameAdapterRef(prior.adapter, adapterRef) &&
      ws.state.chunks.has(prior.rootChunkId)
    ) {
      items.push({ status: 'unchanged', observation, adapter: adapterRef, diagnostics: [], representation: prior });
      continue;
    }

    let product: AdapterProduct;
    try {
      product = await adapter.ingest({
        observation,
        readBytes: async () => confinedReadBytes(ws.root, observation),
      });
    } catch (e) {
      const adapterError = e instanceof AdapterIngestionError ? e : null;
      items.push({
        status: 'failed',
        observation,
        adapter: adapterRef,
        diagnostics: [
          {
            ...diagnostic(
              adapterError?.diagnosticCode ?? 'adapter.ingest-failed',
              'adapt',
              `${observation.relPath}: ${(e as Error).message}`,
            ),
            ...(adapterError ? { retryable: adapterError.retryable } : {}),
          },
        ],
        representation: prior ?? null,
      });
      continue;
    }

    const declaredOutput = adapter.capability.outputs.find(
      (output) => output.relationship === product.relationship && output.mediaType === product.mediaType,
    );
    if (!declaredOutput) {
      items.push({
        status: 'failed',
        observation,
        adapter: adapterRef,
        diagnostics: [
          diagnostic(
            'adapter.undeclared-output',
            'adapt',
            `${adapterRef.id}@${adapterRef.version} returned undeclared ${product.relationship} ${product.mediaType}`,
          ),
        ],
        representation: prior ?? null,
      });
      continue;
    }

    try {
      const ctx = ws.ctxFor(`adapter:${adapter.capability.id}@${adapter.capability.version}`);
      if (product.relationship === 'derived') {
        items.push(
          await materializeDerivedMarkdown({
            ws,
            catalog,
            persist,
            ctx,
            source,
            observation,
            adapter: adapterRef,
            product,
            prior,
          }),
        );
        continue;
      }
      if (product.mediaType === MEDIA_MARKDOWN) {
        const known = existsSync(sidecarPath(ws.root, observation.relPath));
        let rootChunkId: string;
        let status: IngestionStatus;
        let proposalId: string | undefined;
        if (!known && prior) {
          throw new Error(
            `bound Markdown source lost its projection manifest; refusing to import a duplicate: ${observation.relPath}`,
          );
        }
        if (!known) {
          const pending = pendingFor(
            catalog,
            source,
            observation,
            adapterRef,
            product,
            new Date().toISOString(),
          );
          persist();
          const imported = await importMarkdownFile(ctx, {
            workspaceRoot: ws.root,
            relPath: observation.relPath,
            text: product.text,
            operationParams: importOperationParams(pending),
            sidecarPublish: runtime.sidecarPublish,
          });
          rootChunkId = imported.docChunkId;
          status = 'imported';
        } else {
          rootChunkId = markdownDocChunkId(ws.root, observation.relPath);
          const reconciled = await reconcileMarkdownFile(ctx, {
            workspaceRoot: ws.root,
            relPath: observation.relPath,
            text: product.text,
            proposalMarker: sourceObservationMarker(source.id, observation.id),
          });
          proposalId = reconciled.proposalId;
          status = reconciled.action === 'proposal' ? 'proposal' : reconciled.action === 'fast-forward' ? 'updated' : 'unchanged';
        }
        if (status === 'proposal') {
          items.push({
            status,
            observation,
            adapter: adapterRef,
            diagnostics: [diagnostic('source.review-required', 'materialize', 'External changes require review', 'warning')],
            representation: prior ?? null,
            proposalId,
          });
          continue;
        }
        const contentChunkIds = childOccurrences(ws.state, rootChunkId).map((occurrence) => occurrence.chunkId);
        const representation = captureRepresentation(
          ws,
          catalog,
          source,
          observation,
          adapterRef,
          product,
          rootChunkId,
          contentChunkIds,
          new Date().toISOString(),
        );
        const pending = catalog.pendingMaterializations.find(
          (candidate) => candidate.sourceId === source.id && candidate.observationId === observation.id,
        );
        if (pending) clearPending(catalog, pending.token);
        persist();
        items.push({ status, observation, adapter: adapterRef, diagnostics: product.warnings, representation });
        continue;
      }

      if (product.mediaType === MEDIA_TEXT) {
        if (!prior || !ws.state.chunks.has(prior.rootChunkId)) {
          const pending = pendingFor(
            catalog,
            source,
            observation,
            adapterRef,
            product,
            new Date().toISOString(),
          );
          persist();
          const imported = await createComposite(ctx, {
            join: '',
            blocks: [{ text: product.text, mediaType: MEDIA_TEXT }],
            opKind: 'import',
            operationParams: importOperationParams(pending),
          });
          const representation = captureRepresentation(
            ws,
            catalog,
            source,
            observation,
            adapterRef,
            product,
            imported.chunkId,
            imported.blockChunkIds,
            new Date().toISOString(),
            { outputRevisionIds: imported.commit.operation.outputRevisionIds, operationIds: [imported.commit.operation.id] },
          );
          clearPending(catalog, pending.token);
          persist();
          items.push({ status: 'imported', observation, adapter: adapterRef, diagnostics: product.warnings, representation });
          continue;
        }

        const contentChunkId = prior.contentChunkIds[0];
        const content = contentChunkId ? ws.state.chunks.get(contentChunkId) : undefined;
        if (!content || prior.contentChunkIds.length !== 1) throw new Error('plain-text representation shape is no longer one content chunk');
        const head = currentRevision(ws.state, contentChunkId);
        const priorRevision = prior.outputRevisionIds
          .map((revisionId) => ws.state.revisions.get(revisionId))
          .find((revision) => revision?.chunkId === contentChunkId);
        if (!priorRevision) throw new Error('plain-text representation lost its source revision');

        const sourceProposals = openSourceProposals(ws, contentChunkId, source.id);

        // Proposal acceptance or another deliberate operation may already have
        // made kernel truth equal this exact observation. Converge provenance
        // instead of proposing the same external text again.
        if (revisionText(ws.state, head.id) === product.text) {
          for (const proposal of sourceProposals) {
            supersedeProposal(ctx, {
              proposalId: proposal.id,
              reason: `${observation.relPath} now matches the kernel representation`,
            });
          }
          const representation = captureRepresentation(
            ws,
            catalog,
            source,
            observation,
            adapterRef,
            product,
            prior.rootChunkId,
            prior.contentChunkIds,
            new Date().toISOString(),
          );
          persist();
          items.push({ status: 'updated', observation, adapter: adapterRef, diagnostics: product.warnings, representation });
          continue;
        }

        if (head.id !== priorRevision.id) {
          const marker = `${sourceObservationMarker(source.id, observation.id)}[basis:${head.id}]`;
          const existing = sourceProposals.find((proposal) => proposal.note?.startsWith(marker));
          for (const proposal of sourceProposals) {
            if (proposal.id === existing?.id) continue;
            supersedeProposal(ctx, {
              proposalId: proposal.id,
              reason: `${observation.relPath} changed again before the earlier reconciliation was resolved`,
            });
          }
          const proposalId =
            existing?.id ??
            propose(ctx, {
              kind: 'reconciliation',
              basisRevisionIds: [head.id],
              targetChunkIds: [contentChunkId],
              payload: [{ op: 'revise', chunkId: contentChunkId, text: product.text, mediaType: MEDIA_TEXT }],
              note: `${marker} ${observation.relPath} changed externally after the Headspace representation changed; review before replacing it`,
              inputRevisionIds: [head.id],
            }).proposalId;
          items.push({
            status: 'proposal',
            observation,
            adapter: adapterRef,
            diagnostics: [diagnostic('source.review-required', 'materialize', 'Both source and representation changed', 'warning')],
            representation: prior,
            proposalId,
          });
          continue;
        }

        if (revisionText(ws.state, head.id) !== product.text) {
          await revise(ctx, { chunkId: contentChunkId, text: product.text, mediaType: MEDIA_TEXT });
        }
        const representation = captureRepresentation(
          ws,
          catalog,
          source,
          observation,
          adapterRef,
          product,
          prior.rootChunkId,
          prior.contentChunkIds,
          new Date().toISOString(),
        );
        for (const proposal of sourceProposals) {
          supersedeProposal(ctx, {
            proposalId: proposal.id,
            reason: `${observation.relPath} advanced cleanly to a newer source observation`,
          });
        }
        persist();
        items.push({ status: 'updated', observation, adapter: adapterRef, diagnostics: product.warnings, representation });
        continue;
      }

      throw new Error(`adapter returned unsupported product media type: ${product.mediaType}`);
    } catch (e) {
      if (e instanceof IngestionCatalogPersistenceError) throw e;
      items.push({
        status: 'failed',
        observation,
        adapter: adapterRef,
        diagnostics: [diagnostic('adapter.materialize-failed', 'materialize', `${observation.relPath}: ${(e as Error).message}`)],
        representation: prior ?? null,
      });
    }
  }

  const counts: Record<IngestionStatus, number> = {
    imported: 0,
    updated: 0,
    unchanged: 0,
    proposal: 0,
    unsupported: 0,
    failed: 0,
  };
  for (const item of items) counts[item.status]++;
  const report: IngestionRunReport = {
    id: `ingestion_${randomUUID()}`,
    startedAt,
    finishedAt: new Date().toISOString(),
    items,
    counts,
    diagnostics: observed.diagnostics,
  };
  catalog.lastRun = report;
  persist();
  return report;
}
