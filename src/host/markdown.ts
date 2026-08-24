// Markdown filesystem driver. The sidecar is a projection manifest: it records
// both identity and the authority an external edit may exercise.

import { existsSync, lstatSync, readFileSync, realpathSync } from 'node:fs';
import { isAbsolute, join, resolve, sep } from 'node:path';
import { METHOD_BLOCKS, decomposeText } from '../kernel/decompose';
import { blobHashOf, sha256Hex } from '../kernel/hash';
import {
  childOccurrences,
  currentRevision,
  occurrenceRevision,
  renderChunk,
  revisionText,
  type SubstrateState,
} from '../kernel/state';
import { createChunk, createComposite, moveOccurrence, propose, revise, supersedeProposal, type TxCtx } from '../kernel/tx';
import { MEDIA_COMPOSITE, MEDIA_MARKDOWN } from '../kernel/types';
import type {
  BlobHash,
  ChunkId,
  Commit,
  CommitId,
  Occurrence,
  OccurrenceId,
  OccurrenceMode,
  OperationId,
  Proposal,
  ProposalId,
  ProposedChange,
  RevisionId,
} from '../kernel/types';
import { atomicWriteText, type AtomicPublish } from './atomic-file';
import { assessSimilarity } from './similarity';

const SIM_THRESHOLD = 0.5;
const SIDECAR_SCHEMA_VERSION = 2;

export type ReconciliationPolicy = 'revise-leaf' | 'flatten-composite' | 'detach-transclusion';

export interface MarkdownProjectionBlock {
  occurrenceId: OccurrenceId;
  occurrencePath: OccurrenceId[];
  blockOrdinal: number;
  chunkId: ChunkId;
  projectedTextHash: BlobHash;
  projectedText: string;
  sourceRevisionId: RevisionId;
  pin: 'current' | RevisionId;
  mode: OccurrenceMode;
  policy: ReconciliationPolicy;
}

export interface MarkdownSidecar {
  schemaVersion: typeof SIDECAR_SCHEMA_VERSION;
  docChunkId: ChunkId;
  relPath: string;
  blocks: MarkdownProjectionBlock[];
  lastImportedFileHash: string;
  lastProjectedFileHash: string;
}

interface LegacyMarkdownSidecar {
  docChunkId: ChunkId;
  relPath: string;
  blocks: { chunkId: ChunkId; blobHash: BlobHash }[];
  lastImportedFileHash: string;
  lastProjectedFileHash: string;
}

export interface ReconcileResult {
  action: 'noop' | 'fast-forward' | 'proposal';
  proposalId?: ProposalId;
}

export class ProjectionConflictError extends Error {
  override name = 'ProjectionConflictError';
}

interface ProjectionUnit {
  occurrence: Occurrence;
  occurrencePath: OccurrenceId[];
  blockOrdinal: number;
  text: string;
  sourceRevisionId: RevisionId;
  policy: ReconciliationPolicy;
}

interface MarkdownProjection {
  text: string;
  blocks: MarkdownProjectionBlock[];
}

function assertSafeRel(workspaceRoot: string, relPath: string): void {
  if (isAbsolute(relPath)) throw new Error(`relPath must be workspace-relative: ${relPath}`);
  const base = resolve(workspaceRoot);
  const abs = resolve(base, relPath);
  const prefix = base.endsWith(sep) ? base : base + sep;
  if (abs === base || !abs.startsWith(prefix)) throw new Error(`relPath escapes workspace root: ${relPath}`);
}

function sourcePathForProjection(workspaceRoot: string, relPath: string): string {
  assertSafeRel(workspaceRoot, relPath);
  const abs = resolve(workspaceRoot, relPath);
  let sourceStat;
  let rootReal: string;
  let sourceReal: string;
  try {
    sourceStat = lstatSync(abs);
    rootReal = realpathSync(workspaceRoot);
    sourceReal = realpathSync(abs);
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code === 'ENOENT') {
      throw new ProjectionConflictError(`refusing to project ${relPath}: source file is missing`);
    }
    throw e;
  }
  if (sourceStat.isSymbolicLink()) {
    throw new ProjectionConflictError(`refusing to project ${relPath}: source path is a symbolic link`);
  }
  if (!sourceStat.isFile()) {
    throw new ProjectionConflictError(`refusing to project ${relPath}: source path is not a regular file`);
  }
  const prefix = rootReal.endsWith(sep) ? rootReal : rootReal + sep;
  if (!sourceReal.startsWith(prefix)) {
    throw new ProjectionConflictError(`refusing to project ${relPath}: source resolves outside the workspace root`);
  }
  return abs;
}

export function sidecarPath(workspaceRoot: string, relPath: string): string {
  assertSafeRel(workspaceRoot, relPath);
  return join(workspaceRoot, '.substrate', 'sidecars', `${relPath}.json`);
}

function readSidecar(workspaceRoot: string, relPath: string): MarkdownSidecar | LegacyMarkdownSidecar {
  return JSON.parse(readFileSync(sidecarPath(workspaceRoot, relPath), 'utf8')) as MarkdownSidecar | LegacyMarkdownSidecar;
}

function writeSidecar(workspaceRoot: string, relPath: string, sc: MarkdownSidecar, publish?: AtomicPublish): void {
  const path = sidecarPath(workspaceRoot, relPath);
  atomicWriteText(path, `${JSON.stringify(sc, null, 2)}\n`, publish);
}

const blockTextsOf = (text: string): string[] => decomposeText(text, METHOD_BLOCKS).map((s) => text.slice(s.start, s.end));
const canonical = (blocks: string[]): string => (blocks.length ? `${blocks.join('\n\n')}\n` : '');
const normalizeEol = (text: string): string => text.replace(/\r\n/g, '\n');
const fileMarker = (fileHash: string): string => `[file:${fileHash.slice(0, 12)}]`;

function openReconciliation(ctx: TxCtx, docChunkId: ChunkId): Proposal | undefined {
  for (const p of ctx.state.proposals.values()) {
    if (p.status === 'open' && p.kind === 'reconciliation' && p.targetChunkIds.includes(docChunkId)) return p;
  }
  return undefined;
}

function policyFor(state: SubstrateState, occ: Occurrence): ReconciliationPolicy {
  if (occ.mode === 'transclude') return 'detach-transclusion';
  return occurrenceRevision(state, occ).mediaType === MEDIA_COMPOSITE ? 'flatten-composite' : 'revise-leaf';
}

// Start from the occurrence so a leaf pin is honored. Composite occurrences
// expand through the kernel renderer instead of leaking their JSON join blob.
function renderOccurrence(state: SubstrateState, occ: Occurrence): string {
  const rev = occurrenceRevision(state, occ);
  return rev.mediaType === MEDIA_COMPOSITE ? renderChunk(state, occ.chunkId) : revisionText(state, rev.id);
}

function projectionUnits(state: SubstrateState, docChunkId: ChunkId): ProjectionUnit[] {
  const units: ProjectionUnit[] = [];
  for (const occ of childOccurrences(state, docChunkId)) {
    const rendered = renderOccurrence(state, occ);
    const decomposed = blockTextsOf(rendered);
    const texts = decomposed.length || rendered.length === 0 ? decomposed : [rendered];
    const sourceRevisionId = occurrenceRevision(state, occ).id;
    const policy = policyFor(state, occ);
    texts.forEach((text, blockOrdinal) => {
      units.push({ occurrence: occ, occurrencePath: [occ.id], blockOrdinal, text, sourceRevisionId, policy });
    });
  }
  return units;
}

async function buildProjection(state: SubstrateState, docChunkId: ChunkId): Promise<MarkdownProjection> {
  const units = projectionUnits(state, docChunkId);
  const hashes = await Promise.all(units.map((u) => blobHashOf(MEDIA_MARKDOWN, u.text)));
  return {
    text: canonical(units.map((u) => u.text)),
    blocks: units.map((u, i) => ({
      occurrenceId: u.occurrence.id,
      occurrencePath: u.occurrencePath,
      blockOrdinal: u.blockOrdinal,
      chunkId: u.occurrence.chunkId,
      projectedTextHash: hashes[i],
      projectedText: u.text,
      sourceRevisionId: u.sourceRevisionId,
      pin: u.occurrence.pin,
      mode: u.occurrence.mode,
      policy: u.policy,
    })),
  };
}

function manifestSnapshotEqual(a: MarkdownProjectionBlock[], b: MarkdownProjectionBlock[]): boolean {
  return (
    a.length === b.length &&
    a.every((x, i) => {
      const y = b[i];
      return (
        x.occurrenceId === y.occurrenceId &&
        x.occurrencePath.join('/') === y.occurrencePath.join('/') &&
        x.blockOrdinal === y.blockOrdinal &&
        x.chunkId === y.chunkId &&
        x.projectedTextHash === y.projectedTextHash &&
        x.sourceRevisionId === y.sourceRevisionId &&
        x.pin === y.pin &&
        x.mode === y.mode &&
        x.policy === y.policy
      );
    })
  );
}

// V1 sidecars are upgraded in memory. The old hash/text remain the evidence
// used to match the last file projection, while current occurrence metadata
// supplies the authority policy.
function upgradeSidecar(state: SubstrateState, raw: MarkdownSidecar | LegacyMarkdownSidecar): MarkdownSidecar {
  if ('schemaVersion' in raw && raw.schemaVersion === SIDECAR_SCHEMA_VERSION) return raw;
  const legacy = raw as LegacyMarkdownSidecar;
  const unclaimed = childOccurrences(state, legacy.docChunkId);
  const blocks: MarkdownProjectionBlock[] = legacy.blocks.map((old) => {
    const i = unclaimed.findIndex((occ) => occ.chunkId === old.chunkId);
    if (i < 0) throw new Error(`legacy sidecar block ${old.chunkId} no longer occurs in ${legacy.docChunkId}`);
    const [occ] = unclaimed.splice(i, 1);
    const oldRevision = [...state.revisions.values()].find((r) => r.chunkId === old.chunkId && r.blobHash === old.blobHash);
    return {
      occurrenceId: occ.id,
      occurrencePath: [occ.id],
      blockOrdinal: 0,
      chunkId: old.chunkId,
      projectedTextHash: old.blobHash,
      projectedText: state.blobs.get(old.blobHash)?.text ?? '',
      sourceRevisionId: oldRevision?.id ?? occurrenceRevision(state, occ).id,
      pin: occ.pin,
      mode: occ.mode,
      policy: policyFor(state, occ),
    };
  });
  return { ...legacy, schemaVersion: SIDECAR_SCHEMA_VERSION, blocks };
}

export async function importMarkdownFile(
  ctx: TxCtx,
  opts: {
    workspaceRoot: string;
    relPath: string;
    text: string;
    operationParams?: Record<string, unknown>;
    sidecarPublish?: AtomicPublish;
  },
): Promise<{
  docChunkId: ChunkId;
  blockChunkIds: ChunkId[];
  commit: Commit;
  commitId: CommitId;
  operationId: OperationId;
}> {
  const { workspaceRoot, relPath } = opts;
  assertSafeRel(workspaceRoot, relPath);
  const text = normalizeEol(opts.text);
  const blockTexts = blockTextsOf(text);
  const doc = await createComposite(ctx, {
    join: '\n\n',
    blocks: blockTexts.map((t) => ({ text: t, mediaType: MEDIA_MARKDOWN })),
    opKind: 'import',
    operationParams: opts.operationParams,
  });
  const projection = await buildProjection(ctx.state, doc.chunkId);
  writeSidecar(
    workspaceRoot,
    relPath,
    {
      schemaVersion: SIDECAR_SCHEMA_VERSION,
      docChunkId: doc.chunkId,
      relPath,
      blocks: projection.blocks,
      lastImportedFileHash: await sha256Hex(text),
      lastProjectedFileHash: await sha256Hex(projection.text),
    },
    opts.sidecarPublish,
  );
  return {
    docChunkId: doc.chunkId,
    blockChunkIds: doc.blockChunkIds,
    commit: doc.commit,
    commitId: doc.commit.id,
    operationId: doc.commit.operation.id,
  };
}

// Complete the projection half of an import whose kernel operation became
// durable before its sidecar/catalog publication. The caller supplies the
// hash of the exact normalized source text recorded in its write-ahead intent;
// no source bytes need to be duplicated in the catalog.
export async function recoverMarkdownImport(
  state: SubstrateState,
  opts: { workspaceRoot: string; relPath: string; docChunkId: ChunkId; lastImportedFileHash: string },
): Promise<void> {
  const { workspaceRoot, relPath, docChunkId, lastImportedFileHash } = opts;
  assertSafeRel(workspaceRoot, relPath);
  const path = sidecarPath(workspaceRoot, relPath);
  if (existsSync(path)) {
    const existing = readSidecar(workspaceRoot, relPath);
    if (existing.docChunkId !== docChunkId) {
      throw new Error(`Markdown recovery found another projection manifest for ${relPath}`);
    }
    return;
  }
  const projection = await buildProjection(state, docChunkId);
  writeSidecar(workspaceRoot, relPath, {
    schemaVersion: SIDECAR_SCHEMA_VERSION,
    docChunkId,
    relPath,
    blocks: projection.blocks,
    lastImportedFileHash,
    lastProjectedFileHash: await sha256Hex(projection.text),
  });
}

export function projectMarkdown(state: SubstrateState, docChunkId: ChunkId): string {
  return canonical(projectionUnits(state, docChunkId).map((u) => u.text));
}

export async function writeProjection(
  ctx: TxCtx,
  opts: { workspaceRoot: string; relPath: string },
): Promise<{ docChunkId: ChunkId; text: string }> {
  const { workspaceRoot, relPath } = opts;
  const raw = readSidecar(workspaceRoot, relPath);
  const projection = await buildProjection(ctx.state, raw.docChunkId);
  const abs = sourcePathForProjection(workspaceRoot, relPath);
  const sourceText = normalizeEol(readFileSync(abs, 'utf8'));
  const [sourceHash, hash] = await Promise.all([sha256Hex(sourceText), sha256Hex(projection.text)]);
  const sourceIsKnown = sourceHash === raw.lastImportedFileHash || sourceHash === raw.lastProjectedFileHash;
  const sourceIsIntendedProjection = sourceHash === hash;
  if (!sourceIsKnown && !sourceIsIntendedProjection) {
    throw new ProjectionConflictError(
      `refusing to project ${relPath}: source changed since its last import or projection; sync first`,
    );
  }

  // If the destination already contains this projection, a previous attempt
  // may have replaced the source and failed before replacing the sidecar.
  // Advancing only the manifest makes that partial attempt safely retryable.
  if (!sourceIsIntendedProjection) atomicWriteText(abs, projection.text);
  writeSidecar(workspaceRoot, relPath, {
    schemaVersion: SIDECAR_SCHEMA_VERSION,
    docChunkId: raw.docChunkId,
    relPath: raw.relPath,
    blocks: projection.blocks,
    lastImportedFileHash: hash,
    lastProjectedFileHash: hash,
  });
  return { docChunkId: raw.docChunkId, text: projection.text };
}

export async function reconcileMarkdownFile(
  ctx: TxCtx,
  opts: { workspaceRoot: string; relPath: string; text: string; proposalMarker?: string },
): Promise<ReconcileResult> {
  const { workspaceRoot, relPath } = opts;
  const text = normalizeEol(opts.text);
  const state = ctx.state;
  const sc = upgradeSidecar(state, readSidecar(workspaceRoot, relPath));
  const fileHash = await sha256Hex(text);
  if (fileHash === sc.lastImportedFileHash || fileHash === sc.lastProjectedFileHash) return { action: 'noop' };

  const currentProjection = await buildProjection(state, sc.docChunkId);
  const currentProjectionHash = await sha256Hex(currentProjection.text);
  // Proposal acceptance mutates kernel truth, not driver memory. Converge the
  // manifest when accepted truth now renders exactly as the file.
  if (currentProjectionHash === fileHash) {
    const obsolete = openReconciliation(ctx, sc.docChunkId);
    if (obsolete) supersedeProposal(ctx, { proposalId: obsolete.id, reason: `${relPath} now matches the kernel projection` });
    writeSidecar(workspaceRoot, relPath, {
      ...sc,
      blocks: currentProjection.blocks,
      lastImportedFileHash: fileHash,
      lastProjectedFileHash: currentProjectionHash,
    });
    return { action: 'fast-forward' };
  }

  const proposalBasisHash = await sha256Hex(
    JSON.stringify({
      docRevisionId: currentRevision(state, sc.docChunkId).id,
      blocks: currentProjection.blocks.map((block) => ({
        occurrenceId: block.occurrenceId,
        occurrencePath: block.occurrencePath,
        blockOrdinal: block.blockOrdinal,
        chunkId: block.chunkId,
        projectedTextHash: block.projectedTextHash,
        sourceRevisionId: block.sourceRevisionId,
        pin: block.pin,
        mode: block.mode,
        policy: block.policy,
      })),
    }),
  );
  const marker = `${opts.proposalMarker ?? fileMarker(fileHash)}[basis:${proposalBasisHash}]`;
  const standing = openReconciliation(ctx, sc.docChunkId);
  if (standing && standing.note?.startsWith(marker)) return { action: 'proposal', proposalId: standing.id };
  if (standing) {
    supersedeProposal(ctx, { proposalId: standing.id, reason: `${relPath} changed again before the earlier reconciliation was resolved` });
  }

  const fileTexts = blockTextsOf(text);
  const fileHashes = await Promise.all(fileTexts.map((t) => blobHashOf(MEDIA_MARKDOWN, t)));
  const scMatched: boolean[] = sc.blocks.map(() => false);
  const matchOf: number[] = fileTexts.map(() => -1);
  const matchScoreOf: number[] = fileTexts.map(() => 0);
  const approximateMatchOf: boolean[] = fileTexts.map(() => false);
  const ambiguousMatchOf: boolean[] = fileTexts.map(() => false);

  // Pass 1: exact projected-text hash, each manifest entry claimed once.
  for (let i = 0; i < fileTexts.length; i++) {
    const j = sc.blocks.findIndex((b, k) => !scMatched[k] && b.projectedTextHash === fileHashes[i]);
    if (j >= 0) {
      matchOf[i] = j;
      matchScoreOf[i] = 1;
      scMatched[j] = true;
    }
  }
  // Pass 2: remaining entries matched monotonically against recorded text.
  {
    const freeSc = sc.blocks.map((_, k) => k).filter((k) => !scMatched[k]);
    let cursor = 0;
    for (let i = 0; i < fileTexts.length; i++) {
      if (matchOf[i] >= 0) continue;
      let picked: { c: number; k: number; assessment: ReturnType<typeof assessSimilarity> } | undefined;
      let candidateCount = 0;
      for (let c = cursor; c < freeSc.length; c++) {
        const k = freeSc[c];
        const assessment = assessSimilarity(fileTexts[i], sc.blocks[k].projectedText);
        if (assessment.score >= SIM_THRESHOLD) {
          candidateCount++;
          picked ??= { c, k, assessment };
          // One alternative is enough to prove the identity inference is
          // ambiguous; do not pay every remaining comparison merely to count.
          if (candidateCount > 1) break;
        }
      }
      if (picked) {
        matchOf[i] = picked.k;
        matchScoreOf[i] = picked.assessment.score;
        approximateMatchOf[i] = picked.assessment.approximate;
        ambiguousMatchOf[i] = candidateCount > 1;
        scMatched[picked.k] = true;
        cursor = picked.c + 1;
      }
    }
  }

  const vanished = sc.blocks.map((_, k) => k).filter((k) => !scMatched[k]);
  const isChanged = (i: number) => matchOf[i] >= 0 && sc.blocks[matchOf[i]].projectedTextHash !== fileHashes[i];
  const clean = manifestSnapshotEqual(sc.blocks, currentProjection.blocks);
  const occs = childOccurrences(state, sc.docChunkId);
  const occById = new Map(occs.map((occ) => [occ.id, occ]));
  const docHead = currentRevision(state, sc.docChunkId);
  const manifestCount = new Map<OccurrenceId, number>();
  for (const b of sc.blocks) manifestCount.set(b.occurrenceId, (manifestCount.get(b.occurrenceId) ?? 0) + 1);
  const protectedEdit = fileTexts.some((_, i) => isChanged(i) && sc.blocks[matchOf[i]].policy !== 'revise-leaf');
  const multiBlockOccurrence = [...manifestCount.values()].some((n) => n > 1);
  const reviewMatches = fileTexts
    .map((_, i) => ({ i, score: matchScoreOf[i], sampled: approximateMatchOf[i], ambiguous: ambiguousMatchOf[i] }))
    .filter((m) => m.sampled || m.ambiguous);

  if (clean && !protectedEdit && !multiBlockOccurrence && reviewMatches.length === 0) {
    // Only locally-owned leaves are allowed through the direct fast path.
    for (let i = 0; i < fileTexts.length; i++) {
      if (!isChanged(i)) continue;
      const block = sc.blocks[matchOf[i]];
      if (block.policy !== 'revise-leaf') throw new Error(`unsafe direct reconcile policy ${block.policy}`);
      await revise(ctx, { chunkId: block.chunkId, text: fileTexts[i], mediaType: MEDIA_MARKDOWN });
    }

    const vanishedOccurrenceIds = new Set(vanished.map((k) => sc.blocks[k].occurrenceId));
    let prevOccId: OccurrenceId | null = null;
    for (let i = 0; i < fileTexts.length; i++) {
      const j = matchOf[i];
      if (j >= 0) {
        const occId = sc.blocks[j].occurrenceId;
        const ordered = childOccurrences(state, sc.docChunkId).filter((o) => !vanishedOccurrenceIds.has(o.id));
        const idx = ordered.findIndex((o) => o.id === occId);
        const prevIdx = prevOccId ? ordered.findIndex((o) => o.id === prevOccId) : -1;
        if (idx < 0) throw new Error(`manifest occurrence ${occId} is no longer in document ${sc.docChunkId}`);
        if (idx !== prevIdx + 1) moveOccurrence(ctx, { occurrenceId: occId, at: prevOccId ? { after: prevOccId } : 'start' });
        prevOccId = occId;
      } else {
        const made = await createChunk(ctx, {
          text: fileTexts[i],
          mediaType: MEDIA_MARKDOWN,
          containerId: sc.docChunkId,
          at: prevOccId ? { after: prevOccId } : 'start',
        });
        prevOccId = made.occurrenceId!;
      }
    }

    let proposalId: ProposalId | undefined;
    const severs: ProposedChange[] = [];
    for (const occurrenceId of vanishedOccurrenceIds) {
      if (occById.has(occurrenceId)) severs.push({ op: 'sever', occurrenceId });
    }
    if (severs.length) {
      proposalId = propose(ctx, {
        kind: 'reconciliation',
        basisRevisionIds: [docHead.id],
        targetChunkIds: [sc.docChunkId],
        payload: severs,
        note: `${marker} ${severs.length} block occurrence(s) vanished from ${relPath}; accept to sever them`,
      }).proposalId;
    }

    const updatedProjection = await buildProjection(state, sc.docChunkId);
    writeSidecar(workspaceRoot, relPath, {
      ...sc,
      blocks: updatedProjection.blocks.filter((b) => !vanishedOccurrenceIds.has(b.occurrenceId)),
      lastImportedFileHash: fileHash,
      lastProjectedFileHash: await sha256Hex(canonical(fileTexts)),
    });
    return { action: 'fast-forward', proposalId };
  }

  // Dirty state, authority-bound content, and multi-block occurrences cross a
  // structural boundary. Carry the complete file-side delta in one proposal.
  const payload: ProposedChange[] = [];
  const basis = new Set<RevisionId>([docHead.id]);
  const byOccurrence = new Map<OccurrenceId, number[]>();
  sc.blocks.forEach((b, i) => {
    const xs = byOccurrence.get(b.occurrenceId);
    if (xs) xs.push(i);
    else byOccurrence.set(b.occurrenceId, [i]);
  });
  const fileIndicesFor = (occurrenceId: OccurrenceId): number[] =>
    matchOf
      .map((j, i) => ({ j, i }))
      .filter(({ j }) => j >= 0 && sc.blocks[j].occurrenceId === occurrenceId)
      .map(({ i }) => i);
  const handled = new Set<OccurrenceId>();
  const deferredSevers = new Set<OccurrenceId>();
  let tempN = 0;
  let protectedN = 0;
  let lastExistingOcc: OccurrenceId | undefined;
  let inNewRun = false;

  for (let i = 0; i < fileTexts.length; i++) {
    const j = matchOf[i];
    if (j < 0) {
      const tempId = `new-${tempN++}`;
      payload.push({ op: 'create', tempId, text: fileTexts[i], mediaType: MEDIA_MARKDOWN });
      const place: ProposedChange = { op: 'place', containerId: sc.docChunkId, chunkId: { tempId } };
      if (!inNewRun) {
        if (lastExistingOcc) place.after = lastExistingOcc;
        else place.at = 'start';
      }
      payload.push(place);
      inNewRun = true;
      continue;
    }

    const block = sc.blocks[j];
    const occurrenceId = block.occurrenceId;
    const group = byOccurrence.get(occurrenceId)!;
    const matchedFileIndices = fileIndicesFor(occurrenceId);
    const groupChanged = matchedFileIndices.some(isChanged) || matchedFileIndices.length !== group.length;
    if (!handled.has(occurrenceId) && groupChanged) {
      handled.add(occurrenceId);
      const replacementText = matchedFileIndices.map((k) => fileTexts[k]).join('\n\n');
      if (block.policy === 'detach-transclusion') {
        const tempId = `detached-${tempN++}`;
        payload.push({
          op: 'create',
          tempId,
          text: replacementText,
          mediaType: MEDIA_MARKDOWN,
          derivedFrom: { sourceRevisionId: block.sourceRevisionId, via: 'copy' },
        });
        const place: ProposedChange = { op: 'place', containerId: sc.docChunkId, chunkId: { tempId } };
        if (lastExistingOcc) place.after = lastExistingOcc;
        else place.at = 'start';
        payload.push(place);
        deferredSevers.add(occurrenceId);
        basis.add(block.sourceRevisionId);
        protectedN++;
      } else if (block.policy === 'flatten-composite') {
        payload.push({ op: 'revise', chunkId: block.chunkId, text: replacementText, mediaType: MEDIA_MARKDOWN });
        basis.add(currentRevision(state, block.chunkId).id);
        for (const child of childOccurrences(state, block.chunkId)) deferredSevers.add(child.id);
        protectedN++;
      } else {
        payload.push({ op: 'revise', chunkId: block.chunkId, text: replacementText, mediaType: MEDIA_MARKDOWN });
        basis.add(currentRevision(state, block.chunkId).id);
      }
    }
    if (occById.has(occurrenceId)) lastExistingOcc = occurrenceId;
    inNewRun = false;
  }

  // Whole vanished occurrences are local severs. Partial disappearance from a
  // multi-block occurrence was handled above as a replacement/flatten.
  for (const [occurrenceId, group] of byOccurrence) {
    if (group.every((k) => !scMatched[k])) deferredSevers.add(occurrenceId);
  }
  for (const occurrenceId of deferredSevers) {
    if (state.occurrences.has(occurrenceId)) payload.push({ op: 'sever', occurrenceId });
  }

  if (!payload.length) return { action: 'noop' };
  const internalChanged = currentProjection.blocks.filter((b, i) => {
    const old = sc.blocks[i];
    return !old || b.occurrenceId !== old.occurrenceId || b.projectedTextHash !== old.projectedTextHash;
  }).length;
  const nRevise = payload.filter((c) => c.op === 'revise').length;
  const sampledCount = reviewMatches.filter((m) => m.sampled).length;
  const ambiguousCount = reviewMatches.filter((m) => m.ambiguous).length;
  const matchReviewSummary = reviewMatches.length
    ? `${marker} ${relPath} has ${[
        sampledCount ? `${sampledCount} sampled-similarity match(es)` : '',
        ambiguousCount ? `${ambiguousCount} ambiguous match(es)` : '',
      ]
        .filter(Boolean)
        .join(' and ')} (minimum confidence ${Math.min(...reviewMatches.map((m) => m.score)).toFixed(2)}); review identity before applying`
    : null;
  const { proposalId } = propose(ctx, {
    kind: 'reconciliation',
    basisRevisionIds: [...basis],
    targetChunkIds: [sc.docChunkId],
    payload,
    note:
      matchReviewSummary ??
      (protectedN > 0
        ? `${marker} ${relPath} has ${protectedN} edit(s) that require detaching transclusion or flattening composite structure`
        : `${marker} ${relPath} diverged on both sides: file has ${nRevise} edited, ${tempN} new, ${vanished.length} vanished block(s); ` +
          `store has ${internalChanged} changed projection block(s) since last sync`),
  });
  return { action: 'proposal', proposalId };
}
