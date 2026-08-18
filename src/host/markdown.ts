// The markdown filesystem driver (wiki/drivers.md): one file = one composite
// doc chunk + one MEDIA_MARKDOWN chunk per md/blocks@1 block. Round-trip memory
// lives in a sidecar under .substrate/sidecars/ — never as ids in the .md.
// Reconcile matches file blocks back to their chunks (exact hash, then in-order
// similarity >= 0.5); an external-only edit applies directly as driver:fs, a
// two-sided divergence or a vanished block becomes a reconciliation proposal.
// Never silent loss in either direction (wiki/conflicts.md).

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, isAbsolute, join, resolve, sep } from 'node:path';
import { METHOD_BLOCKS, decomposeText } from '../kernel/decompose';
import { blobHashOf, sha256Hex } from '../kernel/hash';
import { childOccurrences, currentRevision, occurrenceRevision, revisionText, type SubstrateState } from '../kernel/state';
import { createChunk, moveOccurrence, propose, revise, type TxCtx } from '../kernel/tx';
import { MEDIA_COMPOSITE, MEDIA_MARKDOWN } from '../kernel/types';
import type { BlobHash, ChunkId, OccurrenceId, ProposalId, ProposedChange } from '../kernel/types';
import { similarity } from './similarity';

const SIM_THRESHOLD = 0.5;

export interface MarkdownSidecar {
  docChunkId: ChunkId;
  relPath: string;
  blocks: { chunkId: ChunkId; blobHash: BlobHash }[]; // in occurrence order
  lastImportedFileHash: string; // sha256Hex of the raw file text last read
  lastProjectedFileHash: string; // sha256Hex of the canonical projection at last sync
}

export interface ReconcileResult {
  action: 'noop' | 'fast-forward' | 'proposal';
  proposalId?: ProposalId; // set on 'proposal', and on 'fast-forward' when blocks vanished (sever proposal)
}

// A workspace-relative path must stay inside the root; '.', '..' escapes, and
// absolute paths are refused before any commit or filesystem write happens.
function assertSafeRel(workspaceRoot: string, relPath: string): void {
  if (isAbsolute(relPath)) throw new Error(`relPath must be workspace-relative: ${relPath}`);
  const base = resolve(workspaceRoot);
  const abs = resolve(base, relPath);
  const prefix = base.endsWith(sep) ? base : base + sep;
  if (abs === base || !abs.startsWith(prefix)) throw new Error(`relPath escapes workspace root: ${relPath}`);
}

export function sidecarPath(workspaceRoot: string, relPath: string): string {
  assertSafeRel(workspaceRoot, relPath);
  return join(workspaceRoot, '.substrate', 'sidecars', `${relPath}.json`);
}

function readSidecar(workspaceRoot: string, relPath: string): MarkdownSidecar {
  return JSON.parse(readFileSync(sidecarPath(workspaceRoot, relPath), 'utf8')) as MarkdownSidecar;
}

function writeSidecar(workspaceRoot: string, relPath: string, sc: MarkdownSidecar): void {
  const path = sidecarPath(workspaceRoot, relPath);
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${JSON.stringify(sc, null, 2)}\n`);
}

const blockTextsOf = (text: string): string[] => decomposeText(text, METHOD_BLOCKS).map((s) => text.slice(s.start, s.end));

// Canonical file form: blocks joined by one blank line, trailing newline.
const canonical = (blocks: string[]): string => (blocks.length ? `${blocks.join('\n\n')}\n` : '');

export async function importMarkdownFile(
  ctx: TxCtx,
  opts: { workspaceRoot: string; relPath: string; text: string },
): Promise<{ docChunkId: ChunkId; blockChunkIds: ChunkId[] }> {
  const { workspaceRoot, relPath, text } = opts;
  assertSafeRel(workspaceRoot, relPath); // fail before minting any identity
  const blockTexts = blockTextsOf(text);
  const doc = await createChunk(ctx, { text: JSON.stringify({ join: '\n\n' }), mediaType: MEDIA_COMPOSITE });
  const blockChunkIds: ChunkId[] = [];
  const blocks: MarkdownSidecar['blocks'] = [];
  for (const t of blockTexts) {
    const b = await createChunk(ctx, { text: t, mediaType: MEDIA_MARKDOWN, containerId: doc.chunkId });
    blockChunkIds.push(b.chunkId);
    blocks.push({ chunkId: b.chunkId, blobHash: ctx.state.revisions.get(b.revisionId)!.blobHash });
  }
  writeSidecar(workspaceRoot, relPath, {
    docChunkId: doc.chunkId,
    relPath,
    blocks,
    lastImportedFileHash: await sha256Hex(text),
    lastProjectedFileHash: await sha256Hex(canonical(blockTexts)),
  });
  return { docChunkId: doc.chunkId, blockChunkIds };
}

export function projectMarkdown(state: SubstrateState, docChunkId: ChunkId): string {
  const parts = childOccurrences(state, docChunkId).map((occ) => revisionText(state, occurrenceRevision(state, occ).id));
  return canonical(parts);
}

export async function writeProjection(
  ctx: TxCtx,
  opts: { workspaceRoot: string; relPath: string },
): Promise<{ docChunkId: ChunkId; text: string }> {
  const { workspaceRoot, relPath } = opts;
  const sc = readSidecar(workspaceRoot, relPath);
  const text = projectMarkdown(ctx.state, sc.docChunkId);
  const abs = join(workspaceRoot, relPath);
  mkdirSync(dirname(abs), { recursive: true });
  writeFileSync(abs, text);
  const hash = await sha256Hex(text);
  const blocks = childOccurrences(ctx.state, sc.docChunkId).map((occ) => ({
    chunkId: occ.chunkId,
    blobHash: occurrenceRevision(ctx.state, occ).blobHash,
  }));
  writeSidecar(workspaceRoot, relPath, { ...sc, blocks, lastImportedFileHash: hash, lastProjectedFileHash: hash });
  return { docChunkId: sc.docChunkId, text };
}

export async function reconcileMarkdownFile(ctx: TxCtx, opts: { workspaceRoot: string; relPath: string; text: string }): Promise<ReconcileResult> {
  const { workspaceRoot, relPath, text } = opts;
  const sc = readSidecar(workspaceRoot, relPath);
  const state = ctx.state;
  const fileHash = await sha256Hex(text);
  if (fileHash === sc.lastImportedFileHash || fileHash === sc.lastProjectedFileHash) return { action: 'noop' };

  const fileTexts = blockTextsOf(text);
  const fileHashes = await Promise.all(fileTexts.map((t) => blobHashOf(MEDIA_MARKDOWN, t)));

  // Pass 1 — exact blobHash, each sidecar block claimed once, earliest first.
  const scMatched: boolean[] = sc.blocks.map(() => false);
  const matchOf: number[] = fileTexts.map(() => -1); // file index -> sidecar index
  for (let i = 0; i < fileTexts.length; i++) {
    const j = sc.blocks.findIndex((b, k) => !scMatched[k] && b.blobHash === fileHashes[i]);
    if (j >= 0) {
      matchOf[i] = j;
      scMatched[j] = true;
    }
  }
  // Pass 2 — leftovers matched monotonically in sidecar order by similarity.
  {
    const freeSc = sc.blocks.map((_, k) => k).filter((k) => !scMatched[k]);
    let cursor = 0;
    for (let i = 0; i < fileTexts.length; i++) {
      if (matchOf[i] >= 0) continue;
      for (let c = cursor; c < freeSc.length; c++) {
        const k = freeSc[c];
        const old = state.blobs.get(sc.blocks[k].blobHash)?.text;
        if (old !== undefined && similarity(fileTexts[i], old) >= SIM_THRESHOLD) {
          matchOf[i] = k;
          scMatched[k] = true;
          cursor = c + 1;
          break;
        }
      }
    }
  }
  const vanished = sc.blocks.map((_, k) => k).filter((k) => !scMatched[k]);
  const isChanged = (i: number) => matchOf[i] >= 0 && sc.blocks[matchOf[i]].blobHash !== fileHashes[i];

  // Internal-clean: no store-side edits since the sidecar was written — every
  // recorded blobHash is still current and the occurrence order still matches.
  const occs = childOccurrences(state, sc.docChunkId);
  const clean =
    occs.length === sc.blocks.length &&
    occs.every((o, i) => o.chunkId === sc.blocks[i].chunkId) &&
    sc.blocks.every((b) => state.chunks.has(b.chunkId) && currentRevision(state, b.chunkId).blobHash === b.blobHash);

  const occByChunk = new Map(occs.map((o) => [o.chunkId, o]));
  const docHead = currentRevision(state, sc.docChunkId);

  if (clean) {
    // Fast path: an external-only edit is just an edit, arriving through the driver.
    for (let i = 0; i < fileTexts.length; i++) {
      if (isChanged(i)) await revise(ctx, { chunkId: sc.blocks[matchOf[i]].chunkId, text: fileTexts[i] });
    }
    // Realize file order: create new blocks in place, move strays. Vanished
    // blocks are ignored for ordering so a pending sever keeps its position.
    const vanishedChunks = new Set(vanished.map((k) => sc.blocks[k].chunkId));
    const newBlocks: MarkdownSidecar['blocks'] = [];
    let prevOccId: OccurrenceId | null = null;
    for (let i = 0; i < fileTexts.length; i++) {
      const j = matchOf[i];
      if (j >= 0) {
        const chunkId = sc.blocks[j].chunkId;
        const ordered = childOccurrences(state, sc.docChunkId).filter((o) => !vanishedChunks.has(o.chunkId));
        const idx = ordered.findIndex((o) => o.chunkId === chunkId);
        const prevIdx = prevOccId ? ordered.findIndex((o) => o.id === prevOccId) : -1;
        const occId = ordered[idx].id;
        if (idx !== prevIdx + 1) moveOccurrence(ctx, { occurrenceId: occId, at: prevOccId ? { after: prevOccId } : 'start' });
        prevOccId = occId;
        newBlocks.push({ chunkId, blobHash: fileHashes[i] });
      } else {
        const made = await createChunk(ctx, {
          text: fileTexts[i],
          mediaType: MEDIA_MARKDOWN,
          containerId: sc.docChunkId,
          at: prevOccId ? { after: prevOccId } : 'start',
        });
        prevOccId = made.occurrenceId!;
        newBlocks.push({ chunkId: made.chunkId, blobHash: fileHashes[i] });
      }
    }
    // A vanished block is never deleted outright: its occurrence stays until a
    // human accepts the sever (never silent loss).
    let proposalId: ProposalId | undefined;
    const severs: ProposedChange[] = [];
    for (const k of vanished) {
      const occ = occByChunk.get(sc.blocks[k].chunkId);
      if (occ) severs.push({ op: 'sever', occurrenceId: occ.id });
    }
    if (severs.length) {
      proposalId = propose(ctx, {
        kind: 'reconciliation',
        basisRevisionIds: [docHead.id],
        targetChunkIds: [sc.docChunkId],
        payload: severs,
        note: `${severs.length} block(s) vanished from ${relPath}; accept to sever their occurrences`,
      }).proposalId;
    }
    writeSidecar(workspaceRoot, relPath, {
      ...sc,
      blocks: newBlocks,
      lastImportedFileHash: fileHash,
      lastProjectedFileHash: await sha256Hex(canonical(fileTexts)),
    });
    return { action: 'fast-forward', proposalId };
  }

  // Internal-dirty: both sides moved since the last sync. No direct writes —
  // the whole file-side delta rides one reconciliation proposal; the store-side
  // edits stay current and become the parents of any accepted revise.
  const payload: ProposedChange[] = [];
  const basis = new Set<string>([docHead.id]);
  let tempN = 0;
  let lastExistingOcc: OccurrenceId | undefined;
  for (let i = 0; i < fileTexts.length; i++) {
    const j = matchOf[i];
    if (j >= 0) {
      const chunkId = sc.blocks[j].chunkId;
      if (isChanged(i) && state.chunks.has(chunkId)) {
        payload.push({ op: 'revise', chunkId, text: fileTexts[i] });
        basis.add(state.chunks.get(chunkId)!.currentRevisionId);
      }
      const occ = occByChunk.get(chunkId);
      if (occ) lastExistingOcc = occ.id;
    } else {
      const tempId = `new-${tempN++}`;
      payload.push({ op: 'create', tempId, text: fileTexts[i], mediaType: MEDIA_MARKDOWN });
      payload.push({ op: 'place', containerId: sc.docChunkId, chunkId: { tempId }, after: lastExistingOcc });
    }
  }
  for (const k of vanished) {
    const occ = occByChunk.get(sc.blocks[k].chunkId);
    if (occ) payload.push({ op: 'sever', occurrenceId: occ.id });
  }
  // A dirty store plus a cosmetic-only file change carries no ops; the next
  // writeProjection reconverges the file, so there is nothing to propose.
  if (!payload.length) return { action: 'noop' };
  const internalChanged = sc.blocks.filter(
    (b) => state.chunks.has(b.chunkId) && currentRevision(state, b.chunkId).blobHash !== b.blobHash,
  ).length;
  const nRevise = payload.filter((c) => c.op === 'revise').length;
  const { proposalId } = propose(ctx, {
    kind: 'reconciliation',
    basisRevisionIds: [...basis],
    targetChunkIds: [sc.docChunkId],
    payload,
    note:
      `${relPath} diverged on both sides: file has ${nRevise} edited, ${tempN} new, ${vanished.length} vanished block(s); ` +
      `store has ${internalChanged} block(s) revised since last sync`,
  });
  return { action: 'proposal', proposalId };
}
