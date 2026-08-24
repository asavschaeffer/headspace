// Filesystem workspace store — the first durable backend behind the store seam
// Rooted at <workspaceRoot>/.headspace/:
//   log.jsonl      append-only commits, one JSON line each (authoritative)
//   blobs/ab/abc…  content-addressed payloads, 2-char fan-out; the log also
//                  carries blobs, so replay never reads this dir — it is the
//                  canonical payload store for integrity and future compaction
//   snapshot.json  { schemaVersion, coveredCommits, state } materialization
//                  (temp + rename)
//   lock           single-writer pid, exclusive-create
// Node-only; writes are sync so a commit is durable before the tx returns.

import {
  appendFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  rmSync,
  truncateSync,
  writeFileSync,
} from 'node:fs';
import { join } from 'node:path';
import { applyCommit, emptyState, materialize, type WorkspaceGraph } from '../kernel/state';
import type { TxCtx } from '../kernel/tx';
import type {
  ActorId,
  Blob,
  Chunk,
  Commit,
  Derivation,
  Link,
  Occurrence,
  Operation,
  Proposal,
  Revision,
} from '../kernel/types';
import { workspaceDataPaths } from './workspace-data';

const SNAPSHOT_EVERY = 50; // appends between auto-snapshots
const SNAPSHOT_SCHEMA_VERSION = 1;

export interface WorkspaceStore {
  root: string; // the user-owned workspace root
  dataDir: string; // the centralized host-owned .headspace directory
  state: WorkspaceGraph;
  ctxFor(actorId: ActorId): TxCtx;
  appendCommit(commit: Commit): void; // make a validated commit durable, before it is folded (the TxCtx.onCommit hook)
  snapshotIfDue(): void; // cadence check; call only once the commit is folded (the TxCtx.afterCommit hook)
  saveSnapshot(): void;
  close(): void; // snapshot, then release the lock; idempotent
}

interface SerializedState {
  chunks: Chunk[];
  revisions: Revision[];
  blobs: Blob[];
  occurrences: Occurrence[];
  links: Link[];
  derivations: Derivation[];
  proposals: Proposal[];
  operations: Operation[];
  head: string | null;
  commitCount: number;
}

interface SnapshotFile {
  schemaVersion: typeof SNAPSHOT_SCHEMA_VERSION;
  coveredCommits: number; // log lines folded into state; replay starts after this
  state: SerializedState;
}

type JsonObject = Record<string, unknown>;

const isJsonObject = (value: unknown): value is JsonObject =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

function invalidSnapshot(message: string): never {
  throw new Error(`invalid workspace snapshot: ${message}`);
}

function readSnapshot(snapshotPath: string): SnapshotFile {
  const parsed: unknown = JSON.parse(readFileSync(snapshotPath, 'utf8'));
  if (!isJsonObject(parsed)) invalidSnapshot('root must be an object');
  if (parsed.schemaVersion !== SNAPSHOT_SCHEMA_VERSION) {
    throw new Error(
      `unsupported workspace snapshot schema: ${String(parsed.schemaVersion)}; expected ${SNAPSHOT_SCHEMA_VERSION}`,
    );
  }
  if (!Number.isInteger(parsed.coveredCommits) || (parsed.coveredCommits as number) < 0) {
    invalidSnapshot('coveredCommits must be a non-negative integer');
  }
  if (!isJsonObject(parsed.state)) invalidSnapshot('state must be an object');
  const state = parsed.state;
  const arrayFields = [
    'chunks',
    'revisions',
    'blobs',
    'occurrences',
    'links',
    'derivations',
    'proposals',
    'operations',
  ] as const;
  for (const field of arrayFields) {
    if (!Array.isArray(state[field])) invalidSnapshot(`state.${field} must be an array`);
  }
  if (state.head !== null && typeof state.head !== 'string') invalidSnapshot('state.head must be a string or null');
  if (!Number.isInteger(state.commitCount) || (state.commitCount as number) < 0) {
    invalidSnapshot('state.commitCount must be a non-negative integer');
  }
  if (parsed.coveredCommits !== state.commitCount) {
    invalidSnapshot('coveredCommits must equal state.commitCount');
  }
  if ((state.commitCount === 0) !== (state.head === null)) {
    invalidSnapshot('state.head does not match state.commitCount');
  }

  const operations = new Map<string, JsonObject>();
  const operationValues = state.operations as unknown[];
  operationValues.forEach((value, index) => {
    if (!isJsonObject(value)) invalidSnapshot(`state.operations[${index}] must be an object`);
    if (typeof value.id !== 'string' || value.id.length === 0) invalidSnapshot(`state.operations[${index}].id must be a string`);
    if (operations.has(value.id)) invalidSnapshot(`duplicate operation ${value.id}`);
    if (typeof value.kind !== 'string') invalidSnapshot(`operation ${value.id}.kind must be a string`);
    if (typeof value.actorId !== 'string') invalidSnapshot(`operation ${value.id}.actorId must be a string`);
    if (typeof value.at !== 'string') invalidSnapshot(`operation ${value.id}.at must be a string`);
    if (!Array.isArray(value.inputRevisionIds) || value.inputRevisionIds.some((id) => typeof id !== 'string')) {
      invalidSnapshot(`operation ${value.id}.inputRevisionIds must be a string array`);
    }
    if (!Array.isArray(value.outputRevisionIds) || value.outputRevisionIds.some((id) => typeof id !== 'string')) {
      invalidSnapshot(`operation ${value.id}.outputRevisionIds must be a string array`);
    }
    operations.set(value.id, value);
  });
  if (operations.size !== state.commitCount) invalidSnapshot('operation count must equal state.commitCount');

  const proposalIds = new Set<string>();
  const proposalValues = state.proposals as unknown[];
  proposalValues.forEach((value, index) => {
    if (!isJsonObject(value)) invalidSnapshot(`state.proposals[${index}] must be an object`);
    if (typeof value.id !== 'string' || value.id.length === 0) invalidSnapshot(`state.proposals[${index}].id must be a string`);
    if (proposalIds.has(value.id)) invalidSnapshot(`duplicate proposal ${value.id}`);
    proposalIds.add(value.id);
    if (typeof value.operationId !== 'string') invalidSnapshot(`proposal ${value.id}.operationId must be a string`);
    if (typeof value.kind !== 'string') invalidSnapshot(`proposal ${value.id}.kind must be a string`);
    if (typeof value.createdBy !== 'string' || typeof value.createdAt !== 'string') {
      invalidSnapshot(`proposal ${value.id} creation provenance is malformed`);
    }
    const proposeOperation = operations.get(value.operationId);
    if (
      !proposeOperation ||
      proposeOperation.kind !== 'propose' ||
      proposeOperation.at !== value.createdAt ||
      !isJsonObject(proposeOperation.params) ||
      proposeOperation.params.kind !== value.kind
    ) {
      invalidSnapshot(`proposal ${value.id} does not resolve to its matching propose operation`);
    }
    if (value.status === 'open') {
      if (value.resolution !== undefined) invalidSnapshot(`open proposal ${value.id} cannot have a resolution`);
      return;
    }
    if (value.status !== 'accepted' && value.status !== 'rejected' && value.status !== 'superseded') {
      invalidSnapshot(`proposal ${value.id}.status is unsupported`);
    }
    if (!isJsonObject(value.resolution)) invalidSnapshot(`terminal proposal ${value.id} requires a resolution`);
    const resolution = value.resolution;
    if (
      typeof resolution.operationId !== 'string' ||
      typeof resolution.by !== 'string' ||
      typeof resolution.at !== 'string'
    ) {
      invalidSnapshot(`proposal ${value.id} resolution provenance is malformed`);
    }
    const resolutionOperation = operations.get(resolution.operationId);
    const expectedKind = value.status === 'accepted' ? 'accept' : 'reject';
    if (
      !resolutionOperation ||
      resolutionOperation.kind !== expectedKind ||
      resolutionOperation.actorId !== resolution.by ||
      resolutionOperation.at !== resolution.at
    ) {
      invalidSnapshot(`proposal ${value.id} resolution does not resolve consistently`);
    }
    if (
      value.status === 'accepted' &&
      (!isJsonObject(resolutionOperation.params) || resolutionOperation.params.proposalId !== value.id)
    ) {
      invalidSnapshot(`proposal ${value.id} acceptance operation names another proposal`);
    }
  });

  return parsed as unknown as SnapshotFile;
}

// Map keys are always the value's own id/hash, so arrays of values round-trip.
function serializeState(s: WorkspaceGraph): SerializedState {
  return {
    chunks: [...s.chunks.values()],
    revisions: [...s.revisions.values()],
    blobs: [...s.blobs.values()],
    occurrences: [...s.occurrences.values()],
    links: [...s.links.values()],
    derivations: [...s.derivations.values()],
    proposals: [...s.proposals.values()],
    operations: [...s.operations.values()],
    head: s.head,
    commitCount: s.commitCount,
  };
}

function deserializeState(d: SerializedState): WorkspaceGraph {
  const s = emptyState();
  for (const c of d.chunks) s.chunks.set(c.id, c);
  for (const r of d.revisions) s.revisions.set(r.id, r);
  for (const b of d.blobs) s.blobs.set(b.hash, b);
  for (const o of d.occurrences) s.occurrences.set(o.id, o);
  for (const l of d.links) s.links.set(l.id, l);
  for (const dv of d.derivations) s.derivations.set(dv.id, dv);
  for (const p of d.proposals) s.proposals.set(p.id, p);
  for (const op of d.operations) s.operations.set(op.id, op);
  s.head = d.head;
  s.commitCount = d.commitCount;
  return s;
}

interface LogTail {
  tail: Commit[]; // parsed commits from line `from` onward
  cleanBytes: number; // byte length of the newline-terminated prefix
  torn: boolean; // a partial final line (crash artifact) follows the prefix
}

// A line is a commit iff newline-terminated: appendCommit writes payload+'\n'
// in one call, so anything after the last '\n' is a torn append and the commit
// it began never became durable. A terminated line that fails to parse is
// corruption and throws.
function readLog(logPath: string, from: number): LogTail {
  if (!existsSync(logPath)) {
    if (from > 0) throw new Error(`snapshot covers ${from} commits but log.jsonl is missing`);
    return { tail: [], cleanBytes: 0, torn: false };
  }
  const raw = readFileSync(logPath, 'utf8');
  const cleanEnd = raw.lastIndexOf('\n') + 1;
  const clean = raw.slice(0, cleanEnd);
  const lines = cleanEnd > 0 ? clean.slice(0, -1).split('\n') : [];
  if (lines.length < from) throw new Error(`snapshot covers ${from} commits but log.jsonl has ${lines.length}`);
  const tail: Commit[] = [];
  for (let i = from; i < lines.length; i++) {
    try {
      tail.push(JSON.parse(lines[i]) as Commit);
    } catch (e) {
      throw new Error(`log.jsonl corrupt at line ${i + 1}: ${(e as Error).message}`);
    }
  }
  return { tail, cleanBytes: Buffer.byteLength(clean, 'utf8'), torn: cleanEnd < raw.length };
}

export async function openWorkspace(rootDir: string, opts: { force?: boolean } = {}): Promise<WorkspaceStore> {
  const paths = workspaceDataPaths(rootDir);
  const dir = paths.dataDir;
  const logPath = paths.logPath;
  const snapPath = paths.snapshotPath;
  const lockPath = paths.lockPath;
  mkdirSync(paths.blobsDir, { recursive: true });

  // Single-writer: exclusive-create wins. A lock whose holder pid is no longer
  // alive is a crash artifact and is taken over; a live holder is respected —
  // force exists for deliberate takeover only, never as a default.
  const acquireLock = (): void => {
    try {
      writeFileSync(lockPath, String(process.pid), { flag: opts.force ? 'w' : 'wx' });
    } catch (e) {
      if ((e as NodeJS.ErrnoException).code !== 'EEXIST') throw e;
      const holder = Number.parseInt(readFileSync(lockPath, 'utf8').trim(), 10);
      let alive = false;
      if (Number.isFinite(holder) && holder > 0) {
        if (holder === process.pid) {
          alive = true; // our own earlier open still holds it — still a second writer
        } else {
          try {
            process.kill(holder, 0);
            alive = true;
          } catch {
            alive = false;
          }
        }
      }
      if (alive) {
        throw new Error(
          `workspace ${rootDir} is locked by running pid ${holder} — close that process (or pass { force: true } to take over)`,
        );
      }
      rmSync(lockPath, { force: true }); // stale lock from a dead process
      writeFileSync(lockPath, String(process.pid), { flag: 'wx' });
    }
  };
  acquireLock();

  const putBlob = (b: Blob): void => {
    const fanout = join(paths.blobsDir, b.hash.slice(0, 2));
    const path = join(fanout, b.hash);
    if (existsSync(path)) return; // immutable: first write wins
    mkdirSync(fanout, { recursive: true });
    const tmp = `${path}.tmp`;
    writeFileSync(tmp, JSON.stringify({ mediaType: b.mediaType, text: b.text }));
    renameSync(tmp, path);
  };

  let state: WorkspaceGraph;
  try {
    let log: LogTail;
    if (existsSync(snapPath)) {
      const snap = readSnapshot(snapPath);
      state = deserializeState(snap.state);
      log = readLog(logPath, snap.coveredCommits);
      for (const c of log.tail) applyCommit(state, c);
    } else {
      log = readLog(logPath, 0);
      state = materialize(log.tail);
    }
    // Drop the crash artifact so future appends stay line-aligned.
    if (log.torn) truncateSync(logPath, log.cleanBytes);
    // Heal blobs a crash may have separated from their logged commit.
    for (const c of log.tail) for (const b of c.facts.blobs ?? []) putBlob(b);
  } catch (e) {
    rmSync(lockPath, { force: true }); // a failed open must not leave a stale lock
    throw e;
  }

  let sinceSnapshot = 0;
  let closed = false;
  let writeFailed: Error | null = null;

  const saveSnapshot = (): void => {
    // coveredCommits is a LINE OFFSET into the log. Recording it while the log
    // is suspect would tell the next open to skip lines that may not be there.
    if (writeFailed) throw new Error(`refusing to snapshot an unwritable workspace: ${writeFailed.message}`);
    const file: SnapshotFile = {
      schemaVersion: SNAPSHOT_SCHEMA_VERSION,
      coveredCommits: state.commitCount,
      state: serializeState(state),
    };
    const tmp = `${snapPath}.tmp`;
    writeFileSync(tmp, JSON.stringify(file));
    renameSync(tmp, snapPath);
    sinceSnapshot = 0;
  };

  const appendCommit = (commit: Commit): void => {
    if (closed) throw new Error('workspace store is closed');
    if (writeFailed) throw new Error(`workspace store is not writable: ${writeFailed.message}`);
    try {
      appendFileSync(logPath, `${JSON.stringify(commit)}\n`);
    } catch (e) {
      // A failed append may have written a partial line. At the tail that is a
      // recoverable crash artifact, but appending PAST it would bury the torn
      // line mid-log, where recovery cannot tell it from corruption. So the
      // store stops writing here, and the caller never folds this commit.
      writeFailed = e as Error;
      throw e;
    }
    // Durable now, and the log carries these blobs, so a failure to mirror them
    // into blobs/ is healed on the next open. It must not fail a transaction
    // that already reached the log.
    for (const b of commit.facts.blobs ?? []) {
      try {
        putBlob(b);
      } catch {
        /* healed by openWorkspace from the log tail */
      }
    }
    sinceSnapshot++;
  };

  // A snapshot records coveredCommits, a line offset into the log, so it may
  // only be taken when the folded state and the log agree — that is, after the
  // fold, never from inside the append.
  const snapshotIfDue = (): void => {
    if (sinceSnapshot >= SNAPSHOT_EVERY) saveSnapshot();
  };

  return {
    root: rootDir,
    dataDir: dir,
    state,
    ctxFor: (actorId) => ({ state, actorId, onCommit: appendCommit, afterCommit: snapshotIfDue }),
    appendCommit,
    snapshotIfDue,
    saveSnapshot,
    close: () => {
      if (closed) return;
      // An unwritable store still releases its lock: refusing to close would
      // leave the workspace held by a process that can no longer write it.
      if (!writeFailed) saveSnapshot();
      closed = true;
      rmSync(lockPath, { force: true });
    },
  };
}
