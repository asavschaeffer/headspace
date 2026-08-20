// Filesystem workspace store — the first durable backend behind the store seam
// (wiki/store.md). Rooted at <workspaceRoot>/.substrate/:
//   log.jsonl      append-only commits, one JSON line each (authoritative)
//   blobs/ab/abc…  content-addressed payloads, 2-char fan-out; the log also
//                  carries blobs, so replay never reads this dir — it is the
//                  canonical payload store for integrity and future compaction
//   snapshot.json  { coveredCommits, state } materialization (temp + rename)
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
import { applyCommit, emptyState, materialize, type SubstrateState } from '../kernel/state';
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

const SNAPSHOT_EVERY = 50; // appends between auto-snapshots

export interface WorkspaceStore {
  root: string; // the workspace root; store files live under <root>/.substrate
  state: SubstrateState;
  ctxFor(actorId: ActorId): TxCtx;
  appendCommit(commit: Commit): void; // persist a commit already folded into state (the TxCtx.onCommit hook)
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
  coveredCommits: number; // log lines folded into state; replay starts after this
  state: SerializedState;
}

// Map keys are always the value's own id/hash, so arrays of values round-trip.
function serializeState(s: SubstrateState): SerializedState {
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

function deserializeState(d: SerializedState): SubstrateState {
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
  const dir = join(rootDir, '.substrate');
  const logPath = join(dir, 'log.jsonl');
  const snapPath = join(dir, 'snapshot.json');
  const lockPath = join(dir, 'lock');
  mkdirSync(join(dir, 'blobs'), { recursive: true });

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
    const fanout = join(dir, 'blobs', b.hash.slice(0, 2));
    const path = join(fanout, b.hash);
    if (existsSync(path)) return; // immutable: first write wins
    mkdirSync(fanout, { recursive: true });
    const tmp = `${path}.tmp`;
    writeFileSync(tmp, JSON.stringify({ mediaType: b.mediaType, text: b.text }));
    renameSync(tmp, path);
  };

  let state: SubstrateState;
  try {
    let log: LogTail;
    if (existsSync(snapPath)) {
      const snap = JSON.parse(readFileSync(snapPath, 'utf8')) as SnapshotFile;
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

  const saveSnapshot = (): void => {
    const file: SnapshotFile = { coveredCommits: state.commitCount, state: serializeState(state) };
    const tmp = `${snapPath}.tmp`;
    writeFileSync(tmp, JSON.stringify(file));
    renameSync(tmp, snapPath);
    sinceSnapshot = 0;
  };

  const appendCommit = (commit: Commit): void => {
    if (closed) throw new Error('workspace store is closed');
    appendFileSync(logPath, `${JSON.stringify(commit)}\n`);
    for (const b of commit.facts.blobs ?? []) putBlob(b);
    if (++sinceSnapshot >= SNAPSHOT_EVERY) saveSnapshot();
  };

  return {
    root: rootDir,
    state,
    ctxFor: (actorId) => ({ state, actorId, onCommit: appendCommit }),
    appendCommit,
    saveSnapshot,
    close: () => {
      if (closed) return;
      saveSnapshot();
      closed = true;
      rmSync(lockPath, { force: true });
    },
  };
}
