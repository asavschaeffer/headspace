// End-to-end check against a running development host (npm run dev), simulating
// the browser: fetch state, apply a commit locally, post it, verify host truth.
// Not part of the auto-run suite (no .test suffix): it needs the live host.
import assert from 'node:assert';
import { OFFLINE_COLLABORATOR, stubCompleter } from '../src/collaboration/stub';
import { deserializeState } from '../src/kernel/serialize';
import { renderChunk } from '../src/kernel/state';
import { generateProposal } from '../src/kernel/select';
import { acceptProposal, revise, type TxCtx } from '../src/kernel/tx';
import type { Commit } from '../src/kernel/types';

const base = process.argv[2] ?? 'http://localhost:5173';
const get = async () => {
  const j = await (await fetch(`${base}/api/state`)).json();
  return { state: deserializeState(j.state), bindings: j.bindings as { docChunkId: string; relPath: string }[] };
};

const { state, bindings } = await get();
const sent: Commit[] = [];
const ctx: TxCtx = { state, actorId: 'human:asa', onCommit: (c) => sent.push(c) };

const doc = bindings.find((b) => b.relPath === 'wiki/kernel.md')!;
const before = renderChunk(state, doc.docChunkId);

// A generation proposal, accepted — two commits, exercising create+place replay.
const gen = await generateProposal(ctx, {
  focusChunkId: doc.docChunkId,
  instruction: 'live-loop check',
  complete: stubCompleter,
  modelActorId: OFFLINE_COLLABORATOR.actorId,
});
const acc = await acceptProposal(ctx, { proposalId: gen.proposalId });
assert.ok(acc.applied);
// And a direct block revise.
const newBlock = acc.createdChunkIds[0];
await revise(ctx, { chunkId: newBlock, text: 'live-loop verified block' });

const r = await fetch(`${base}/api/commits`, {
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ commits: sent }),
});
assert.equal(r.status, 200, `host refused: ${r.status} ${await r.text()}`);

const after = await get();
assert.equal(after.state.head, state.head, 'host head matches client head after replay');
const hostText = renderChunk(after.state, doc.docChunkId);
assert.ok(hostText.includes('live-loop verified block'), 'host holds the revised block');
assert.ok(hostText.startsWith(before.slice(0, 200)), 'original doc content intact');
console.log(`live loop OK — ${sent.length} commits replayed, host head ${after.state.head?.slice(0, 12)}…`);
