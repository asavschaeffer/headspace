import assert from 'node:assert';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { emptyState } from '../src/kernel/state';
import { revise, type TxCtx } from '../src/kernel/tx';
import { importMarkdownFile, writeProjection } from '../src/host/markdown';

const envelope = mkdtempSync(join(tmpdir(), 'headspace-projection-confinement-'));
let linkExercised = false;
try {
  const root = join(envelope, 'workspace');
  const sourceDir = join(root, 'notes');
  const outside = join(envelope, 'outside');
  const relPath = 'notes/bound.md';
  const original = '# Original\n';
  mkdirSync(sourceDir, { recursive: true });
  mkdirSync(outside, { recursive: true });
  writeFileSync(join(sourceDir, 'bound.md'), original);
  writeFileSync(join(outside, 'bound.md'), original);

  const state = emptyState();
  const filesystemAdapter: TxCtx = { state, actorId: 'adapter:filesystem' };
  const human: TxCtx = { state, actorId: 'human:test' };
  const imported = await importMarkdownFile(filesystemAdapter, { workspaceRoot: root, relPath, text: original });
  await revise(human, { chunkId: imported.blockChunkIds[0], text: '# Internal revision' });

  // Replace an ordinary imported ancestor with a link/junction to an external
  // directory. The external file deliberately has the manifest-known bytes,
  // proving that fingerprint checks alone are not a path-confinement defense.
  rmSync(sourceDir, { recursive: true, force: true });
  try {
    symlinkSync(outside, sourceDir, process.platform === 'win32' ? 'junction' : 'dir');
    linkExercised = true;
  } catch (e) {
    const unavailable = new Set(['EACCES', 'EINVAL', 'ENOSYS', 'ENOTSUP', 'EPERM', 'UNKNOWN']);
    if (!unavailable.has((e as NodeJS.ErrnoException).code ?? '')) throw e;
  }

  if (linkExercised) {
    await assert.rejects(
      writeProjection(filesystemAdapter, { workspaceRoot: root, relPath }),
      /source resolves outside the workspace root/,
    );
    assert.equal(readFileSync(join(outside, 'bound.md'), 'utf8'), original, 'external target remains untouched');
  }
} finally {
  rmSync(envelope, { recursive: true, force: true });
}

console.log(`projection confinement OK — ${Number(linkExercised)} ancestor link escape path(s) exercised`);
