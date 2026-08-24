import assert from 'node:assert';
import { mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { atomicWriteText } from '../src/host/atomic-file';

const root = mkdtempSync(join(tmpdir(), 'headspace-atomic-file-'));
try {
  const destination = join(root, 'binding.json');
  writeFileSync(destination, 'old');

  assert.throws(
    () =>
      atomicWriteText(destination, 'unpublished', () => {
        throw new Error('injected publish failure');
      }),
    /injected publish failure/,
  );
  assert.equal(readFileSync(destination, 'utf8'), 'old', 'failed publish preserves the previous destination');
  assert.deepEqual(
    readdirSync(root).filter((name) => name.startsWith('.substrate-write-')),
    [],
    'failed publish cleans its unpublished temporary file',
  );

  atomicWriteText(destination, 'new');
  assert.equal(readFileSync(destination, 'utf8'), 'new');
  assert.deepEqual(readdirSync(root), ['binding.json']);

  console.log('atomic file replacement OK — publish failure preserves prior data');
} finally {
  rmSync(root, { recursive: true, force: true });
}
