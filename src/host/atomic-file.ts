import { randomUUID } from 'node:crypto';
import { mkdirSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import { basename, dirname, join } from 'node:path';

export type AtomicPublish = (temporaryPath: string, destinationPath: string) => void;

// Same-directory temporary write followed by publish-by-rename. The optional
// publisher is a deliberately tiny fault-injection seam: callers use the
// default, while tests can prove a failed publish preserves the old file.
export function atomicWriteText(path: string, text: string, publish: AtomicPublish = renameSync): void {
  const dir = dirname(path);
  mkdirSync(dir, { recursive: true });
  const tmp = join(dir, `.substrate-write-${basename(path)}-${process.pid}-${randomUUID()}.tmp`);
  try {
    writeFileSync(tmp, text, { flag: 'wx' });
    publish(tmp, path);
  } finally {
    // After a successful rename the temporary path is already absent. On any
    // earlier failure, remove only the unpublished file and preserve the old
    // destination intact.
    rmSync(tmp, { force: true });
  }
}
