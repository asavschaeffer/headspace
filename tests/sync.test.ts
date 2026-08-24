import assert from 'node:assert';
import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { syncWorkspace } from '../src/host/sync';
import { openWorkspace, type WorkspaceStore } from '../src/host/store-fs';

const envelopes: string[] = [];
const freshEnvelope = () => {
  const dir = mkdtempSync(join(tmpdir(), 'substrate-sync-'));
  envelopes.push(dir);
  return dir;
};

const close = (ws: WorkspaceStore | null) => {
  if (ws) ws.close();
};

const linkUnavailable = new Set(['EACCES', 'EINVAL', 'ENOSYS', 'ENOTSUP', 'EPERM', 'UNKNOWN']);
const trySymlink = (target: string, path: string, type: 'file' | 'dir' | 'junction'): boolean => {
  try {
    symlinkSync(target, path, type);
    return true;
  } catch (e) {
    const code = (e as NodeJS.ErrnoException).code ?? '';
    if (linkUnavailable.has(code)) return false;
    throw e;
  }
};

let linksExercised = 0;
try {
  // The compatibility sync summary now rides over native Markdown and text
  // adapters while preserving recursive, case-insensitive Markdown handling.
  {
    const envelope = freshEnvelope();
    const root = join(envelope, 'workspace');
    mkdirSync(join(root, 'notes', 'deep'), { recursive: true });
    writeFileSync(join(root, 'notes', 'alpha.md'), '# Alpha\n');
    writeFileSync(join(root, 'notes', 'deep', 'beta.MD'), '# Beta\n');
    writeFileSync(join(root, 'notes', 'deep', 'note.txt'), 'plain text is a native source');

    let ws: WorkspaceStore | null = await openWorkspace(root);
    try {
      const first = await syncWorkspace(ws, { contentDirs: ['notes'] });
      assert.deepEqual(first.imported, ['notes/alpha.md', 'notes/deep/beta.MD', 'notes/deep/note.txt']);
      assert.equal(ws.state.chunks.size, 6, 'three native documents with one block each were ingested');
      assert.equal(first.ingestion.counts.unsupported, 0);

      const second = await syncWorkspace(ws, { contentDirs: ['notes'] });
      assert.equal(second.unchanged, 3, 'a second nested sweep recognizes every native document');
      assert.deepEqual(second.imported, []);
    } finally {
      close(ws);
      ws = null;
    }
  }

  // Lexical traversal and absolute paths outside the workspace are always
  // rejected. This remains the confinement test on Windows hosts where the
  // current account is not allowed to create symbolic links.
  {
    const envelope = freshEnvelope();
    const root = join(envelope, 'workspace');
    const outside = join(envelope, 'outside');
    mkdirSync(root, { recursive: true });
    mkdirSync(outside, { recursive: true });
    const secret = join(outside, 'secret.md');
    writeFileSync(secret, '# Outside\n');

    let ws: WorkspaceStore | null = await openWorkspace(root);
    try {
      const report = await syncWorkspace(ws, {
        contentDirs: ['../outside', outside],
        contentFiles: ['../outside/secret.md', secret],
      });
      assert.deepEqual(report.imported, []);
      assert.equal(ws.state.chunks.size, 0, 'outside Markdown never enters the substrate');
    } finally {
      close(ws);
      ws = null;
    }
  }

  // When the host permits links, exercise both a file symlink and a Windows-
  // friendly directory junction. Neither recursive discovery nor an explicit
  // contentFiles path may follow them outside the real workspace root.
  {
    const envelope = freshEnvelope();
    const root = join(envelope, 'workspace');
    const safe = join(root, 'safe');
    const outside = join(envelope, 'outside');
    mkdirSync(safe, { recursive: true });
    mkdirSync(outside, { recursive: true });
    writeFileSync(join(safe, 'inside.md'), '# Inside\n');
    const outsideFile = join(outside, 'outside.md');
    writeFileSync(outsideFile, '# Outside file\n');
    writeFileSync(join(outside, 'nested.md'), '# Outside directory\n');

    const fileLink = join(root, 'linked-file.md');
    const dirLink = join(root, 'linked-dir');
    const madeFileLink = trySymlink(outsideFile, fileLink, 'file');
    const madeDirLink = trySymlink(outside, dirLink, process.platform === 'win32' ? 'junction' : 'dir');
    linksExercised += Number(madeFileLink) + Number(madeDirLink);

    let ws: WorkspaceStore | null = await openWorkspace(root);
    try {
      const report = await syncWorkspace(ws, {
        contentDirs: ['.'],
        contentFiles: [
          ...(madeFileLink ? ['linked-file.md'] : []),
          ...(madeDirLink ? ['linked-dir/nested.md'] : []),
        ],
      });
      assert.deepEqual(report.imported, ['safe/inside.md']);
      assert.equal(ws.state.chunks.size, 2, 'only the ordinary in-root document was ingested');
    } finally {
      close(ws);
      ws = null;
    }
  }
} finally {
  for (const dir of envelopes) rmSync(dir, { recursive: true, force: true });
}

console.log(`sync confinement OK — ${linksExercised} symlink/junction escape path(s) exercised`);
