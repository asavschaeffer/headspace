// Filesystem driver: reflects a real folder into a chunk-tree snapshot.
// The filesystem remains authoritative; every chunk binds back to its source.
// STUB: write-back, fs watching, non-text file types, LLM-conversation import.
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import type { Chunk } from '../src/kernel/chunk';

const ROOT = resolve(process.argv[2] ?? '.');
const SKIP = new Set(['node_modules', '.git', 'dist', 'public']);
const at = new Date().toISOString();
const chunks: Chunk[] = [];

const add = (c: Chunk) => (chunks.push(c), c.id);

function ingestFile(path: string, parent: string): string {
  const rel = relative(ROOT, path).replaceAll('\\', '/');
  const blocks = readFileSync(path, 'utf8')
    .split(/\r?\n\s*\r?\n/)
    .map((s) => s.trim())
    .filter(Boolean)
    .map((text, i) =>
      add({
        id: `${rel}#${i}`,
        kind: 'block',
        text,
        children: [],
        version: 1,
        provenance: { author: 'driver:fs', at },
        binding: { path },
        parent: rel,
      }),
    );
  return add({
    id: rel,
    kind: 'doc',
    text: rel,
    children: blocks,
    version: 1,
    provenance: { author: 'driver:fs', at },
    binding: { path },
    parent,
  });
}

function ingestDir(path: string, parent?: string): string | null {
  const rel = path === ROOT ? '.' : relative(ROOT, path).replaceAll('\\', '/');
  const children: string[] = [];
  for (const e of readdirSync(path, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
    if (SKIP.has(e.name) || e.name.startsWith('.')) continue;
    const full = join(path, e.name);
    if (e.isDirectory()) {
      const id = ingestDir(full, rel);
      if (id) children.push(id);
    } else if (/\.(md|txt)$/i.test(e.name)) {
      children.push(ingestFile(full, rel));
    }
  }
  if (children.length === 0) return null; // prune empty branches of the tree
  return add({
    id: rel,
    kind: 'dir',
    text: rel,
    children,
    version: 1,
    provenance: { author: 'driver:fs', at },
    binding: { path },
    parent,
  });
}

const rootId = ingestDir(ROOT);
if (!rootId) {
  console.error(`No .md/.txt files found under ${ROOT}`);
  process.exit(1);
}
mkdirSync('public', { recursive: true });
writeFileSync('public/snapshot.json', JSON.stringify({ root: rootId, chunks }, null, 1));
console.log(`Ingested ${chunks.length} chunks from ${ROOT} → public/snapshot.json`);
