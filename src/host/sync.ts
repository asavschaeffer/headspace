// Reflect a folder of markdown into the workspace: new files import, known
// files reconcile, watched sources raise update proposals. The filesystem
// remains authoritative for external edits; nothing here silently loses either
// side (see wiki/drivers.md, wiki/conflicts.md).

import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import { scanWatchedSources } from '../kernel/tx';
import { importMarkdownFile, reconcileMarkdownFile, sidecarPath } from './markdown';
import type { WorkspaceStore } from './store-fs';

const SKIP = new Set(['node_modules', '.git', '.substrate', 'dist', 'public']);

export interface SyncReport {
  imported: string[];
  fastForwarded: string[];
  proposals: string[]; // proposal ids raised by reconciliation
  sourceUpdates: string[]; // proposal ids raised by watched-source scanning
  unchanged: number;
}

function* mdFiles(root: string, dir: string): Generator<string> {
  for (const entry of readdirSync(dir, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
    if (SKIP.has(entry.name) || entry.name.startsWith('.')) continue;
    const full = join(dir, entry.name);
    if (entry.isDirectory()) yield* mdFiles(root, full);
    else if (/\.md$/i.test(entry.name)) yield relative(root, full).replaceAll('\\', '/');
  }
}

export async function syncWorkspace(
  ws: WorkspaceStore,
  opts: { contentDirs?: string[]; contentFiles?: string[] },
): Promise<SyncReport> {
  const report: SyncReport = { imported: [], fastForwarded: [], proposals: [], sourceUpdates: [], unchanged: 0 };
  const ctx = ws.ctxFor('driver:fs');
  const relPaths = new Set<string>();
  for (const dir of opts.contentDirs ?? []) {
    const abs = join(ws.root, dir);
    try {
      if (!statSync(abs).isDirectory()) continue;
    } catch {
      continue;
    }
    for (const rel of mdFiles(ws.root, abs)) relPaths.add(rel);
  }
  for (const f of opts.contentFiles ?? []) {
    try {
      if (statSync(join(ws.root, f)).isFile()) relPaths.add(f.replaceAll('\\', '/'));
    } catch {
      /* absent optional file */
    }
  }

  for (const relPath of relPaths) {
    const text = readFileSync(join(ws.root, relPath), 'utf8');
    let known = true;
    try {
      readFileSync(sidecarPath(ws.root, relPath), 'utf8');
    } catch {
      known = false;
    }
    if (!known) {
      await importMarkdownFile(ctx, { workspaceRoot: ws.root, relPath, text });
      report.imported.push(relPath);
      continue;
    }
    const res = await reconcileMarkdownFile(ctx, { workspaceRoot: ws.root, relPath, text });
    if (res.action === 'noop') report.unchanged++;
    else if (res.action === 'fast-forward') report.fastForwarded.push(relPath);
    // A fast-forward can still raise a sever proposal for vanished blocks.
    if (res.action !== 'noop' && res.proposalId) report.proposals.push(res.proposalId);
  }

  report.sourceUpdates = scanWatchedSources(ctx);
  ws.saveSnapshot();
  return report;
}
