// CLI: reflect a folder's markdown into its Substrate workspace.
//   npm run ingest -- [folder] [contentDir ...]
// Defaults to the repo itself with wiki/ and headspace-brief.md as content.
import { resolve } from 'node:path';
import { openWorkspace } from '../src/host/store-fs';
import { syncWorkspace } from '../src/host/sync';

const args = process.argv.slice(2).filter((a) => a !== '--force');
const force = process.argv.includes('--force');
const root = resolve(args[0] ?? '.');
const contentDirs = args.slice(1);
const opts = contentDirs.length
  ? { contentDirs }
  : { contentDirs: ['wiki'], contentFiles: ['headspace-brief.md'] };

// A live lock (e.g. the dev server) is respected; stale locks from dead
// processes are taken over automatically. --force is a deliberate override.
const ws = await openWorkspace(root, { force });
const report = await syncWorkspace(ws, opts);
ws.close();
console.log(
  `Synced ${root}: ${report.imported.length} imported, ${report.fastForwarded.length} fast-forwarded, ` +
    `${report.proposals.length} reconciliation proposals, ${report.sourceUpdates.length} source updates, ` +
    `${report.unchanged} unchanged.`,
);
