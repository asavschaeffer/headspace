// CLI: reflect a folder's markdown into its Substrate workspace.
//   npm run ingest -- [folder] [contentDir ...]
// Defaults to the repo itself with wiki/ and headspace-brief.md as content.
import { resolve } from 'node:path';
import { openWorkspace } from '../src/host/store-fs';
import { syncWorkspace } from '../src/host/sync';

const root = resolve(process.argv[2] ?? '.');
const contentDirs = process.argv.slice(3);
const opts = contentDirs.length
  ? { contentDirs }
  : { contentDirs: ['wiki'], contentFiles: ['headspace-brief.md'] };

const ws = await openWorkspace(root, { force: true });
const report = await syncWorkspace(ws, opts);
ws.close();
console.log(
  `Synced ${root}: ${report.imported.length} imported, ${report.fastForwarded.length} fast-forwarded, ` +
    `${report.proposals.length} reconciliation proposals, ${report.sourceUpdates.length} source updates, ` +
    `${report.unchanged} unchanged.`,
);
