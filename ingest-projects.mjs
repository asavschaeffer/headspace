// Point the fs-driver at the real Projects folder — READ-ONLY, metadata only (peek:false,
// so no file body is ever opened), secrets flagged and never touched. Output stays local.
import { ingestDir } from './src/fs-driver.mjs';

const root = process.argv[2] || 'C:\\Users\\asas\\Projects';
const t0 = Date.now();
const { store, summary } = ingestDir(root, { peek: false });
const ms = Date.now() - t0;
const MB = (b) => (b / 1e6).toFixed(1) + 'MB';

console.log(`\n═══ fs-driver ⟶ ${root}`);
console.log(`   ${summary.dirs} dirs · ${summary.files} files · ${MB(summary.bytes)} · ${ms}ms`);
console.log(`   chunks minted: ${store.all().length}   ·   dirs skipped (node_modules/.git/…): ${summary.skipped}`);

console.log(`\n   projects detected (each = ONE chunk, interior black-boxed):`);
summary.projects.sort((a, b) => b.files - a.files).forEach(p =>
  console.log(`     ${String(p.type).padEnd(7)} ${p.name.padEnd(28)} ${String(p.files).padStart(5)} files  ${MB(p.bytes)}`));
if (!summary.projects.length) console.log('     (none — no project markers found at this level)');

console.log(`\n   loose files by kind: ${JSON.stringify(summary.kinds)}`);
console.log(`   secrets flagged & left untouched: ${summary.secrets}`);
console.log(`\n   (peek:false — not a single file body was opened. structure + metadata only.)`);
