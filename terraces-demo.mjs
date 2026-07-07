// Climb the free terraces over the real Projects scan — no API, nothing leaves the machine.
// T8 gives each black-boxed project a real identity; T4/T7 name and peek loose files;
// T6 finds exact duplicates.
import { ingestDir } from './src/fs-driver.mjs';
import { t_manifest, t_role, t_peek, t_contentHash, t_isText } from './src/terraces.mjs';

const root = process.argv[2] || 'C:\\Users\\asas\\Projects';
const { store } = ingestDir(root, { peek: false });
const pad = (s, n) => String(s).slice(0, n).padEnd(n);

// ── T8 · project identity from manifests ──────────────────────────────────────
const projects = store.all().filter(c => c.kind === 'project').sort((a, b) => b.meta.files - a.meta.files);
console.log('T8 · project identity, read from manifests (no API, nothing left the machine):');
let named = 0;
for (const p of projects.slice(0, 16)) {
  const m = t_manifest(p.binding);
  if (m) { named++; const d = m.description ? ' — ' + m.description.slice(0, 40) : '';
    console.log(`  ${pad(p.text, 22)} ${pad(m.lang, 6)} ${pad(m.name, 22)} ${String(m.deps).padStart(3)} deps${d}`); }
  else console.log(`  ${pad(p.text, 22)} (no manifest — identity from structure only)`);
}
console.log(`  → ${named}/${projects.length} projects gained a real identity, for free.\n`);

// ── T4 + T7 · role + peek on loose files ──────────────────────────────────────
const files = store.all().filter(c => c.kind === 'file');
console.log('T4 + T7 · role + bounded peek on loose files (deterministic, local):');
for (const f of files.slice(0, 9)) {
  const peek = t_peek(f.binding, f.meta);
  console.log(`  ${pad(f.text, 20)} ${pad(t_role(f.text), 10)} ${peek ? '“' + peek.slice(0, 46) + '”' : ''}`);
}

// ── T6 · exact-duplicate detection ────────────────────────────────────────────
const seen = new Map(); let dups = 0;
for (const f of files) { const h = t_contentHash(f.binding, f.meta.size); if (!h) continue;
  if (seen.has(h)) dups++; else seen.set(h, f.text); }
console.log(`\nT6 · exact-dup scan of ${files.length} loose files → ${dups} exact duplicate(s) found by content hash.`);
console.log('  (this is the weapon for Downloads — where the same installer lives five times.)');
