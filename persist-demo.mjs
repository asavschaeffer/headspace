// Prove the floor holds: durability, integrity, content-addressed dedup — then put the
// REAL Projects workshop on disk and bring it back with its Merkle root intact.
import fs from 'node:fs';
import { Store, parse } from './src/substrate.mjs';
import { saveStore, loadStore } from './src/store-disk.mjs';
import { ingestDir } from './src/fs-driver.mjs';

const tmp = ['.a.db.json', '.b.db.json', '.projects.db.json'];

// 1 — durability + integrity: save a tree, reload it in a clean Store, compare Merkle roots
const s1 = new Store();
const r1 = parse('## Games\n- mog\n- heard chef\n\n## Rust ports\n- spacetime physics port', s1, { actor: { model: 'x' }, source_id: 'm' });
const before = s1.get(r1).content_hash;
const info = saveStore(s1, tmp[0]);
const s2 = loadStore(tmp[0]);
const after = s2.get(r1).content_hash;
console.log('① durability   ', `${info.chunks} chunks → ${info.bytes} bytes on disk → reloaded ${s2.all().length}`);
console.log('  integrity    ', `root merkle ${before} ${after === before ? '== survived the round trip ✓' : '≠ MISMATCH ✗'}`);

// 2 — content-addressed dedup: identical text is stored once
const s3 = new Store();
parse('- duplicate line\n- duplicate line\n- duplicate line\n- unique line', s3, { actor: { model: 'x' }, source_id: 'm' });
const d = saveStore(s3, tmp[1]);
const textChunks = s3.all().filter(c => c.text != null).length;
console.log('② dedup        ', `${textChunks} text chunks → ${d.blobs} unique blobs (identical content stored once)`);

// 3 — the real thing: your whole workshop map, onto the hard drive and back
console.log('\n③ putting your real workshop on disk…');
const t0 = Date.now();
const { store } = ingestDir('C:\\Users\\asas\\Projects', { peek: false });
const rootHash = store.roots()[0].content_hash;
const big = saveStore(store, tmp[2]);
const reload = loadStore(tmp[2]);
console.log('  persisted    ', `${big.chunks} chunks → ${(big.bytes / 1e6).toFixed(2)}MB on disk (${Date.now() - t0}ms)`);
console.log('  reloaded     ', `${reload.all().length} chunks, root merkle ${reload.roots()[0].content_hash === rootHash ? 'matches — your map survived ✓' : 'MISMATCH ✗'}`);
console.log('  meaning      ', 'close the tab, reopen tomorrow — the workshop is still there.');

tmp.forEach(f => { try { fs.unlinkSync(f); } catch {} });
