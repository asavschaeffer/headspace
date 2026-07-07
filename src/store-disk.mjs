// store-disk — syscall(2). The store made durable. Memory that forgets isn't memory.
// Content-addressed: a chunk's text lives in a blob keyed by its hash, so identical
// content is stored exactly once (real dedup — the mog-army pays rent only once).
// The floor the drivers pour into. Written in TS-flavoured JS; the seam to a Rust/SQLite
// backend is a straight swap of saveStore/loadStore — the kernel never learns which.
import fs from 'node:fs';
import crypto from 'node:crypto';
import { Store, fnv1a, syncSeq } from './substrate.mjs';

// ── encryption at rest (opt-in) — AES-256-GCM, key derived by scrypt ──────────
// The envelope carries everything needed to decrypt EXCEPT the passphrase; GCM's
// auth tag means tampering or a wrong passphrase fails loudly, never silently.
function seal(json, passphrase) {
  const salt = crypto.randomBytes(16), iv = crypto.randomBytes(12);
  const key = crypto.scryptSync(passphrase, salt, 32);
  const c = crypto.createCipheriv('aes-256-gcm', key, iv);
  const ct = Buffer.concat([c.update(json, 'utf8'), c.final()]);
  const b64 = (b) => b.toString('base64');
  return JSON.stringify({ v: 1, enc: 'aes-256-gcm', salt: b64(salt), iv: b64(iv), tag: b64(c.getAuthTag()), ct: b64(ct) });
}

function unseal(env, passphrase) {
  if (!passphrase) throw new Error('store is encrypted — pass { passphrase } to loadStore');
  const b64 = (s) => Buffer.from(s, 'base64');
  const key = crypto.scryptSync(passphrase, b64(env.salt), 32);
  const d = crypto.createDecipheriv('aes-256-gcm', key, b64(env.iv));
  d.setAuthTag(b64(env.tag));
  try { return Buffer.concat([d.update(b64(env.ct)), d.final()]).toString('utf8'); }
  catch { throw new Error('could not decrypt store — wrong passphrase or corrupted file'); }
}

export function saveStore(store, file, { passphrase } = {}) {
  const blobs = {};            // hash -> text  (content-addressed, deduped)
  const chunks = [];
  for (const c of store.all()) {
    let textRef = null;
    if (c.text != null) { textRef = fnv1a(c.text); blobs[textRef] = c.text; } // same text => same blob
    chunks.push({ ...c, text: undefined, textRef });
  }
  const db = { version: 1, savedAt: Date.now(), chunks, blobs };
  const json = JSON.stringify(db);
  fs.writeFileSync(file, passphrase ? seal(json, passphrase) : json);   // no passphrase = plaintext (legacy)
  return { file, chunks: chunks.length, blobs: Object.keys(blobs).length, bytes: fs.statSync(file).size };
}

export function loadStore(file, { passphrase } = {}) {
  let db = JSON.parse(fs.readFileSync(file, 'utf8'));
  if (db.enc) db = JSON.parse(unseal(db, passphrase));   // encrypted envelope; plaintext loads as before
  const store = new Store();
  let maxSeq = 0;
  for (const rec of db.chunks) {
    const c = { ...rec, text: rec.textRef != null ? db.blobs[rec.textRef] : null };
    delete c.textRef;
    store.map.set(c.id, c);
    const n = parseInt(c.causal_seq, 10); if (n > maxSeq) maxSeq = n;
  }
  syncSeq(maxSeq);             // resume the clock where the saved world left off
  store.rehashAll();           // recompute Merkle from scratch — integrity check, not trust
  return store;
}
