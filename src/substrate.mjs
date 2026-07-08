// substrate — the chunk kernel. syscall(0) made runnable.
// No external deps: this same source is inlined into the artifact UI.
// Four zones per chunk: IDENTITY (immutable) · CONTENT (versioned/hashed) ·
// VIEW (mutable parent+order_key) · PROVENANCE (append-only).

// ── hashing: FNV-1a 32-bit hex. Content-addressing + Merkle roll-up. ──────────
export function fnv1a(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = (h + ((h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24))) >>> 0;
  }
  return ('0000000' + h.toString(16)).slice(-8);
}

// ── identity + logical clock ─────────────────────────────────────────────────
let _seq = 0;
const nextSeq = () => String(++_seq).padStart(6, '0'); // causal birth order (immutable)
const newId = (seq) => `c_${seq}_${Math.random().toString(36).slice(2, 7)}`;
// after loading a persisted store, advance the clock so new chunks never collide with old
export function syncSeq(n) { if (n > _seq) _seq = n; }

// ── fractional order keys: insert between siblings without renumbering ────────
// View order. Decimal fractions in (0,1); lexicographic == numeric for our range.
export function keyBetween(a, b) {
  const lo = a == null ? 0 : parseFloat(a);
  const hi = b == null ? 1 : parseFloat(b);
  return String((lo + hi) / 2);
}

// ── the Chunk factory ────────────────────────────────────────────────────────
export function makeChunk({ kind, text = null, parent_id = null, order_key = '0.5', actor, source_id }) {
  const causal_seq = nextSeq();
  return {
    // IDENTITY (immutable for life)
    id: newId(causal_seq),
    causal_seq,
    // CONTENT (versioned; hash filled in by rehash)
    kind,
    text,
    content_hash: null,
    // VIEW (mutable)
    parent_id,
    order_key,
    // PROVENANCE (append-only)
    origin: { actor, at: now(), source_id },
    derived_from: undefined,
    edits: [],
  };
}
const now = () => new Date().toISOString();

// ── the Store: id -> chunk, plus tree helpers ────────────────────────────────
export class Store {
  constructor() { this.map = new Map(); }
  put(c) { this.map.set(c.id, c); return c; }
  get(id) { return this.map.get(id); }
  all() { return [...this.map.values()]; }
  roots() { return this.all().filter(c => c.parent_id == null); }
  children(id) {
    return this.all()
      .filter(c => c.parent_id === id)
      .sort((x, y) => x.order_key.localeCompare(y.order_key, undefined, { numeric: true }));
  }
  ancestors(id) {
    const out = []; let c = this.get(id);
    while (c && c.parent_id != null) { c = this.get(c.parent_id); if (c) out.push(c); }
    return out;
  }
  descendants(id) {
    const out = []; const walk = (pid) => this.children(pid).forEach(k => { out.push(k); walk(k.id); });
    walk(id); return out;
  }
  // Merkle content hash: leaves hash text, containers hash ordered child hashes.
  hashOf(id) {
    const c = this.get(id);
    const kids = this.children(id);
    c.content_hash = kids.length
      ? fnv1a(c.kind + '|' + kids.map(k => this.hashOf(k.id)).join(','))
      : fnv1a(c.kind + '|' + (c.text ?? ''));
    return c.content_hash;
  }
  rehashFrom(id) { // bubble a change up its ancestor spine only
    let c = this.get(id);
    while (c) { this.hashOf(c.id); c = c.parent_id == null ? null : this.get(c.parent_id); }
  }
  rehashAll() { this.roots().forEach(r => this.hashOf(r.id)); }
}

// ── structure-aware parser: markdown -> chunk tree (the AST *is* the tree) ────
// Handles: ## headings (nested by level), - list items, paragraphs, ``` code.
export function parse(md, store, { actor, source_id }) {
  const root = store.put(makeChunk({ kind: 'message', actor, source_id }));
  const lines = md.replace(/\r/g, '').split('\n');
  // stack of section containers by heading depth; index 0 = message root
  const sections = [{ depth: 0, id: root.id }];
  let curList = null;                 // active list container id
  const orderCounters = new Map();    // parentId -> running fraction for initial layout
  const nextKey = (pid) => {
    const n = (orderCounters.get(pid) ?? 0) + 1; orderCounters.set(pid, n);
    return String(n / 100); // spaced apart; plenty of room to insert between
  };
  const parentSection = () => sections[sections.length - 1].id;
  const add = (kind, text, parent) =>
    store.put(makeChunk({ kind, text, parent_id: parent, order_key: nextKey(parent), actor, source_id }));

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const h = /^(#{1,6})\s+(.*)$/.exec(line);
    if (h) {
      const depth = h[1].length;
      while (sections.length > 1 && sections[sections.length - 1].depth >= depth) sections.pop();
      const sec = add('heading_section', h[2].trim(), parentSection());
      sections.push({ depth, id: sec.id });
      curList = null; continue;
    }
    const li = /^\s*[-*]\s+(.*)$/.exec(line);
    if (li) {
      if (!curList) curList = add('list', null, parentSection()).id;
      add('list_item', li[1].trim(), curList); continue;
    }
    if (/^```/.test(line)) {
      const buf = []; i++;
      while (i < lines.length && !/^```/.test(lines[i])) buf.push(lines[i++]);
      add('code_block', buf.join('\n'), parentSection()); curList = null; continue;
    }
    if (line.trim() === '') { curList = null; continue; }
    add('paragraph', line.trim(), parentSection()); curList = null;
  }
  store.rehashAll();
  return root.id;
}

// ── SYSCALLS ─────────────────────────────────────────────────────────────────

// select(predicate) -> Chunk[]
export const select = (store, pred) => store.all().filter(pred);

// reduce(Chunk[], budget) -> Context : linearize by view order, respect nesting
export function reduce(store, ids, budget = Infinity) {
  const want = new Set(ids);
  const lines = [];
  const walk = (id, depth) => {
    const c = store.get(id);
    if (want.has(id) && c.text != null) lines.push('  '.repeat(depth) + c.text);
    const nd = c.kind === 'message' ? depth : depth + (c.text != null ? 1 : 0);
    store.children(id).forEach(k => walk(k.id, want.has(id) ? nd : depth));
  };
  store.roots().forEach(r => walk(r.id, 0));
  let ctx = lines.join('\n');
  if (ctx.length > budget) ctx = ctx.slice(0, budget) + '\n…[truncated to budget]';
  return ctx;
}

// edit(id, text) -> Chunk : new content, same identity, provenance appended
export function edit(store, id, text, actor) {
  const c = store.get(id);
  const from = c.content_hash;
  c.text = text;
  store.rehashFrom(id);
  c.edits.push({ at: now(), actor, from, to: c.content_hash });
  return c;
}

// rearrange(id, newParent, newOrderKey) -> Chunk : VIEW only; causal_seq untouched
export function rearrange(store, id, newParent, newOrderKey) {
  const c = store.get(id);
  const oldParent = c.parent_id;
  c.parent_id = newParent;
  c.order_key = newOrderKey;
  store.rehashFrom(oldParent ?? id);
  store.rehashFrom(newParent ?? id);
  return c;
}

// fork(Chunk[]) -> new root id : copy-on-write, new identities + lineage edge
export function fork(store, rootId, { actor, source_id }) {
  const copyMap = new Map();
  const copy = (srcId, newParent) => {
    const src = store.get(srcId);
    const seq = nextSeq();
    const c = {
      ...structuredCloneSafe(src),
      id: newId(seq), causal_seq: seq,
      parent_id: newParent, order_key: src.order_key,
      origin: { actor, at: now(), source_id },
      derived_from: { id: src.id, op: 'fork' },
      edits: [],
    };
    store.put(c); copyMap.set(srcId, c.id);
    store.children(srcId).forEach(k => copy(k.id, c.id));
    return c.id;
  };
  const newRoot = copy(rootId, null);
  store.rehashFrom(newRoot);
  return newRoot;
}
function structuredCloneSafe(o) { return JSON.parse(JSON.stringify(o)); }

// reingest(store) -> Index : one entry per chunk; container match = contains child match
export function reingest(store) {
  const index = { store, tokens: new Map() }; // token -> Set(leafId)
  for (const c of store.all()) {
    if (c.text == null) continue;
    for (const t of tokenize(c.text)) {
      if (!index.tokens.has(t)) index.tokens.set(t, new Set());
      index.tokens.get(t).add(c.id);
    }
  }
  return index;
}
const tokenize = (s) => (s.toLowerCase().match(/[a-z0-9.]+/g) || []);

// search(index, query) -> { leaves, parents } : leaf hits rolled up to parents
export function search(index, query) {
  const store = index.store;
  const q = query.toLowerCase().trim();
  const leaves = store.all().filter(c => c.text != null && c.text.toLowerCase().includes(q));
  const parentCounts = new Map(); // ancestorId -> count of matching leaves under it
  for (const leaf of leaves) {
    for (const anc of store.ancestors(leaf.id)) {
      parentCounts.set(anc.id, (parentCounts.get(anc.id) || 0) + 1);
    }
  }
  const parents = [...parentCounts.entries()]
    .map(([id, count]) => ({ chunk: store.get(id), count }))
    .filter(p => p.chunk.kind !== 'message')          // roll up to real containers
    .sort((a, b) => b.count - a.count);
  return { leaves, parents };
}
