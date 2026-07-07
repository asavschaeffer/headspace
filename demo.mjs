// Proves the substrate runs: ingest the triage message, then exercise every syscall
// against the hero scenario (search "iOS" -> two leaves + parent; rearrange; fork; edit).
import { Store, parse, reduce, select, edit, rearrange, fork, reingest, search }
  from './src/substrate.mjs';

const MSG = `Here's your workshop, clustered.

## Games
- mog template character & ability creator
- educational cooking survival game
- heard chef (iOS)
- healthkit survivor.io — steps-as-movement (iOS)
- doom.ing — shipped, needs domain

## LLM / AI infrastructure
- game grammar transformer
- angry llm research — experimental design audit
- headspace — chunked chats with provenance
- hopper — the folder reorganizer

## Rust ports
- box2d/3d physics to spacetime, rust + typescript
- gooner nexus rust port`;

const store = new Store();
const rootId = parse(MSG, store, { actor: { model: 'claude-opus-4-8' }, source_id: 'msg-triage' });

const line = (s) => console.log(s);
line('═══ INGEST ════════════════════════════════════════════════');
line(`chunks minted: ${store.all().length}   root merkle: ${store.get(rootId).content_hash}`);

line('\n═══ syscall: search("iOS") — leaves roll up to parent ═════');
const idx = reingest(store);
const res = search(idx, 'iOS');
res.leaves.forEach(l => line(`  ● leaf   ${l.text}   [${l.id}]`));
res.parents.forEach(p => line(`  ▸ parent "${p.chunk.text}" contains ${p.count} match(es)`));

line('\n═══ syscall: rearrange — VIEW order diverges from CAUSAL ══');
// Move "healthkit survivor" to sit right after "heard chef".
const heard = select(store, c => /heard chef/.test(c.text || ''))[0];
const hk = select(store, c => /healthkit/.test(c.text || ''))[0];
const sibs = store.children(heard.parent_id);
const afterHeard = sibs[sibs.indexOf(sibs.find(s => s.id === heard.id)) + 1];
const { keyBetween } = await import('./src/substrate.mjs');
rearrange(store, hk.id, heard.parent_id, keyBetween(heard.order_key, afterHeard.order_key));
const causal = store.children(heard.parent_id).slice().sort((a, b) => a.causal_seq.localeCompare(b.causal_seq));
line('  VIEW  order: ' + store.children(heard.parent_id).map(c => short(c.text)).join(' | '));
line('  CAUSAL order: ' + causal.map(c => short(c.text)).join(' | '));
line('  (identity stable; both orderings preserved)');

line('\n═══ syscall: edit — content changes, identity + provenance kept ══');
const before = hk.content_hash;
edit(store, hk.id, hk.text + '  ✎ (my note: this is the flagship)', 'human');
line(`  same id ${hk.id}`);
line(`  content_hash ${before} -> ${hk.content_hash}`);
line(`  provenance: born by ${JSON.stringify(hk.origin.actor)}, edited by "${hk.edits[0].actor}"`);

line('\n═══ syscall: fork — copy-on-write section into a new tree ═══');
const gamesSection = select(store, c => c.kind === 'heading_section' && /Games/.test(c.text))[0];
const forkedRoot = fork(store, gamesSection.id, { actor: 'human', source_id: 'fork-games' });
const f = store.get(forkedRoot);
line(`  forked "${f.text}"  new id ${f.id}  derived_from ${f.derived_from.id} (${f.derived_from.op})`);
line(`  origin untouched: original still id ${gamesSection.id}`);

line('\n═══ syscall: reduce — curated view -> linear context ══════');
const gameLeaves = store.descendants(gamesSection.id).map(c => c.id).concat(gamesSection.id);
line(reduce(store, gameLeaves).split('\n').map(s => '  ' + s).join('\n'));

function short(t) { return (t || '').split(' ').slice(0, 2).join(' '); }
