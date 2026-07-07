// The first heartbeat, for real. Ingest -> select a chunk -> generate against a live model.
// Captures real replies into generated-seeds.json so the sandboxed "home" can replay them.
import fs from 'node:fs';
import { Store, parse } from './src/substrate.mjs';
import { drivers, gather, assemble, generate } from './src/model-driver.mjs';

const aiMsg =
`Here's your workshop, clustered — projects that share a spine sitting together.

## 🎮 Games
- mog — template character & ability creator
- educational cooking survival game
- heard chef — recipe companion (iOS)
- healthkit survivor.io — your real steps are your only moves (iOS)
- doom.ing — shipped, just needs a domain

## 🧠 LLM / AI infrastructure
- game grammar transformer
- angry llm — experimental-design audit
- headspace — every chat chunked, with provenance
- hopper — the folder reorganizer

## 🦀 Rust ports
- box2d/3d physics → spacetime, in rust + typescript
- gooner nexus rust port`;

const store = new Store();
const root = parse(aiMsg, store, { actor: { model: 'claude-opus-4-8' }, source_id: 'msg-2' });

const driver = drivers.nim('meta/llama-3.1-70b-instruct');
const targets = ['mog —', 'healthkit survivor', 'hopper —'];
const seeds = {};

for (const needle of targets) {
  const t = store.all().find(c => (c.text || '').includes(needle));
  const g = gather(store, t.id);
  const { manifest, tokens } = assemble(store, g);
  console.log(`\n═══ generate ⟶ "${t.text}"`);
  console.log(`   select: ${g.length} chunks · reduce: ~${tokens} tokens`);
  const t0 = Date.now();
  try {
    const res = await generate(store, { targetId: t.id, driver });
    console.log(`   pulse: ${driver.name}/${driver.model} · ${Date.now() - t0}ms`);
    console.log(res.reply.split('\n').map(s => '   │ ' + s).join('\n'));
    // record the freshly-minted model chunks (the reply, as a tree)
    const kids = store.descendants(res.rootId).map(c => ({ kind: c.kind, text: c.text }));
    seeds[t.text] = { manifest: manifest.length, tokens, model: driver.model, chunks: kids };
  } catch (e) {
    console.log('   ✗', e.message);
  }
}

fs.writeFileSync('generated-seeds.json', JSON.stringify(seeds, null, 2));
console.log(`\n✓ wrote generated-seeds.json (${Object.keys(seeds).length} real replies)`);
