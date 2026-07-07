#!/usr/bin/env node
// Hopper — v0 ingestion.
// Walk a folder, treat each project as the atomic unit, emit one chunk-row per unit.
// Deterministic only: no model, no embeddings. This is the tier-1 metadata pass.
//
//   node ingest.mjs [rootDir]   (defaults to the parent of hopper/)

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

const ROOT = path.resolve(process.argv[2] ?? path.join(process.cwd(), '..'));
const OUT  = path.join(process.cwd(), 'index.json');

// dirs we never descend into — the noise the naive walker drowns in
const SKIP = new Set(['node_modules','.git','dist','build','out','target','.next','.venv',
  'venv','__pycache__','.cache','.turbo','coverage','Pods','DerivedData','.gradle','bin','obj']);

// files whose presence marks a directory as a project root, → a kind guess
const ROOT_MARKERS = [
  ['Cargo.toml',      'Rust'],
  ['package.json',    'Node / TS'],
  ['pyproject.toml',  'Python'],
  ['requirements.txt','Python'],
  ['Package.swift',   'Swift / iOS'],
  ['go.mod',          'Go'],
  ['pubspec.yaml',    'Flutter'],
  ['index.html',      'Web'],
];

const EXT_KIND = {
  '.rs':'Rust', '.ts':'TypeScript', '.tsx':'TypeScript', '.js':'JavaScript', '.jsx':'JavaScript',
  '.py':'Python', '.ipynb':'Notebook', '.swift':'Swift', '.go':'Go', '.html':'Web', '.css':'Web',
  '.md':'Docs', '.txt':'Docs', '.json':'Data', '.png':'Image', '.jpg':'Image', '.jpeg':'Image',
  '.gif':'Image', '.svg':'Image', '.wav':'Audio', '.mp3':'Audio', '.mp4':'Video',
};

const human = b => b < 1024 ? b+' B'
  : b < 1048576 ? (b/1024).toFixed(1)+' KB'
  : b < 1073741824 ? (b/1048576).toFixed(1)+' MB'
  : (b/1073741824).toFixed(2)+' GB';
const days = ms => Math.round((Date.now()-ms)/86400000);
const hash = s => crypto.createHash('sha1').update(s).digest('hex').slice(0,12);

// crack open one project dir: sum size, histogram extensions, find newest mtime
function crawl(dir){
  let size=0, files=0, newest=0, oldest=Infinity;
  const exts = Object.create(null);
  const markers = new Set();
  (function rec(d, depth){
    if (depth > 8) return;
    let entries;
    try { entries = fs.readdirSync(d, { withFileTypes:true }); } catch { return; }
    for (const e of entries){
      if (e.isDirectory()){
        if (SKIP.has(e.name)) continue;
        rec(path.join(d, e.name), depth+1);
      } else if (e.isFile()){
        const full = path.join(d, e.name);
        if (depth===0) for (const [m] of ROOT_MARKERS) if (e.name===m) markers.add(m);
        if (depth===0 && e.name.endsWith('.xcodeproj')) markers.add('.xcodeproj');
        let st; try { st = fs.statSync(full); } catch { continue; }
        files++; size += st.size;
        newest = Math.max(newest, st.mtimeMs);
        oldest = Math.min(oldest, st.birthtimeMs || st.mtimeMs);
        const ext = path.extname(e.name).toLowerCase();
        if (ext) exts[ext] = (exts[ext]||0)+1;
      }
    }
  })(dir, 0);
  return { size, files, newest, oldest, exts, markers };
}

function kindOf(markers, exts){
  if (markers.has('.xcodeproj') || markers.has('Package.swift')) return 'Swift / iOS';
  for (const [m,k] of ROOT_MARKERS) if (markers.has(m) && m!=='index.html') return k;
  // no root marker: infer from the dominant source extension
  const rank = Object.entries(exts)
    .filter(([e]) => EXT_KIND[e] && !['.md','.txt','.json'].includes(e))
    .sort((a,b)=>b[1]-a[1]);
  if (rank.length) return EXT_KIND[rank[0][0]];
  if (markers.has('index.html')) return 'Web';
  return 'Unknown';
}

// v0 cluster key = the stack, which is all metadata can honestly tell us
const CLUSTER = {
  'Rust':'Rust', 'Go':'Systems', 'Swift / iOS':'Mobile',
  'TypeScript':'JS/TS', 'JavaScript':'JS/TS', 'Node / TS':'JS/TS', 'Web':'JS/TS',
  'Python':'Python', 'Notebook':'Python', 'Docs':'Docs', 'Unknown':'Other',
};

const rows = [];
let top;
try { top = fs.readdirSync(ROOT, { withFileTypes:true }); }
catch (e){ console.error('cannot read', ROOT, e.message); process.exit(1); }

for (const e of top){
  if (e.name.startsWith('.') || SKIP.has(e.name)) continue;
  const full = path.join(ROOT, e.name);
  if (e.isDirectory()){
    const c = crawl(full);
    if (!c.files){                          // empty or fully-skipped dir: fall back to the folder's own stat
      try { const s = fs.statSync(full); c.newest = s.mtimeMs; c.oldest = s.birthtimeMs || s.mtimeMs; }
      catch { c.newest = Date.now(); c.oldest = Date.now(); }
    }
    const kind = kindOf(c.markers, c.exts);
    const isProject = c.markers.size > 0;
    const langs = Object.fromEntries(Object.entries(c.exts).sort((a,b)=>b[1]-a[1]).slice(0,5));
    rows.push({
      id: hash(full+'|'+c.newest),      // identity: path + latest activity (v0 stand-in for content hash)
      name: e.name,
      type: isProject ? 'project' : 'folder',
      kind,
      cluster: CLUSTER[kind] || 'Other',
      langs,
      files: c.files,
      size: c.size, sizeHuman: human(c.size),
      created: new Date(c.oldest).toISOString().slice(0,10),
      modified: new Date(c.newest).toISOString().slice(0,10),
      ageDays: days(c.newest),
    });
  } else if (e.isFile()){
    let st; try { st = fs.statSync(full); } catch { continue; }
    const ext = path.extname(e.name).toLowerCase();
    const kind = EXT_KIND[ext] || 'Unknown';
    rows.push({
      id: hash(full+'|'+st.mtimeMs), name:e.name, type:'file', kind,
      cluster: CLUSTER[kind] || 'Other', langs:{[ext||'(none)']:1},
      files:1, size:st.size, sizeHuman:human(st.size),
      created:new Date(st.birthtimeMs||st.mtimeMs).toISOString().slice(0,10),
      modified:new Date(st.mtimeMs).toISOString().slice(0,10),
      ageDays:days(st.mtimeMs),
    });
  }
}

rows.sort((a,b)=>a.ageDays-b.ageDays);   // freshest first
fs.writeFileSync(OUT, JSON.stringify({ root:ROOT, ingestedAt:new Date().toISOString(), count:rows.length, rows }, null, 2));

// ---- human summary ----
const by = (arr,k)=>arr.reduce((m,r)=>((m[r[k]]=(m[r[k]]||0)+1),m),{});
const pad = (s,n)=>String(s).padEnd(n);
console.log(`\n  Hopper v0 ingest  ·  ${ROOT}`);
console.log(`  ${rows.length} units  ·  ${rows.filter(r=>r.type==='project').length} projects  ·  index.json written\n`);

console.log('  BY CLUSTER (stack — what metadata can see)');
for (const [k,n] of Object.entries(by(rows,'cluster')).sort((a,b)=>b[1]-a[1]))
  console.log(`    ${pad(k,10)} ${'█'.repeat(n)} ${n}`);

console.log('\n  BY KIND');
for (const [k,n] of Object.entries(by(rows,'kind')).sort((a,b)=>b[1]-a[1]))
  console.log(`    ${pad(k,14)} ${n}`);

console.log('\n  FRESHEST 8');
for (const r of rows.slice(0,8))
  console.log(`    ${pad(r.name,32)} ${pad(r.kind,13)} ${pad(r.ageDays+'d',6)} ${r.sizeHuman}`);

console.log('\n  BIGGEST 5');
for (const r of [...rows].sort((a,b)=>b.size-a.size).slice(0,5))
  console.log(`    ${pad(r.name,32)} ${pad(r.sizeHuman,10)} ${r.files} files`);
console.log();
