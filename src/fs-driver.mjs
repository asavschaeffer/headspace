// fs-driver — the filesystem driver. World (a folder tree) -> chunks.
// Read-only. It never opens a file body unless asked (peek:true) and never touches
// anything flagged secret. This is hopper's "atomic unit is the project" discipline,
// plus the tiered-reading budget from the design notes, made real.
import fs from 'node:fs';
import path from 'node:path';
import { Store, makeChunk, fnv1a } from './substrate.mjs';
import { gate } from './redact.mjs';

// dirs we black-box entirely — never recurse, never chunk
const SKIP = new Set(['node_modules','.git','.svn','.hg','target','dist','build','out','.next','.nuxt',
  '.venv','venv','env','.cache','vendor','.idea','.vscode','coverage','.pytest_cache','__pycache__','bin','obj','.gradle']);
// the presence of any of these makes a folder a *project* — one unit, not a pile of files
const MARKERS = new Set(['.git','package.json','Cargo.toml','pyproject.toml','go.mod','pom.xml',
  'build.gradle','Gemfile','requirements.txt','pubspec.yaml','composer.json','deno.json']);
const CODE = new Set(['.js','.mjs','.cjs','.ts','.tsx','.jsx','.py','.rs','.go','.java','.rb','.c','.cc','.cpp','.h','.hpp','.cs','.php','.swift','.kt','.scala','.sh','.lua','.dart','.vue','.svelte']);
const TEXT = new Set(['.md','.txt','.rst','.json','.yaml','.yml','.toml','.ini','.cfg','.csv','.html','.htm','.css','.scss','.xml','.sql']);
const IMAGE = new Set(['.png','.jpg','.jpeg','.gif','.webp','.svg','.ico','.bmp','.tiff','.psd']);
const DOCS  = new Set(['.pdf','.doc','.docx','.ppt','.pptx','.xls','.xlsx','.odt']);
const MEDIA = new Set(['.mp4','.mov','.avi','.mkv','.webm','.mp3','.wav','.flac','.zip','.tar','.gz','.7z','.rar','.exe','.dll','.iso','.dmg']);
// never peek these, never even name them in output — the secret-safety line
const SECRET = /(^\.env)|secret|credential|password|\.pem$|\.key$|\.pfx$|\.p12$|id_rsa|token|\.crt$/i;

const kindOf = (ext) => CODE.has(ext) ? 'code' : TEXT.has(ext) ? 'text' : IMAGE.has(ext) ? 'image'
  : DOCS.has(ext) ? 'doc' : MEDIA.has(ext) ? 'media' : 'other';

// default-deny content policy: a file BODY is never opened unless a policy explicitly
// allows it. { readPeek, hashDupes, allowExt:Set('.md',…), allowUnder:[path substrings] }.
// No policy (the default) = metadata only. Legacy peek:true maps to { readPeek:true }.
const contentAllowed = (pol, ext, fp) => !!pol
  && (!pol.allowExt || pol.allowExt.has(ext))
  && (!pol.allowUnder || pol.allowUnder.some(p => fp.includes(p)));

export function ingestDir(root, { maxDepth = 6, maxFiles = 25000, peek = false, content = null } = {}) {
  const policy = content || (peek ? { readPeek: true } : null);   // default posture: deny
  const store = new Store();
  const summary = { root, dirs: 0, files: 0, bytes: 0, projects: [], kinds: {}, secrets: 0, skipped: 0, bodiesRead: 0 };
  const rootChunk = store.put(makeChunk({ kind: 'root', text: path.basename(root), actor: { driver: 'fs' }, source_id: root }));
  let counter = 0; const key = () => String(++counter / 100000);

  // shallow scan of a project's interior — counts, no per-file chunking (the black box)
  function summarize(dir) {
    let files = 0, bytes = 0; const kinds = {}; const stack = [dir]; let scanned = 0;
    while (stack.length && scanned < 6000) {
      const d = stack.pop(); let ents;
      try { ents = fs.readdirSync(d, { withFileTypes: true }); } catch { continue; }
      for (const e of ents) {
        if (e.isSymbolicLink()) continue;
        const fp = path.join(d, e.name);
        if (e.isDirectory()) { if (!SKIP.has(e.name)) stack.push(fp); }
        else { scanned++; files++; try { bytes += fs.statSync(fp).size; } catch {}
          kinds[kindOf(path.extname(e.name).toLowerCase())] = (kinds[kindOf(path.extname(e.name).toLowerCase())] || 0) + 1; }
      }
    }
    return { files, bytes, kinds };
  }

  function walk(dir, parentId, depth) {
    if (depth > maxDepth || summary.files > maxFiles) return;
    let entries; try { entries = fs.readdirSync(dir, { withFileTypes: true }); }
    catch { summary.skipped++; return; }
    const names = entries.map(e => e.name);

    // a project? emit ONE chunk and stop — don't shred a coherent unit into dust
    if (depth > 0 && names.some(n => MARKERS.has(n))) {
      const s = summarize(dir);
      const type = names.includes('Cargo.toml') ? 'rust'
        : names.includes('package.json') ? 'node'
        : (names.includes('pyproject.toml') || names.includes('requirements.txt')) ? 'python'
        : names.includes('go.mod') ? 'go' : 'repo';
      const c = makeChunk({ kind: 'project', text: path.basename(dir), parent_id: parentId, order_key: key(), actor: { driver: 'fs' }, source_id: dir });
      c.binding = dir; c.meta = { type, git: names.includes('.git'), ...s };
      store.put(c);
      summary.projects.push({ name: path.basename(dir), type, files: s.files, bytes: s.bytes });
      summary.files += s.files; summary.bytes += s.bytes;
      return;
    }

    summary.dirs++;
    const dc = makeChunk({ kind: depth === 0 ? 'root' : 'directory', text: path.basename(dir) || dir, parent_id: parentId, order_key: key(), actor: { driver: 'fs' }, source_id: dir });
    dc.binding = dir; if (depth > 0) store.put(dc);
    const pid = depth === 0 ? parentId : dc.id;

    for (const e of entries) {
      if (e.isSymbolicLink()) continue;
      const fp = path.join(dir, e.name);
      if (e.isDirectory()) { if (SKIP.has(e.name)) { summary.skipped++; continue; } walk(fp, pid, depth + 1); }
      else {
        let st; try { st = fs.statSync(fp); } catch { continue; }
        const ext = path.extname(e.name).toLowerCase(); const kind = kindOf(ext); const secret = SECRET.test(e.name);
        summary.files++; summary.bytes += st.size; summary.kinds[kind] = (summary.kinds[kind] || 0) + 1; if (secret) summary.secrets++;
        let peekText = null, redactions = 0, chash = null;
        if (!secret && contentAllowed(policy, ext, fp)) {
          if (policy.readPeek && kind === 'text' && st.size < 65536) {
            try {
              summary.bodiesRead++;
              const raw = fs.readFileSync(fp, 'utf8').split('\n').slice(0, 3).join(' ');
              const g = gate(raw, 'store');            // redact BEFORE truncating or storing
              peekText = g.content.slice(0, 140); redactions = g.hits.length;
            } catch {}
          }
          if (policy.hashDupes && st.size < 5e6) {     // opt-in dup detection, bounded
            try { summary.bodiesRead++; chash = fnv1a(fs.readFileSync(fp, 'latin1')); } catch {}
          }
        }
        const fc = makeChunk({ kind: 'file', text: e.name, parent_id: pid, order_key: key(), actor: { driver: 'fs' }, source_id: fp });
        fc.binding = fp; fc.meta = { ext, kind, size: st.size, mtime: st.mtimeMs, secret, peek: peekText };
        if (redactions > 0) fc.meta.redactions = redactions;
        if (chash != null) fc.meta.chash = chash;
        store.put(fc);
      }
    }
  }

  walk(root, rootChunk.id, 0);
  store.rehashAll();
  return { store, rootId: rootChunk.id, summary };
}
