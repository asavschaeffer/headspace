// terraces — the free, deterministic extraction levels. Each is an independent brick you
// climb; none call an API, none leave the machine. Compose them over the fs scan.
// Everything here is BELOW the API line: private, offline, instant.
import fs from 'node:fs';
import path from 'node:path';
import { fnv1a } from './substrate.mjs';
import { scan } from './redact.mjs';

// ── T4 · role from name convention (a filename carries huge signal for free) ──
export function t_role(name) {
  const n = name.toLowerCase();
  if (/^readme/.test(n)) return 'readme';
  if (/^licen[sc]e/.test(n)) return 'license';
  if (/(^|[.\-])(test|spec)[.\-]|^test_/.test(n)) return 'test';
  if (/(package-lock|yarn\.lock|cargo\.lock|\.lock$|poetry\.lock)/.test(n)) return 'lockfile';
  if (/^(index|main|app|__init__|mod|lib)\.[a-z]+$/.test(n)) return 'entrypoint';
  if (/\.(json|ya?ml|toml|ini|cfg|conf|config|xml)$/.test(n)) return 'config';
  if (n.startsWith('.env')) return 'secret';
  if (n.startsWith('.')) return 'dotfile';
  return 'file';
}

// ── T5 · is it really text? (extensions lie; a null byte means binary) ─────────
export function t_isText(fp) {
  try {
    const fd = fs.openSync(fp, 'r'); const buf = Buffer.alloc(512);
    const n = fs.readSync(fd, buf, 0, 512, 0); fs.closeSync(fd);
    for (let i = 0; i < n; i++) if (buf[i] === 0) return false;
    return true;
  } catch { return null; }
}

// ── T6 · content hash → exact-duplicate detection across the whole machine ─────
export function t_contentHash(fp, size, cap = 5e6) {
  if (size > cap) return null;                 // bounded: don't slurp huge media to hash
  try { return fnv1a(fs.readFileSync(fp, 'latin1')); } catch { return null; }
}

// ── T7 · bounded peek — the cheap identity, guarded (never reads a secret) ─────
// Every candidate line passes through the redactor BEFORE truncation, so a stored
// peek can never carry a raw secret (name-based flags catch files; this catches content).
export function t_peek(fp, meta) {
  const { kind, size, secret } = meta;
  if (secret || size > 65536 || !(kind === 'text' || kind === 'code')) return null;
  let s; try { s = fs.readFileSync(fp, 'utf8'); } catch { return null; }
  const clean = (t) => scan(t).redacted;      // redact first, slice after — no half-cut secrets
  const lines = s.split('\n');
  const md = lines.find(l => /^#{1,3}\s+\S/.test(l));
  if (md) return clean(md.replace(/^#+\s+/, '').trim()).slice(0, 80);
  const html = (/<title>([^<]+)<\/title>/i.exec(s) || [])[1];
  if (html) return clean(html.trim()).slice(0, 80);
  const first = (lines.find(l => l.trim()) || '').trim();
  return clean(first).slice(0, 80) || null;
}

// ── T8 · a project's identity, from its manifest (one small file, no API) ──────
export function t_manifest(dir) {
  const read = (f) => { try { return fs.readFileSync(path.join(dir, f), 'utf8'); } catch { return null; } };
  const pkg = read('package.json');
  if (pkg) { try { const j = JSON.parse(pkg);
    return { lang: 'node', name: j.name || path.basename(dir), description: j.description || '',
      deps: Object.keys({ ...j.dependencies, ...j.devDependencies }).length }; } catch {} }
  const cargo = read('Cargo.toml');
  if (cargo) return { lang: 'rust', name: (/name\s*=\s*"([^"]+)"/.exec(cargo) || [])[1] || path.basename(dir),
    description: (/description\s*=\s*"([^"]+)"/.exec(cargo) || [])[1] || '',
    deps: (cargo.split(/\[dependencies\]/)[1]?.match(/^\s*[\w-]+\s*=/gm) || []).length };
  const py = read('pyproject.toml');
  if (py) return { lang: 'python', name: (/name\s*=\s*"([^"]+)"/.exec(py) || [])[1] || path.basename(dir),
    description: (/description\s*=\s*"([^"]+)"/.exec(py) || [])[1] || '',
    deps: (py.match(/^\s*"[\w.\-]+[><=~]/gm) || []).length };
  const req = read('requirements.txt');
  if (req) return { lang: 'python', name: path.basename(dir), description: '',
    deps: req.split('\n').filter(l => l.trim() && !l.startsWith('#')).length };
  return null;
}
