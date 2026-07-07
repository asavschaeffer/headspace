// redact — the safety terrace. Content-based secret detection (not just filename).
// This is the guard that makes reading Downloads survivable, and the enforcement point
// for the API line: content passes through here before it is stored, and MUST pass before
// it is ever allowed to leave the machine.
const PATTERNS = [
  ['private-key',        /-----BEGIN (?:RSA |EC |OPENSSH |PGP |DSA )?PRIVATE KEY-----/g],
  ['aws-access-key',     /\bAKIA[0-9A-Z]{16}\b/g],
  ['openai-key',         /\bsk-(?:or-)?[A-Za-z0-9_-]{20,}\b/g],
  ['nvidia-key',         /\bnvapi-[A-Za-z0-9_-]{20,}\b/g],
  ['github-token',       /\bgh[pousr]_[A-Za-z0-9]{20,}\b/g],
  ['slack-token',        /\bxox[baprs]-[A-Za-z0-9-]{10,}\b/g],
  ['jwt',                /\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{6,}\b/g],
  ['bearer-token',       /\bBearer\s+[A-Za-z0-9._-]{20,}/g],
  ['secret-assignment',  /\b[A-Z0-9_]*(?:API_?KEY|SECRET|TOKEN|PASSWORD|PASSWD|PRIVATE_?KEY|CLIENT_?SECRET)[A-Z0-9_]*\s*[=:]\s*['"]?[^\s'"]{6,}/gi],
  ['credit-card',        /\b(?:\d[ -]?){13,16}\b/g],
];

// scan(text) -> { safe, hits:[{type,sample}], redacted }
export function scan(text) {
  const hits = []; let redacted = text;
  for (const [type, re] of PATTERNS) {
    redacted = redacted.replace(re, (m) => {
      hits.push({ type, sample: m.slice(0, 4) + '…' + m.slice(-2) }); // never keep the full secret
      return `‹REDACTED:${type}›`;
    });
  }
  return { safe: hits.length === 0, hits, redacted };
}

// the API-line gate. mode 'store' redacts and keeps; mode 'send' blocks entirely on any hit.
export function gate(content, mode = 'store') {
  const s = scan(content);
  if (s.safe) return { allowed: true, content, hits: [] };
  if (mode === 'send') return { allowed: false, content: null, hits: s.hits, reason: `${s.hits.length} secret(s) — refused to leave the machine` };
  return { allowed: true, content: s.redacted, hits: s.hits, reason: `${s.hits.length} secret(s) redacted before storing` };
}
