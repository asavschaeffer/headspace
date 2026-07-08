// Prove the guard on SYNTHETIC secrets (all fake/example values — no real key is used).
import { scan, gate } from './src/redact.mjs';

const samples = {
  'a normal note':          'Remember to finish the mog ability editor and buy milk.',
  'a downloaded invoice':   'Invoice #4471 — total due $1,240.00 to Acme Corp by Friday.',
  'an aws leak':            'export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\nregion=us-east-1',
  'an .env file':           'OPENROUTER_API_KEY=sk-or-v1-0123456789abcdefghijklmno\nDB_PASSWORD=hunter2trustno1sesame',
  'a nim key (like the real one lives in env)': 'NVIDIA_NIM_API_KEY=nvapi-FAKE0000abcdefghijklmnopqrstuvwx',
  'a private key file':     '-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAAB...',
};

console.log('CONTENT SECRET SCAN (synthetic inputs):\n');
for (const [label, text] of Object.entries(samples)) {
  const s = scan(text);
  const mark = s.safe ? '✅ safe ' : '🛑 SECRET';
  console.log(`${mark}  ${label}`);
  if (!s.safe) {
    console.log(`         found: ${s.hits.map(h => h.type + ' (' + h.sample + ')').join(', ')}`);
    console.log(`         stored as: ${JSON.stringify(s.redacted.replace(/\n/g, '⏎').slice(0, 72))}`);
  }
}

console.log('\nTHE API-LINE GATE (mode="send" — about to leave the machine):');
const leak = 'my key is sk-or-v1-0123456789abcdefghijklmno please summarize';
const g = gate(leak, 'send');
console.log(`  content with a secret → allowed to send? ${g.allowed}  (${g.reason || 'clean'})`);
const clean = gate('summarize my notes about the mog character editor', 'send');
console.log(`  clean content        → allowed to send? ${clean.allowed}`);
