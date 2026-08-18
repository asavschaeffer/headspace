// Zero-dependency test harness: every tests/*.test.ts is an assert-script that
// throws on failure. Files run sequentially so persistence tests can own the
// filesystem without interleaving.
import { readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const dir = dirname(fileURLToPath(import.meta.url));
const files = readdirSync(dir)
  .filter((f) => f.endsWith('.test.ts'))
  .sort();

let failed = 0;
for (const f of files) {
  try {
    await import(pathToFileURL(join(dir, f)).href);
    console.log(`PASS ${f}`);
  } catch (e) {
    failed++;
    console.error(`FAIL ${f}`);
    console.error(e);
  }
}
if (failed > 0) {
  console.error(`${failed}/${files.length} test files failed`);
  process.exit(1);
}
console.log(`${files.length} test files passed`);
