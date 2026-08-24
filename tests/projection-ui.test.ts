import assert from 'node:assert';
import { HostActionError, actionErrorMessage, projectSource } from '../src/Star';

const refusalMessage = 'refusing to project notes/demo.md: source changed since its last import or projection; sync first';
const observed: { request?: { input: string; init?: RequestInit } } = {};
const refusalFetch: typeof globalThis.fetch = async (input, init) => {
  observed.request = { input: String(input), init };
  return new Response(
    JSON.stringify({ code: 'projection-conflict', error: refusalMessage }),
    { status: 409, headers: { 'content-type': 'application/json' } },
  );
};

let refusal: unknown;
try {
  await projectSource('notes/demo.md', refusalFetch);
  assert.fail('projection refusal should reject');
} catch (error) {
  refusal = error;
}

assert.ok(refusal instanceof HostActionError);
assert.equal(refusal.code, 'projection-conflict');
assert.equal(refusal.hostMessage, refusalMessage);
assert.equal(refusal.status, 409);
assert.equal(
  actionErrorMessage('project', refusal),
  `project: projection-conflict — ${refusalMessage}`,
  'Star notice retains the host diagnostic code and actionable message',
);
assert.equal(observed.request?.input, '/api/project');
assert.equal(observed.request?.init?.method, 'POST');
assert.deepEqual(JSON.parse(String(observed.request?.init?.body)), { relPath: 'notes/demo.md' });

await assert.rejects(
  () => projectSource('notes/demo.md', async () => new Response('<html>gateway failure</html>', { status: 502 })),
  (error: unknown) =>
    error instanceof HostActionError &&
    error.code === null &&
    error.hostMessage === 'projection request failed: HTTP 502' &&
    error.status === 502,
);

await projectSource('notes/demo.md', async () => new Response(null, { status: 200 }));

console.log('projection UI diagnostics OK — structured host refusals remain actionable in Star');
