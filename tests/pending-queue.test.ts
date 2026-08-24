import assert from 'node:assert';
import {
  DispatchBarrier,
  preventPendingUnload,
  SingleFlightDrain,
  StateReplacementMutex,
  TruthQuarantine,
} from '../src/client/useSubstrate';

let prevented = false;
const event = {
  returnValue: undefined,
  preventDefault: () => { prevented = true; },
} as unknown as BeforeUnloadEvent;
preventPendingUnload(event);
assert.equal(prevented, true);
assert.equal(event.returnValue, '');

const barrier = new DispatchBarrier();
const releaseFirst = barrier.enter();
const releaseSecond = barrier.enter();
let replacementRan = false;
const replacement = barrier.wait().then(() => { replacementRan = true; });
await Promise.resolve();
assert.equal(replacementRan, false, 'state replacement waits while any dispatch owns the captured state');
releaseFirst();
await Promise.resolve();
assert.equal(replacementRan, false, 'all concurrent dispatches must finish');
releaseSecond();
await replacement;
assert.equal(replacementRan, true);
assert.equal(barrier.active, 0);
releaseSecond();
assert.equal(barrier.active, 0, 'a release handle is idempotent');

const commits = ['proposal'];
const drain = new SingleFlightDrain();
let acknowledge!: () => void;
const acknowledged = new Promise<void>((resolve) => { acknowledge = resolve; });
let posts = 0;
const pump = () => drain.run(async () => {
  posts++;
  await acknowledged;
  commits.shift();
});
void pump();
let replacementAfterAck = false;
const waitingReplacement = (async () => {
  await pump();
  assert.equal(commits.length, 0, 'a waiting reload observes the completed active drain');
  replacementAfterAck = true;
})();
await Promise.resolve();
assert.equal(replacementAfterAck, false);
assert.equal(posts, 1, 'concurrent pumps share the one active network drain');
acknowledge();
await waitingReplacement;
assert.equal(replacementAfterAck, true);
assert.equal(drain.busy, false);

// A 409 recovery must run after—not inside—the active commit drain. A model
// that finishes while recovery waits is dropped as old-state work, then the
// authoritative replacement completes without the drain awaiting itself.
const divergenceBarrier = new DispatchBarrier();
const finishProvider = divergenceBarrier.enter();
const divergenceDrain = new SingleFlightDrain();
const staleQueue = ['rejected-edit'];
let divergenceRecovery!: Promise<void>;
let authoritativeReloaded = false;
await divergenceDrain.run(async () => {
  staleQueue.splice(0);
  divergenceRecovery = (async () => {
    await divergenceBarrier.wait();
    staleQueue.splice(0);
    authoritativeReloaded = true;
  })();
});
staleQueue.push('proposal-produced-on-old-state');
finishProvider();
await divergenceRecovery;
assert.equal(divergenceDrain.busy, false);
assert.deepEqual(staleQueue, []);
assert.equal(authoritativeReloaded, true);

const replacementMutex = new StateReplacementMutex();
const order: string[] = [];
let finishFirstReplacement!: () => void;
const firstReplacement = new Promise<void>((resolve) => { finishFirstReplacement = resolve; });
const firstRun = replacementMutex.run(async () => {
  order.push('first-start');
  await firstReplacement;
  order.push('first-end');
});
const secondRun = replacementMutex.run(async () => { order.push('second'); });
await Promise.resolve();
assert.deepEqual(order, ['first-start']);
finishFirstReplacement();
await Promise.all([firstRun, secondRun]);
assert.deepEqual(order, ['first-start', 'first-end', 'second'], 'workspace replacements serialize');

const truth = new TruthQuarantine();
assert.equal(truth.unknown, false);
truth.quarantine();
assert.equal(truth.unknown, true, 'a 409 quarantines optimistic state synchronously');
// A failed authoritative GET changes nothing: only installing a successful
// response is allowed to restore mutation authority.
assert.equal(truth.unknown, true, 'failed recovery leaves truth quarantined');
truth.restore();
assert.equal(truth.unknown, false, 'a later successful authoritative reload restores writes');
assert.equal(truth.failedReplacement('GET'), false);
assert.equal(truth.unknown, false, 'a failed read does not make previously known truth ambiguous');
assert.equal(truth.failedReplacement('POST'), true);
assert.equal(truth.unknown, true, 'a failed mutating replacement quarantines its unknown server outcome');

console.log('client durability gates OK — reload preserves queued commits and in-flight proposal state');
