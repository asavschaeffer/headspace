import assert from 'node:assert';
import { assessSimilarity, similarity } from '../src/host/similarity';

const exact = assessSimilarity('a short block', 'a short block edited');
assert.equal(exact.method, 'exact-levenshtein');
assert.equal(exact.approximate, false);
assert.equal(exact.score, similarity('a short block', 'a short block edited'));

const lengthBound = assessSimilarity('short', 'x'.repeat(100));
assert.equal(lengthBound.method, 'length-bound');
assert.equal(lengthBound.approximate, false);
assert.equal(lengthBound.score, 0);

const long = 'recognizable head '.repeat(300) + 'middle' + ' recognizable tail'.repeat(300);
const sampled = assessSimilarity(long, long.replace('middle', 'entirely different middle'));
assert.equal(sampled.method, 'sampled-levenshtein');
assert.equal(sampled.approximate, true);
assert.ok(sampled.score >= 0.5);

console.log('similarity assessment OK — exact and sampled evidence remain distinguishable');
