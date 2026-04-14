import { describe, expect, it } from 'vitest';
import { validateBrainWeights } from '../js/brain.js';
import { HIDDEN_SIZE, NUM_SENSORS } from '../js/nn.js';

function makeValidWeights() {
  return {
    w1: Array.from({ length: NUM_SENSORS }, () => Array.from({ length: HIDDEN_SIZE }, () => 0.1)),
    b1: Array.from({ length: HIDDEN_SIZE }, () => 0.2),
    w2: Array.from({ length: HIDDEN_SIZE }, () => [0.3, -0.3]),
    b2: [0.1, -0.1],
  };
}

describe('validateBrainWeights', () => {
  it('accepts a correctly shaped payload', () => {
    const result = validateBrainWeights(makeValidWeights());
    expect(result.ok).toBe(true);
    expect(result.errors).toEqual([]);
  });

  it('rejects malformed payloads', () => {
    const broken = makeValidWeights();
    broken.w2[0] = [1];
    const result = validateBrainWeights(broken);
    expect(result.ok).toBe(false);
    expect(result.errors[0]).toMatch(/w2/);
  });

  it('rejects non-finite values', () => {
    const broken = makeValidWeights();
    broken.b2[1] = Infinity;
    const result = validateBrainWeights(broken);
    expect(result.ok).toBe(false);
    expect(result.errors[0]).toMatch(/b2/);
  });
});
