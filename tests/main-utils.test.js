import { describe, expect, it } from 'vitest';
import { parseBrainJson, isDebugMode } from '../js/main-utils.js';
import { HIDDEN_SIZE, NUM_SENSORS } from '../js/nn.js';

function makeBrainJson() {
  const payload = {
    w1: Array.from({ length: NUM_SENSORS }, () => Array.from({ length: HIDDEN_SIZE }, () => 0)),
    b1: Array.from({ length: HIDDEN_SIZE }, () => 0),
    w2: Array.from({ length: HIDDEN_SIZE }, () => [0, 0]),
    b2: [0, 0],
  };
  return JSON.stringify(payload);
}

describe('parseBrainJson', () => {
  it('returns weights for valid JSON payloads', () => {
    const parsed = parseBrainJson(makeBrainJson());
    expect(parsed.ok).toBe(true);
    expect(parsed.weights).toBeTruthy();
  });

  it('returns structured error for invalid JSON', () => {
    const parsed = parseBrainJson('not-json');
    expect(parsed.ok).toBe(false);
    expect(parsed.error).toMatch(/JSON/);
  });

  it('returns structured error for invalid brain shapes', () => {
    const parsed = parseBrainJson(JSON.stringify({ hello: 'world' }));
    expect(parsed.ok).toBe(false);
    expect(parsed.error).toMatch(/w1/);
  });
});

describe('isDebugMode', () => {
  it('only enables debug mode when debug=1', () => {
    expect(isDebugMode('?debug=1')).toBe(true);
    expect(isDebugMode('?debug=0')).toBe(false);
    expect(isDebugMode('?other=1')).toBe(false);
  });
});
