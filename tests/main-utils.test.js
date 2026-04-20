import { describe, expect, it } from 'vitest';
import { createQuitController, parseBrainJson, isDebugMode, parseVisualRuntimeParams } from '../js/main-utils.js';
import { HIDDEN_SIZE, NUM_INPUTS } from '../js/nn.js';

function makeBrainJson() {
  const payload = {
    w1: Array.from({ length: NUM_INPUTS }, () => Array.from({ length: HIDDEN_SIZE }, () => 0)),
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

describe('createQuitController', () => {
  it('starts not quit, then requests quit once', () => {
    let calls = 0;
    const qc = createQuitController(() => { calls++; });
    expect(qc.isQuitRequested()).toBe(false);

    expect(qc.requestQuit()).toBe(true);
    expect(qc.isQuitRequested()).toBe(true);
    expect(calls).toBe(1);

    expect(qc.requestQuit()).toBe(false);
    expect(calls).toBe(1);
  });
});

describe('parseVisualRuntimeParams', () => {
  it('parses and clamps supported params', () => {
    const parsed = parseVisualRuntimeParams(
      '?fresh=1&track=monaco&cars=999&speed=99&mutation=-1&timeout=0&timeoutFrames=999999',
      { allowedTracks: ['monaco', 'suzuka'] }
    );
    expect(parsed.fresh).toBe(true);
    expect(parsed.trackType).toBe('monaco');
    expect(parsed.numCars).toBe(100);
    expect(parsed.speedMult).toBe(3);
    expect(parsed.mutationRate).toBe(0.02);
    expect(parsed.timeoutEnabled).toBe(false);
    expect(parsed.timeoutDuration).toBe(5000);
  });

  it('ignores unknown/invalid values', () => {
    const parsed = parseVisualRuntimeParams('?track=not_a_track&cars=nope&timeout=maybe', {
      allowedTracks: ['monaco'],
    });
    expect(parsed.trackType).toBe(null);
    expect(parsed.numCars).toBe(null);
    expect(parsed.timeoutEnabled).toBe(null);
  });
});
