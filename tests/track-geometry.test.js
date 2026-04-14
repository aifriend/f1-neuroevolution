import { describe, expect, it } from 'vitest';
import { computeInterpolationSteps } from '../js/track-geometry.js';

describe('computeInterpolationSteps', () => {
  it('increases interpolation for long control-point segments', () => {
    const silverstoneLike = [
      [0, 0],
      [120, 10],
      [180, 40],
      [200, 100],
      [185, 160],
    ];
    const monacoLike = [
      [0, 0],
      [40, 10],
      [70, 30],
      [90, 60],
      [80, 95],
    ];

    const silverstoneSteps = computeInterpolationSteps(silverstoneLike, 4);
    const monacoSteps = computeInterpolationSteps(monacoLike, 4);

    expect(silverstoneSteps).toBeGreaterThan(monacoSteps);
    expect(silverstoneSteps).toBeGreaterThanOrEqual(28);
    expect(monacoSteps).toBeGreaterThanOrEqual(20);
  });

  it('clamps interpolation density to configured bounds', () => {
    const tinyLoop = [
      [0, 0],
      [2, 0],
      [2, 2],
      [0, 2],
    ];
    const hugeLoop = [
      [0, 0],
      [1000, 0],
      [1000, 1000],
      [0, 1000],
    ];

    expect(computeInterpolationSteps(tinyLoop, 4, 20, 64)).toBe(20);
    expect(computeInterpolationSteps(hugeLoop, 4, 20, 64)).toBe(64);
  });
});
