import { describe, expect, it } from 'vitest';
import { HIDDEN_SIZE, NeuralCar, NUM_SENSORS } from '../js/nn.js';

describe('NeuralCar', () => {
  it('produces bounded outputs from think()', () => {
    const brain = new NeuralCar();
    const output = brain.think([0, 0.25, 0.5, 0.75, 1]);
    expect(output.steer).toBeGreaterThanOrEqual(-1);
    expect(output.steer).toBeLessThanOrEqual(1);
    expect(output.gas).toBeGreaterThanOrEqual(-1);
    expect(output.gas).toBeLessThanOrEqual(1);
  });

  it('returns deep-copied weights', () => {
    const brain = new NeuralCar();
    const weights = brain.getWeights();
    weights.base.w1[0][0] = 999;
    weights.base.b1[0] = 999;
    const fresh = brain.getWeights();
    expect(fresh.base.w1[0][0]).not.toBe(999);
    expect(fresh.base.b1[0]).not.toBe(999);
  });

  it('keeps weights unchanged when mutation rate is zero', () => {
    const brain = new NeuralCar({
      w1: Array.from({ length: NUM_SENSORS }, () => Array.from({ length: HIDDEN_SIZE }, () => 0.1)),
      b1: Array.from({ length: HIDDEN_SIZE }, () => 0.2),
      w2: Array.from({ length: HIDDEN_SIZE }, () => [0.3, 0.4]),
      b2: [0.5, 0.6],
    });

    const before = brain.getWeights();
    brain.mutate(0);
    expect(brain.getWeights().base).toEqual(before.base);
    expect(brain.getWeights().adapters).toEqual(before.adapters);
    expect(brain.getWeights().currentLevel).toEqual(before.currentLevel);
  });
});
