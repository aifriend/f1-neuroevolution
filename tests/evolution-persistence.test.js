import { describe, expect, it } from 'vitest';
import { buildPersistPayload } from '../js/persistence.js';
import { NeuralCar } from '../js/nn.js';

function makeState({ cars = [] } = {}) {
  return {
    cars,
    generation: 42,
    frameCounter: 123,
    bestLapTime: 88.4,
    allTimeBest: 901,
    bestScore: 777,
    genBestLap: 90.2,
    lapHistory: [{ gen: 41, bestLap: 90.5, avgProgress: 0.7 }],
    _bestWeights: new NeuralCar().getWeights(),
    _difficultyLevel: 2,
    _currentMutation: 0.11,
    settings: {
      trackType: 'suzuka',
      numCars: 30,
      speedMult: 1.2,
      mutationRate: 0.08,
      timeoutEnabled: true,
      timeoutDuration: 3500,
    },
  };
}

describe('buildPersistPayload', () => {
  it('stores lightweight progress fields by default', () => {
    const state = makeState();
    const payload = buildPersistPayload(state);
    expect(payload.generation).toBe(42);
    expect(payload.settings.trackType).toBe('suzuka');
    expect(payload.populationWeights).toBeUndefined();
    expect(payload.lapHistory).toBeUndefined();
  });

  it('stores full population snapshot when full mode is enabled', () => {
    const carA = { brain: new NeuralCar() };
    const carB = { brain: new NeuralCar() };
    const state = makeState({ cars: [carA, carB] });
    const payload = buildPersistPayload(state, { full: true });
    expect(payload.populationWeights).toHaveLength(2);
    expect(payload.lapHistory).toEqual(state.lapHistory);
  });
});
