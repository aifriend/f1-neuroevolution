import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { NeuralCar } from '../js/nn.js';

// evolution.js imports its Three.js-dependent siblings with a `?v=29`
// cache-busting query, so the mock paths must include the query to intercept.
vi.mock('../js/car.js?v=29', () => ({
  Car: class MockCar {
    constructor(track, brain, teamIdx, speedMult) {
      this.track = track;
      this.brain = brain;
      this.teamIdx = teamIdx;
      this.speedMult = speedMult;
      this.group = { isMockGroup: true };
      this.totalProgress = 0;
      this.finished = false;
      this.lapTime = Infinity;
      this.score = 0;
    }

    disposeSensorLines() {}
  },
}));

vi.mock('../js/track.js?v=29', () => ({
  Track: class MockTrack {
    constructor(type, widthOverride = null) {
      this.type = type;
      this.widthOverride = widthOverride;
      this.mesh = { isMockMesh: true };
    }

    dispose() {}
  },
}));

// Wrap the shared evolution core with a spy so the test can assert what the
// browser call site actually forwards, while keeping the real decision logic.
vi.mock('../js/evolution-core.js?v=29', async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    computeAdaptiveMutation: vi.fn(actual.computeAdaptiveMutation),
  };
});

function makeCar({ finished = false } = {}) {
  return {
    totalProgress: 0.5,
    finished,
    lapTime: finished ? 90 : Infinity,
    score: finished ? 5000 : 100,
    brain: new NeuralCar(),
    group: { isMockGroup: true },
    disposeSensorLines: () => {},
  };
}

// Builds a state that is one nextGeneration() call away from a confirmed
// plateau: flat avg-progress/finisher-rate histories well past the 25-gen
// minimum, and a valid lap from earlier generations (bestLapTime < Infinity).
function makeState({ finishedCount = 0, numCars = 10 } = {}) {
  const cars = [];
  for (let i = 0; i < numCars; i++) cars.push(makeCar({ finished: i < finishedCount }));
  const finishedRate = finishedCount / numCars;
  return {
    cars,
    generation: 40,
    frameCounter: 0,
    bestLapTime: 90,
    allTimeBest: 5000,
    bestScore: 5000,
    genBestLap: Infinity,
    lapHistory: [],
    settings: {
      trackType: 'monaco',
      numCars,
      speedMult: 1,
      mutationRate: 0.1,
      timeoutEnabled: true,
      timeoutDuration: 3500,
    },
    track: { type: 'monaco', mesh: { isMockMesh: true }, dispose() {} },
    scene: { add: () => {}, remove: () => {} },
    _difficultyLevel: 0,
    _stagnantGens: 5,
    _currentMutation: 0.1,
    _bestWeights: new NeuralCar().getWeights(),
    _plateauAvgHistory: Array(40).fill(0.5),
    _plateauFinishedRateHistory: Array(40).fill(finishedRate),
    _onEscalation: vi.fn(),
  };
}

describe('nextGeneration curriculum escalation (browser call site)', () => {
  let nextGeneration;
  let mutationSpy;
  let DIFFICULTY_LADDER;

  beforeAll(async () => {
    ({ nextGeneration } = await import('../js/evolution.js'));
    const core = await import('../js/evolution-core.js?v=29');
    mutationSpy = core.computeAdaptiveMutation;
    DIFFICULTY_LADDER = core.DIFFICULTY_LADDER;
  });

  beforeEach(() => {
    mutationSpy.mockClear();
  });

  it('forwards the generation finisher rate and escalates on a confirmed plateau', () => {
    // 5/10 finishers clears the 10% robustness gate. Before the fix, the
    // browser call site never forwarded currentFinisherRate, so the gate
    // always failed and the visual app could never leave level 0.
    const state = makeState({ finishedCount: 5, numCars: 10 });
    nextGeneration(state);

    expect(mutationSpy).toHaveBeenCalledTimes(1);
    expect(mutationSpy.mock.calls[0][0].currentFinisherRate).toBeCloseTo(0.5);

    expect(state._onEscalation).toHaveBeenCalledTimes(1);
    expect(state._onEscalation).toHaveBeenCalledWith(DIFFICULTY_LADDER[1], 1);
    expect(state._difficultyLevel).toBe(1);
    expect(state.settings.trackType).toBe('suzuka');
    expect(state.track.type).toBe('suzuka');
  });

  it('keeps the robustness gate closed when the latest generation had no finishers', () => {
    const state = makeState({ finishedCount: 0, numCars: 10 });
    nextGeneration(state);

    expect(mutationSpy).toHaveBeenCalledTimes(1);
    expect(mutationSpy.mock.calls[0][0].currentFinisherRate).toBe(0);
    expect(state._escalationStatus.reason).toBe('fragile_adapter');
    expect(state._onEscalation).not.toHaveBeenCalled();
    expect(state._difficultyLevel).toBe(0);
  });
});
