import { Car } from './car.js?v=29';
import { NeuralCar } from './nn.js?v=29';
import { validateBrainWeights } from './brain.js?v=29';
import {
  computeAdaptiveMutation,
  evaluatePlateauStatus,
  tournamentSelect,
  getTieredMutationRate,
  DIFFICULTY_LADDER,
  TRACK_DEFAULT_WIDTHS,
} from './evolution-core.js?v=29';
import { Track } from './track.js?v=29';
import { buildPersistPayload } from './persistence.js?v=29';

const STORAGE_KEY = 'f1-neuroevo-state';

function _persistState(state, options = {}) {
  try {
    const save = buildPersistPayload(state, options);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(save));
  } catch { /* ignore quota errors */ }
}

export function restoreState(state) {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return false;
    const save = JSON.parse(raw);
    if (save?.settings && typeof save.settings === 'object') {
      const s = save.settings;
      if (Number.isFinite(s.numCars)) state.settings.numCars = s.numCars;
      if (Number.isFinite(s.speedMult)) state.settings.speedMult = s.speedMult;
      if (Number.isFinite(s.mutationRate)) state.settings.mutationRate = s.mutationRate;
      if (typeof s.timeoutEnabled === 'boolean') state.settings.timeoutEnabled = s.timeoutEnabled;
      if (Number.isFinite(s.timeoutDuration)) state.settings.timeoutDuration = s.timeoutDuration;
      if (typeof s.trackType === 'string' && Object.hasOwn(TRACK_DEFAULT_WIDTHS, s.trackType)) {
        state.settings.trackType = s.trackType;
      }
    }

    let restoreTrack = save.trackType;
    if (!restoreTrack || !Object.hasOwn(TRACK_DEFAULT_WIDTHS, restoreTrack)) {
      restoreTrack = state.settings.trackType;
    }
    let widthOverride = null;
    if (save.bestWeights) {
      state._bestWeights = save.bestWeights;
      state._loadedWeights = save.bestWeights; // inject into gen 0
    }
    if (save.difficultyLevel > 0) {
      state._difficultyLevel = save.difficultyLevel;
      const step = DIFFICULTY_LADDER[save.difficultyLevel];
      if (step) {
        const defaultWidth = TRACK_DEFAULT_WIDTHS[step.track];
        widthOverride = step.widthDelta !== 0 ? defaultWidth + step.widthDelta : null;
        restoreTrack = step.track;
      }
    }
    if (restoreTrack) {
      state.track.dispose();
      state.scene.remove(state.track.mesh);
      state.settings.trackType = restoreTrack;
      state.track = new Track(restoreTrack, widthOverride);
      state.scene.add(state.track.mesh);
    }

    state.generation = save.generation || 1;
    state.frameCounter = save.frameCounter || 0;
    state.bestLapTime = save.bestLapTime || Infinity;
    state.allTimeBest = save.allTimeBest || 0;
    state.bestScore = save.bestScore || 0;
    state.genBestLap = save.genBestLap || Infinity;
    state._currentMutation = save.currentMutation || state.settings.mutationRate;
    if (Array.isArray(save.lapHistory)) state.lapHistory = save.lapHistory;

    if (Array.isArray(save.populationWeights) && save.populationWeights.length > 0) {
      const validPopulation = save.populationWeights
        .filter((weights) => validateBrainWeights(weights).ok);
      if (validPopulation.length > 0) {
        state._loadedPopulation = validPopulation;
      }
    }
    return true;
  } catch { return false; }
}

// Public: save state on demand (called from main.js beforeunload)
export function persistState(state) { _persistState(state, { full: true }); }

export function initialCars(state) {
  const cars = [];
  const level = state._difficultyLevel || 0;
  const restoredPopulation = Array.isArray(state._loadedPopulation) ? state._loadedPopulation : null;
  for (let i = 0; i < state.settings.numCars; i++) {
    let brain = null;
    if (restoredPopulation && i < restoredPopulation.length) {
      brain = new NeuralCar(restoredPopulation[i]);
    } else if (i < 2 && state._loadedWeights) {
      // If restored weights exist, inject as car #0 and #1 (elite pair)
      brain = new NeuralCar(state._loadedWeights);
    } else {
      brain = new NeuralCar();
    }
    // Make sure every brain knows which level it's training (controls
    // which adapter is mutable in nn.js).
    brain.setLevel(level);
    const car = new Car(state.track, brain, i, state.settings.speedMult);
    cars.push(car);
    state.scene.add(car.group);
  }
  // Clear loaded weights after injection so nextGeneration doesn't double-inject
  if (state._loadedWeights) state._loadedWeights = null;
  if (state._loadedPopulation) state._loadedPopulation = null;
  return cars;
}

export function nextGeneration(state) {
  const { cars, settings } = state;

  // Record generation stats
  const avgProgress = cars.reduce((s, c) => s + c.totalProgress, 0) / cars.length;
  const finishedRate = cars.filter((c) => c.finished).length / Math.max(1, cars.length);
  const genBestLap = cars
    .filter((c) => c.finished)
    .reduce((best, c) => Math.min(best, c.lapTime), Infinity);
  const finishers = cars.filter((c) => c.finished);

  state.lapHistory.push({
    gen: state.generation,
    bestLap: genBestLap < Infinity ? genBestLap : null,
    avgProgress,
  });
  if (state.lapHistory.length > 100) state.lapHistory.shift();
  state._plateauAvgHistory = state._plateauAvgHistory || [];
  state._plateauFinishedRateHistory = state._plateauFinishedRateHistory || [];
  state._plateauAvgHistory.push(avgProgress);
  state._plateauFinishedRateHistory.push(finishedRate);
  if (state._plateauAvgHistory.length > 160) state._plateauAvgHistory.shift();
  if (state._plateauFinishedRateHistory.length > 160) state._plateauFinishedRateHistory.shift();

  // Update all-time best lap + track best-lap stagnation + count IMPROVEMENTS
  if (genBestLap < state.bestLapTime) {
    state.bestLapTime = genBestLap;
    state._bestLapStagnantGens = 0;
    // Count lap improvements on current level (resets on escalation)
    state._lapImprovementsOnLevel = (state._lapImprovementsOnLevel || 0) + 1;
  } else if (state.bestLapTime < Infinity) {
    state._bestLapStagnantGens = (state._bestLapStagnantGens || 0) + 1;
  }

  // Sort by score for stats
  const sorted = [...cars].sort((a, b) => b.score - a.score);
  const bestScore = sorted[0]?.score || 0;
  const bestCar = sorted[0] || null;
  const plateauStatus = evaluatePlateauStatus({
    avgProgressHistory: state._plateauAvgHistory,
    finishedRateHistory: state._plateauFinishedRateHistory,
    consecutiveChecks: state._plateauConsecutiveChecks || 0,
  });
  state._plateauConsecutiveChecks = plateauStatus.consecutiveChecks || 0;
  state._plateauStatus = plateauStatus;

  // Adaptive mutation + curriculum escalation
  const mutationState = computeAdaptiveMutation({
    bestScore,
    allTimeBest: state.allTimeBest,
    stagnantGens: state._stagnantGens || 0,
    baseMutation: settings.mutationRate,
    currentMutation: state._currentMutation || settings.mutationRate,
    bestLapStagnantGens: state._bestLapStagnantGens || 0,
    currentDifficultyLevel: state._difficultyLevel || 0,
    escalationGens: state._escalationGens || 0,
    lapImprovementsOnLevel: state._lapImprovementsOnLevel || 0,
    hasCompletedLap: state.bestLapTime < Infinity,
    plateauStatus,
  });
  state._escalationGens = (state._escalationGens || 0) + 1;
  state.allTimeBest = mutationState.allTimeBest;
  state._stagnantGens = mutationState.stagnantGens;
  state._currentMutation = mutationState.mutationRate;
  state._escalationStatus = mutationState.escalationStatus;
  const baseMut = mutationState.mutationRate;
  if (!state._bestWeights || mutationState.improved) {
    state._bestWeights = sorted[0].brain.getWeights();
    // Auto-save to localStorage so training survives reloads
    _persistState(state);
  }

  // ─── Curriculum escalation: swap to harder track ───
  if (mutationState.escalate) {
    const nextLevel = (state._difficultyLevel || 0) + 1;
    const step = DIFFICULTY_LADDER[nextLevel];
    const defaultWidth = TRACK_DEFAULT_WIDTHS[step.track];
    const newWidth = step.widthDelta !== 0 ? defaultWidth + step.widthDelta : null;

    state.track.dispose();
    state.scene.remove(state.track.mesh);
    state.settings.trackType = step.track;
    state.track = new Track(step.track, newWidth);
    state.scene.add(state.track.mesh);

    state._difficultyLevel = nextLevel;
    state._stagnantGens = 0;
    state._escalationGens = 0;
    state._bestLapStagnantGens = 0;
    state._lapImprovementsOnLevel = 0;
    state._plateauConsecutiveChecks = 0;
    state._plateauAvgHistory = [];
    state._plateauFinishedRateHistory = [];
    state._plateauStatus = null;
    state._escalationStatus = null;
    state.bestLapTime = Infinity;
    state.genBestLap = Infinity;
    state.lapHistory = [];
    state.allTimeBest = 0;
    // _bestWeights intentionally preserved — carry over brain

    _persistState(state);
    if (state._onEscalation) state._onEscalation(step, nextLevel);
  }

  // Dispose old cars
  for (const c of cars) {
    c.disposeSensorLines(state.scene);
    state.scene.remove(c.group);
  }

  // Create next generation
  const newCars = [];
  const currentLevel = state._difficultyLevel || 0;

  // At level >= 1, ALL children share the champion's base (the locked,
  // frozen "level-0 universal feature extractor"). This collapses any base
  // diversity that existed pre-escalation and ensures the post-escalation
  // population can never drift away from the laps-Monaco brain. Children
  // still differ in their adapters, providing tournament diversity.
  const championBase = currentLevel >= 1 && state._bestWeights && state._bestWeights.base
    ? state._bestWeights.base
    : null;

  // Helper: clone a brain and align it with the current curriculum level
  // (so the right adapter is mutable). Used for every brain we instantiate
  // here — ensures LoRA's "freeze base, mutate current adapter" invariant.
  const makeBrain = (weights) => {
    const b = new NeuralCar(weights);
    if (championBase) b.setBase(championBase);
    b.setLevel(currentLevel);
    return b;
  };

  // Car #0: preserve all-time champion
  // Car #1: preserve best current-generation strategy
  const eliteBrain = makeBrain(state._bestWeights || sorted[0].brain.getWeights());
  const elite2Brain = makeBrain(sorted[0].brain.getWeights());

  // Handle loaded weights
  if (state._loadedWeights) {
    const validation = validateBrainWeights(state._loadedWeights);
    if (validation.ok) {
      const loaded = makeBrain(state._loadedWeights);
      const car = new Car(state.track, loaded, 0, settings.speedMult);
      newCars.push(car);
      state.scene.add(car.group);
    } else {
      const car = new Car(state.track, eliteBrain, 0, settings.speedMult);
      newCars.push(car);
      state.scene.add(car.group);
      console.warn(`Loaded brain rejected: ${validation.errors[0]}`);
    }
    state._loadedWeights = null;
  } else {
    const car = new Car(state.track, eliteBrain, 0, settings.speedMult);
    newCars.push(car);
    state.scene.add(car.group);
    // 2nd elite
    const car2 = new Car(state.track, elite2Brain, 1, settings.speedMult);
    newCars.push(car2);
    state.scene.add(car2.group);
  }

  // Partial restart: keep top 20% as micro-mutated elites,
  // fill remaining 80% with heavily mutated offspring (not fully random,
  // so they inherit some driving ability and recover faster)
  if (mutationState.restart) {
    const eliteCount = Math.max(3, Math.ceil(settings.numCars * 0.2));
    // Micro-mutated elites — use BASE mutation rate, not adaptive rate
    // (adaptive rate can be 0.4 during stagnation, too high for elites)
    const eliteMutRate = settings.mutationRate * 0.1;
    for (let i = 1; i < Math.min(eliteCount, settings.numCars); i++) {
      const elite = makeBrain(sorted[Math.min(i, sorted.length - 1)].brain.getWeights());
      elite.mutate(eliteMutRate);
      const car = new Car(state.track, elite, i, settings.speedMult);
      newCars.push(car);
      state.scene.add(car.group);
    }
    // Heavily mutated offspring (not fully random — inherit some structure)
    for (let i = newCars.length; i < settings.numCars; i++) {
      const parent = sorted[Math.floor(Math.random() * eliteCount)];
      const child = makeBrain(parent.brain.getWeights());
      child.mutate(baseMut * 3.0); // moderate heavy mutation, not fully random
      const car = new Car(state.track, child, i, settings.speedMult);
      newCars.push(car);
      state.scene.add(car.group);
    }
    state._stagnantGens = 0;
  } else {
    // Normal evolution: tournament selection + tiered mutation
    for (let i = newCars.length; i < settings.numCars; i++) {
      const parent = tournamentSelect(sorted, 3);
      const child = makeBrain(parent.brain.getWeights());
      const mutRate = getTieredMutationRate(i, settings.numCars, baseMut);
      child.mutate(mutRate);
      const car = new Car(state.track, child, i, settings.speedMult);
      newCars.push(car);
      state.scene.add(car.group);
    }
  }

  state.cars = newCars;
  state.generation++;
  state.frameCounter = 0;
  state.genBestLap = genBestLap;
  state.bestScore = bestScore;

  // Periodic save so reload doesn't lose more than ~20 gens of progress
  if (state.generation % 20 === 0) _persistState(state);
}
