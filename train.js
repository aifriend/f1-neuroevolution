#!/usr/bin/env node
// Headless F1 neuroevolution trainer — no rendering, pure math
// Uses the SAME evolution logic as the visual frontend (evolution-core.js)
// Runs 1000x faster than visual mode. Outputs best brain as JSON.
//
// Usage:
//   node train.js                          # defaults: monaco, 30 cars, 500 gens
//   node train.js --track suzuka --cars 80 --gens 1000
//   node train.js --output my-brain.json

// ─── Argument parsing ────────────────────────────────
const args = process.argv.slice(2);
function flag(name, fallback) {
  const i = args.indexOf('--' + name);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : fallback;
}

const TRACK_TYPE = flag('track', 'monaco');
const NUM_CARS = parseInt(flag('cars', '30'));
const NUM_GENS = parseInt(flag('gens', '500'));
const MUTATION_RATE = parseFloat(flag('mutation', '0.05'));
const TIMEOUT = parseInt(flag('timeout', '3500'));
const SPEED_MULT = parseFloat(flag('speed', '1'));
const OUTPUT_FILE = flag('output', `models/best-brain-${TRACK_TYPE}.json`);
const LOAD_FILE = flag('load', null);
// Two-phase training: each curriculum level trains at SLOW_SPEED first
// (gentler physics → faster to learn track shape), then refines at SPEED_MULT
// (full race speed). Escalation to the NEXT level only fires after both
// phases on the current level plateau.
//   Enable with: --twoPhase
//   Slow speed:  --slow 0.5  (default)
const SLOW_SPEED = parseFloat(flag('slow', '0.5'));
const TWO_PHASE = args.includes('--twoPhase');
// Disable LoRA freezing — clean A/B baseline against the LoRA-protected
// runs. With --noLora, brains stay at level 0 internally (their full base
// always mutates) regardless of which curriculum track they're on. This
// is the catastrophic-forgetting baseline.
const NO_LORA = args.includes('--noLora');
// LoRA rank: how much capacity each per-level adapter has.
//   rank 2 = 88 params/level (tight, max protection, low adaptation)
//   rank 4 = 176 params/level (balanced)
//   rank 8 = 352 params/level (high adaptation, still less than full base)
const LORA_RANK_OVERRIDE = parseInt(flag('rank', '2'), 10);
// Soft freeze: at level >= 1, mutate the base at this fraction of the
// adapter's mutation rate. 0 = strict freeze (no forgetting); >0 = base
// drifts slowly so the universal feature extractor can still improve as
// new tracks are encountered.
const SOFT_FREEZE = parseFloat(flag('softFreeze', '0'));
// Optional explicit track-width override. Useful for targeting a specific
// curriculum level's geometry without going through escalation
// (e.g. --track ironcliff --width 16 to dwell on level 9).
const WIDTH_OVERRIDE = args.includes('--width') ? parseInt(flag('width'), 10) : null;
// Override the starting curriculum level. By default we honor the loaded
// brain's currentLevel; with --startLevel N we force it. Used for targeted
// dwell training: load an L{K} checkpoint and dwell on level K+1 (the next
// adapter to refine) without going through escalation first.
const START_LEVEL_OVERRIDE = args.includes('--startLevel')
  ? parseInt(flag('startLevel'), 10)
  : null;
// Escalation robustness gate: don't promote to the next track until this
// fraction of cars in the latest generation is finishing laps. Prevents
// "lucky elite" plateaus from advancing fragile adapters. 0 disables.
const MIN_FINISHER_RATE_FOR_ESCALATION = parseFloat(flag('minFinisherRate', '0.10'));
// Rolling window for robust-elite tracking: bestWeights becomes the brain
// with the highest (finisherRate, bestScore) within the last N generations.
const ROBUST_ELITE_WINDOW_SIZE = parseInt(flag('robustWindow', '20'), 10);
// Clone-test elite gating. Every CLONE_TEST_EVERY gens we take the top-K
// brains from the robust window, re-spawn each as N fresh cars on the
// current track, and pick the one with the most finishers as the new
// bestWeights. Closes the "single fastest brain has fragile spawn-position
// generalization" gap (13/80 from a 71/80-population elite).
//   --cloneTestEvery 0 disables.
const CLONE_TEST_EVERY = parseInt(flag('cloneTestEvery', '10'), 10);
const CLONE_TEST_CARS = parseInt(flag('cloneTestCars', '16'), 10);
const CLONE_TEST_K = parseInt(flag('cloneTestK', '3'), 10);

// ─── Import shared evolution + neural-network logic ──
// Headless and visual trainers share the EXACT same NeuralCar (with LoRA
// continual learning), so brain JSON saved by either side loads in the other.
const {
  computeAdaptiveMutation,
  evaluatePlateauStatus,
  tournamentSelect,
  getTieredMutationRate,
  computeScore,
  DIFFICULTY_LADDER,
  TRACK_DEFAULT_WIDTHS,
} = await import('./js/evolution-core.js');
const { NeuralCar, NUM_INPUTS, HIDDEN_SIZE } = await import('./js/nn.js');
const { TRACK_IDS } = await import('./js/track-data.js');
const { HeadlessTrack, HEADLESS_SENSOR_LENGTH, HEADLESS_LAP_COMPLETION_PROGRESS } = await import('./js/headless-track.js');
// HeadlessCar + runGeneration come from a SHARED module (js/headless-car.js)
// so cross-track-eval.js sees byte-identical physics. Eliminates the
// "trains lap, eval doesn't" gap that plagued earlier dwell runs.
const { HeadlessCar, runGeneration, SENSOR_ANGLES, NUM_SENSORS, STUCK_LIMIT } = await import('./js/headless-car.js');

// adapter" inside NeuralCar.mutate().
function evolve(sorted, numCars, baseMut, track, speedMult, doRestart, hallOfFameWeights, level) {
  const newCars = [];

  // LoRA mode: freeze base + ALL prior-level adapters by collapsing every
  // child to the champion's versions. Without this, tournament selection
  // (parents come from the previous gen's mixed elite/random population)
  // scatters base + adapter diversity, and a lucky child whose ancestry
  // skipped some prior level can win bestWeights — silently dropping
  // mastery on every level its lineage didn't carry.
  // --noLora mode: skip everything — base stays mutable (forgetting baseline).
  const championBase = !NO_LORA && level >= 1 && hallOfFameWeights && hallOfFameWeights.base
    ? hallOfFameWeights.base
    : null;
  const championAdapters = !NO_LORA && level >= 1 && hallOfFameWeights && hallOfFameWeights.adapters
    ? hallOfFameWeights.adapters
    : null;

  const makeBrain = (weights) => {
    const b = new NeuralCar(weights, { rank: LORA_RANK_OVERRIDE, softFreezeFactor: SOFT_FREEZE });
    if (championBase) b.setBase(championBase);
    if (!NO_LORA) b.setLevel(level);
    // setPriorAdapters AFTER setLevel so currentLevel is set; setPriorAdapters
    // skips currentLevel's adapter (preserves evolutionary search there).
    if (championAdapters) b.setPriorAdapters(championAdapters);
    return b;
  };

  // Cars #0-1: preserve all-time champion + best current generation.
  const championWeights = hallOfFameWeights || sorted[0].brain.getWeights();
  const currentBestWeights = sorted[0].brain.getWeights();
  newCars.push(new HeadlessCar(track, makeBrain(championWeights), 0, speedMult));
  newCars.push(new HeadlessCar(track, makeBrain(currentBestWeights), 1, speedMult));

  if (doRestart) {
    // Partial restart: keep top 20% micro-mutated, rest heavily mutated
    const eliteCount = Math.max(3, Math.ceil(numCars * 0.2));
    // Use configured base mutation for parity with visual runtime.
    const eliteMutRate = MUTATION_RATE * 0.1;
    for (let i = 1; i < Math.min(eliteCount, numCars); i++) {
      const elite = makeBrain(sorted[Math.min(i, sorted.length - 1)].brain.getWeights());
      elite.mutate(eliteMutRate);
      newCars.push(new HeadlessCar(track, elite, i, speedMult));
    }
    for (let i = newCars.length; i < numCars; i++) {
      const parent = sorted[Math.floor(Math.random() * eliteCount)];
      const child = makeBrain(parent.brain.getWeights());
      child.mutate(baseMut * 3.0);
      newCars.push(new HeadlessCar(track, child, i, speedMult));
    }
  } else {
    // Tournament selection + tiered mutation
    for (let i = newCars.length; i < numCars; i++) {
      const parent = tournamentSelect(sorted, 3);
      const child = makeBrain(parent.brain.getWeights());
      const mutRate = getTieredMutationRate(i, numCars, baseMut);
      child.mutate(mutRate);
      newCars.push(new HeadlessCar(track, child, i, speedMult));
    }
  }
  if (newCars.length !== numCars) {
    throw new Error(`Evolution size mismatch: expected ${numCars}, got ${newCars.length}`);
  }
  return newCars;
}

// ─── Main Training Loop ─────────────────────────────
const { writeFileSync, mkdirSync } = await import('fs');
const { dirname } = await import('path');

// Per-level checkpoints: at every escalation we snapshot the just-mastered
// brain to OUTPUT_FILE with a "-Lv{N}" suffix. Lets you replay any earlier
// curriculum state if a later level destabilizes the run.
function checkpointPathForLevel(outputPath, level) {
  if (outputPath.endsWith('.json')) {
    return outputPath.replace(/\.json$/, `-L${level}.json`);
  }
  return `${outputPath}-L${level}.json`;
}
function saveLevelCheckpoint(level, weights) {
  if (!weights) return;
  const path = checkpointPathForLevel(OUTPUT_FILE, level);
  try {
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, JSON.stringify(weights, null, 2));
    console.log(`  >>> CHECKPOINT saved: L${level} → ${path}`);
  } catch (e) {
    console.warn(`  >>> CHECKPOINT failed for L${level}: ${e.message}`);
  }
}

console.log(`\n  F1 Neuroevolution — Headless Trainer v2`);
console.log(`  ─────────────────────────────────────────`);
console.log(`  Track:      ${TRACK_TYPE}${WIDTH_OVERRIDE != null ? ` (width override: ${WIDTH_OVERRIDE})` : ''}`);
console.log(`  Cars:       ${NUM_CARS}`);
console.log(`  Generations:${NUM_GENS}`);
console.log(`  Mutation:   ${MUTATION_RATE}`);
console.log(`  Timeout:    ${TIMEOUT} frames`);
console.log(`  Speed mult: ${SPEED_MULT}x`);
console.log(`  Output:     ${OUTPUT_FILE}`);
console.log(`  Evolution:  tournament(k=3) + tiered mutation + curriculum learning`);
console.log(`  LoRA:       ${NO_LORA ? 'OFF (catastrophic-forgetting baseline)' : `ON rank=${LORA_RANK_OVERRIDE}` + (SOFT_FREEZE > 0 ? ` softFreeze=${SOFT_FREEZE}` : ' (strict freeze)')}`);
console.log(`  Two-phase:  ${TWO_PHASE ? `ON (slow ${SLOW_SPEED}x → fast ${SPEED_MULT}x per level)` : 'OFF'}\n`);

let track = new HeadlessTrack(TRACK_TYPE, WIDTH_OVERRIDE);

// Optional warm-start from a saved brain. The first 2 cars are cloned from
// the loaded weights (elite pair), the rest are random.
let warmBrain = null;
if (LOAD_FILE) {
  const { readFileSync } = await import('fs');
  warmBrain = JSON.parse(readFileSync(LOAD_FILE, 'utf-8'));
  console.log(`  Warm-starting from: ${LOAD_FILE}\n`);
}

// Warm-start brain might already encode a higher level (saved from a
// previous run that escalated). Honor its currentLevel so adapter mutability
// matches the loaded state — unless --startLevel was passed explicitly.
let difficultyLevel;
if (Number.isFinite(START_LEVEL_OVERRIDE)) {
  difficultyLevel = START_LEVEL_OVERRIDE;
} else if (warmBrain && Number.isFinite(warmBrain.currentLevel)) {
  difficultyLevel = warmBrain.currentLevel;
} else {
  difficultyLevel = 0;
}
if (Number.isFinite(START_LEVEL_OVERRIDE)) {
  console.log(`  --startLevel override: training adapter[${START_LEVEL_OVERRIDE}]`);
}

// Two-phase speed curriculum within each level. Phase 0 = slow (warmup),
// phase 1 = full race speed. When TWO_PHASE is off, every gen runs at
// SPEED_MULT (existing behavior).
let currentPhase = TWO_PHASE ? 0 : 1;
let currentSpeed = TWO_PHASE ? SLOW_SPEED : SPEED_MULT;

let cars = [];
for (let i = 0; i < NUM_CARS; i++) {
  let b;
  const opts = { rank: LORA_RANK_OVERRIDE, softFreezeFactor: SOFT_FREEZE };
  // Warm-load mode: ALL initial cars clone the warm brain so every child
  // starts with the correct frozen base + frozen prior adapters. Diversity
  // comes from per-car mutation of adapter[currentLevel], not from random
  // base/adapter init that destroys mastery on prior levels.
  if (warmBrain) {
    b = new NeuralCar(warmBrain, opts);
  } else {
    b = new NeuralCar(null, opts);
  }
  if (!NO_LORA) b.setLevel(difficultyLevel);
  cars.push(new HeadlessCar(track, b, i, currentSpeed));
}

let allTimeBest = 0;
let bestLapTime = Infinity;
let stagnantGens = 0;
let currentMutation = MUTATION_RATE;
// Seed bestWeights from the warm-loaded brain so championBase always
// inherits the carefully-trained base from gen 1. Without this, a random
// gen-1 elite could overwrite bestWeights with a random base, destroying
// every prior level's mastery (the base-drift bug seen in dwell runs).
let bestWeights = warmBrain
  ? JSON.parse(JSON.stringify(warmBrain))
  : null;
// Rolling window of recent gens' elites for robust-elite selection. Each
// entry: { finisherRate, bestScore, weights }. Cleared on escalation so the
// new level builds its own robustness statistics.
let robustEliteWindow = [];
// Clone-test ratchet: tracks the clone-test stats of the brain currently
// in bestWeights, so subsequent clone-tests can only IMPROVE it (not
// regress to a faster-but-fragile strategy). Cleared on escalation.
let bestWeightsCloneStats = null;
let bestLapStagnantGens = 0;
let escalationGens = 0;
let lapImprovementsOnLevel = 0;
let plateauConsecutiveChecks = 0;
let plateauAvgHistory = [];
let plateauFinishedRateHistory = [];
let escalationStatus = null;

const t0 = performance.now();

for (let gen = 1; gen <= NUM_GENS; gen++) {
  runGeneration(track, cars, TIMEOUT);

  // Sort by score
  cars.sort((a, b) => b.score - a.score);
  const bestScore = cars[0].score;
  const avgProgress = cars.reduce((s, c) => s + c.totalProgress, 0) / cars.length;
  const finishers = cars.filter((c) => c.finished);
  const bestCar = cars[0];
  const finishedRate = finishers.length / Math.max(1, cars.length);
  const genBestLap = finishers.length > 0
    ? Math.min(...finishers.map((c) => c.lapTime))
    : Infinity;
  plateauAvgHistory.push(avgProgress);
  plateauFinishedRateHistory.push(finishedRate);
  if (plateauAvgHistory.length > 160) plateauAvgHistory.shift();
  if (plateauFinishedRateHistory.length > 160) plateauFinishedRateHistory.shift();

  if (genBestLap < bestLapTime) {
    bestLapTime = genBestLap;
    bestLapStagnantGens = 0;
    lapImprovementsOnLevel++;
  } else if (bestLapTime < Infinity) {
    bestLapStagnantGens++;
  }

  const plateauStatus = evaluatePlateauStatus({
    avgProgressHistory: plateauAvgHistory,
    finishedRateHistory: plateauFinishedRateHistory,
    consecutiveChecks: plateauConsecutiveChecks,
  });
  plateauConsecutiveChecks = plateauStatus.consecutiveChecks || 0;

  // Adaptive mutation + curriculum escalation
  const mutState = computeAdaptiveMutation({
    bestScore,
    allTimeBest,
    stagnantGens,
    baseMutation: MUTATION_RATE,
    currentMutation,
    bestLapStagnantGens,
    currentDifficultyLevel: difficultyLevel,
    escalationGens,
    lapImprovementsOnLevel,
    hasCompletedLap: bestLapTime < Infinity,
    plateauStatus,
    // Robustness gate: refuse to escalate while only a "lucky elite"
    // 1-3/N cars are lapping. Adapter must be broadly capable first.
    currentFinisherRate: finishedRate,
    minFinisherRateForEscalation: MIN_FINISHER_RATE_FOR_ESCALATION,
  });
  escalationGens++;
  allTimeBest = mutState.allTimeBest;
  stagnantGens = mutState.stagnantGens;
  currentMutation = mutState.mutationRate;
  escalationStatus = mutState.escalationStatus || null;

  // Robust-elite tracking: bestWeights captures the brain that has the
  // best ROLLING-WINDOW finisher rate (not just the single fastest car).
  // This stores a brain that finishes consistently across many spawn
  // positions, which is what we want to ship to the next level.
  robustEliteWindow.push({
    finisherRate: finishedRate,
    bestScore,
    weights: bestCar.brain.getWeights(),
  });
  if (robustEliteWindow.length > ROBUST_ELITE_WINDOW_SIZE) robustEliteWindow.shift();
  // Choose elite by (finisherRate, then bestScore) lexicographic max.
  const robustElite = robustEliteWindow.reduce((best, cur) => {
    if (!best) return cur;
    if (cur.finisherRate > best.finisherRate) return cur;
    if (cur.finisherRate === best.finisherRate && cur.bestScore > best.bestScore) return cur;
    return best;
  }, null);

  // Only replace bestWeights when the new candidate is actually lapping
  // (finisherRate > 0). Otherwise a random gen-1 elite with no laps would
  // overwrite a warm-loaded brain that's already known-good — destroying
  // the carefully-trained base and prior adapters.
  // ALSO: when clone-test gating is active, the per-gen update is suppressed
  // once we have a clone-tested bestWeights so it can't get clobbered by
  // an un-tested brain that won the per-gen sort by raw lap-time alone.
  const candidate = robustElite ? robustElite.weights : cars[0].brain.getWeights();
  const candidateLaps = robustElite ? robustElite.finisherRate > 0 : false;
  const cloneTestActive = CLONE_TEST_EVERY > 0 && CLONE_TEST_K > 0 && CLONE_TEST_CARS > 0;
  if (!bestWeights) {
    bestWeights = candidate;
  } else if (!cloneTestActive && mutState.improved && candidateLaps) {
    // Without clone-test gating, fall back to per-gen elite update.
    bestWeights = candidate;
  }
  // (When clone-test is active, the periodic clone-test block below is the
  //  ONLY path that updates bestWeights — guarantees the saved brain has
  //  passed a fresh-clone robustness check.)

  // ─── Clone-test elite gating ──────────────────────────────
  // Every CLONE_TEST_EVERY gens, take the top-K candidates from the rolling
  // window and re-spawn each as N fresh-clone cars on the current track.
  // Pick the one with the most finishers (tiebreak: highest avg score).
  // This catches the "single fastest brain has fragile spawn-position
  // generalization" gap — the saved brain must lap from many spawn rows,
  // not just the row that happened to be fastest in some gen.
  if (
    CLONE_TEST_EVERY > 0
    && CLONE_TEST_K > 0
    && CLONE_TEST_CARS > 0
    && gen % CLONE_TEST_EVERY === 0
    && robustEliteWindow.length >= 1
  ) {
    const topK = [...robustEliteWindow]
      .filter((e) => e.finisherRate > 0) // only test candidates that lapped
      .sort((a, b) => {
        if (b.finisherRate !== a.finisherRate) return b.finisherRate - a.finisherRate;
        return b.bestScore - a.bestScore;
      })
      .slice(0, CLONE_TEST_K);
    if (topK.length > 0) {
      let winner = null;
      for (const cand of topK) {
        const cars2 = [];
        for (let i = 0; i < CLONE_TEST_CARS; i++) {
          const b = new NeuralCar(cand.weights, { rank: LORA_RANK_OVERRIDE });
          if (!NO_LORA) b.setLevel(difficultyLevel);
          cars2.push(new HeadlessCar(track, b, i, currentSpeed));
        }
        runGeneration(track, cars2, TIMEOUT);
        const fin = cars2.filter((c) => c.finished).length;
        const avg = cars2.reduce((s, c) => s + c.score, 0) / Math.max(1, cars2.length);
        if (
          !winner
          || fin > winner.fin
          || (fin === winner.fin && avg > winner.avg)
        ) {
          winner = { fin, avg, weights: cand.weights };
        }
      }
      if (winner && winner.fin > 0) {
        // RATCHET: only replace bestWeights when the new winner is at least
        // as robust (more finishers, or equal finishers with better avg
        // score). Otherwise a faster-but-less-robust strategy could erase
        // a previously-validated 16/16 winner. Tracking via bestWeightsCloneStats.
        if (
          !bestWeightsCloneStats
          || winner.fin > bestWeightsCloneStats.fin
          || (winner.fin === bestWeightsCloneStats.fin
              && winner.avg > bestWeightsCloneStats.avg)
        ) {
          bestWeights = winner.weights;
          bestWeightsCloneStats = { fin: winner.fin, avg: winner.avg };
          if (gen % (CLONE_TEST_EVERY * 5) === 0 || winner.fin === CLONE_TEST_CARS) {
            console.log(
              `  >>> CLONE-TEST gen ${gen}: ${topK.length} candidates, winner ${winner.fin}/${CLONE_TEST_CARS} fin avg ${winner.avg.toFixed(0)} (UPDATED)`
            );
          }
        } else if (gen % (CLONE_TEST_EVERY * 10) === 0) {
          console.log(
            `  >>> CLONE-TEST gen ${gen}: ${winner.fin}/${CLONE_TEST_CARS} avg ${winner.avg.toFixed(0)} — kept ${bestWeightsCloneStats.fin}/${CLONE_TEST_CARS} avg ${bestWeightsCloneStats.avg.toFixed(0)}`
          );
        }
      }
    }
  }

  // Curriculum escalation, with optional two-phase speed curriculum:
  //   PHASE 0 (slow) plateaus → switch to PHASE 1 (fast) on SAME level
  //   PHASE 1 (fast) plateaus → escalate to NEXT level (back to phase 0)
  // Special case: at the MAX curriculum level, computeAdaptiveMutation()
  // forces escalate=false (notMaxLevel guard). But we still want a phase
  // transition from slow → fast at max level. Detect plateau directly.
  const atMaxLevel = difficultyLevel >= DIFFICULTY_LADDER.length - 1;
  const slowPhasePlateauedAtMax = TWO_PHASE
    && currentPhase === 0
    && atMaxLevel
    && bestLapStagnantGens >= 80
    && escalationGens >= 150;

  // Clone-test escalation gate: when clone-test is active, refuse to
  // escalate or phase-transition until bestWeights has been validated at
  // the CURRENT phase by passing a clone-test with at least 75% finishers.
  // Without this gate, mutState.escalate can fire on population-wide
  // plateau (e.g. fin:20/80 at fast phase) while bestWeights is still the
  // slow-phase 16/16 winner — and the saved checkpoint fails at fast speed.
  const cloneTestThresholdFin = Math.max(1, Math.floor(CLONE_TEST_CARS * 0.75));
  const cloneTestValidatedAtCurrentPhase =
    !cloneTestActive
    || (bestWeightsCloneStats && bestWeightsCloneStats.fin >= cloneTestThresholdFin);

  if ((mutState.escalate || slowPhasePlateauedAtMax) && cloneTestValidatedAtCurrentPhase) {
    if (TWO_PHASE && currentPhase === 0) {
      // Phase transition (same level, switch to fast speed). Adapter persists.
      currentPhase = 1;
      currentSpeed = SPEED_MULT;
      bestLapTime = Infinity;
      bestLapStagnantGens = 0;
      stagnantGens = 0;
      escalationGens = 0;
      lapImprovementsOnLevel = 0;
      plateauConsecutiveChecks = 0;
      plateauAvgHistory = [];
      plateauFinishedRateHistory = [];
      escalationStatus = null;
      allTimeBest = 0;
      // CRITICAL: reset robust window + clone-test ratchet at phase transition.
      // Slow-phase brains often lap 16/16 at 0.5x but 0/16 at 1.0x. Without
      // this reset, the ratchet locks in slow-phase 16/16 as the bar, then
      // every fast-phase candidate (who legitimately laps at 1.0x but might
      // only achieve 12/16 in clone-test) gets rejected as "less robust" —
      // and the saved checkpoint is the slow-phase-only brain that fails
      // at race speed in eval. Reset clears the bar so fast-phase can win.
      robustEliteWindow = [];
      bestWeightsCloneStats = null;
      console.log(`\n  >>> PHASE TRANSITION: Level ${difficultyLevel} -> fast (${currentSpeed.toFixed(1)}x); ratchet reset, refining for race speed\n`);
    } else {
      // True escalation: SNAPSHOT the just-mastered level's brain BEFORE any
      // state changes. The saved file is a fully-loadable v2 brain.
      saveLevelCheckpoint(difficultyLevel, bestWeights);

      difficultyLevel++;
      const step = DIFFICULTY_LADDER[difficultyLevel];
      const defaultWidth = TRACK_DEFAULT_WIDTHS[step.track];
      const newWidth = step.widthDelta !== 0 ? defaultWidth + step.widthDelta : null;
      track = new HeadlessTrack(step.track, newWidth);
      bestLapTime = Infinity;
      bestLapStagnantGens = 0;
      stagnantGens = 0;
      escalationGens = 0;
      lapImprovementsOnLevel = 0;
      plateauConsecutiveChecks = 0;
      plateauAvgHistory = [];
      plateauFinishedRateHistory = [];
      escalationStatus = null;
      allTimeBest = 0;
      currentPhase = TWO_PHASE ? 0 : 1;
      currentSpeed = TWO_PHASE ? SLOW_SPEED : SPEED_MULT;
      // New level = new robustness statistics. Old window is irrelevant.
      robustEliteWindow = [];
      // Reset clone-test ratchet — the new level's adapter starts fresh,
      // so any prior clone-test stats are meaningless.
      bestWeightsCloneStats = null;
      const widthNote = step.widthDelta !== 0 ? ` (width: ${defaultWidth + step.widthDelta})` : '';
      const speedNote = TWO_PHASE ? ` @ ${currentSpeed.toFixed(1)}x slow` : '';
      console.log(`\n  >>> ESCALATION: Level ${difficultyLevel} -> ${step.track}${widthNote}${speedNote}`);
      console.log(`  >>> Base frozen; new LoRA adapter (rank ${LORA_RANK_OVERRIDE}, ~${(NUM_INPUTS + HIDDEN_SIZE) * LORA_RANK_OVERRIDE + (HIDDEN_SIZE + 2) * LORA_RANK_OVERRIDE} params) added for this level\n`);
    }
  }

  // Progress output
  const lapStr = genBestLap < Infinity ? (genBestLap / 60).toFixed(1) + 's' : '--';
  const bestLapStr = bestLapTime < Infinity ? (bestLapTime / 60).toFixed(1) + 's' : '--';
  const restartTag = mutState.restart ? ' RESTART' : (mutState.escalate ? ' ESCALATE' : '');

  if (gen % 10 === 0 || gen === 1 || genBestLap < Infinity || mutState.restart) {
    const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
    const finStr = finishers.length > 0 ? ` fin:${finishers.length}` : '';
    const plateauStr = escalationStatus
      ? ` pl:${Math.round((escalationStatus.confidence || 0) * 100)
        .toString()
        .padStart(2)}% chk:${escalationStatus.consecutiveChecks || 0}/${escalationStatus.requiredChecks || 0}`
      : '';
    console.log(
      `  Gen ${String(gen).padStart(4)} | ` +
      `score: ${bestScore.toFixed(2).padStart(8)} | ` +
      `avg: ${(avgProgress * 100).toFixed(1).padStart(5)}% | ` +
      `lap: ${lapStr.padStart(5)} | ` +
      `best: ${bestLapStr.padStart(5)} | ` +
      `mut: ${currentMutation.toFixed(3)}${finStr}${plateauStr}${restartTag} | ${elapsed}s`
    );
  }

  // Reset stagnation counter after restart
  if (mutState.restart) stagnantGens = 0;

  // Evolve next generation — pass current curriculum level so LoRA adapters
  // get switched/created and only the right parameters mutate. currentSpeed
  // tracks the two-phase speed schedule (slow → fast within each level).
  if (gen < NUM_GENS) {
    cars = evolve(cars, NUM_CARS, currentMutation, track, currentSpeed, mutState.restart, bestWeights, difficultyLevel);
  }
}

// Save best brain — both as the canonical OUTPUT_FILE and as a final-level
// checkpoint, so models/ has a complete -L0..-L{N} progression on disk.
mkdirSync(dirname(OUTPUT_FILE), { recursive: true });
writeFileSync(OUTPUT_FILE, JSON.stringify(bestWeights, null, 2));
saveLevelCheckpoint(difficultyLevel, bestWeights);
const totalTime = ((performance.now() - t0) / 1000).toFixed(1);
console.log(`\n  Done! ${NUM_GENS} generations in ${totalTime}s`);
console.log(`  Best lap: ${bestLapTime < Infinity ? (bestLapTime / 60).toFixed(1) + 's' : 'none'}`);
console.log(`  Best score: ${allTimeBest.toFixed(2)}`);
console.log(`  Saved to: ${OUTPUT_FILE}\n`);
console.log(`  Load in visual mode: open browser -> Load Brain -> select ${OUTPUT_FILE}\n`);
