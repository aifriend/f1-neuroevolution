#!/usr/bin/env node
// Headless F1 neuroevolution trainer — no rendering, pure math
// Uses the SAME evolution logic as the visual frontend (evolution-core.js)
// Runs 1000x faster than visual mode. Outputs best brain as JSON.
//
// Usage:
//   node train.js                          # defaults: monaco, 50 cars, 500 gens
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
const { computeInterpolationSteps } = await import('./js/track-geometry.js');
const { NeuralCar, NUM_INPUTS, HIDDEN_SIZE } = await import('./js/nn.js');
// Canonical curriculum track definitions (shared with js/track.js).
// Drift between headless and visual physics is now architecturally impossible.
const { TRACKS } = await import('./js/track-data.js');
const NUM_SENSORS = 9;

// ─── Track Physics (from track.js, no Three.js) ──────
const SENSOR_LENGTH = 220;

class HeadlessTrack {
  constructor(type, widthOverride = null) {
    const cfg = TRACKS[type] || TRACKS.monaco;
    this.name = cfg.name;
    this.trackWidth = widthOverride !== null ? widthOverride : cfg.width;
    this.gridSize = 5;
    const stepsPerSeg = computeInterpolationSteps(cfg.points, this.gridSize * 0.8);
    this.points = this._interpolate(cfg.points, stepsPerSeg);
    this.tangents = this._computeTangents();

    const sp = this.points[0];
    const st = this.tangents[0];
    this.startX = sp[0];
    this.startZ = sp[1];
    this.startAngle = Math.atan2(st[1], st[0]);

    this.grid = {};
    this._buildGrid();
  }

  _interpolate(pts, steps) {
    const result = [];
    const n = pts.length;
    for (let i = 0; i < n; i++) {
      const p0 = pts[(i - 1 + n) % n];
      const p1 = pts[i];
      const p2 = pts[(i + 1) % n];
      const p3 = pts[(i + 2) % n];
      for (let s = 0; s < steps; s++) {
        const t = s / steps;
        const t2 = t * t, t3 = t2 * t;
        const c0 = -0.5 * t3 + t2 - 0.5 * t;
        const c1 = 1.5 * t3 - 2.5 * t2 + 1;
        const c2 = -1.5 * t3 + 2 * t2 + 0.5 * t;
        const c3 = 0.5 * t3 - 0.5 * t2;
        result.push([
          c0 * p0[0] + c1 * p1[0] + c2 * p2[0] + c3 * p3[0],
          c0 * p0[1] + c1 * p1[1] + c2 * p2[1] + c3 * p3[1],
        ]);
      }
    }
    return result;
  }

  _computeTangents() {
    const n = this.points.length;
    const tangents = [];
    for (let i = 0; i < n; i++) {
      const next = this.points[(i + 1) % n];
      const curr = this.points[i];
      const dx = next[0] - curr[0], dz = next[1] - curr[1];
      const len = Math.sqrt(dx * dx + dz * dz) || 1;
      tangents.push([dx / len, dz / len]);
    }
    return tangents;
  }

  _buildGrid() {
    const n = this.points.length;
    const gs = this.gridSize;
    const hw = this.trackWidth;
    for (let i = 0; i < n; i++) {
      const px = this.points[i][0], pz = this.points[i][1];
      const t = this.tangents[i];
      const nx = -t[1], nz = t[0];
      for (let w = -hw; w <= hw; w += 1) {
        const gx = Math.floor((px + nx * w) / gs);
        const gz = Math.floor((pz + nz * w) / gs);
        this.grid[gx + ',' + gz] = true;
      }
    }
  }

  isOnTrack(x, z) {
    return this.grid[Math.floor(x / this.gridSize) + ',' + Math.floor(z / this.gridSize)] === true;
  }

  castRay(x, z, angle) {
    const cosA = Math.cos(angle), sinA = Math.sin(angle);
    for (let d = 2; d < SENSOR_LENGTH; d += 2) {
      if (!this.isOnTrack(x + cosA * d, z + sinA * d)) return d;
    }
    return SENSOR_LENGTH;
  }

  getProgress(x, z) {
    let minDist = Infinity, bestIdx = 0;
    const n = this.points.length;
    for (let i = 0; i < n; i += 4) {
      const dx = x - this.points[i][0], dz = z - this.points[i][1];
      const dist = dx * dx + dz * dz;
      if (dist < minDist) { minDist = dist; bestIdx = i; }
    }
    for (let offset = -6; offset <= 6; offset++) {
      const i = ((bestIdx + offset) % n + n) % n;
      const dx = x - this.points[i][0], dz = z - this.points[i][1];
      const dist = dx * dx + dz * dz;
      if (dist < minDist) { minDist = dist; bestIdx = i; }
    }
    return bestIdx / n;
  }

  getProgressLocal(x, z, hintIdx, searchRadius = 20) {
    let minDist = Infinity, bestIdx = hintIdx;
    const n = this.points.length;
    for (let offset = -searchRadius; offset <= searchRadius; offset++) {
      const i = ((hintIdx + offset) % n + n) % n;
      const dx = x - this.points[i][0], dz = z - this.points[i][1];
      const dist = dx * dx + dz * dz;
      if (dist < minDist) { minDist = dist; bestIdx = i; }
    }
    return { progress: bestIdx / n, idx: bestIdx };
  }

  getStartPos() {
    return { x: this.startX, z: this.startZ, angle: this.startAngle };
  }
}

// ─── Car Physics (from car.js, no Three.js) ──────────
const SENSOR_ANGLES = [
  -Math.PI / 2, -Math.PI * 3 / 8, -Math.PI / 4, -Math.PI / 8,
  0,
  Math.PI / 8, Math.PI / 4, Math.PI * 3 / 8, Math.PI / 2,
];
const STUCK_LIMIT = 120;
const LAP_COMPLETION_PROGRESS = 0.995;

class HeadlessCar {
  constructor(track, brain, teamIdx, speedMult) {
    this.track = track;
    this.brain = brain || new NeuralCar();
    this.speedMult = speedMult;

    const start = track.getStartPos();
    const t = track.tangents[0];
    const nx = -t[1], nz = t[0];
    const row = Math.floor(teamIdx / 2);
    const col = (teamIdx % 2) - 0.5;

    this.x = start.x - t[0] * row * 4 + nx * col * 8;
    this.z = start.z - t[1] * row * 4 + nz * col * 8;
    this.angle = start.angle;
    this.speed = 0;
    this.sensors = new Array(SENSOR_ANGLES.length).fill(0);
    this.lapTime = 0;
    this.totalProgress = 0;
    const initProg = track.getProgress(this.x, this.z);
    this.initialProgress = initProg;
    this.lastProgress = initProg;
    this.lastProgressIdx = Math.round(initProg * track.points.length);
    this.progressAccum = 0;
    this.score = 0;
    this.alive = true;
    this.finished = false;
    this.killReason = 'active';
    this.frameCounter = 0;
    this.stuckFrames = 0;
    this.reverseAccum = 0;
    this.wrongWayFrames = 0;
    // Lap checkpoints — all must be passed before finish-line crossing counts.
    this.passedQuarter = false;
    this.passedHalf = false;
    this.passedThreeQuarter = false;
  }

  update() {
    if (!this.alive || this.finished) return;

    for (let i = 0; i < SENSOR_ANGLES.length; i++) {
      this.sensors[i] = this.track.castRay(this.x, this.z, this.angle + SENSOR_ANGLES[i]) / SENSOR_LENGTH;
    }

    const inputs = [...this.sensors, this.speed / 8.1];
    const decision = this.brain.think(inputs);
    this.angle += decision.steer * 0.08;
    this.speed = (2.5 + (decision.gas + 1) * 2.8) * this.speedMult;

    const cosA = Math.cos(this.angle);
    const sinA = Math.sin(this.angle);
    const newX = this.x + cosA * this.speed;
    const newZ = this.z + sinA * this.speed;

    // Midpoint collision to prevent wall tunneling at high speeds
    const midX = (this.x + newX) * 0.5;
    const midZ = (this.z + newZ) * 0.5;
    if (!this.track.isOnTrack(midX, midZ) || !this.track.isOnTrack(newX, newZ)) {
      this.alive = false;
      this.killReason = 'offtrack';
      this.score = computeScore(this);
      return;
    }

    this.x = newX;
    this.z = newZ;

    const { progress, idx } = this.track.getProgressLocal(this.x, this.z, this.lastProgressIdx);
    this.lastProgressIdx = idx;
    const rawDelta = progress - this.lastProgress;
    let delta = rawDelta;
    const crossedFinish = rawDelta < -0.5;
    if (crossedFinish) delta += 1.0;
    if (delta > 0.5) delta -= 1.0;

    // Cap delta to prevent progress aliasing on overlapping track geometry
    if (delta > 0.05) delta = 0.05;
    if (delta < -0.05) delta = -0.05;

    if (delta > 0) { this.progressAccum += delta; this.stuckFrames = 0; this.reverseAccum = 0; }
    else { this.stuckFrames++; this.reverseAccum += delta; }
    this.lastProgress = progress;
    this.totalProgress = this.progressAccum;

    if (this.stuckFrames > STUCK_LIMIT) { this.alive = false; this.killReason = 'stuck'; return; }

    // Reverse driving — kill within ~10 frames of wrong-way driving
    if (this.reverseAccum < -0.05) { this.alive = false; this.killReason = 'reverse'; return; }

    // Wrong-way via velocity-tangent alignment — catches aliasing at overlaps
    if (this.speed > 0.5) {
      const tg = this.track.tangents[idx];
      const dot = cosA * tg[0] + sinA * tg[1];
      if (dot < -0.3) {
        this.wrongWayFrames++;
        if (this.wrongWayFrames > 5) { this.alive = false; this.killReason = 'wrong_way'; return; }
      } else {
        this.wrongWayFrames = 0;
      }
    }

    // Scaled loitering detection
    if (this.frameCounter > 200 && this.progressAccum < this.frameCounter * 0.00015) {
      this.alive = false;
      this.killReason = 'loitering';
      this.score = computeScore(this);
      return;
    }

    this.frameCounter++;

    // Ordered checkpoints — each must be passed before the next counts
    if (!this.passedQuarter && progress >= 0.20 && progress <= 0.35) {
      this.passedQuarter = true;
    }
    if (this.passedQuarter && !this.passedHalf && progress >= 0.45 && progress <= 0.60) {
      this.passedHalf = true;
    }
    if (this.passedHalf && !this.passedThreeQuarter && progress >= 0.70 && progress <= 0.85) {
      this.passedThreeQuarter = true;
    }

    // Lap completion requires ALL checkpoints + finish-line crossing (or
    // accumulated progress reaching 1.0 AND all checkpoints visited)
    const allCheckpointsHit =
      this.passedQuarter && this.passedHalf && this.passedThreeQuarter;
    const reachedTarget =
      allCheckpointsHit && this.progressAccum >= LAP_COMPLETION_PROGRESS;
    const crossedFinishLine = allCheckpointsHit && crossedFinish;
    if (reachedTarget || crossedFinishLine) {
      this.finished = true;
      this.killReason = 'finished';
      this.lapTime = this.frameCounter;
    }

    // Use shared scoring function (same as visual frontend)
    this.score = computeScore(this);
  }
}

// ─── Generation Runner ──────────────────────────────
function runGeneration(track, cars, timeout) {
  for (let frame = 0; frame < timeout; frame++) {
    let anyAlive = false;
    for (const car of cars) {
      car.update();
      if (car.alive && !car.finished) anyAlive = true;
    }
    if (!anyAlive) break;
  }
  for (const c of cars) {
    if (c.alive && !c.finished) {
      c.alive = false;
      if (!c.killReason || c.killReason === 'active') c.killReason = 'timeout';
    }
  }
}

// ─── Evolution (uses shared evolution-core.js) ───────
// `level` is the current curriculum level — every cloned brain has its
// active LoRA adapter switched to this level (and a fresh adapter created
// if none exists yet). This is what enforces "freeze base, mutate current
// adapter" inside NeuralCar.mutate().
function evolve(sorted, numCars, baseMut, track, speedMult, doRestart, hallOfFameWeights, level) {
  const newCars = [];

  // LoRA mode: freeze base at level >= 1 by collapsing all child bases to the
  // champion's. Without this, tournament selection would scatter base
  // diversity and we'd lose the lapping brain's exact base.
  // --noLora mode: skip everything — base stays mutable (catastrophic
  // forgetting baseline for A/B comparison).
  const championBase = !NO_LORA && level >= 1 && hallOfFameWeights && hallOfFameWeights.base
    ? hallOfFameWeights.base
    : null;

  const makeBrain = (weights) => {
    const b = new NeuralCar(weights, { rank: LORA_RANK_OVERRIDE, softFreezeFactor: SOFT_FREEZE });
    if (championBase) b.setBase(championBase);
    if (!NO_LORA) b.setLevel(level);
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
// matches the loaded state.
let difficultyLevel = (warmBrain && Number.isFinite(warmBrain.currentLevel))
  ? warmBrain.currentLevel
  : 0;

// Two-phase speed curriculum within each level. Phase 0 = slow (warmup),
// phase 1 = full race speed. When TWO_PHASE is off, every gen runs at
// SPEED_MULT (existing behavior).
let currentPhase = TWO_PHASE ? 0 : 1;
let currentSpeed = TWO_PHASE ? SLOW_SPEED : SPEED_MULT;

let cars = [];
for (let i = 0; i < NUM_CARS; i++) {
  let b;
  const opts = { rank: LORA_RANK_OVERRIDE, softFreezeFactor: SOFT_FREEZE };
  if (warmBrain && i < 2) {
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
let bestWeights = null;
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
  });
  escalationGens++;
  allTimeBest = mutState.allTimeBest;
  stagnantGens = mutState.stagnantGens;
  currentMutation = mutState.mutationRate;
  escalationStatus = mutState.escalationStatus || null;

  if (!bestWeights || mutState.improved) {
    bestWeights = cars[0].brain.getWeights();
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

  if (mutState.escalate || slowPhasePlateauedAtMax) {
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
      console.log(`\n  >>> PHASE TRANSITION: Level ${difficultyLevel} -> fast (${currentSpeed.toFixed(1)}x); same adapter, refining for race speed\n`);
    } else {
      // True escalation: next track + reset to slow phase (if two-phase)
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
      const widthNote = step.widthDelta !== 0 ? ` (width: ${defaultWidth + step.widthDelta})` : '';
      const speedNote = TWO_PHASE ? ` @ ${currentSpeed.toFixed(1)}x slow` : '';
      console.log(`\n  >>> ESCALATION: Level ${difficultyLevel} -> ${step.track}${widthNote}${speedNote}`);
      console.log(`  >>> Base frozen; new LoRA adapter (rank 2, ~88 params) added for this level\n`);
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

// Save best brain
mkdirSync(dirname(OUTPUT_FILE), { recursive: true });
writeFileSync(OUTPUT_FILE, JSON.stringify(bestWeights, null, 2));
const totalTime = ((performance.now() - t0) / 1000).toFixed(1);
console.log(`\n  Done! ${NUM_GENS} generations in ${totalTime}s`);
console.log(`  Best lap: ${bestLapTime < Infinity ? (bestLapTime / 60).toFixed(1) + 's' : 'none'}`);
console.log(`  Best score: ${allTimeBest.toFixed(2)}`);
console.log(`  Saved to: ${OUTPUT_FILE}\n`);
console.log(`  Load in visual mode: open browser -> Load Brain -> select ${OUTPUT_FILE}\n`);
