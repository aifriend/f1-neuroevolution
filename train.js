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
const NUM_CARS = parseInt(flag('cars', '50'));
const NUM_GENS = parseInt(flag('gens', '500'));
const MUTATION_RATE = parseFloat(flag('mutation', '0.05'));
const TIMEOUT = parseInt(flag('timeout', '1500'));
const SPEED_MULT = parseFloat(flag('speed', '1'));
const OUTPUT_FILE = flag('output', `best-brain-${TRACK_TYPE}.json`);

// ─── Import shared evolution logic ───────────────────
// These functions are the SAME ones used by the visual frontend,
// ensuring headless and visual training produce identical behavior.
const {
  computeAdaptiveMutation,
  tournamentSelect,
  getTieredMutationRate,
  computeScore,
  DIFFICULTY_LADDER,
  TRACK_DEFAULT_WIDTHS,
} = await import('./js/evolution-core.js');
const { computeInterpolationSteps } = await import('./js/track-geometry.js');

// ─── Neural Network (from nn.js) ─────────────────────
const NUM_SENSORS = 9;
const NUM_INPUTS = NUM_SENSORS + 1; // sensors + speed
const HIDDEN_SIZE = 16;

function randomGaussian(mean = 0, std = 1) {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return mean + std * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

class NeuralCar {
  constructor(weights) {
    if (!weights) {
      const scale1 = 0.8;
      const scale2 = 0.6;
      this.w1 = this._randomMatrix(NUM_INPUTS, HIDDEN_SIZE, scale1);
      this.b1 = new Array(HIDDEN_SIZE).fill(0);
      this.w2 = this._randomMatrix(HIDDEN_SIZE, 2, scale2);
      this.b2 = new Array(2).fill(0);
    } else {
      this.w1 = weights.w1.map((r) => [...r]);
      this.b1 = [...(weights.b1 || new Array(HIDDEN_SIZE).fill(0))];
      this.w2 = weights.w2.map((r) => [...r]);
      this.b2 = [...(weights.b2 || new Array(2).fill(0))];
    }
  }

  _randomMatrix(rows, cols, scale = 1) {
    const m = [];
    for (let i = 0; i < rows; i++) {
      const row = [];
      for (let j = 0; j < cols; j++) row.push(randomGaussian(0, scale));
      m.push(row);
    }
    return m;
  }

  _randomVector(size) {
    return Array.from({ length: size }, () => randomGaussian(0, 0.5));
  }

  think(sensors) {
    const hidden = [];
    for (let j = 0; j < HIDDEN_SIZE; j++) {
      let sum = this.b1[j];
      for (let i = 0; i < sensors.length; i++) sum += sensors[i] * this.w1[i][j];
      hidden.push(Math.tanh(sum));
    }
    const output = [];
    for (let j = 0; j < 2; j++) {
      let sum = this.b2[j];
      for (let i = 0; i < HIDDEN_SIZE; i++) sum += hidden[i] * this.w2[i][j];
      output.push(Math.tanh(sum));
    }
    return { steer: output[0], gas: output[1] };
  }

  mutate(rate) {
    for (let i = 0; i < this.w1.length; i++)
      for (let j = 0; j < this.w1[i].length; j++)
        this.w1[i][j] += randomGaussian(0, 1) * rate;
    for (let i = 0; i < this.b1.length; i++)
      this.b1[i] += randomGaussian(0, 1) * rate;
    for (let i = 0; i < this.w2.length; i++)
      for (let j = 0; j < this.w2[i].length; j++)
        this.w2[i][j] += randomGaussian(0, 1) * rate;
    for (let i = 0; i < this.b2.length; i++)
      this.b2[i] += randomGaussian(0, 1) * rate;
  }

  getWeights() {
    return {
      w1: this.w1.map((r) => [...r]),
      b1: [...this.b1],
      w2: this.w2.map((r) => [...r]),
      b2: [...this.b2],
    };
  }
}

// ─── Track Physics (from track.js, no Three.js) ──────
const SENSOR_LENGTH = 220;

const TRACKS = {
  monaco: {
    name: 'Monaco', width: 28,
    points: [
      [0,0],[100,5],[160,35],[180,85],
      [170,145],[130,180],[80,185],[50,160],
      [35,120],[55,85],[40,45],[10,15],
    ],
  },
  suzuka: {
    name: 'Suzuka', width: 30,
    points: [
      [-80,0],[-115,95],[-125,200],[-110,305],
      [-70,370],[0,405],[55,370],[75,320],
      [100,375],[155,410],[215,360],
      [235,275],[225,185],[200,100],[170,25],
      [140,-60],[110,25],[80,-60],
      [50,25],[20,-60],[-20,15],[-55,-10],
    ],
  },
  silverstone: {
    name: 'Silverstone', width: 32,
    points: [
      [0,0],[120,10],[180,40],[200,100],[185,160],
      [150,200],[170,260],[200,320],[180,390],[130,430],
      [60,440],[0,420],[-30,370],[-60,320],[-40,270],
      [-70,210],[-90,140],[-80,70],[-50,25],[-20,5],
    ],
  },
  spaghetti: {
    name: 'Spaghetti', width: 26,
    points: [
      [0,0],[70,-10],[130,-45],[150,-100],
      [125,-160],[160,-210],[130,-270],[165,-320],
      [120,-360],[60,-370],[10,-340],[-20,-280],
      [15,-230],[-25,-175],[10,-120],[-20,-60],[-5,-10],
    ],
  },
  serpentine: {
    name: 'Serpentine', width: 24,
    points: [
      [0,0],[55,-20],[85,15],[140,-10],[160,25],
      [200,55],[170,100],[215,135],[185,180],[225,215],
      [195,265],[155,285],[180,325],
      [135,345],[100,315],[65,350],
      [25,320],[50,280],[-10,240],
      [30,200],[-20,155],[20,110],
      [-30,70],[10,35],[-10,10],
    ],
  },
  inferno: {
    name: 'Inferno', width: 22,
    points: [
      [0,0],[70,25],[-10,60],[80,105],[5,135],
      [55,180],[-20,220],[75,250],[10,295],
      [65,335],[-15,365],[45,410],[0,445],
      [60,475],[120,480],[160,455],
      [190,420],[250,395],[170,355],[240,320],
      [160,280],[260,250],[175,210],[235,170],
      [155,140],[245,105],[180,70],[250,35],
      [185,0],[140,-25],[80,-30],[30,-15],
    ],
  },
};

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
const LAP_COMPLETION_PROGRESS = 1.0;

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
    this.frameCounter = 0;
    this.stuckFrames = 0;
    this.reverseAccum = 0;
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
      this.score = computeScore(this);
      return;
    }

    this.x = newX;
    this.z = newZ;

    const { progress, idx } = this.track.getProgressLocal(this.x, this.z, this.lastProgressIdx);
    this.lastProgressIdx = idx;
    let delta = progress - this.lastProgress;
    if (delta < -0.5) delta += 1.0;
    if (delta > 0.5) delta -= 1.0;

    // Cap delta to prevent progress aliasing on overlapping track geometry
    if (delta > 0.05) delta = 0.05;
    if (delta < -0.05) delta = -0.05;

    if (delta > 0) { this.progressAccum += delta; this.stuckFrames = 0; this.reverseAccum = 0; }
    else { this.stuckFrames++; this.reverseAccum += delta; }
    this.lastProgress = progress;
    this.totalProgress = this.progressAccum;

    if (this.stuckFrames > STUCK_LIMIT) { this.alive = false; return; }

    // Reverse driving — kill within ~10 frames of wrong-way driving
    if (this.reverseAccum < -0.05) { this.alive = false; return;
    }

    // Scaled loitering detection
    if (this.frameCounter > 200 && this.progressAccum < this.frameCounter * 0.00015) {
      this.alive = false;
      this.score = computeScore(this);
      return;
    }

    this.frameCounter++;

    if (this.progressAccum >= LAP_COMPLETION_PROGRESS) {
      this.finished = true;
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
  for (const c of cars) { if (c.alive && !c.finished) c.alive = false; }
}

// ─── Evolution (uses shared evolution-core.js) ───────
function evolve(sorted, numCars, baseMut, track, speedMult, doRestart, hallOfFameWeights) {
  const newCars = [];

  // Cars #0-1: preserve all-time champion + best current generation.
  const championWeights = hallOfFameWeights || sorted[0].brain.getWeights();
  const currentBestWeights = sorted[0].brain.getWeights();
  newCars.push(new HeadlessCar(track, new NeuralCar(championWeights), 0, speedMult));
  newCars.push(new HeadlessCar(track, new NeuralCar(currentBestWeights), 1, speedMult));

  if (doRestart) {
    // Partial restart: keep top 20% micro-mutated, rest heavily mutated
    const eliteCount = Math.max(3, Math.ceil(numCars * 0.2));
    // Use configured base mutation for parity with visual runtime.
    const eliteMutRate = MUTATION_RATE * 0.1;
    for (let i = 1; i < Math.min(eliteCount, numCars); i++) {
      const elite = new NeuralCar(sorted[Math.min(i, sorted.length - 1)].brain.getWeights());
      elite.mutate(eliteMutRate);
      newCars.push(new HeadlessCar(track, elite, i, speedMult));
    }
    for (let i = newCars.length; i < numCars; i++) {
      const parent = sorted[Math.floor(Math.random() * eliteCount)];
      const child = new NeuralCar(parent.brain.getWeights());
      child.mutate(baseMut * 3.0);
      newCars.push(new HeadlessCar(track, child, i, speedMult));
    }
  } else {
    // Tournament selection + tiered mutation
    for (let i = newCars.length; i < numCars; i++) {
      const parent = tournamentSelect(sorted, 3);
      const child = new NeuralCar(parent.brain.getWeights());
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
const { writeFileSync } = await import('fs');

console.log(`\n  F1 Neuroevolution — Headless Trainer v2`);
console.log(`  ─────────────────────────────────────────`);
console.log(`  Track:      ${TRACK_TYPE}`);
console.log(`  Cars:       ${NUM_CARS}`);
console.log(`  Generations:${NUM_GENS}`);
console.log(`  Mutation:   ${MUTATION_RATE}`);
console.log(`  Timeout:    ${TIMEOUT} frames`);
console.log(`  Speed mult: ${SPEED_MULT}x`);
console.log(`  Output:     ${OUTPUT_FILE}`);
console.log(`  Evolution:  tournament(k=3) + tiered mutation + curriculum learning\n`);

let track = new HeadlessTrack(TRACK_TYPE);
let cars = [];
for (let i = 0; i < NUM_CARS; i++) {
  cars.push(new HeadlessCar(track, null, i, SPEED_MULT));
}

let allTimeBest = 0;
let bestLapTime = Infinity;
let stagnantGens = 0;
let currentMutation = MUTATION_RATE;
let bestWeights = null;
let difficultyLevel = 0;
let bestLapStagnantGens = 0;
let escalationGens = 0;
let lapImprovementsOnLevel = 0;

const t0 = performance.now();

for (let gen = 1; gen <= NUM_GENS; gen++) {
  runGeneration(track, cars, TIMEOUT);

  // Sort by score
  cars.sort((a, b) => b.score - a.score);
  const bestScore = cars[0].score;
  const avgProgress = cars.reduce((s, c) => s + c.totalProgress, 0) / cars.length;
  const finishers = cars.filter((c) => c.finished);
  const genBestLap = finishers.length > 0
    ? Math.min(...finishers.map((c) => c.lapTime))
    : Infinity;

  if (genBestLap < bestLapTime) {
    bestLapTime = genBestLap;
    bestLapStagnantGens = 0;
    lapImprovementsOnLevel++;
  } else if (bestLapTime < Infinity) {
    bestLapStagnantGens++;
  }

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
  });
  escalationGens++;
  allTimeBest = mutState.allTimeBest;
  stagnantGens = mutState.stagnantGens;
  currentMutation = mutState.mutationRate;

  if (!bestWeights || mutState.improved) {
    bestWeights = cars[0].brain.getWeights();
  }

  // Curriculum escalation
  if (mutState.escalate) {
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
    allTimeBest = 0;
    const widthNote = step.widthDelta !== 0 ? ` (width: ${defaultWidth + step.widthDelta})` : '';
    console.log(`\n  >>> ESCALATION: Level ${difficultyLevel} -> ${step.track}${widthNote}`);
    console.log(`  >>> Carrying over best brain weights (210 params)\n`);
  }

  // Progress output
  const lapStr = genBestLap < Infinity ? (genBestLap / 60).toFixed(1) + 's' : '--';
  const bestLapStr = bestLapTime < Infinity ? (bestLapTime / 60).toFixed(1) + 's' : '--';
  const restartTag = mutState.restart ? ' RESTART' : (mutState.escalate ? ' ESCALATE' : '');

  if (gen % 10 === 0 || gen === 1 || genBestLap < Infinity || mutState.restart) {
    const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
    const finStr = finishers.length > 0 ? ` fin:${finishers.length}` : '';
    console.log(
      `  Gen ${String(gen).padStart(4)} | ` +
      `score: ${bestScore.toFixed(2).padStart(8)} | ` +
      `avg: ${(avgProgress * 100).toFixed(1).padStart(5)}% | ` +
      `lap: ${lapStr.padStart(5)} | ` +
      `best: ${bestLapStr.padStart(5)} | ` +
      `mut: ${currentMutation.toFixed(3)}${finStr}${restartTag} | ${elapsed}s`
    );
  }

  // Reset stagnation counter after restart
  if (mutState.restart) stagnantGens = 0;

  // Evolve next generation
  if (gen < NUM_GENS) {
    cars = evolve(cars, NUM_CARS, currentMutation, track, SPEED_MULT, mutState.restart, bestWeights);
  }
}

// Save best brain
writeFileSync(OUTPUT_FILE, JSON.stringify(bestWeights, null, 2));
const totalTime = ((performance.now() - t0) / 1000).toFixed(1);
console.log(`\n  Done! ${NUM_GENS} generations in ${totalTime}s`);
console.log(`  Best lap: ${bestLapTime < Infinity ? (bestLapTime / 60).toFixed(1) + 's' : 'none'}`);
console.log(`  Best score: ${allTimeBest.toFixed(2)}`);
console.log(`  Saved to: ${OUTPUT_FILE}\n`);
console.log(`  Load in visual mode: open browser -> Load Brain -> select ${OUTPUT_FILE}\n`);
