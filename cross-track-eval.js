#!/usr/bin/env node
// Measure how well a saved brain performs on EVERY curriculum level.
// Loads a v2 brain JSON (with base + adapters), runs N evaluation laps on
// each level by switching the active adapter, and reports per-level scores.
//
// This is the empirical check for catastrophic forgetting: with LoRA,
// scores on previously-trained levels should NOT degrade over time, because
// each level keeps its own frozen adapter.
//
// Usage:
//   node cross-track-eval.js path/to/brain.json
//   node cross-track-eval.js path/to/brain.json --laps 20 --cars 8

import { readFileSync } from 'fs';

const args = process.argv.slice(2);
function flag(name, fallback) {
  const i = args.indexOf('--' + name);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : fallback;
}

const BRAIN_PATH = args.find((a) => !a.startsWith('--')) || './best-brain-monaco.json';
const NUM_CARS = parseInt(flag('cars', '8'));
const TIMEOUT = parseInt(flag('timeout', '3000'));
const SPEED_MULT = parseFloat(flag('speed', '1'));

const { NeuralCar, NUM_INPUTS, HIDDEN_SIZE } = await import('./js/nn.js');
const { computeScore, DIFFICULTY_LADDER, TRACK_DEFAULT_WIDTHS } = await import('./js/evolution-core.js');
const { computeInterpolationSteps } = await import('./js/track-geometry.js');

// We re-use train.js's HeadlessTrack/HeadlessCar by directly importing parts.
// To avoid a bigger refactor, inline a minimal evaluator here — same physics
// as train.js but no evolution loop.
const NUM_SENSORS = 9;
const SENSOR_LENGTH = 220;
const SENSOR_ANGLES = [
  -Math.PI/2, -Math.PI*3/8, -Math.PI/4, -Math.PI/8, 0,
  Math.PI/8, Math.PI/4, Math.PI*3/8, Math.PI/2,
];
const STUCK_LIMIT = 120;
const LAP_COMPLETION_PROGRESS = 1.0;

// Load track definitions from the shared source of truth. Both the browser
// runtime (js/track.js) and the headless trainer (train.js) also use this.
const { TRACKS } = await import('./js/track-data.js');

class HeadlessTrack {
  constructor(type, widthOverride = null) {
    const cfg = TRACKS[type];
    if (!cfg) throw new Error(`Unknown track: ${type}`);
    this.name = cfg.name;
    this.trackWidth = widthOverride ?? cfg.width;
    this.gridSize = 5;
    const stepsPerSeg = computeInterpolationSteps(cfg.points, this.gridSize * 0.8);
    this.points = this._interpolate(cfg.points, stepsPerSeg);
    this.tangents = this._tangents();
    const sp = this.points[0], st = this.tangents[0];
    this.startX = sp[0]; this.startZ = sp[1];
    this.startAngle = Math.atan2(st[1], st[0]);
    this.grid = {};
    this._buildGrid();
  }
  _interpolate(pts, steps) {
    const out = []; const n = pts.length;
    for (let i = 0; i < n; i++) {
      const p0 = pts[(i-1+n)%n], p1 = pts[i], p2 = pts[(i+1)%n], p3 = pts[(i+2)%n];
      for (let s = 0; s < steps; s++) {
        const t = s/steps, t2 = t*t, t3 = t2*t;
        const c0 = -0.5*t3 + t2 - 0.5*t;
        const c1 = 1.5*t3 - 2.5*t2 + 1;
        const c2 = -1.5*t3 + 2*t2 + 0.5*t;
        const c3 = 0.5*t3 - 0.5*t2;
        out.push([
          c0*p0[0]+c1*p1[0]+c2*p2[0]+c3*p3[0],
          c0*p0[1]+c1*p1[1]+c2*p2[1]+c3*p3[1],
        ]);
      }
    }
    return out;
  }
  _tangents() {
    const n = this.points.length, t = [];
    for (let i = 0; i < n; i++) {
      const a = this.points[i], b = this.points[(i+1)%n];
      const dx = b[0]-a[0], dz = b[1]-a[1];
      const L = Math.sqrt(dx*dx + dz*dz) || 1;
      t.push([dx/L, dz/L]);
    }
    return t;
  }
  _buildGrid() {
    const n = this.points.length, gs = this.gridSize, hw = this.trackWidth;
    for (let i = 0; i < n; i++) {
      const px = this.points[i][0], pz = this.points[i][1];
      const t = this.tangents[i], nx = -t[1], nz = t[0];
      for (let w = -hw; w <= hw; w += 1) {
        const gx = Math.floor((px + nx*w) / gs);
        const gz = Math.floor((pz + nz*w) / gs);
        this.grid[gx + ',' + gz] = true;
      }
    }
  }
  isOnTrack(x, z) {
    return this.grid[Math.floor(x/this.gridSize) + ',' + Math.floor(z/this.gridSize)] === true;
  }
  castRay(x, z, a) {
    const ca = Math.cos(a), sa = Math.sin(a);
    for (let d = 2; d < SENSOR_LENGTH; d += 2) {
      if (!this.isOnTrack(x + ca*d, z + sa*d)) return d;
    }
    return SENSOR_LENGTH;
  }
  getProgressLocal(x, z, hint, r = 20) {
    let best = hint, min = Infinity;
    const n = this.points.length;
    for (let off = -r; off <= r; off++) {
      const i = ((hint + off) % n + n) % n;
      const dx = x - this.points[i][0], dz = z - this.points[i][1];
      const d = dx*dx + dz*dz;
      if (d < min) { min = d; best = i; }
    }
    return { progress: best / n, idx: best };
  }
  getProgress(x, z) {
    let min = Infinity, best = 0;
    const n = this.points.length;
    for (let i = 0; i < n; i += 4) {
      const dx = x - this.points[i][0], dz = z - this.points[i][1];
      const d = dx*dx + dz*dz;
      if (d < min) { min = d; best = i; }
    }
    return best / n;
  }
}

function runCar(brain, track, timeout, speedMult, idx) {
  const t = track.tangents[0];
  const nx = -t[1], nz = t[0];
  const row = Math.floor(idx / 2);
  const col = (idx % 2) - 0.5;
  let x = track.startX - t[0]*row*4 + nx*col*8;
  let z = track.startZ - t[1]*row*4 + nz*col*8;
  let angle = track.startAngle, speed = 0;
  let progressAccum = 0, lastProgress = track.getProgress(x, z);
  let lastProgressIdx = Math.round(lastProgress * track.points.length);
  let stuck = 0, alive = true, finished = false, frame = 0, lapTime = 0;
  const sensors = new Array(NUM_SENSORS);
  let passedQ = false, passedH = false, passed3Q = false;

  while (alive && !finished && frame < timeout) {
    for (let i = 0; i < NUM_SENSORS; i++) {
      sensors[i] = track.castRay(x, z, angle + SENSOR_ANGLES[i]) / SENSOR_LENGTH;
    }
    const dec = brain.think([...sensors, speed / 8.1]);
    angle += dec.steer * 0.08;
    speed = (2.5 + (dec.gas + 1) * 2.8) * speedMult;
    const ca = Math.cos(angle), sa = Math.sin(angle);
    const nxX = x + ca*speed, nzZ = z + sa*speed;
    if (!track.isOnTrack((x+nxX)*0.5, (z+nzZ)*0.5) || !track.isOnTrack(nxX, nzZ)) {
      alive = false; break;
    }
    x = nxX; z = nzZ;
    const pr = track.getProgressLocal(x, z, lastProgressIdx);
    lastProgressIdx = pr.idx;
    let delta = pr.progress - lastProgress;
    const wrap = delta < -0.5;
    if (wrap) delta += 1.0;
    if (delta > 0.5) delta -= 1.0;
    if (delta > 0.05) delta = 0.05;
    if (delta < -0.05) delta = -0.05;
    if (delta > 0) { progressAccum += delta; stuck = 0; }
    else stuck++;
    lastProgress = pr.progress;
    if (stuck > STUCK_LIMIT) { alive = false; break; }
    if (pr.progress >= 0.20 && pr.progress <= 0.35) passedQ = true;
    if (passedQ && pr.progress >= 0.45 && pr.progress <= 0.60) passedH = true;
    if (passedH && pr.progress >= 0.70 && pr.progress <= 0.85) passed3Q = true;
    const allHit = passedQ && passedH && passed3Q;
    if (allHit && (progressAccum >= LAP_COMPLETION_PROGRESS || wrap)) {
      finished = true; lapTime = frame;
    }
    frame++;
  }
  return {
    score: computeScore({ totalProgress: progressAccum, finished, lapTime }),
    finished, lapTime: finished ? lapTime : null, progress: progressAccum,
  };
}

// ─── Main ──────────────────────────────────────────────
const brainData = JSON.parse(readFileSync(BRAIN_PATH, 'utf-8'));
console.log(`\n  Cross-track evaluation`);
console.log(`  ─────────────────────────`);
console.log(`  Brain: ${BRAIN_PATH}`);
console.log(`  Format: ${brainData.version === 2 ? 'v2 (LoRA)' : 'v1 (legacy)'}`);
if (brainData.adapters) {
  console.log(`  Adapters: ${Object.keys(brainData.adapters).join(', ') || '(none)'}`);
}
console.log(`  Cars per level: ${NUM_CARS}, timeout: ${TIMEOUT}\n`);

const results = [];
for (let level = 0; level < DIFFICULTY_LADDER.length; level++) {
  const step = DIFFICULTY_LADDER[level];
  const defaultWidth = TRACK_DEFAULT_WIDTHS[step.track];
  const width = step.widthDelta !== 0 ? defaultWidth + step.widthDelta : null;
  const track = new HeadlessTrack(step.track, width);

  // Two modes:
  //   1. LoRA brain: has per-level adapters. Switch to this level's adapter
  //      and evaluate. If no adapter exists for this level, the brain was
  //      never trained here → skip.
  //   2. Baseline brain: NO adapters at all. Its single mutable base has
  //      been training on whatever the LAST level was. Evaluate that base
  //      on every track to measure catastrophic forgetting.
  const isBaseline = !brainData.adapters || Object.keys(brainData.adapters).length === 0;
  const hasAdapterForLevel = brainData.adapters && brainData.adapters[String(level)];

  let brain;
  if (isBaseline) {
    // Baseline: always use base only, on every track
    brain = new NeuralCar(brainData);
    brain.setLevel(0); // ensures _effectiveWeights returns base
  } else if (level === 0 || hasAdapterForLevel) {
    brain = new NeuralCar(brainData);
    brain.setLevel(level);
  } else {
    results.push({ level, track: step.track, status: 'no adapter', scores: null });
    continue;
  }

  const scores = [];
  let finishers = 0;
  for (let c = 0; c < NUM_CARS; c++) {
    const r = runCar(brain, track, TIMEOUT, SPEED_MULT, c);
    scores.push(r.score);
    if (r.finished) finishers++;
  }
  const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
  const max = Math.max(...scores);
  results.push({ level, track: step.track, status: 'ok', avg, max, finishers, total: NUM_CARS });
}

console.log(`  Lv | Track       | Status      | Avg score | Best score | Finishers`);
console.log(`  ───┼─────────────┼─────────────┼───────────┼────────────┼──────────`);
for (const r of results) {
  if (r.status === 'no adapter') {
    console.log(`  ${String(r.level).padStart(2)} | ${r.track.padEnd(11)} | (no adapter — skipped)`);
  } else {
    console.log(
      `  ${String(r.level).padStart(2)} | ${r.track.padEnd(11)} | ok          | ` +
      `${r.avg.toFixed(2).padStart(9)} | ${r.max.toFixed(2).padStart(10)} | ${r.finishers}/${r.total}`
    );
  }
}
console.log('');
