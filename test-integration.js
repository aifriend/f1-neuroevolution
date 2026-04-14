#!/usr/bin/env node
// Integration tests for F1 neuroevolution simulation
// Tests the full pipeline: track → cars → sensors → NN → evolution → scoring
// Uses seeded random for reproducible results
//
// Usage: node test-integration.js

const { computeScore, computeAdaptiveMutation, tournamentSelect, getTieredMutationRate } = await import('./js/evolution-core.js');

let passed = 0, failed = 0, total = 0;

function assert(condition, name, detail = '') {
  total++;
  if (condition) {
    passed++;
    console.log(`  ✓ ${name}`);
  } else {
    failed++;
    console.log(`  ✗ ${name}${detail ? ' — ' + detail : ''}`);
  }
}

// ─── Seeded PRNG for reproducible tests ──────────────
let seed = 42;
function seededRandom() {
  seed = (seed * 16807 + 0) % 2147483647;
  return (seed - 1) / 2147483646;
}

// Override Math.random for reproducible training
const originalRandom = Math.random;
function useSeed(s) { seed = s; Math.random = seededRandom; }
function useRealRandom() { Math.random = originalRandom; }

// ─── Import track and car physics from train.js ─────
// We inline the minimal physics needed for integration testing

const NUM_SENSORS = 9;
const NUM_INPUTS = NUM_SENSORS + 1;
const HIDDEN_SIZE = 16;
const SENSOR_LENGTH = 220;
const STUCK_LIMIT = 120;
const LAP_COMPLETION_PROGRESS = 1.0;
const SENSOR_ANGLES = [
  -Math.PI / 2, -Math.PI * 3 / 8, -Math.PI / 4, -Math.PI / 8,
  0,
  Math.PI / 8, Math.PI / 4, Math.PI * 3 / 8, Math.PI / 2,
];

const TRACKS = {
  monaco: {
    name: 'Monaco', width: 28,
    points: [[0,0],[80,5],[150,25],[200,60],[230,110],[240,170],[220,230],[180,270],[130,280],[70,260],[20,220],[-20,170],[-40,120],[-50,70],[-45,30],[-25,10]],
  },
  suzuka: {
    name: 'Suzuka', width: 30,
    points: [[0,0],[100,5],[190,25],[260,70],[300,140],[310,215],[290,280],[250,330],[190,360],[120,365],[60,340],[20,295],[-10,240],[-30,180],[-55,125],[-65,70],[-55,25],[-30,5]],
  },
  silverstone: {
    name: 'Silverstone', width: 32,
    points: [[0,0],[100,15],[195,50],[270,110],[310,190],[315,275],[285,350],[230,400],[165,430],[90,435],[15,410],[-45,365],[-85,305],[-105,240],[-110,170],[-95,105],[-70,50],[-35,15]],
  },
};

function interpolate(pts, steps) {
  const result = []; const n = pts.length;
  for (let i = 0; i < n; i++) {
    const p0=pts[(i-1+n)%n],p1=pts[i],p2=pts[(i+1)%n],p3=pts[(i+2)%n];
    for (let s = 0; s < steps; s++) {
      const t=s/steps,t2=t*t,t3=t2*t;
      result.push([(-0.5*t3+t2-0.5*t)*p0[0]+(1.5*t3-2.5*t2+1)*p1[0]+(-1.5*t3+2*t2+0.5*t)*p2[0]+(0.5*t3-0.5*t2)*p3[0],
                    (-0.5*t3+t2-0.5*t)*p0[1]+(1.5*t3-2.5*t2+1)*p1[1]+(-1.5*t3+2*t2+0.5*t)*p2[1]+(0.5*t3-0.5*t2)*p3[1]]);
    }
  }
  return result;
}

function buildTrack(type) {
  const cfg = TRACKS[type];
  // Use enough interpolation steps so point spacing < gridSize (5 units)
  const stepsPerSeg = 20;
  const points = interpolate(cfg.points, stepsPerSeg);
  const n = points.length;
  const tangents = [];
  for (let i = 0; i < n; i++) {
    const next = points[(i+1)%n], curr = points[i];
    const dx = next[0]-curr[0], dz = next[1]-curr[1];
    const len = Math.sqrt(dx*dx+dz*dz)||1;
    tangents.push([dx/len, dz/len]);
  }
  const grid = {}; const gs = 5;
  for (let i = 0; i < n; i++) {
    const px = points[i][0], pz = points[i][1];
    const t = tangents[i], nx = -t[1], nz = t[0];
    for (let w = -cfg.width; w <= cfg.width; w += 1) {
      grid[Math.floor((px+nx*w)/gs)+','+Math.floor((pz+nz*w)/gs)] = true;
    }
  }
  const sp = points[0], st = tangents[0];
  return {
    name: cfg.name, points, tangents, grid, gridSize: gs, width: cfg.width,
    startX: sp[0], startZ: sp[1],
    startAngle: Math.atan2(st[1], st[0]),
    isOnTrack(x, z) { return grid[Math.floor(x/gs)+','+Math.floor(z/gs)] === true; },
    getProgress(x, z) {
      let minDist = Infinity, bestIdx = 0;
      for (let i = 0; i < n; i += 4) {
        const dx = x-points[i][0], dz = z-points[i][1];
        const dist = dx*dx+dz*dz;
        if (dist < minDist) { minDist = dist; bestIdx = i; }
      }
      for (let off = -6; off <= 6; off++) {
        const i = ((bestIdx+off)%n+n)%n;
        const dx = x-points[i][0], dz = z-points[i][1];
        const dist = dx*dx+dz*dz;
        if (dist < minDist) { minDist = dist; bestIdx = i; }
      }
      return bestIdx / n;
    },
    getProgressLocal(x, z, hintIdx, r = 20) {
      let minDist = Infinity, bestIdx = hintIdx;
      for (let off = -r; off <= r; off++) {
        const i = ((hintIdx+off)%n+n)%n;
        const dx = x-points[i][0], dz = z-points[i][1];
        const dist = dx*dx+dz*dz;
        if (dist < minDist) { minDist = dist; bestIdx = i; }
      }
      return { progress: bestIdx/n, idx: bestIdx };
    },
    castRay(x, z, angle) {
      const cosA = Math.cos(angle), sinA = Math.sin(angle);
      for (let d = 2; d < SENSOR_LENGTH; d += 2) {
        if (!this.isOnTrack(x+cosA*d, z+sinA*d)) return d;
      }
      return SENSOR_LENGTH;
    },
  };
}

// ═══════════════════════════════════════════════════════
console.log('\n  F1 Neuroevolution — Integration Tests');
console.log('  ═══════════════════════════════════════\n');

// ─── TEST 1: Track geometry ──────────────────────────
console.log('  1. Track Geometry');
for (const [type, cfg] of Object.entries(TRACKS)) {
  const track = buildTrack(type);

  // Start position is on track
  assert(track.isOnTrack(track.startX, track.startZ),
    `${type}: start position is on track`);

  // First 10 car grid positions are on track
  let allOnTrack = true;
  for (let i = 0; i < 10; i++) {
    const t = track.tangents[0], nx = -t[1], nz = t[0];
    const row = Math.floor(i/2), col = (i%2)-0.5;
    const x = track.startX - t[0]*row*4 + nx*col*8;
    const z = track.startZ - t[1]*row*4 + nz*col*8;
    if (!track.isOnTrack(x, z)) allOnTrack = false;
  }
  assert(allOnTrack, `${type}: first 10 grid positions on track`);

  // Car going straight survives at least 5 frames
  let x = track.startX, z = track.startZ, angle = track.startAngle;
  let survived = 0;
  for (let f = 0; f < 10; f++) {
    const speed = 2.5;
    const nx = x + Math.cos(angle)*speed, nz = z + Math.sin(angle)*speed;
    if (!track.isOnTrack(nx, nz)) break;
    x = nx; z = nz; survived++;
  }
  assert(survived >= 5, `${type}: car survives ≥5 frames going straight`,
    `survived ${survived}`);

  // No self-intersection within track width
  let closeCount = 0;
  const pts = track.points;
  for (let i = 0; i < pts.length; i += 8) {
    for (let j = i + 30; j < pts.length; j += 8) {
      const idxDist = Math.min(j-i, pts.length-(j-i));
      if (idxDist < 30) continue;
      const dx = pts[i][0]-pts[j][0], dz = pts[i][1]-pts[j][1];
      // Detect true self-crossing (< 5 units = roads physically on top of each other)
      if (Math.sqrt(dx*dx+dz*dz) < 5) closeCount++;
    }
  }
  assert(closeCount === 0, `${type}: no track self-intersection`,
    closeCount > 0 ? `${closeCount} close pairs` : '');
}

// ─── TEST 2: Neural Network dimensions ──────────────
console.log('\n  2. Neural Network Dimensions');
useSeed(123);

function randomGaussian(mean = 0, std = 1) {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return mean + std * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

// Create a random brain
const scale1 = Math.sqrt(2 / (NUM_INPUTS + HIDDEN_SIZE));
const w1 = []; for (let i = 0; i < NUM_INPUTS; i++) { const r = []; for (let j = 0; j < HIDDEN_SIZE; j++) r.push(randomGaussian(0, scale1)); w1.push(r); }
const b1 = new Array(HIDDEN_SIZE).fill(0);
const scale2 = Math.sqrt(2 / (HIDDEN_SIZE + 2));
const w2 = []; for (let i = 0; i < HIDDEN_SIZE; i++) { const r = []; for (let j = 0; j < 2; j++) r.push(randomGaussian(0, scale2)); w2.push(r); }
const b2 = new Array(2).fill(0);

assert(w1.length === NUM_INPUTS, `w1 has ${NUM_INPUTS} rows (inputs)`, `got ${w1.length}`);
assert(w1[0].length === HIDDEN_SIZE, `w1[0] has ${HIDDEN_SIZE} cols (hidden)`, `got ${w1[0].length}`);
assert(b1.length === HIDDEN_SIZE, `b1 has ${HIDDEN_SIZE} biases`);
assert(w2.length === HIDDEN_SIZE, `w2 has ${HIDDEN_SIZE} rows`);
assert(w2[0].length === 2, `w2[0] has 2 cols (outputs)`);
assert(b2.length === 2, `b2 has 2 biases`);

const totalParams = NUM_INPUTS*HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE*2 + 2;
assert(totalParams === 210, `total params = 210`, `got ${totalParams}`);

// Think produces valid outputs
const testInputs = new Array(NUM_INPUTS).fill(0.5);
const hidden = [];
for (let j = 0; j < HIDDEN_SIZE; j++) {
  let sum = b1[j];
  for (let i = 0; i < NUM_INPUTS; i++) sum += testInputs[i] * w1[i][j];
  hidden.push(Math.tanh(sum));
}
const output = [];
for (let j = 0; j < 2; j++) {
  let sum = b2[j];
  for (let i = 0; i < HIDDEN_SIZE; i++) sum += hidden[i] * w2[i][j];
  output.push(Math.tanh(sum));
}
assert(output.length === 2, 'think() produces 2 outputs');
assert(Math.abs(output[0]) <= 1, 'steer output in [-1, 1]', `got ${output[0]}`);
assert(Math.abs(output[1]) <= 1, 'gas output in [-1, 1]', `got ${output[1]}`);

// ─── TEST 3: Scoring function ───────────────────────
console.log('\n  3. Scoring Function');
assert(computeScore({ totalProgress: 0.5, finished: false }) === 0.5,
  'unfinished car: score = progress');
assert(computeScore({ totalProgress: 0, finished: false }) === 0,
  'zero progress: score = 0');

const finishedCar = { totalProgress: 1.0, finished: true, lapTime: 200 };
const finishedScore = computeScore(finishedCar);
assert(finishedScore > 3000, 'finished car: score > 3000', `got ${finishedScore.toFixed(0)}`);

const fasterCar = { totalProgress: 1.0, finished: true, lapTime: 100 };
const fasterScore = computeScore(fasterCar);
assert(fasterScore > finishedScore, 'faster car scores higher than slower car',
  `${fasterScore.toFixed(0)} vs ${finishedScore.toFixed(0)}`);

// ─── TEST 4: Adaptive mutation ──────────────────────
console.log('\n  4. Adaptive Mutation');
const mut1 = computeAdaptiveMutation({
  bestScore: 100, allTimeBest: 50, stagnantGens: 0,
  baseMutation: 0.08, currentMutation: 0.08,
});
assert(mut1.improved === true, 'improvement detected when score > allTimeBest');
assert(mut1.mutationRate === 0.08, 'mutation resets to base on improvement');
assert(mut1.stagnantGens === 0, 'stagnant gens reset on improvement');

const mut2 = computeAdaptiveMutation({
  bestScore: 50, allTimeBest: 100, stagnantGens: 59,
  baseMutation: 0.08, currentMutation: 0.3,
});
assert(mut2.restart === true, 'restart triggered at 60 stagnant gens');
assert(mut2.mutationRate === 0.08, 'mutation resets to base on restart');

const mut3 = computeAdaptiveMutation({
  bestScore: 50, allTimeBest: 100, stagnantGens: 10,
  baseMutation: 0.08, currentMutation: 0.08,
});
assert(mut3.mutationRate > 0.08, 'mutation increases after 8+ stagnant gens');
assert(mut3.mutationRate <= 0.25, 'mild mutation capped at 0.25',
  `got ${mut3.mutationRate.toFixed(3)}`);

// ─── TEST 5: Tournament selection ───────────────────
console.log('\n  5. Tournament Selection');
useSeed(999);
const mockCars = Array.from({length: 20}, (_, i) => ({ score: i * 10 }));
let bestWins = 0;
for (let trial = 0; trial < 100; trial++) {
  const selected = tournamentSelect(mockCars, 3);
  if (selected.score >= 150) bestWins++;
}
assert(bestWins > 30, 'tournament selects top performers most often',
  `top selected ${bestWins}/100 trials`);
assert(bestWins < 90, 'tournament allows weaker cars sometimes',
  `top selected ${bestWins}/100 trials`);

// ─── TEST 6: Tiered mutation rates ──────────────────
console.log('\n  6. Tiered Mutation Rates');
const low = getTieredMutationRate(5, 80, 0.08);
const mid = getTieredMutationRate(50, 80, 0.08);
const high = getTieredMutationRate(75, 80, 0.08);
assert(low < mid, 'fine-tune tier < standard tier', `${low.toFixed(3)} < ${mid.toFixed(3)}`);
assert(mid < high, 'standard tier < wild tier', `${mid.toFixed(3)} < ${high.toFixed(3)}`);
assert(low < 0.02, 'fine-tune tier is very gentle', `got ${low.toFixed(4)}`);

// ─── TEST 7: Full training run (seeded, deterministic) ──
console.log('\n  7. Full Training Integration (seeded)');
useSeed(42);

// Quick 100-gen training on Monaco
const track = buildTrack('monaco');
let cars = [];
for (let i = 0; i < 40; i++) {
  // Create minimal car with random brain
  const t = track.tangents[0], nx = -t[1], nz = t[0];
  const row = Math.floor(i/2), col = (i%2)-0.5;
  const brain_w1 = []; for (let r = 0; r < NUM_INPUTS; r++) { const ro = []; for (let c = 0; c < HIDDEN_SIZE; c++) ro.push(randomGaussian(0, scale1)); brain_w1.push(ro); }
  const brain_b1 = new Array(HIDDEN_SIZE).fill(0);
  const brain_w2 = []; for (let r = 0; r < HIDDEN_SIZE; r++) { const ro = []; for (let c = 0; c < 2; c++) ro.push(randomGaussian(0, scale2)); brain_w2.push(ro); }
  const brain_b2 = new Array(2).fill(0);

  cars.push({
    x: track.startX - t[0]*row*4 + nx*col*8,
    z: track.startZ - t[1]*row*4 + nz*col*8,
    angle: track.startAngle,
    speed: 0,
    sensors: new Array(NUM_SENSORS).fill(0),
    brain: { w1: brain_w1, b1: brain_b1, w2: brain_w2, b2: brain_b2,
      think(inputs) {
        const hid = [];
        for (let j = 0; j < HIDDEN_SIZE; j++) {
          let s = this.b1[j];
          for (let ii = 0; ii < inputs.length; ii++) s += inputs[ii] * this.w1[ii][j];
          hid.push(Math.tanh(s));
        }
        const out = [];
        for (let j = 0; j < 2; j++) {
          let s = this.b2[j];
          for (let ii = 0; ii < HIDDEN_SIZE; ii++) s += hid[ii] * this.w2[ii][j];
          out.push(Math.tanh(s));
        }
        return { steer: out[0], gas: out[1] };
      },
      getWeights() {
        return { w1: this.w1.map(r=>[...r]), b1: [...this.b1], w2: this.w2.map(r=>[...r]), b2: [...this.b2] };
      },
      mutate(rate) {
        for (let i = 0; i < this.w1.length; i++) for (let j = 0; j < this.w1[i].length; j++) this.w1[i][j] += randomGaussian(0,1)*rate;
        for (let i = 0; i < this.b1.length; i++) this.b1[i] += randomGaussian(0,1)*rate;
        for (let i = 0; i < this.w2.length; i++) for (let j = 0; j < this.w2[i].length; j++) this.w2[i][j] += randomGaussian(0,1)*rate;
        for (let i = 0; i < this.b2.length; i++) this.b2[i] += randomGaussian(0,1)*rate;
      }
    },
    alive: true, finished: false, frameCounter: 0, stuckFrames: 0,
    totalProgress: 0, progressAccum: 0, lapTime: 0, score: 0,
    lastProgressIdx: Math.round(track.getProgress(
      track.startX - t[0]*row*4 + nx*col*8,
      track.startZ - t[1]*row*4 + nz*col*8
    ) * track.points.length),
    lastProgress: track.getProgress(
      track.startX - t[0]*row*4 + nx*col*8,
      track.startZ - t[1]*row*4 + nz*col*8
    ),
  });
}

// Run 100 frames on gen 1
let anyAlive = true;
for (let frame = 0; frame < 500 && anyAlive; frame++) {
  anyAlive = false;
  for (const car of cars) {
    if (!car.alive || car.finished) continue;
    anyAlive = true;
    // Cast sensors
    for (let i = 0; i < SENSOR_ANGLES.length; i++) {
      car.sensors[i] = track.castRay(car.x, car.z, car.angle + SENSOR_ANGLES[i]) / SENSOR_LENGTH;
    }
    // Think
    const inputs = [...car.sensors, car.speed / 8.1];
    const decision = car.brain.think(inputs);
    car.angle += decision.steer * 0.08;
    car.speed = (2.5 + (decision.gas + 1) * 2.8);
    const newX = car.x + Math.cos(car.angle) * car.speed;
    const newZ = car.z + Math.sin(car.angle) * car.speed;
    const midX = (car.x+newX)*0.5, midZ = (car.z+newZ)*0.5;
    if (!track.isOnTrack(midX, midZ) || !track.isOnTrack(newX, newZ)) { car.alive = false; continue; }
    car.x = newX; car.z = newZ;
    // Progress
    const { progress, idx } = track.getProgressLocal(car.x, car.z, car.lastProgressIdx);
    car.lastProgressIdx = idx;
    let delta = progress - car.lastProgress;
    if (delta < -0.5) delta += 1.0;
    if (delta > 0.5) delta -= 1.0;
    if (delta > 0.05) delta = 0.05;
    if (delta < -0.05) delta = -0.05;
    if (delta > 0) { car.progressAccum += delta; car.stuckFrames = 0; }
    else car.stuckFrames++;
    car.lastProgress = progress;
    car.totalProgress = car.progressAccum;
    if (car.stuckFrames > STUCK_LIMIT) { car.alive = false; continue; }
    car.frameCounter++;
    if (car.progressAccum >= LAP_COMPLETION_PROGRESS) { car.finished = true; car.lapTime = car.frameCounter; }
    car.score = computeScore(car);
  }
}

const aliveCars = cars.filter(c => c.alive || c.finished);
const maxProgress = Math.max(...cars.map(c => c.totalProgress));
const avgProgress = cars.reduce((s, c) => s + c.totalProgress, 0) / cars.length;
const finishers = cars.filter(c => c.finished);

assert(maxProgress > 0, 'at least some cars made progress',
  `max progress: ${(maxProgress*100).toFixed(1)}%`);
assert(avgProgress > 0, 'average progress > 0',
  `avg: ${(avgProgress*100).toFixed(1)}%`);
assert(cars.some(c => c.score > 0), 'at least one car scored > 0');

// ─── TEST 8: Sensor coverage ────────────────────────
console.log('\n  8. Sensor Coverage');
assert(SENSOR_ANGLES.length === 9, '9 sensor angles defined');
assert(Math.abs(SENSOR_ANGLES[0] - (-Math.PI/2)) < 0.001, 'leftmost sensor at -90°');
assert(Math.abs(SENSOR_ANGLES[4]) < 0.001, 'center sensor at 0°');
assert(Math.abs(SENSOR_ANGLES[8] - Math.PI/2) < 0.001, 'rightmost sensor at +90°');

// Check sensors detect walls
const testTrack = buildTrack('monaco');
const centerRay = testTrack.castRay(testTrack.startX, testTrack.startZ, testTrack.startAngle);
assert(centerRay > 10, 'center sensor detects open track ahead', `ray = ${centerRay}`);
const sideRay = testTrack.castRay(testTrack.startX, testTrack.startZ, testTrack.startAngle + Math.PI/2);
assert(sideRay < centerRay, 'side sensor detects closer wall than forward');

// ─── SUMMARY ────────────────────────────────────────
useRealRandom();
console.log('\n  ═══════════════════════════════════════');
console.log(`  Results: ${passed}/${total} passed, ${failed} failed`);
if (failed === 0) {
  console.log('  All tests passed! ✓');
} else {
  console.log(`  ${failed} test(s) FAILED ✗`);
  process.exit(1);
}
console.log('');
