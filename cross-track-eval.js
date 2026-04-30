#!/usr/bin/env node
// Measure how well a saved brain performs on EVERY curriculum level.
// Loads a v2 brain JSON (with base + adapters), runs N evaluation laps on
// each level by switching the active adapter, and reports per-level scores.
//
// Uses the SAME HeadlessCar + runGeneration as train.js — guaranteed
// physics parity between training and evaluation. A brain that lapped
// during training will lap during eval (no more eval-vs-train gap).
//
// Usage:
//   node cross-track-eval.js path/to/brain.json
//   node cross-track-eval.js path/to/brain.json --cars 16 --timeout 4000

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

const { NeuralCar } = await import('./js/nn.js');
const { DIFFICULTY_LADDER, TRACK_DEFAULT_WIDTHS } = await import('./js/evolution-core.js');
const { HeadlessTrack } = await import('./js/headless-track.js');
const { HeadlessCar, runGeneration } = await import('./js/headless-car.js');

// ─── Per-level evaluation ─────────────────────────────
// Spawn `count` cars all running the same brain at the requested level,
// run until everyone finishes/dies/timeouts, return the population stats.
function evaluateLevel(brainData, level, count, timeout, speedMult) {
  const step = DIFFICULTY_LADDER[level];
  const defaultWidth = TRACK_DEFAULT_WIDTHS[step.track];
  const width = step.widthDelta !== 0 ? defaultWidth + step.widthDelta : null;
  const track = new HeadlessTrack(step.track, width);

  // Two modes:
  //   1. LoRA brain with adapter for this level → use that adapter.
  //   2. Baseline brain (no adapters) → use base only.
  const isBaseline = !brainData.adapters || Object.keys(brainData.adapters).length === 0;
  const hasAdapterForLevel = brainData.adapters && brainData.adapters[String(level)];
  if (!isBaseline && level > 0 && !hasAdapterForLevel) {
    return { level, track: step.track, status: 'no adapter' };
  }

  const cars = [];
  for (let i = 0; i < count; i++) {
    const brain = new NeuralCar(brainData);
    brain.setLevel(isBaseline ? 0 : level);
    cars.push(new HeadlessCar(track, brain, i, speedMult));
  }
  runGeneration(track, cars, timeout);

  const finishers = cars.filter((c) => c.finished).length;
  const scores = cars.map((c) => c.score);
  const avg = scores.reduce((a, b) => a + b, 0) / Math.max(1, scores.length);
  const max = scores.length ? Math.max(...scores) : 0;
  return {
    level,
    track: step.track,
    status: 'ok',
    avg,
    max,
    finishers,
    total: count,
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
console.log(`  Cars per level: ${NUM_CARS}, timeout: ${TIMEOUT}, speed: ${SPEED_MULT}x\n`);

const results = [];
for (let level = 0; level < DIFFICULTY_LADDER.length; level++) {
  results.push(evaluateLevel(brainData, level, NUM_CARS, TIMEOUT, SPEED_MULT));
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
