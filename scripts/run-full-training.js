#!/usr/bin/env node
import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { TRACK_IDS } from '../js/track-data.js';

const TRACKS = new Set(TRACK_IDS);

const INT_FLAGS = ['cars', 'gens', 'timeout', 'rank', 'width', 'evalCars', 'evalTimeout', 'robustWindow', 'cloneTestEvery', 'cloneTestCars', 'cloneTestK', 'startLevel'];
const FLOAT_FLAGS = ['mutation', 'speed', 'slow', 'softFreeze', 'minFinisherRate'];
const VALUE_FLAGS = ['track', ...INT_FLAGS, ...FLOAT_FLAGS, 'output', 'load'];
const BOOL_FLAGS = ['noTwoPhase', 'noLora', 'skipTests', 'skipEval', 'dryRun'];
const KNOWN_FLAGS = new Set([...VALUE_FLAGS, ...BOOL_FLAGS]);

function parseArgs(argv) {
  const args = argv.slice();
  const out = {
    track: 'monaco',
    cars: 80,
    // 2000 gens is the empirical budget for a full 10-level curriculum with
    // clone-test gating ON: ~400 gens to find a lapping L0 brain, then ~150
    // gens per validated level escalation. Bumped from the pre-clone-test
    // default of 1000 (which got stuck at L5 in the failing v3 e2e run).
    gens: 2000,
    mutation: 0.05,
    // 4000-frame timeout matches what the v8 silverstone/serpentine dwells
    // used and clears even the longer ironcliff laps comfortably. Pre-v8
    // default was 3500; clone-test runs benefit from the extra headroom.
    timeout: 4000,
    speed: 1,
    slow: 0.5,
    rank: 2,
    softFreeze: 0,
    twoPhase: true,
    noLora: false,
    output: '',
    load: '',
    width: null,
    skipTests: false,
    skipEval: false,
    dryRun: false,
    // Robustness controls (forwarded to train.js).
    minFinisherRate: 0.10,
    robustWindow: 20,
    // Clone-test elite gating (forwarded to train.js). Every K gens we
    // re-spawn the top candidates as fresh-clone cars to verify the brain
    // we're saving actually generalizes across spawn positions.
    cloneTestEvery: 10,
    cloneTestCars: 16,
    cloneTestK: 3,
    // Optional explicit starting level (forwarded to train.js for dwell runs)
    startLevel: null,
    // Post-training cross-track-eval defaults.
    evalCars: 12,
    evalTimeout: 4000,
  };

  for (let i = 0; i < args.length; i++) {
    const token = args[i];
    if (!token.startsWith('--')) continue;
    const name = token.slice(2);
    if (!KNOWN_FLAGS.has(name)) {
      throw new Error(`Unknown flag: --${name}`);
    }
  }

  const readValue = (name) => {
    const idx = args.indexOf(`--${name}`);
    if (idx === -1) return null;
    if (idx + 1 >= args.length || args[idx + 1].startsWith('--')) {
      throw new Error(`Flag --${name} requires a value`);
    }
    return args[idx + 1];
  };

  const has = (name) => args.includes(`--${name}`);

  const track = readValue('track');
  if (track) out.track = track;

  for (const name of INT_FLAGS) {
    const value = readValue(name);
    if (value === null) continue;
    const parsed = Number.parseInt(value, 10);
    if (!Number.isFinite(parsed)) throw new Error(`Invalid integer for --${name}: ${value}`);
    out[name] = parsed;
  }

  for (const name of FLOAT_FLAGS) {
    const value = readValue(name);
    if (value === null) continue;
    const parsed = Number.parseFloat(value);
    if (!Number.isFinite(parsed)) throw new Error(`Invalid number for --${name}: ${value}`);
    out[name] = parsed;
  }

  const output = readValue('output');
  if (output) out.output = output;

  const load = readValue('load');
  if (load) out.load = load;

  out.twoPhase = !has('noTwoPhase');
  out.noLora = has('noLora');
  out.skipTests = has('skipTests');
  out.skipEval = has('skipEval');
  out.dryRun = has('dryRun');

  return out;
}

function assertPathWithinProject(projectRoot, candidatePath, flagName) {
  const root = resolve(projectRoot);
  const target = resolve(projectRoot, candidatePath);
  const rootWithSep = root.endsWith('/') ? root : `${root}/`;
  if (target !== root && !target.startsWith(rootWithSep)) {
    throw new Error(`${flagName} must stay within project root`);
  }
  return target;
}

function validateConfig(config) {
  if (!TRACKS.has(config.track)) {
    throw new Error(`Unsupported track: ${config.track}`);
  }
  if (config.cars < 2 || config.cars > 1000) {
    throw new Error(`--cars must be between 2 and 1000, got ${config.cars}`);
  }
  if (config.gens < 1 || config.gens > 200000) {
    throw new Error(`--gens must be between 1 and 200000, got ${config.gens}`);
  }
  if (config.mutation < 0 || config.mutation > 1) {
    throw new Error(`--mutation must be in [0, 1], got ${config.mutation}`);
  }
  if (config.timeout < 200 || config.timeout > 50000) {
    throw new Error(`--timeout must be between 200 and 50000, got ${config.timeout}`);
  }
  if (config.speed <= 0 || config.speed > 25) {
    throw new Error(`--speed must be in (0, 25], got ${config.speed}`);
  }
  if (config.slow <= 0 || config.slow > config.speed) {
    throw new Error(`--slow must be > 0 and <= --speed, got ${config.slow}`);
  }
  if (config.rank < 1 || config.rank > 64) {
    throw new Error(`--rank must be between 1 and 64, got ${config.rank}`);
  }
  if (config.softFreeze < 0 || config.softFreeze > 1) {
    throw new Error(`--softFreeze must be in [0, 1], got ${config.softFreeze}`);
  }
  if (config.width !== null && (config.width < 4 || config.width > 200)) {
    throw new Error(`--width must be between 4 and 200, got ${config.width}`);
  }
  if (config.minFinisherRate < 0 || config.minFinisherRate > 1) {
    throw new Error(`--minFinisherRate must be in [0, 1], got ${config.minFinisherRate}`);
  }
  if (config.robustWindow < 1 || config.robustWindow > 1000) {
    throw new Error(`--robustWindow must be in [1, 1000], got ${config.robustWindow}`);
  }
  if (config.evalCars < 1 || config.evalCars > 200) {
    throw new Error(`--evalCars must be in [1, 200], got ${config.evalCars}`);
  }
  if (config.evalTimeout < 200 || config.evalTimeout > 50000) {
    throw new Error(`--evalTimeout must be in [200, 50000], got ${config.evalTimeout}`);
  }
  if (config.cloneTestEvery < 0 || config.cloneTestEvery > 10000) {
    throw new Error(`--cloneTestEvery must be in [0, 10000], got ${config.cloneTestEvery}`);
  }
  if (config.cloneTestCars < 1 || config.cloneTestCars > 200) {
    throw new Error(`--cloneTestCars must be in [1, 200], got ${config.cloneTestCars}`);
  }
  if (config.cloneTestK < 1 || config.cloneTestK > 50) {
    throw new Error(`--cloneTestK must be in [1, 50], got ${config.cloneTestK}`);
  }
  if (config.startLevel !== null && (config.startLevel < 0 || config.startLevel > 30)) {
    throw new Error(`--startLevel must be in [0, 30], got ${config.startLevel}`);
  }
}

function buildTrainArgs(config, outputPath) {
  const args = [
    'train.js',
    '--track',
    config.track,
    '--cars',
    String(config.cars),
    '--gens',
    String(config.gens),
    '--mutation',
    String(config.mutation),
    '--timeout',
    String(config.timeout),
    '--speed',
    String(config.speed),
    '--slow',
    String(config.slow),
    '--rank',
    String(config.rank),
    '--softFreeze',
    String(config.softFreeze),
    '--output',
    outputPath,
  ];

  if (config.twoPhase) args.push('--twoPhase');
  if (config.noLora) args.push('--noLora');
  if (config.load) args.push('--load', config.load);
  if (config.width !== null) args.push('--width', String(config.width));
  args.push('--minFinisherRate', String(config.minFinisherRate));
  args.push('--robustWindow', String(config.robustWindow));
  args.push('--cloneTestEvery', String(config.cloneTestEvery));
  args.push('--cloneTestCars', String(config.cloneTestCars));
  args.push('--cloneTestK', String(config.cloneTestK));
  if (Number.isFinite(config.startLevel)) {
    args.push('--startLevel', String(config.startLevel));
  }

  return args;
}

// Build the cross-track-eval invocation. Eval runs the saved brain on every
// curriculum level it has an adapter for and prints per-level finisher rates.
// This is the "did we actually retain capability?" check.
function buildEvalArgs(config, outputPath) {
  return [
    'cross-track-eval.js',
    outputPath,
    '--cars',
    String(config.evalCars),
    '--timeout',
    String(config.evalTimeout),
    '--speed',
    String(config.speed),
  ];
}

function validateBrainArtifact(outputPath) {
  if (!existsSync(outputPath)) {
    throw new Error(`Training finished but output artifact is missing: ${outputPath}`);
  }
  const raw = readFileSync(outputPath, 'utf8');
  const parsed = JSON.parse(raw);
  if (!parsed || typeof parsed !== 'object') {
    throw new Error('Output brain artifact is not a valid JSON object');
  }
  if (!parsed.base || !parsed.base.w1 || !parsed.base.w2) {
    throw new Error('Output brain artifact is missing required base weights');
  }
}

function resolveTrainingPaths(projectRoot, config) {
  const ts = new Date().toISOString().replace(/[:.]/g, '-');
  const defaultOutput = resolve(projectRoot, 'models', `best-brain-${config.track}-${ts}.json`);
  const outputPath = config.output
    ? assertPathWithinProject(projectRoot, config.output, '--output')
    : defaultOutput;
  const loadPath = config.load ? assertPathWithinProject(projectRoot, config.load, '--load') : '';
  if (loadPath && !existsSync(loadPath)) {
    throw new Error(`Warm-start file not found: ${loadPath}`);
  }
  return { outputPath, loadPath };
}

function runCommand(command, args, cwd, label) {
  const result = spawnSync(command, args, {
    cwd,
    stdio: 'inherit',
    env: process.env,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`${label} failed with exit code ${result.status}`);
  }
}

function main(argv = process.argv.slice(2)) {
  const scriptDir = dirname(fileURLToPath(import.meta.url));
  const projectRoot = resolve(scriptDir, '..');
  const config = parseArgs(argv);
  validateConfig(config);

  const { outputPath, loadPath } = resolveTrainingPaths(projectRoot, config);

  const trainArgs = buildTrainArgs({ ...config, load: loadPath }, outputPath);

  console.log('\nF1 E2E Training Loop');
  console.log(`track=${config.track} cars=${config.cars} gens=${config.gens}`);
  console.log(`mutation=${config.mutation} timeout=${config.timeout} speed=${config.speed}`);
  console.log(`twoPhase=${config.twoPhase} noLora=${config.noLora} output=${outputPath}\n`);

  if (config.dryRun) {
    console.log(`Dry run command: node ${trainArgs.join(' ')}`);
    return;
  }

  if (!config.skipTests) {
    runCommand('npm', ['test'], projectRoot, 'Preflight tests');
  }

  runCommand('node', trainArgs, projectRoot, 'Training loop');
  validateBrainArtifact(outputPath);

  console.log(`\nTraining completed successfully.`);
  console.log(`Validated artifact: ${outputPath}\n`);

  // Auto-run cross-track-eval against the final brain so the per-level
  // retention table is part of every training run, not a manual follow-up.
  if (!config.skipEval) {
    console.log(`\n──── Cross-track retention eval (${config.evalCars} cars, timeout ${config.evalTimeout}) ────`);
    const evalArgs = buildEvalArgs(config, outputPath);
    runCommand('node', evalArgs, projectRoot, 'Cross-track eval');
  }
}

const isCliEntry = (() => {
  if (!process.argv[1]) return false;
  const currentFile = fileURLToPath(import.meta.url);
  return resolve(currentFile) === resolve(process.argv[1]);
})();

if (isCliEntry) {
  try {
    main();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(`Error: ${message}`);
    process.exit(1);
  }
}

export { buildTrainArgs, buildEvalArgs, parseArgs, resolveTrainingPaths, validateConfig, validateBrainArtifact };
