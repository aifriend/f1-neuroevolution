import { describe, expect, it } from 'vitest';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  buildTrainArgs,
  parseArgs,
  resolveTrainingPaths,
  validateConfig,
  validateBrainArtifact,
} from '../scripts/run-full-training.js';

describe('run-full-training config', () => {
  it('parses defaults with twoPhase + clone-test gating enabled', () => {
    const cfg = parseArgs([]);
    expect(cfg.track).toBe('monaco');
    expect(cfg.cars).toBe(80);
    // 2000 gens is the post-clone-test budget for full 10-level mastery.
    expect(cfg.gens).toBe(2000);
    expect(cfg.timeout).toBe(4000);
    expect(cfg.twoPhase).toBe(true);
    // Robustness gate + clone-test gating are ON by default.
    expect(cfg.minFinisherRate).toBeCloseTo(0.10);
    expect(cfg.robustWindow).toBe(20);
    expect(cfg.cloneTestEvery).toBe(10);
    expect(cfg.cloneTestCars).toBe(16);
    expect(cfg.cloneTestK).toBe(3);
  });

  it('parses booleans and numeric overrides', () => {
    const cfg = parseArgs([
      '--track',
      'suzuka',
      '--cars',
      '120',
      '--noTwoPhase',
      '--noLora',
      '--mutation',
      '0.12',
      '--width',
      '18',
    ]);
    expect(cfg.track).toBe('suzuka');
    expect(cfg.cars).toBe(120);
    expect(cfg.twoPhase).toBe(false);
    expect(cfg.noLora).toBe(true);
    expect(cfg.mutation).toBeCloseTo(0.12);
    expect(cfg.width).toBe(18);
  });

  it('rejects invalid track values', () => {
    expect(() => validateConfig({ ...parseArgs([]), track: 'bad-track' })).toThrow(
      /Unsupported track/
    );
  });

  it('builds node train.js arguments with toggles', () => {
    const cfg = parseArgs([
      '--track',
      'monaco',
      '--cars',
      '50',
      '--gens',
      '200',
      '--noTwoPhase',
      '--noLora',
      '--width',
      '16',
      '--load',
      'models/base.json',
    ]);
    const args = buildTrainArgs(cfg, '/tmp/out.json');
    expect(args).toContain('--track');
    expect(args).toContain('monaco');
    expect(args).not.toContain('--twoPhase');
    expect(args).toContain('--noLora');
    expect(args).toContain('--load');
    expect(args).toContain('models/base.json');
    expect(args).toContain('--width');
    expect(args).toContain('16');
  });

  it('rejects unknown flags', () => {
    expect(() => parseArgs(['--notARealFlag'])).toThrow(/Unknown flag/);
  });

  it('rejects missing value flags', () => {
    expect(() => parseArgs(['--cars'])).toThrow(/requires a value/);
  });

  it('validates numeric bounds', () => {
    expect(() => validateConfig({ ...parseArgs([]), cars: 1 })).toThrow(/--cars/);
    expect(() => validateConfig({ ...parseArgs([]), speed: 0 })).toThrow(/--speed/);
    expect(() => validateConfig({ ...parseArgs([]), slow: 2, speed: 1 })).toThrow(/--slow/);
  });

  it('rejects output/load paths that escape project root', () => {
    const cfg = parseArgs([]);
    expect(() =>
      resolveTrainingPaths('/tmp/project', { ...cfg, output: '../escape.json' })
    ).toThrow(/project root/);
    expect(() =>
      resolveTrainingPaths('/tmp/project', { ...cfg, load: '../escape.json' })
    ).toThrow(/project root/);
  });
});

describe('run-full-training artifact validation', () => {
  it('rejects missing artifact files', () => {
    expect(() => validateBrainArtifact('/tmp/this-file-does-not-exist.json')).toThrow(
      /output artifact is missing/i
    );
  });

  it('rejects malformed JSON artifacts', () => {
    const dir = mkdtempSync(join(tmpdir(), 'runner-artifact-'));
    const badPath = join(dir, 'bad.json');
    try {
      writeFileSync(badPath, '{not-json');
      expect(() => validateBrainArtifact(badPath)).toThrow();
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('rejects artifacts missing base weights', () => {
    const dir = mkdtempSync(join(tmpdir(), 'runner-artifact-'));
    const badPath = join(dir, 'incomplete.json');
    try {
      writeFileSync(badPath, JSON.stringify({ base: { w1: [] } }));
      expect(() => validateBrainArtifact(badPath)).toThrow(/missing required base weights/i);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
