import { describe, expect, it } from 'vitest';
import {
  DIFFICULTY_LADDER,
  TRACK_DEFAULT_WIDTHS,
  computeAdaptiveMutation,
  evaluatePlateauStatus,
  getTieredMutationRate,
  PLATEAU_DEFAULTS,
  selectTopPerformers,
} from '../js/evolution-core.js';

describe('selectTopPerformers', () => {
  it('sorts by descending score and selects top 20%', () => {
    const cars = [
      { score: 4 },
      { score: 10 },
      { score: 2 },
      { score: 6 },
      { score: 1 },
    ];

    const { scored, top } = selectTopPerformers(cars, cars.length);
    expect(scored.map((c) => c.score)).toEqual([10, 6, 4, 2, 1]);
    expect(top).toHaveLength(1);
    expect(top[0].score).toBe(10);
  });
});

describe('computeAdaptiveMutation', () => {
  it('resets stagnation and mutation when score improves', () => {
    const result = computeAdaptiveMutation({
      bestScore: 20,
      allTimeBest: 10,
      stagnantGens: 3,
      baseMutation: 0.15,
      currentMutation: 0.3,
    });
    expect(result.improved).toBe(true);
    expect(result.stagnantGens).toBe(0);
    expect(result.mutationRate).toBe(0.15);
    expect(result.allTimeBest).toBe(20);
  });

  it('ramps mutation during stagnation with configured cap', () => {
    const result = computeAdaptiveMutation({
      bestScore: 10,
      allTimeBest: 10,
      stagnantGens: 8,
      baseMutation: 0.15,
      currentMutation: 0.24,
    });
    expect(result.improved).toBe(false);
    expect(result.stagnantGens).toBe(9);
    expect(result.mutationRate).toBeCloseTo(0.25);
    expect(result.restart).toBe(false);
  });

  it('triggers restart at threshold and resets mutation to base', () => {
    const result = computeAdaptiveMutation({
      bestScore: 10,
      allTimeBest: 10,
      stagnantGens: 59,
      baseMutation: 0.15,
      currentMutation: 0.35,
    });
    expect(result.improved).toBe(false);
    expect(result.stagnantGens).toBe(60);
    expect(result.mutationRate).toBe(0.15);
    expect(result.restart).toBe(true);
  });

  it('escalates only when lap exists and plateau is confirmed', () => {
    const result = computeAdaptiveMutation({
      bestScore: 10,
      allTimeBest: 10,
      stagnantGens: 4,
      baseMutation: 0.15,
      currentMutation: 0.2,
      hasCompletedLap: true,
      currentDifficultyLevel: 2,
      plateauStatus: {
        isPlateau: true,
        confidence: 0.87,
        remainingChecks: 0,
        requiredChecks: 3,
        consecutiveChecks: 3,
      },
    });
    expect(result.escalate).toBe(true);
    expect(result.escalationStatus.reason).toBe('plateau_confirmed');
    expect(result.escalationStatus.confidence).toBeCloseTo(0.87);
  });

  it('does not escalate before first valid lap', () => {
    const result = computeAdaptiveMutation({
      bestScore: 10,
      allTimeBest: 10,
      stagnantGens: 4,
      baseMutation: 0.15,
      currentMutation: 0.2,
      hasCompletedLap: false,
      currentDifficultyLevel: 2,
      plateauStatus: {
        isPlateau: true,
        confidence: 0.8,
        remainingChecks: 0,
        requiredChecks: 3,
        consecutiveChecks: 3,
      },
    });
    expect(result.escalate).toBe(false);
    expect(result.escalationStatus.reason).toBe('waiting_for_lap');
  });
});

describe('evaluatePlateauStatus', () => {
  it('confirms plateau when slope/gain/variance are flat for required checks', () => {
    const history = Array.from({ length: 70 }, (_, i) => 0.42 + (i % 2 === 0 ? 0.00015 : -0.00015));
    const finishHistory = Array.from({ length: 70 }, () => 0.08);
    const result = evaluatePlateauStatus({
      avgProgressHistory: history,
      finishedRateHistory: finishHistory,
      consecutiveChecks: 0,
    });
    expect(result.ready).toBe(true);
    expect(result.isPlateau).toBe(true);
    expect(result.remainingChecks).toBe(0);
    expect(result.confidence).toBeGreaterThan(0.5);
  });

  it('rejects plateau when progress is still improving quickly', () => {
    const history = Array.from({ length: 70 }, (_, i) => 0.1 + i * 0.01);
    const finishHistory = Array.from({ length: 70 }, (_, i) => Math.min(1, i * 0.01));
    const result = evaluatePlateauStatus({
      avgProgressHistory: history,
      finishedRateHistory: finishHistory,
      consecutiveChecks: 2,
    });
    expect(result.ready).toBe(true);
    expect(result.isPlateau).toBe(false);
    expect(result.flatNow).toBe(false);
    expect(result.consecutiveChecks).toBe(0);
    expect(result.confidence).toBeLessThan(0.5);
  });
});

describe('curriculum tracks', () => {
  it('includes Ironcliff as the hardest curriculum stage', () => {
    expect(TRACK_DEFAULT_WIDTHS.serpentine_bay).toBe(20);
    expect(TRACK_DEFAULT_WIDTHS.ironcliff).toBe(18);
    expect(TRACK_DEFAULT_WIDTHS.stormfront_gp).toBe(20);
    const lastStage = DIFFICULTY_LADDER[DIFFICULTY_LADDER.length - 1];
    expect(lastStage.track).toBe('ironcliff');
    expect(lastStage.widthDelta).toBe(-2);
  });
});

describe('getTieredMutationRate', () => {
  it('uses fine-tune tier for first half of population', () => {
    expect(getTieredMutationRate(10, 80, 0.15)).toBeCloseTo(0.015);
  });

  it('uses standard tier for middle slice', () => {
    expect(getTieredMutationRate(50, 80, 0.15)).toBeCloseTo(0.075);
  });

  it('uses wild tier for tail slice', () => {
    expect(getTieredMutationRate(70, 80, 0.15)).toBeCloseTo(0.3);
  });
});
