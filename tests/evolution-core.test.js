import { describe, expect, it } from 'vitest';
import {
  computeAdaptiveMutation,
  getTieredMutationRate,
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

  it('applies mild stagnation cap at 0.3', () => {
    const result = computeAdaptiveMutation({
      bestScore: 10,
      allTimeBest: 10,
      stagnantGens: 4,
      baseMutation: 0.15,
      currentMutation: 0.4,
    });
    expect(result.improved).toBe(false);
    expect(result.stagnantGens).toBe(5);
    expect(result.mutationRate).toBe(0.3);
    expect(result.restart).toBe(false);
  });

  it('applies severe stagnation cap at 0.4', () => {
    const result = computeAdaptiveMutation({
      bestScore: 10,
      allTimeBest: 10,
      stagnantGens: 14,
      baseMutation: 0.15,
      currentMutation: 0.35,
    });
    expect(result.improved).toBe(false);
    expect(result.stagnantGens).toBe(15);
    expect(result.mutationRate).toBe(0.4);
    expect(result.restart).toBe(false);
  });

  it('triggers restart at threshold and resets mutation to base', () => {
    const result = computeAdaptiveMutation({
      bestScore: 10,
      allTimeBest: 10,
      stagnantGens: 44,
      baseMutation: 0.15,
      currentMutation: 0.4,
      weakSignalStreak: 4,
      finisherRatio: 0.0,
      avgProgress: 0.02,
    });
    expect(result.improved).toBe(false);
    expect(result.stagnantGens).toBe(45);
    expect(result.mutationRate).toBe(0.15);
    expect(result.restart).toBe(true);
    expect(result.restartCooldown).toBe(20);
    expect(result.weakSignalStreak).toBe(0);
  });

  it('does not restart at threshold without sustained weak-signal streak', () => {
    const result = computeAdaptiveMutation({
      bestScore: 10,
      allTimeBest: 10,
      stagnantGens: 44,
      baseMutation: 0.15,
      currentMutation: 0.4,
      weakSignalStreak: 0,
      finisherRatio: 0.0,
      avgProgress: 0.02,
    });
    expect(result.restart).toBe(false);
    expect(result.mutationRate).toBe(0.35);
    expect(result.weakSignalStreak).toBe(1);
  });

  it('avoids hard restart when finishers indicate healthy exploration', () => {
    const result = computeAdaptiveMutation({
      bestScore: 10,
      allTimeBest: 10,
      stagnantGens: 44,
      baseMutation: 0.15,
      currentMutation: 0.4,
      weakSignalStreak: 7,
      finisherRatio: 0.2,
      avgProgress: 0.14,
    });
    expect(result.improved).toBe(false);
    expect(result.stagnantGens).toBe(45);
    expect(result.restart).toBe(false);
    expect(result.mutationRate).toBe(0.35);
    expect(result.restartCooldown).toBe(0);
    expect(result.weakSignalStreak).toBe(0);
  });

  it('respects restart cooldown after a recent restart', () => {
    const result = computeAdaptiveMutation({
      bestScore: 10,
      allTimeBest: 10,
      stagnantGens: 50,
      baseMutation: 0.15,
      currentMutation: 0.3,
      restartCooldown: 5,
      weakSignalStreak: 10,
      finisherRatio: 0.0,
      avgProgress: 0.01,
    });
    expect(result.restart).toBe(false);
    expect(result.mutationRate).toBeCloseTo(0.33);
    expect(result.restartCooldown).toBe(4);
    expect(result.weakSignalStreak).toBe(11);
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
