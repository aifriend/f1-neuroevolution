// Core evolution logic shared between visual (evolution.js) and headless (train.js)
// No Three.js dependency — pure math

// ─── Curriculum Difficulty Ladder ────────────────────
export const DIFFICULTY_LADDER = [
  { track: 'monaco',      widthDelta: 0 },   // Lv0: boot + hairpin
  { track: 'suzuka',      widthDelta: 0 },    // Lv1: apple + M-bottom
  { track: 'silverstone', widthDelta: 0 },    // Lv2: triangle + braking
  { track: 'spaghetti',   widthDelta: 0 },    // Lv3: hook + bumps
  { track: 'serpentine',  widthDelta: 0 },    // Lv4: asymmetric gear-teeth
  { track: 'inferno',     widthDelta: 0 },    // Lv5: asymmetric zigzag
  { track: 'inferno',     widthDelta: -4 },   // Lv6: THE FINAL BOSS (width 18)
];

export const TRACK_DEFAULT_WIDTHS = {
  monaco: 28, suzuka: 30, silverstone: 32, spaghetti: 26, serpentine: 24, inferno: 22,
};

// ─── Tournament Selection ────────────────────────────
export function tournamentSelect(cars, k = 3) {
  let best = null;
  for (let i = 0; i < k; i++) {
    const candidate = cars[Math.floor(Math.random() * cars.length)];
    if (!best || candidate.score > best.score) best = candidate;
  }
  return best;
}

// ─── Fitness Scoring ─────────────────────────────────
export function computeScore(car) {
  const progressScore = car.totalProgress;
  if (car.finished) {
    const refTime = 200;
    return progressScore + 3000 + 8000 * (refTime / car.lapTime);
  }
  return progressScore;
}

// ─── Adaptive Mutation + Curriculum Escalation ───────
export function computeAdaptiveMutation({
  bestScore,
  allTimeBest,
  stagnantGens,
  baseMutation,
  currentMutation,
  bestLapStagnantGens = 0,
  currentDifficultyLevel = 0,
  escalationGens = 50,
  lapImprovementsOnLevel = 0,
}) {
  if (bestScore > allTimeBest) {
    return {
      allTimeBest: bestScore,
      stagnantGens: 0,
      mutationRate: baseMutation,
      improved: true,
      restart: false,
      escalate: false,
    };
  }

  const nextStagnant = stagnantGens + 1;
  let nextMutation = currentMutation;
  let restart = false;

  if (nextStagnant >= 60) {
    restart = true;
    nextMutation = baseMutation;
  } else if (nextStagnant >= 25) {
    nextMutation = Math.min(0.35, currentMutation * 1.08);
  } else if (nextStagnant >= 8) {
    nextMutation = Math.min(0.25, currentMutation * 1.06);
  }

  // Curriculum escalation: detect REAL mastery before moving on.
  // Requirements (ALL must be true):
  //   1. Minimum 150 gens on this level (give time to actually learn)
  //   2. At least 3 lap-time improvements on this level (not just inherited)
  //   3. Stagnation ratio >= 40% (plateau is persistent)
  //   4. Minimum 30 stagnant gens (absolute floor)
  //   5. Not at max level
  // Rationale: carried-over brains can lap immediately, but true mastery
  // requires the car to discover IMPROVEMENTS on the current track.
  const minGensOnLevel = 150;
  const minImprovementsOnLevel = 3;
  const minStagnantFloor = 30;
  const escalate = (escalationGens || 0) >= minGensOnLevel
    && lapImprovementsOnLevel >= minImprovementsOnLevel
    && bestLapStagnantGens >= minStagnantFloor
    && bestLapStagnantGens >= Math.round((escalationGens || 50) * 0.4)
    && currentDifficultyLevel < DIFFICULTY_LADDER.length - 1;

  return {
    allTimeBest,
    stagnantGens: nextStagnant,
    mutationRate: nextMutation,
    improved: false,
    restart,
    escalate,
  };
}

// ─── Per-Individual Mutation Rate Tiers ──────────────
export function getTieredMutationRate(childIndex, totalCars, baseMutation) {
  const ratio = childIndex / totalCars;
  if (ratio < 0.5) return baseMutation * 0.1;
  if (ratio < 0.85) return baseMutation * 0.5;
  return baseMutation * 2.0;
}

// ─── Legacy selector ────────────────────────────────
export function selectTopPerformers(cars, numCars) {
  const scored = [...cars].sort((a, b) => b.score - a.score);
  const topCount = Math.max(1, Math.ceil(numCars / 5));
  return { scored, top: scored.slice(0, topCount) };
}
