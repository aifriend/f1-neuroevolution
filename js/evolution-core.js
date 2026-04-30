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
  { track: 'serpentine_bay', widthDelta: 0 }, // Lv6: heavy chicane density
  { track: 'serpentine_bay', widthDelta: -2 }, // Lv7: narrow-bay setup (width 18)
  { track: 'ironcliff', widthDelta: 0 }, // Lv8: 23-turn anti-clockwise gauntlet
  { track: 'ironcliff', widthDelta: -2 }, // Lv9: THE FINAL BOSS (width 16)
];

export const TRACK_DEFAULT_WIDTHS = {
  monaco: 28, suzuka: 30, silverstone: 32, spaghetti: 26,
  serpentine: 24, inferno: 22, serpentine_bay: 20, ironcliff: 18, stormfront_gp: 20,
};

export const PLATEAU_DEFAULTS = {
  minHistory: 25,
  ewmaAlpha: 0.2,
  slopeLookback: 20,
  relativeWindow: 30,
  varianceWindow: 30,
  maxSlope: 0.008,
  maxRelativeGain: 0.10,
  maxVariance: 0.016,
  minFinishedRateGain: 0.10,
  requiredConsecutiveChecks: 1,
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

function computeEwmaSeries(values, alpha) {
  if (!Array.isArray(values) || values.length === 0) return [];
  const ewma = [values[0]];
  for (let i = 1; i < values.length; i++) {
    ewma.push(alpha * values[i] + (1 - alpha) * ewma[i - 1]);
  }
  return ewma;
}

function varianceOf(values) {
  if (!values.length) return 0;
  const mean = values.reduce((s, x) => s + x, 0) / values.length;
  let sumSq = 0;
  for (let i = 0; i < values.length; i++) {
    const d = values[i] - mean;
    sumSq += d * d;
  }
  return sumSq / values.length;
}

export function evaluatePlateauStatus({
  avgProgressHistory = [],
  finishedRateHistory = [],
  config = {},
  consecutiveChecks = 0,
}) {
  const cfg = { ...PLATEAU_DEFAULTS, ...config };
  const len = avgProgressHistory.length;
  if (len < cfg.minHistory) {
    return {
      ready: false,
      isPlateau: false,
      confidence: 0,
      slope: 0,
      relativeGain: 0,
      variance: 0,
      finishedRateGain: 0,
      consecutiveChecks,
      remainingChecks: cfg.requiredConsecutiveChecks,
      requiredChecks: cfg.requiredConsecutiveChecks,
      reason: 'insufficient_history',
    };
  }

  const ewma = computeEwmaSeries(avgProgressHistory, cfg.ewmaAlpha);
  const slopeStart = Math.max(0, ewma.length - 1 - cfg.slopeLookback);
  const slopeDenom = Math.max(1, (ewma.length - 1) - slopeStart);
  const slope = (ewma[ewma.length - 1] - ewma[slopeStart]) / slopeDenom;

  const relWindow = Math.min(cfg.relativeWindow, Math.floor(len / 2));
  const recent = avgProgressHistory.slice(len - relWindow);
  const previous = avgProgressHistory.slice(len - relWindow * 2, len - relWindow);
  const recentMean = recent.reduce((s, x) => s + x, 0) / relWindow;
  const previousMean = previous.reduce((s, x) => s + x, 0) / relWindow;
  const relativeGain = (recentMean - previousMean) / Math.max(1e-6, Math.abs(previousMean));

  const varWindow = Math.min(cfg.varianceWindow, len);
  const variance = varianceOf(avgProgressHistory.slice(len - varWindow));

  let finishedRateGain = 0;
  if (Array.isArray(finishedRateHistory) && finishedRateHistory.length >= relWindow * 2) {
    const fLen = finishedRateHistory.length;
    const fRecent = finishedRateHistory.slice(fLen - relWindow);
    const fPrevious = finishedRateHistory.slice(fLen - relWindow * 2, fLen - relWindow);
    const recentFinished = fRecent.reduce((s, x) => s + x, 0) / relWindow;
    const previousFinished = fPrevious.reduce((s, x) => s + x, 0) / relWindow;
    finishedRateGain = recentFinished - previousFinished;
  }

  const slopeFlat = Math.abs(slope) <= cfg.maxSlope;
  const relativeFlat = relativeGain <= cfg.maxRelativeGain;
  const varianceFlat = variance <= cfg.maxVariance;
  const finishersFlat = finishedRateGain <= cfg.minFinishedRateGain;
  const flatNow = slopeFlat && relativeFlat && varianceFlat && finishersFlat;
  const nextConsecutive = flatNow ? consecutiveChecks + 1 : 0;
  const isPlateau = nextConsecutive >= cfg.requiredConsecutiveChecks;

  let confidence = 0;
  const slopeRatio = Math.min(1, Math.abs(slope) / Math.max(1e-6, cfg.maxSlope));
  const relRatio = Math.min(1, Math.max(0, relativeGain) / Math.max(1e-6, cfg.maxRelativeGain));
  const varRatio = Math.min(1, variance / Math.max(1e-6, cfg.maxVariance));
  const finRatio = Math.min(1, Math.max(0, finishedRateGain) / Math.max(1e-6, cfg.minFinishedRateGain));
  // Less sensitivity near "flat": square ratios so small deviations contribute much less.
  confidence += slopeRatio * slopeRatio;
  confidence += relRatio * relRatio;
  confidence += varRatio * varRatio;
  confidence += finRatio * finRatio;
  confidence = Math.max(0, Math.min(1, 1 - confidence / 4));

  return {
    ready: true,
    isPlateau,
    flatNow,
    confidence,
    slope,
    relativeGain,
    variance,
    finishedRateGain,
    consecutiveChecks: nextConsecutive,
    remainingChecks: Math.max(0, cfg.requiredConsecutiveChecks - nextConsecutive),
    requiredChecks: cfg.requiredConsecutiveChecks,
    reason: isPlateau ? 'plateau_confirmed' : 'plateau_monitoring',
  };
}

// ─── Adaptive Mutation + Curriculum Escalation ───────
// `currentFinisherRate` is the fraction of cars in the latest generation that
//   completed a lap. Used to gate escalation: a "lucky elite" plateau (only
//   1-3 of 80 cars lap) shouldn't promote to the next track because the
//   adapter is too fragile to retain. Default 0 disables the gate.
// `minFinisherRateForEscalation` is the threshold (default 0.10 = 10% of cars).
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
  hasCompletedLap = false,
  plateauStatus = null,
  currentFinisherRate = 0,
  minFinisherRateForEscalation = 0.10,
}) {
  if (bestScore > allTimeBest) {
    return {
      allTimeBest: bestScore,
      stagnantGens: 0,
      mutationRate: baseMutation,
      improved: true,
      restart: false,
      escalate: false,
      escalationStatus: {
        canEscalate: hasCompletedLap,
        isPlateau: false,
        reason: hasCompletedLap ? 'improved' : 'waiting_for_lap',
        confidence: plateauStatus?.confidence || 0,
        remainingChecks: plateauStatus?.remainingChecks || PLATEAU_DEFAULTS.requiredConsecutiveChecks,
        requiredChecks: plateauStatus?.requiredChecks || PLATEAU_DEFAULTS.requiredConsecutiveChecks,
        consecutiveChecks: plateauStatus?.consecutiveChecks || 0,
      },
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

  const notMaxLevel = currentDifficultyLevel < DIFFICULTY_LADDER.length - 1;
  // Robustness gate: don't escalate on a "lucky elite" plateau where only a
  // handful of cars lap. The adapter must be broadly capable. minFinisherRateForEscalation=0
  // disables the gate (legacy behavior).
  const isRobustlyCapable =
    minFinisherRateForEscalation <= 0 ||
    currentFinisherRate >= minFinisherRateForEscalation;
  const canEscalate = hasCompletedLap && notMaxLevel && isRobustlyCapable;
  const escalate = canEscalate && Boolean(plateauStatus?.isPlateau);

  let escalationReason = 'plateau_monitoring';
  if (!notMaxLevel) escalationReason = 'max_level';
  else if (!hasCompletedLap) escalationReason = 'waiting_for_lap';
  else if (!isRobustlyCapable) escalationReason = 'fragile_adapter';
  else if (escalate) escalationReason = 'plateau_confirmed';

  return {
    allTimeBest,
    stagnantGens: nextStagnant,
    mutationRate: nextMutation,
    improved: false,
    restart,
    escalate,
    escalationStatus: {
      canEscalate,
      isPlateau: Boolean(plateauStatus?.isPlateau),
      reason: escalationReason,
      confidence: plateauStatus?.confidence || 0,
      slope: plateauStatus?.slope || 0,
      relativeGain: plateauStatus?.relativeGain || 0,
      variance: plateauStatus?.variance || 0,
      finishedRateGain: plateauStatus?.finishedRateGain || 0,
      currentFinisherRate,
      minFinisherRateForEscalation,
      isRobustlyCapable,
      remainingChecks: plateauStatus?.remainingChecks || PLATEAU_DEFAULTS.requiredConsecutiveChecks,
      requiredChecks: plateauStatus?.requiredChecks || PLATEAU_DEFAULTS.requiredConsecutiveChecks,
      consecutiveChecks: plateauStatus?.consecutiveChecks || 0,
      legacyBestLapStagnantGens: bestLapStagnantGens,
      legacyEscalationGens: escalationGens,
      legacyLapImprovementsOnLevel: lapImprovementsOnLevel,
    },
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
