// Neural network for self-driving F1 cars with LoRA-style continual learning.
//
// Architecture:
//   inputs (10) → W1 (10×16) + b1 → tanh → hidden (16)
//   hidden (16) → W2 (16×2)  + b2 → tanh → output (2: steer, gas)
//
//   Base weights: 210 params (mutable while currentLevel === 0,
//                              FROZEN at currentLevel >= 1)
//
//   Per-level LoRA adapter (rank r=2):
//     W1_eff = W1_base + A1·B1     (A1 is 10×r, B1 is r×16, 52 params)
//     W2_eff = W2_base + A2·B2     (A2 is 16×r, B2 is r×2,  36 params)
//     Per-level cost: 88 params · only the CURRENT level's adapter mutates
//
// Catastrophic forgetting is impossible by construction: previously-trained
// levels keep their adapter weights; the base never changes after level 0.

export const NUM_SENSORS = 9;
export const NUM_INPUTS = NUM_SENSORS + 1; // sensors + current speed
export const HIDDEN_SIZE = 16;
export const OUTPUT_SIZE = 2;
// Default LoRA rank for fresh brains. Loaded brains derive rank from their
// existing adapter shape, so heterogeneous-rank brains can coexist (useful
// for experiments comparing rank-2, rank-4, rank-8, etc.).
export const LORA_RANK = 2;

function randomGaussian(mean = 0, std = 1) {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return mean + std * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function randomMatrix(rows, cols, scale = 1) {
  const m = [];
  for (let i = 0; i < rows; i++) {
    const row = [];
    for (let j = 0; j < cols; j++) row.push(randomGaussian(0, scale));
    m.push(row);
  }
  return m;
}

function zeroMatrix(rows, cols) {
  const m = [];
  for (let i = 0; i < rows; i++) {
    const row = new Array(cols).fill(0);
    m.push(row);
  }
  return m;
}

function cloneMatrix(m) {
  return m.map((row) => [...row]);
}

function matMulAdd(base, A, B) {
  // Returns base + A·B (a fresh matrix, doesn't modify inputs)
  const rows = base.length;
  const cols = base[0].length;
  const r = A[0].length; // rank
  const out = [];
  for (let i = 0; i < rows; i++) {
    const baseRow = base[i];
    const ARow = A[i];
    const outRow = new Array(cols);
    for (let j = 0; j < cols; j++) {
      let delta = 0;
      for (let k = 0; k < r; k++) delta += ARow[k] * B[k][j];
      outRow[j] = baseRow[j] + delta;
    }
    out.push(outRow);
  }
  return out;
}

// Standard LoRA init: A is small random, B is zeros. The adapter contributes
// nothing at init time, so a freshly-escalated brain behaves identically to
// the base. Mutation then drifts A and B from this neutral starting point.
function makeAdapter(rank) {
  return {
    A1: randomMatrix(NUM_INPUTS, rank, 0.05),
    B1: zeroMatrix(rank, HIDDEN_SIZE),
    A2: randomMatrix(HIDDEN_SIZE, rank, 0.05),
    B2: zeroMatrix(rank, OUTPUT_SIZE),
  };
}

// Auto-detect LoRA rank from any existing adapter (every adapter shares the
// same rank, so we just look at A1's column count).
function detectRank(adapters) {
  for (const k of Object.keys(adapters || {})) {
    const a = adapters[k];
    if (a && Array.isArray(a.A1) && Array.isArray(a.A1[0])) return a.A1[0].length;
  }
  return null;
}

function cloneAdapter(a) {
  return {
    A1: cloneMatrix(a.A1),
    B1: cloneMatrix(a.B1),
    A2: cloneMatrix(a.A2),
    B2: cloneMatrix(a.B2),
  };
}

function cloneAdapters(adapters) {
  const out = {};
  for (const k of Object.keys(adapters || {})) {
    out[k] = cloneAdapter(adapters[k]);
  }
  return out;
}

export class NeuralCar {
  // `opts.rank` overrides default LORA_RANK for fresh brains. If loaded brain
  //   already has adapters, rank is auto-detected from their shape (ignored).
  // `opts.softFreezeFactor` lets the base mutate at level >= 1 at this
  //   fraction of the adapter's mutation rate (0 = strict freeze; 0.05 =
  //   slow drift; 1.0 = no protection at all). Default 0.
  constructor(weights = null, opts = {}) {
    if (!weights) {
      // Fresh random base, no adapters, level 0 (Monaco)
      this.base = {
        w1: randomMatrix(NUM_INPUTS, HIDDEN_SIZE, 0.8),
        b1: new Array(HIDDEN_SIZE).fill(0),
        w2: randomMatrix(HIDDEN_SIZE, OUTPUT_SIZE, 0.6),
        b2: new Array(OUTPUT_SIZE).fill(0),
      };
      this.adapters = {};
      this.currentLevel = 0;
      this.rank = opts.rank ?? LORA_RANK;
    } else if (weights.base) {
      // V2 format: explicit base + adapters
      this.base = {
        w1: cloneMatrix(weights.base.w1),
        b1: [...weights.base.b1],
        w2: cloneMatrix(weights.base.w2),
        b2: [...weights.base.b2],
      };
      this.adapters = cloneAdapters(weights.adapters);
      this.currentLevel = weights.currentLevel ?? 0;
      // Auto-detect rank from existing adapters; fall back to hint or default.
      this.rank = detectRank(this.adapters) ?? opts.rank ?? LORA_RANK;
    } else {
      // V1 (legacy) format: flat w1/b1/w2/b2 — treat as base, no adapters
      this.base = {
        w1: cloneMatrix(weights.w1),
        b1: [...(weights.b1 || new Array(HIDDEN_SIZE).fill(0))],
        w2: cloneMatrix(weights.w2),
        b2: [...(weights.b2 || new Array(OUTPUT_SIZE).fill(0))],
      };
      this.adapters = {};
      this.currentLevel = 0;
      this.rank = opts.rank ?? LORA_RANK;
    }

    this.softFreezeFactor = opts.softFreezeFactor ?? 0;
    this._lastHidden = new Array(HIDDEN_SIZE).fill(0);
    this._lastOutput = [0, 0];
  }

  // Switch which adapter is active. Used on curriculum escalation.
  // If no adapter exists for the new level, create one (LoRA-init: zero delta).
  setLevel(level) {
    this.currentLevel = level;
    if (level > 0 && !this.adapters[level]) {
      this.adapters[level] = makeAdapter(this.rank);
    }
  }

  // Replace the frozen base with one from another brain. Used at curriculum
  // escalation: every car in the new population should share the SAME base
  // (the champion's base, locked in as the "level-0 universal feature
  // extractor"). Without this, tournament selection scatters base diversity
  // across the population and we lose the lapping brain's exact base.
  setBase(externalBase) {
    this.base = {
      w1: cloneMatrix(externalBase.w1),
      b1: [...externalBase.b1],
      w2: cloneMatrix(externalBase.w2),
      b2: [...externalBase.b2],
    };
  }

  // Compute effective W1 and W2 for the current level.
  // Level 0: just base. Level >= 1: base + A·B for the current adapter.
  _effectiveWeights() {
    if (this.currentLevel === 0) return this.base;
    const adapter = this.adapters[this.currentLevel];
    if (!adapter) return this.base;
    return {
      w1: matMulAdd(this.base.w1, adapter.A1, adapter.B1),
      b1: this.base.b1, // biases shared across levels
      w2: matMulAdd(this.base.w2, adapter.A2, adapter.B2),
      b2: this.base.b2,
    };
  }

  think(inputs) {
    const { w1, b1, w2, b2 } = this._effectiveWeights();

    const hidden = new Array(HIDDEN_SIZE);
    for (let j = 0; j < HIDDEN_SIZE; j++) {
      let sum = b1[j];
      for (let i = 0; i < inputs.length; i++) sum += inputs[i] * w1[i][j];
      hidden[j] = Math.tanh(sum);
    }

    const output = new Array(OUTPUT_SIZE);
    for (let j = 0; j < OUTPUT_SIZE; j++) {
      let sum = b2[j];
      for (let i = 0; i < HIDDEN_SIZE; i++) sum += hidden[i] * w2[i][j];
      output[j] = Math.tanh(sum);
    }

    this._lastHidden = hidden;
    this._lastOutput = output;
    return { steer: output[0], gas: output[1] };
  }

  // Mutate the trainable parameters.
  //   Level 0: mutate the base (w1, b1, w2, b2).
  //   Level >= 1: mutate the current level's adapter (A1, B1, A2, B2).
  //   Plus, if softFreezeFactor > 0 at level >= 1, also mutate the base at
  //   (rate × softFreezeFactor) — lets the "universal feature extractor"
  //   slowly improve as new tracks are seen, while heavy specialization
  //   still happens in the per-level adapter. Strict freeze = factor 0.
  mutate(rate) {
    if (this.currentLevel === 0) {
      this._mutateBase(rate);
      return;
    }
    const adapter = this.adapters[this.currentLevel];
    if (!adapter) return;
    this._mutateMatrix(adapter.A1, rate);
    this._mutateMatrix(adapter.B1, rate);
    this._mutateMatrix(adapter.A2, rate);
    this._mutateMatrix(adapter.B2, rate);
    if (this.softFreezeFactor > 0) {
      this._mutateBase(rate * this.softFreezeFactor);
    }
  }

  _mutateBase(rate) {
    this._mutateMatrix(this.base.w1, rate);
    this._mutateVector(this.base.b1, rate);
    this._mutateMatrix(this.base.w2, rate);
    this._mutateVector(this.base.b2, rate);
  }

  _mutateMatrix(m, rate) {
    for (let i = 0; i < m.length; i++) {
      const row = m[i];
      for (let j = 0; j < row.length; j++) {
        row[j] += randomGaussian(0, 1) * rate;
      }
    }
  }

  _mutateVector(v, rate) {
    for (let i = 0; i < v.length; i++) v[i] += randomGaussian(0, 1) * rate;
  }

  // V2 serialization: base + adapter library + current level + rank.
  // Loadable by both visual and headless trainers.
  getWeights() {
    return {
      version: 2,
      base: {
        w1: cloneMatrix(this.base.w1),
        b1: [...this.base.b1],
        w2: cloneMatrix(this.base.w2),
        b2: [...this.base.b2],
      },
      adapters: cloneAdapters(this.adapters),
      currentLevel: this.currentLevel,
      rank: this.rank,
    };
  }
}

export const F1_TEAMS = [
  { name: 'Red Bull',     main: 0x3671c6, accent: 0xffcd00 },
  { name: 'Ferrari',      main: 0xe8002d, accent: 0xffffff },
  { name: 'McLaren',      main: 0xff8000, accent: 0x000000 },
  { name: 'Mercedes',     main: 0x27f4d2, accent: 0x000000 },
  { name: 'Aston Martin', main: 0x006f62, accent: 0xcedc00 },
  { name: 'Alpine',       main: 0x0093cc, accent: 0xff57a4 },
  { name: 'Williams',     main: 0x64c4ff, accent: 0x003778 },
  { name: 'RB',           main: 0x6692ff, accent: 0xff0000 },
  { name: 'Kick Sauber',  main: 0x52e252, accent: 0x000000 },
  { name: 'Haas',         main: 0xb6babd, accent: 0xe8002d },
];
