import { HIDDEN_SIZE, NUM_INPUTS, OUTPUT_SIZE, LORA_RANK } from './nn.js';

function isFiniteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function validateMatrixShape(matrix, rows, cols, label) {
  if (!Array.isArray(matrix) || matrix.length !== rows) {
    return `${label} must be an array with ${rows} rows`;
  }
  for (let i = 0; i < rows; i++) {
    const row = matrix[i];
    if (!Array.isArray(row) || row.length !== cols) {
      return `${label}[${i}] must be an array with ${cols} values`;
    }
    for (let j = 0; j < cols; j++) {
      if (!isFiniteNumber(row[j])) {
        return `${label}[${i}][${j}] must be a finite number`;
      }
    }
  }
  return null;
}

function validateVectorShape(vector, size, label) {
  if (!Array.isArray(vector) || vector.length !== size) {
    return `${label} must be an array with ${size} values`;
  }
  for (let i = 0; i < size; i++) {
    if (!isFiniteNumber(vector[i])) {
      return `${label}[${i}] must be a finite number`;
    }
  }
  return null;
}

function validateBase(base) {
  const errors = [];
  const w1Error = validateMatrixShape(base.w1, NUM_INPUTS, HIDDEN_SIZE, 'base.w1');
  if (w1Error) errors.push(w1Error);
  const b1Error = validateVectorShape(base.b1, HIDDEN_SIZE, 'base.b1');
  if (b1Error) errors.push(b1Error);
  const w2Error = validateMatrixShape(base.w2, HIDDEN_SIZE, OUTPUT_SIZE, 'base.w2');
  if (w2Error) errors.push(w2Error);
  const b2Error = validateVectorShape(base.b2, OUTPUT_SIZE, 'base.b2');
  if (b2Error) errors.push(b2Error);
  return errors;
}

function validateAdapter(adapter, levelKey, expectedRank) {
  const errors = [];
  const labelPrefix = `adapters.${levelKey}`;
  // Derive rank from the adapter itself if no expectation was passed in.
  // This is what allows heterogeneous-rank brains to load (e.g. rank-4 brain
  // valid even when LORA_RANK constant says 2).
  const r = expectedRank
    || (Array.isArray(adapter.A1) && Array.isArray(adapter.A1[0]) ? adapter.A1[0].length : LORA_RANK);
  const a1 = validateMatrixShape(adapter.A1, NUM_INPUTS, r, `${labelPrefix}.A1`);
  if (a1) errors.push(a1);
  const b1 = validateMatrixShape(adapter.B1, r, HIDDEN_SIZE, `${labelPrefix}.B1`);
  if (b1) errors.push(b1);
  const a2 = validateMatrixShape(adapter.A2, HIDDEN_SIZE, r, `${labelPrefix}.A2`);
  if (a2) errors.push(a2);
  const b2 = validateMatrixShape(adapter.B2, r, OUTPUT_SIZE, `${labelPrefix}.B2`);
  if (b2) errors.push(b2);
  return errors;
}

export function validateBrainWeights(weights) {
  if (!weights || typeof weights !== 'object') {
    return { ok: false, errors: ['brain payload must be an object'] };
  }

  // V2 format: { version: 2, base: {...}, adapters: {...}, currentLevel: N, rank: R }
  if (weights.base) {
    const errors = validateBase(weights.base);
    if (weights.adapters && typeof weights.adapters === 'object') {
      // Use brain's own rank if recorded; otherwise infer per-adapter.
      const expectedRank = Number.isFinite(weights.rank) ? weights.rank : null;
      for (const key of Object.keys(weights.adapters)) {
        errors.push(...validateAdapter(weights.adapters[key], key, expectedRank));
      }
    }
    return { ok: errors.length === 0, errors };
  }

  // V1 (legacy) format: flat w1/b1/w2/b2 — accept and treat as base
  return { ok: validateBase(weights).length === 0, errors: validateBase(weights) };
}
