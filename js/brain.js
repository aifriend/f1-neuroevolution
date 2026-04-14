import { HIDDEN_SIZE, NUM_INPUTS } from './nn.js';

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

export function validateBrainWeights(weights) {
  if (!weights || typeof weights !== 'object') {
    return { ok: false, errors: ['brain payload must be an object'] };
  }

  const errors = [];

  const w1Error = validateMatrixShape(weights.w1, NUM_INPUTS, HIDDEN_SIZE, 'w1');
  if (w1Error) errors.push(w1Error);

  const b1Error = validateVectorShape(weights.b1, HIDDEN_SIZE, 'b1');
  if (b1Error) errors.push(b1Error);

  const w2Error = validateMatrixShape(weights.w2, HIDDEN_SIZE, 2, 'w2');
  if (w2Error) errors.push(w2Error);

  const b2Error = validateVectorShape(weights.b2, 2, 'b2');
  if (b2Error) errors.push(b2Error);

  return { ok: errors.length === 0, errors };
}
