import { validateBrainWeights } from './brain.js?v=5';

export function isDebugMode(search) {
  const params = new URLSearchParams(search || '');
  return params.get('debug') === '1';
}

export function parseBrainJson(text) {
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch {
    return { ok: false, error: 'Invalid JSON format' };
  }

  const validation = validateBrainWeights(parsed);
  if (!validation.ok) {
    return { ok: false, error: validation.errors[0] || 'Invalid brain format' };
  }

  return { ok: true, weights: parsed };
}
