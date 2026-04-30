import { validateBrainWeights } from './brain.js?v=29';

export function isDebugMode(search) {
  const params = new URLSearchParams(search || '');
  return params.get('debug') === '1';
}

function clamp(n, lo, hi) {
  return Math.max(lo, Math.min(hi, n));
}

function parseNumberParam(params, key) {
  const raw = params.get(key);
  if (raw == null || raw === '') return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

export function getViewportSize() {
  // Viewport may report 0 in hidden iframes/background tabs.
  return {
    width: window.innerWidth || document.documentElement.clientWidth || 1024,
    height: window.innerHeight || document.documentElement.clientHeight || 768,
  };
}

/**
 * Visual runtime URL params (all optional):
 * - track: one of allowedTracks
 * - cars: population size (5..100, step 1)
 * - speed: speed multiplier (0.5..3)
 * - mutation: base mutation rate (0.02..0.3)
 * - timeout: 1/0 (enable/disable)
 * - timeoutFrames: frame limit (100..5000)
 * - fresh: 1/0 (ignore persisted localStorage state)
 */
export function parseVisualRuntimeParams(search, { allowedTracks = [] } = {}) {
  const params = new URLSearchParams(search || '');

  const out = {
    fresh: params.get('fresh') === '1',
    trackType: null,
    numCars: null,
    speedMult: null,
    mutationRate: null,
    timeoutEnabled: null,
    timeoutDuration: null,
  };

  const track = params.get('track');
  if (track && allowedTracks.includes(track)) out.trackType = track;

  const cars = parseNumberParam(params, 'cars');
  if (cars != null) out.numCars = Math.round(clamp(cars, 5, 100));

  const speed = parseNumberParam(params, 'speed');
  if (speed != null) out.speedMult = clamp(speed, 0.5, 3);

  const mut = parseNumberParam(params, 'mutation');
  if (mut != null) out.mutationRate = clamp(mut, 0.02, 0.3);

  const timeout = params.get('timeout');
  if (timeout === '1') out.timeoutEnabled = true;
  if (timeout === '0') out.timeoutEnabled = false;

  const tf = parseNumberParam(params, 'timeoutFrames');
  if (tf != null) out.timeoutDuration = Math.round(clamp(tf, 100, 5000));

  return out;
}

export function createQuitController(onQuit) {
  let quitRequested = false;
  let quitHandled = false;

  return {
    isQuitRequested() {
      return quitRequested;
    },
    requestQuit() {
      if (quitRequested) return false;
      quitRequested = true;
      if (!quitHandled) {
        quitHandled = true;
        if (typeof onQuit === 'function') onQuit();
      }
      return true;
    },
  };
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
