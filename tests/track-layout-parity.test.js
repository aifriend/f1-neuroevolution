import { beforeAll, describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { computeInterpolationSteps } from '../js/track-geometry.js';

function catmullRomInterpolate(points, steps) {
  const out = [];
  const n = points.length;
  for (let i = 0; i < n; i++) {
    const p0 = points[(i - 1 + n) % n];
    const p1 = points[i];
    const p2 = points[(i + 1) % n];
    const p3 = points[(i + 2) % n];
    for (let s = 0; s < steps; s++) {
      const t = s / steps;
      const t2 = t * t;
      const t3 = t2 * t;
      const c0 = -0.5 * t3 + t2 - 0.5 * t;
      const c1 = 1.5 * t3 - 2.5 * t2 + 1;
      const c2 = -1.5 * t3 + 2 * t2 + 0.5 * t;
      const c3 = 0.5 * t3 - 0.5 * t2;
      out.push([
        c0 * p0[0] + c1 * p1[0] + c2 * p2[0] + c3 * p3[0],
        c0 * p0[1] + c1 * p1[1] + c2 * p2[1] + c3 * p3[1],
      ]);
    }
  }
  return out;
}

function tangents(points) {
  const out = [];
  const n = points.length;
  for (let i = 0; i < n; i++) {
    const a = points[i];
    const b = points[(i + 1) % n];
    const dx = b[0] - a[0];
    const dz = b[1] - a[1];
    const len = Math.hypot(dx, dz) || 1;
    out.push([dx / len, dz / len]);
  }
  return out;
}

const GRID_SIZE = 5;
function buildGrid(points, tgs, width) {
  const grid = new Set();
  for (let i = 0; i < points.length; i++) {
    const [px, pz] = points[i];
    const [tx, tz] = tgs[i];
    const nx = -tz;
    const nz = tx;
    for (let w = -width; w <= width; w += 1) {
      const gx = Math.floor((px + nx * w) / GRID_SIZE);
      const gz = Math.floor((pz + nz * w) / GRID_SIZE);
      grid.add(`${gx},${gz}`);
    }
  }
  return grid;
}

function onTrack(grid, x, z) {
  return grid.has(`${Math.floor(x / GRID_SIZE)},${Math.floor(z / GRID_SIZE)}`);
}

function castRay(grid, x, z, angle) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  for (let d = 2; d < 220; d += 2) {
    if (!onTrack(grid, x + c * d, z + s * d)) return d;
  }
  return 220;
}

// Oracle driver: aims at a centerline lookahead and brakes on close walls.
// Mirrors the physics in js/car.js (min speed 2.5, max 8.1, steer 0.08 rad/f).
function runOracleLap(controlPoints, trackWidth) {
  const steps = computeInterpolationSteps(controlPoints, GRID_SIZE * 0.8);
  const ip = catmullRomInterpolate(controlPoints, steps);
  const tgs = tangents(ip);
  const grid = buildGrid(ip, tgs, trackWidth);
  const SENSOR_ANGLES = [
    -Math.PI / 2, -Math.PI * 3 / 8, -Math.PI / 4, -Math.PI / 8,
    0, Math.PI / 8, Math.PI / 4, Math.PI * 3 / 8, Math.PI / 2,
  ];

  let x = ip[0][0];
  let z = ip[0][1];
  let angle = Math.atan2(tgs[0][1], tgs[0][0]);
  let lastIdx = 0;
  let maxIdx = 0;
  let crossedLap = false;

  for (let frame = 0; frame < 5000; frame++) {
    const sensors = SENSOR_ANGLES.map((a) => castRay(grid, x, z, angle + a) / 220);
    let minD = Infinity;
    let idx = lastIdx;
    for (let o = -30; o <= 30; o++) {
      const i = (lastIdx + o + ip.length) % ip.length;
      const d = (ip[i][0] - x) ** 2 + (ip[i][1] - z) ** 2;
      if (d < minD) { minD = d; idx = i; }
    }
    lastIdx = idx;
    if (idx > maxIdx) maxIdx = idx;
    if (maxIdx > ip.length * 0.9 && idx < ip.length * 0.1) crossedLap = true;

    const look = ip[(idx + 10) % ip.length];
    const dx = look[0] - x;
    const dz = look[1] - z;
    const desired = Math.atan2(dz, dx);
    let diff = desired - angle;
    while (diff > Math.PI) diff -= 2 * Math.PI;
    while (diff < -Math.PI) diff += 2 * Math.PI;
    const steer = Math.max(-1, Math.min(1, diff / 0.2));
    const gas = sensors[4] < 0.25 ? -1 : sensors[4] < 0.5 ? -0.3 : 1;

    angle += steer * 0.08;
    const speed = 2.5 + ((gas + 1) / 2) * 5.6;
    const nx = x + Math.cos(angle) * speed;
    const nz = z + Math.sin(angle) * speed;
    if (!onTrack(grid, (x + nx) / 2, (z + nz) / 2) || !onTrack(grid, nx, nz)) {
      return { completed: false, frame, progress: maxIdx / ip.length, diedAt: [x, z] };
    }
    x = nx;
    z = nz;
    if (crossedLap) return { completed: true, frame, progress: 1 };
  }
  return { completed: false, frame: 5000, progress: maxIdx / ip.length, diedAt: 'timeout' };
}

// All 9 curriculum tracks MUST come from js/track-data.js — the single
// source of truth imported by BOTH the browser runtime (js/track.js) and
// the headless trainer (train.js). Drift is architecturally impossible,
// but we still verify:
//   (a) each file actually imports from track-data.js
//   (b) each file does NOT redefine TRACKS inline
// If either guard breaks, a future refactor could silently re-introduce
// duplicated geometry.
describe('track-data sourcing', () => {
  const runtimeSrc = readFileSync(resolve('js/track.js'), 'utf8');
  const trainerSrc = readFileSync(resolve('train.js'), 'utf8');

  it('js/track.js imports TRACKS from track-data.js', () => {
    expect(runtimeSrc).toMatch(/from\s+['"]\.\/track-data\.js(?:\?[^'"]*)?['"]/);
    expect(runtimeSrc).not.toMatch(/^const TRACKS = \{/m);
  });

  it('train.js imports TRACKS from track-data.js', () => {
    expect(trainerSrc).toMatch(/track-data\.js/);
    expect(trainerSrc).not.toMatch(/^const TRACKS = \{/m);
  });

  const allTracks = [
    'monaco', 'suzuka', 'silverstone', 'spaghetti', 'serpentine',
    'inferno', 'serpentine_bay', 'ironcliff', 'stormfront_gp',
  ];

  // Sanity: make sure track-data.js actually exposes every curriculum track
  // with the required shape. Catches typos, missing tracks, malformed data.
  it('track-data.js exposes all 9 curriculum tracks with valid shape', async () => {
    const { TRACKS } = await import('../js/track-data.js');
    for (const name of allTracks) {
      expect(TRACKS, `missing track: ${name}`).toHaveProperty(name);
      const t = TRACKS[name];
      expect(typeof t.name).toBe('string');
      expect(typeof t.width).toBe('number');
      expect(t.width).toBeGreaterThan(0);
      expect(Array.isArray(t.points)).toBe(true);
      expect(t.points.length).toBeGreaterThanOrEqual(4);
      for (const pt of t.points) {
        expect(Array.isArray(pt)).toBe(true);
        expect(pt.length).toBe(2);
        expect(Number.isFinite(pt[0])).toBe(true);
        expect(Number.isFinite(pt[1])).toBe(true);
      }
    }
  });
});

// Drivability invariant: a simple lookahead-oracle policy must finish a full
// lap at stock physics on every track. Previous serpentine_bay geometry
// failed at ~26% progress at the BL bay entrance because Catmull-Rom turned
// its densely clustered control points into sub-vehicle-min-radius cusps.
// This regression test prevents future control-point edits from reintroducing
// undriveable geometry.
describe('track drivability (oracle policy)', () => {
  // Import TRACKS directly from the shared source of truth — no regex parsing.
  let TRACKS;
  beforeAll(async () => {
    ({ TRACKS } = await import('../js/track-data.js'));
  });

  for (const name of [
    'monaco',
    'suzuka',
    'silverstone',
    'spaghetti',
    'serpentine',
    'inferno',
    'serpentine_bay',
    'ironcliff',
    'stormfront_gp',
  ]) {
    it(`${name} is completable by oracle policy at stock physics`, () => {
      const cfg = TRACKS[name];
      const result = runOracleLap(cfg.points, cfg.width);
      expect(result).toMatchObject({ completed: true });
    });
  }
});
