import { computeInterpolationSteps } from './track-geometry.js';
import { TRACKS } from './track-data.js';

export const HEADLESS_SENSOR_LENGTH = 220;
export const HEADLESS_LAP_COMPLETION_PROGRESS = 0.995;

export class HeadlessTrack {
  constructor(type, widthOverride = null) {
    const cfg = TRACKS[type] || TRACKS.monaco;
    this.name = cfg.name;
    this.trackWidth = widthOverride !== null ? widthOverride : cfg.width;
    this.gridSize = 5;
    const stepsPerSeg = computeInterpolationSteps(cfg.points, this.gridSize * 0.8);
    this.points = this._interpolate(cfg.points, stepsPerSeg);
    this.tangents = this._computeTangents();

    const sp = this.points[0];
    const st = this.tangents[0];
    this.startX = sp[0];
    this.startZ = sp[1];
    this.startAngle = Math.atan2(st[1], st[0]);

    this.grid = {};
    this._buildGrid();
  }

  _interpolate(pts, steps) {
    const result = [];
    const n = pts.length;
    for (let i = 0; i < n; i++) {
      const p0 = pts[(i - 1 + n) % n];
      const p1 = pts[i];
      const p2 = pts[(i + 1) % n];
      const p3 = pts[(i + 2) % n];
      for (let s = 0; s < steps; s++) {
        const t = s / steps;
        const t2 = t * t;
        const t3 = t2 * t;
        const c0 = -0.5 * t3 + t2 - 0.5 * t;
        const c1 = 1.5 * t3 - 2.5 * t2 + 1;
        const c2 = -1.5 * t3 + 2 * t2 + 0.5 * t;
        const c3 = 0.5 * t3 - 0.5 * t2;
        result.push([
          c0 * p0[0] + c1 * p1[0] + c2 * p2[0] + c3 * p3[0],
          c0 * p0[1] + c1 * p1[1] + c2 * p2[1] + c3 * p3[1],
        ]);
      }
    }
    return result;
  }

  _computeTangents() {
    const n = this.points.length;
    const tangents = [];
    for (let i = 0; i < n; i++) {
      const next = this.points[(i + 1) % n];
      const curr = this.points[i];
      const dx = next[0] - curr[0];
      const dz = next[1] - curr[1];
      const len = Math.sqrt(dx * dx + dz * dz) || 1;
      tangents.push([dx / len, dz / len]);
    }
    return tangents;
  }

  _buildGrid() {
    const n = this.points.length;
    const gs = this.gridSize;
    const hw = this.trackWidth;
    for (let i = 0; i < n; i++) {
      const px = this.points[i][0];
      const pz = this.points[i][1];
      const t = this.tangents[i];
      const nx = -t[1];
      const nz = t[0];
      for (let w = -hw; w <= hw; w += 1) {
        const gx = Math.floor((px + nx * w) / gs);
        const gz = Math.floor((pz + nz * w) / gs);
        this.grid[`${gx},${gz}`] = true;
      }
    }
  }

  isOnTrack(x, z) {
    return this.grid[`${Math.floor(x / this.gridSize)},${Math.floor(z / this.gridSize)}`] === true;
  }

  castRay(x, z, angle) {
    const cosA = Math.cos(angle);
    const sinA = Math.sin(angle);
    for (let d = 2; d < HEADLESS_SENSOR_LENGTH; d += 2) {
      if (!this.isOnTrack(x + cosA * d, z + sinA * d)) return d;
    }
    return HEADLESS_SENSOR_LENGTH;
  }

  getProgress(x, z) {
    let minDist = Infinity;
    let bestIdx = 0;
    const n = this.points.length;
    for (let i = 0; i < n; i += 4) {
      const dx = x - this.points[i][0];
      const dz = z - this.points[i][1];
      const dist = dx * dx + dz * dz;
      if (dist < minDist) {
        minDist = dist;
        bestIdx = i;
      }
    }
    for (let offset = -6; offset <= 6; offset++) {
      const i = ((bestIdx + offset) % n + n) % n;
      const dx = x - this.points[i][0];
      const dz = z - this.points[i][1];
      const dist = dx * dx + dz * dz;
      if (dist < minDist) {
        minDist = dist;
        bestIdx = i;
      }
    }
    return bestIdx / n;
  }

  getProgressLocal(x, z, hintIdx, searchRadius = 20) {
    let minDist = Infinity;
    let bestIdx = hintIdx;
    const n = this.points.length;
    for (let offset = -searchRadius; offset <= searchRadius; offset++) {
      const i = ((hintIdx + offset) % n + n) % n;
      const dx = x - this.points[i][0];
      const dz = z - this.points[i][1];
      const dist = dx * dx + dz * dz;
      if (dist < minDist) {
        minDist = dist;
        bestIdx = i;
      }
    }
    return { progress: bestIdx / n, idx: bestIdx };
  }

  getStartPos() {
    return { x: this.startX, z: this.startZ, angle: this.startAngle };
  }
}
