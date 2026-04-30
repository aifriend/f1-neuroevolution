// SHARED HEADLESS PHYSICS — used by both train.js (population evolution)
// and cross-track-eval.js (single-brain evaluation). Centralizing here
// guarantees the SAME physics in both contexts, eliminating any
// training-vs-evaluation discrepancy in lap completion, death conditions,
// or progress tracking.

import { NeuralCar } from './nn.js';
import { computeScore } from './evolution-core.js';
import {
  HEADLESS_SENSOR_LENGTH,
  HEADLESS_LAP_COMPLETION_PROGRESS,
} from './headless-track.js';

// 9 forward-hemisphere sensor rays at 22.5° increments.
export const SENSOR_ANGLES = [
  -Math.PI / 2, -Math.PI * 3 / 8, -Math.PI / 4, -Math.PI / 8,
  0,
  Math.PI / 8, Math.PI / 4, Math.PI * 3 / 8, Math.PI / 2,
];
export const NUM_SENSORS = SENSOR_ANGLES.length;
export const STUCK_LIMIT = 120;

export class HeadlessCar {
  constructor(track, brain, teamIdx, speedMult) {
    this.track = track;
    this.brain = brain || new NeuralCar();
    this.speedMult = speedMult;

    const start = track.getStartPos();
    const t = track.tangents[0];
    const nx = -t[1], nz = t[0];
    const row = Math.floor(teamIdx / 2);
    const col = (teamIdx % 2) - 0.5;

    this.x = start.x - t[0] * row * 4 + nx * col * 8;
    this.z = start.z - t[1] * row * 4 + nz * col * 8;
    this.angle = start.angle;
    this.speed = 0;
    this.sensors = new Array(SENSOR_ANGLES.length).fill(0);
    this.lapTime = 0;
    this.totalProgress = 0;
    const initProg = track.getProgress(this.x, this.z);
    this.initialProgress = initProg;
    this.lastProgress = initProg;
    this.lastProgressIdx = Math.round(initProg * track.points.length);
    this.progressAccum = 0;
    this.score = 0;
    this.alive = true;
    this.finished = false;
    this.killReason = 'active';
    this.frameCounter = 0;
    this.stuckFrames = 0;
    this.reverseAccum = 0;
    this.wrongWayFrames = 0;
    // Lap checkpoints — all must be passed before finish-line crossing counts.
    this.passedQuarter = false;
    this.passedHalf = false;
    this.passedThreeQuarter = false;
  }

  update() {
    if (!this.alive || this.finished) return;

    for (let i = 0; i < SENSOR_ANGLES.length; i++) {
      this.sensors[i] = this.track.castRay(this.x, this.z, this.angle + SENSOR_ANGLES[i]) / HEADLESS_SENSOR_LENGTH;
    }

    const inputs = [...this.sensors, this.speed / 8.1];
    const decision = this.brain.think(inputs);
    this.angle += decision.steer * 0.08;
    this.speed = (2.5 + (decision.gas + 1) * 2.8) * this.speedMult;

    const cosA = Math.cos(this.angle);
    const sinA = Math.sin(this.angle);
    const newX = this.x + cosA * this.speed;
    const newZ = this.z + sinA * this.speed;

    // Midpoint collision to prevent wall tunneling at high speeds
    const midX = (this.x + newX) * 0.5;
    const midZ = (this.z + newZ) * 0.5;
    if (!this.track.isOnTrack(midX, midZ) || !this.track.isOnTrack(newX, newZ)) {
      this.alive = false;
      this.killReason = 'offtrack';
      this.score = computeScore(this);
      return;
    }

    this.x = newX;
    this.z = newZ;

    const { progress, idx } = this.track.getProgressLocal(this.x, this.z, this.lastProgressIdx);
    this.lastProgressIdx = idx;
    const rawDelta = progress - this.lastProgress;
    let delta = rawDelta;
    const crossedFinish = rawDelta < -0.5;
    if (crossedFinish) delta += 1.0;
    if (delta > 0.5) delta -= 1.0;

    // Cap delta to prevent progress aliasing on overlapping track geometry
    if (delta > 0.05) delta = 0.05;
    if (delta < -0.05) delta = -0.05;

    if (delta > 0) { this.progressAccum += delta; this.stuckFrames = 0; this.reverseAccum = 0; }
    else { this.stuckFrames++; this.reverseAccum += delta; }
    this.lastProgress = progress;
    this.totalProgress = this.progressAccum;

    if (this.stuckFrames > STUCK_LIMIT) { this.alive = false; this.killReason = 'stuck'; return; }

    // Reverse driving — kill within ~10 frames of wrong-way driving
    if (this.reverseAccum < -0.05) { this.alive = false; this.killReason = 'reverse'; return; }

    // Wrong-way via velocity-tangent alignment — catches aliasing at overlaps
    if (this.speed > 0.5) {
      const tg = this.track.tangents[idx];
      const dot = cosA * tg[0] + sinA * tg[1];
      if (dot < -0.3) {
        this.wrongWayFrames++;
        if (this.wrongWayFrames > 5) { this.alive = false; this.killReason = 'wrong_way'; return; }
      } else {
        this.wrongWayFrames = 0;
      }
    }

    // Scaled loitering detection
    if (this.frameCounter > 200 && this.progressAccum < this.frameCounter * 0.00015) {
      this.alive = false;
      this.killReason = 'loitering';
      this.score = computeScore(this);
      return;
    }

    this.frameCounter++;

    // Ordered checkpoints — each must be passed before the next counts
    if (!this.passedQuarter && progress >= 0.20 && progress <= 0.35) {
      this.passedQuarter = true;
    }
    if (this.passedQuarter && !this.passedHalf && progress >= 0.45 && progress <= 0.60) {
      this.passedHalf = true;
    }
    if (this.passedHalf && !this.passedThreeQuarter && progress >= 0.70 && progress <= 0.85) {
      this.passedThreeQuarter = true;
    }

    // Lap completion requires ALL checkpoints + finish-line crossing (or
    // accumulated progress reaching threshold AND all checkpoints visited)
    const allCheckpointsHit =
      this.passedQuarter && this.passedHalf && this.passedThreeQuarter;
    const reachedTarget =
      allCheckpointsHit && this.progressAccum >= HEADLESS_LAP_COMPLETION_PROGRESS;
    const crossedFinishLine = allCheckpointsHit && crossedFinish;
    if (reachedTarget || crossedFinishLine) {
      this.finished = true;
      this.killReason = 'finished';
      this.lapTime = this.frameCounter;
    }

    this.score = computeScore(this);
  }
}

// Run all cars in a population for one generation. Returns when every car
// has finished or died, or the timeout is reached.
export function runGeneration(track, cars, timeout) {
  for (let frame = 0; frame < timeout; frame++) {
    let anyAlive = false;
    for (const car of cars) {
      car.update();
      if (car.alive && !car.finished) anyAlive = true;
    }
    if (!anyAlive) break;
  }
  for (const c of cars) {
    if (c.alive && !c.finished) {
      c.alive = false;
      if (!c.killReason || c.killReason === 'active') c.killReason = 'timeout';
    }
  }
}
