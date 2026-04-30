import { describe, expect, it } from 'vitest';
import {
  HeadlessTrack,
  HEADLESS_LAP_COMPLETION_PROGRESS,
  HEADLESS_SENSOR_LENGTH,
} from '../js/headless-track.js';

describe('HeadlessTrack', () => {
  it('constructs from canonical track data and exposes start pose', () => {
    const track = new HeadlessTrack('monaco');
    const start = track.getStartPos();
    expect(Number.isFinite(start.x)).toBe(true);
    expect(Number.isFinite(start.z)).toBe(true);
    expect(Number.isFinite(start.angle)).toBe(true);
    expect(track.points.length).toBeGreaterThan(0);
    expect(track.tangents.length).toBe(track.points.length);
  });

  it('respects explicit width override', () => {
    const track = new HeadlessTrack('monaco', 16);
    expect(track.trackWidth).toBe(16);
  });

  it('keeps raycast distances within sensor bounds', () => {
    const track = new HeadlessTrack('monaco');
    const start = track.getStartPos();
    const d = track.castRay(start.x, start.z, start.angle);
    expect(d).toBeGreaterThanOrEqual(2);
    expect(d).toBeLessThanOrEqual(HEADLESS_SENSOR_LENGTH);
  });
});

describe('headless shared constants', () => {
  it('exports expected lap completion threshold', () => {
    expect(HEADLESS_LAP_COMPLETION_PROGRESS).toBeCloseTo(0.995);
  });
});
