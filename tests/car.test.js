import { beforeAll, describe, expect, it, vi } from 'vitest';

class MockGroup {
  constructor() {
    this.children = [];
    this.position = { set: () => {} };
    this.rotation = { y: 0 };
  }

  add(child) {
    this.children.push(child);
  }
}

class MockMesh {
  constructor() {
    this.position = { set: () => {}, y: 0 };
    this.rotation = { x: 0, y: 0, z: 0 };
    this.castShadow = false;
    this.receiveShadow = false;
    this.material = {};
  }

  add() {}

  traverse() {}
}

vi.mock('three', () => ({
  Group: MockGroup,
  Mesh: MockMesh,
  MeshStandardMaterial: class {},
  BoxGeometry: class {},
  SphereGeometry: class {},
  TorusGeometry: class {},
  CylinderGeometry: class {},
}));

function createMockTrack(progressSeries) {
  const points = Array.from({ length: 100 }, (_, i) => [i, 0]);
  const tangents = Array.from({ length: 100 }, () => [1, 0]);
  let i = 0;
  return {
    points,
    tangents,
    getStartPos: () => ({ x: 0, z: 0, angle: 0 }),
    getProgress: () => 0,
    getProgressLocal: () => {
      const progress = progressSeries[Math.min(i, progressSeries.length - 1)];
      i++;
      return { progress, idx: Math.round(progress * (points.length - 1)) };
    },
    castRay: () => 220,
    isOnTrack: () => true,
  };
}

describe('Car lap completion', () => {
  let Car;

  beforeAll(async () => {
    ({ Car } = await import('../js/car.js'));
  });

  it('finishes when accumulated lap milestones are met even with sparse sampled progress windows', () => {
    const sparseProgressSamples = [
      0.10, 0.40, 0.70, 0.90, 0.98,
      0.15, 0.40, 0.70, 0.90, 0.98,
      0.15, 0.40, 0.70, 0.90, 0.98,
      0.15, 0.40, 0.70, 0.90, 0.98,
    ];
    const track = createMockTrack(sparseProgressSamples);
    const car = new Car(track, { think: () => ({ steer: 0, gas: -1 }) }, 0, 1);

    for (let frame = 0; frame < sparseProgressSamples.length; frame++) {
      car.update();
      if (car.finished) break;
    }

    expect(car.finished).toBe(true);
    expect(car.killReason).toBe('finished');
    expect(car.lapTime).toBeGreaterThan(0);
  });
});
