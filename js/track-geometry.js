const DEFAULT_MIN_STEPS_PER_SEGMENT = 20;
const DEFAULT_MAX_STEPS_PER_SEGMENT = 64;
const DEFAULT_TARGET_POINT_SPACING = 4;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

// Choose interpolation density from control-point spacing so grid collision
// coverage remains contiguous on long straights (notably Silverstone/Suzuka).
export function computeInterpolationSteps(
  controlPoints,
  targetPointSpacing = DEFAULT_TARGET_POINT_SPACING,
  minSteps = DEFAULT_MIN_STEPS_PER_SEGMENT,
  maxSteps = DEFAULT_MAX_STEPS_PER_SEGMENT
) {
  if (!Array.isArray(controlPoints) || controlPoints.length < 2) {
    return minSteps;
  }

  let maxSegmentLength = 0;
  const pointCount = controlPoints.length;
  for (let i = 0; i < pointCount; i++) {
    const a = controlPoints[i];
    const b = controlPoints[(i + 1) % pointCount];
    const dx = b[0] - a[0];
    const dz = b[1] - a[1];
    const segmentLength = Math.hypot(dx, dz);
    if (segmentLength > maxSegmentLength) {
      maxSegmentLength = segmentLength;
    }
  }

  const spacing = Math.max(1, targetPointSpacing);
  const rawSteps = Math.ceil(maxSegmentLength / spacing);
  return clamp(rawSteps, minSteps, maxSteps);
}
