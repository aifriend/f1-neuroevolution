// SINGLE SOURCE OF TRUTH for curriculum track layouts.
//
// Both the browser runtime (js/track.js) and the headless trainer
// (train.js) import TRACKS from here. This makes drift between the two
// physically impossible — there is no duplicated geometry to keep in
// sync anymore.
//
// Each entry: { name: displayLabel, width: trackWidth, points: [[x,z], ...] }
// Control points are interpolated with Catmull-Rom; width defines the
// perpendicular half-distance from centerline to track edge.

export const TRACKS = {
  monaco: {
    name: 'Monaco',
    width: 28,
    points: [
      [0, 0], [100, 5], [160, 35], [180, 85],
      [170, 145], [130, 180], [80, 185], [50, 160],
      [35, 120], [55, 85], [40, 45], [10, 15],
    ],
  },
  suzuka: {
    name: 'Suzuka',
    width: 30,
    // Z-SHAPE — every corner is a multi-point rounded arc, not a single
    // pivot point. Each corner has 3-4 sweep points so Catmull-Rom builds a
    // genuine wide radius instead of a sharp bend. Verified clearances at
    // width 30 (surface 60 wide): every centerline pair > 60 units apart.
    points: [
      // TOP BAR: TL → TR (going right)
      [-200, 200], [-100, 205], [0, 205], [100, 205], [180, 200],
      // TR corner — 4 arc sweep points (gentle 135° turn)
      [215, 188], [240, 162], [253, 130], [250, 95],
      // DIAGONAL — gentle descent
      [195, 60], [120, 10], [40, -50], [-30, -110], [-100, -150], [-150, -170],
      // BL corner — 3 arc sweep points (gentle 135° turn)
      [-185, -185], [-210, -202], [-215, -222], [-200, -232],
      // BOTTOM BAR (shorter on left to keep BL sweep clearance ≥ 60)
      [-130, -218], [-50, -218], [50, -218], [150, -215], [200, -210],
      // BR corner — 4 arc sweep points (gentle 90° turn into right return)
      [260, -195], [295, -165], [315, -125], [325, -75],
      // RETURN ARC: right side going up (smoother arc apex)
      [335, 0], [342, 90], [335, 175], [320, 230],
      // RETURN ARC: top going left (smoother apex with extra mid-point)
      [285, 275], [225, 305], [130, 325], [0, 330], [-130, 325], [-225, 305],
      // NW return → close back to start (rounded corner)
      [-265, 268], [-265, 225],
    ],
  },
  silverstone: {
    name: 'Silverstone',
    width: 32,
    // S-SHAPE: top bar going right → curve down on right → middle bar
    // going LEFT → curve down on left → bottom bar going right
    // → return arc wrapping outside (right + top) with ≥30u gap.
    // Surface width = 64; all parallel sections separated by ≥30u.
    points: [
      // TOP BAR going RIGHT
      [-180, 200], [-90, 205], [0, 205], [90, 205], [180, 200],
      // Right S-curve descending to middle
      [240, 170], [255, 110], [240, 60],
      // MIDDLE BAR going LEFT
      [180, 30], [90, 25], [0, 25], [-90, 25], [-180, 30],
      // Left S-curve descending to bottom
      [-240, 0], [-255, -60], [-240, -110],
      // BOTTOM BAR going RIGHT
      [-180, -200], [-90, -205], [0, -205], [90, -205], [180, -200],
      // Return: right side going up (pushed to x≈340 — clear of any overlap)
      [290, -180], [340, -100], [350, 0], [350, 110], [330, 220],
      // Return: top going left (y=300 — 35u clear of top bar at y=200)
      [240, 300], [120, 320], [0, 325], [-120, 320], [-240, 300],
      // Close back toward TL
      [-290, 250],
    ],
  },
  spaghetti: {
    name: 'Spaghetti',
    width: 26,
    // L-SHAPE: vertical bar (left) + horizontal bar (bottom).
    // Vertical bar 90 wide (right rail x=-30, left rail x=-120).
    // Horizontal bar 90 tall (top rail y=-100, bottom rail y=-190).
    // Surface gaps: vertical lanes 90-52=38u apart, horizontal lanes 38u apart.
    // The outline IS a closed L letter — no return arc.
    points: [
      // TOP of vertical bar going RIGHT
      [-100, 300], [-30, 300],
      // Right side of vertical bar going DOWN
      [-30, 230], [-30, 160], [-30, 90], [-30, 20], [-30, -50],
      // Inner corner (smoothed)
      [-15, -100], [40, -110],
      // TOP of horizontal bar going RIGHT
      [110, -110], [180, -110], [240, -110],
      // Right side of horizontal bar going DOWN
      [270, -140], [270, -180],
      // BOTTOM of horizontal bar going LEFT
      [200, -200], [120, -200], [40, -200], [-40, -200], [-100, -200],
      // Outer BL going UP (left side of L going up)
      [-130, -170], [-130, -100], [-130, -30],
      // Continue UP left side
      [-130, 50], [-130, 130], [-130, 210], [-130, 280],
    ],
  },
  serpentine: {
    name: 'Serpentine',
    width: 24,
    points: [
      [0, 0], [55, -20], [85, 15], [140, -10], [160, 25],
      [200, 55], [170, 100], [215, 135], [185, 180], [225, 215],
      [195, 265], [155, 285], [180, 325],
      [135, 345], [100, 315], [65, 350],
      [25, 320], [50, 280], [-10, 240],
      [30, 200], [-20, 155], [20, 110],
      [-30, 70], [10, 35], [-10, 10],
    ],
  },
  inferno: {
    name: 'Inferno',
    width: 22,
    // FULLY ASYMMETRIC: 12 distinct corners around a counter-clockwise loop.
    // No two corners share angle, radius, or direction-pattern. Tight point
    // clusters create sharp corners; sparse points create sweepers.
    //   C1:  ultra-tight 90° hairpin (radius ~25, snap)
    //   C2:  gentle 25° kink right
    //   C3:  60° decreasing-radius (sharpens through corner)
    //   C4:  wide 135° sweeping arc
    //   C5:  long parabolic (continuous ~120° curve)
    //   C6a: small chicane left (40°)
    //   C6b: tight chicane right (75°) — different size from C6a
    //   C7:  snap 50° flick
    //   C8:  long-radius 100° sweeper
    //   C9:  85° increasing-radius (opens through corner)
    //   C10: 110° tight bend
    //   C11: 30° gentle closing bend
    points: [
      // START at top going RIGHT (start point is first in array)
      [10, 270], [70, 275], [130, 273], [180, 263],
      // C1: ultra-tight 90° HAIRPIN (radius ~25, sharp snap)
      [225, 240], [240, 210], [240, 175],
      // S1: short straight DOWN
      [220, 140],
      // C2: gentle 25° KINK right (single point shift)
      [228, 95],
      // S2: continuing down-right
      [240, 50], [248, 0],
      // C3: 60° DECREASING-RADIUS (gradual then snap)
      [248, -45], [240, -85], [222, -120], [195, -145],
      // C4: wide 135° SWEEPER going LEFT (long arc, sparse points)
      [150, -180], [90, -200], [25, -210],
      // C5: long PARABOLIC going up-left (continuous curvature)
      [-40, -205], [-100, -190], [-150, -160], [-190, -120],
      // S5: short straight up
      [-215, -75],
      // C6a: SMALL chicane LEFT (40°, gentle)
      [-225, -30], [-215, 5],
      // C6b: TIGHT chicane RIGHT (75°, tighter than C6a)
      [-195, 15], [-175, 45],
      // S6: straight up-left
      [-200, 90],
      // C7: SNAP 50° flick
      [-235, 115], [-265, 130],
      // C8: long-radius 100° sweeper going UP-RIGHT
      [-280, 165], [-275, 200], [-250, 230],
      // C9: 85° opening turn — gently shifts NE direction
      [-210, 248], [-170, 235], [-135, 220],
      // C10: 100° tight bend — smooth chicane (no Catmull-Rom loop artifact)
      [-100, 232], [-75, 218], [-50, 230],
      // C11: 30° gentle closing bend — last point aligned with start (z=270)
      // for smooth Catmull-Rom wrap to first point at (10, 270)
      [-30, 268],
    ],
  },
  serpentine_bay: {
    name: 'Serpentine Bay',
    width: 20,
    // Bay-style CCW loop: top-left start heading west, descending the west
    // coast, sweeping the southern bay, climbing the east coast, and
    // returning along the top. Control points are spaced 30-70 u apart with
    // no reverse-direction kinks so Catmull-Rom never creates sub-vehicle
    // turning-radius cusps. Verified: oracle policy completes a full lap
    // at default physics (min speed 2.5 u/f, steer rate 0.08 rad/f).
    points: [
      // TL start heading west
      [-200, 200],
      // West coast descending
      [-245, 205], [-275, 170], [-290, 110], [-285, 40],
      [-270, -30], [-240, -100],
      // BL sweep across the southern bay
      [-200, -155], [-140, -195], [-70, -215], [10, -220], [85, -215],
      // BR climb up the east coast
      [160, -200], [225, -170], [265, -120], [285, -55],
      [285, 15], [275, 85], [255, 140],
      // Top return heading west back to the start
      [220, 180], [165, 205], [95, 215], [15, 218],
      [-65, 215], [-135, 210],
    ],
  },
  ironcliff: {
    name: 'Ironcliff',
    width: 18,
    // Technical cliffside loop with fast coastal sweep + tight infield exits.
    // Re-authored from the original dense SVG trace to remove Catmull-Rom
    // cusp artifacts while preserving the same start sector and clockwise flow.
    points: [
      [-140, -215], [-220, -220], [-275, -180], [-295, -120],
      [-292, -55], [-275, 5], [-245, 60], [-205, 105],
      [-165, 145], [-130, 185], [-85, 215], [-30, 228],
      [30, 225], [90, 210], [140, 185], [180, 150],
      [210, 105], [230, 50], [235, -10], [220, -65],
      [190, -110], [150, -145], [105, -170], [70, -205],
      [20, -225], [-40, -228], [-95, -223],
    ],
  },
  stormfront_gp: {
    name: 'Stormfront GP',
    width: 20,
    // Stormfront-wide GP loop: long top straight, east descent, south basin,
    // west climb, then high-speed top return. Simplified from the dense SVG
    // trace to maintain technical character without sub-radius spline kinks.
    points: [
      [-10, 216], [90, 220], [180, 218], [250, 200],
      [295, 160], [315, 100], [320, 30], [305, -35],
      [275, -95], [230, -145], [170, -185], [100, -210],
      [20, -220], [-70, -220], [-150, -205], [-215, -175],
      [-265, -130], [-300, -70], [-315, -5], [-310, 60],
      [-285, 125], [-240, 180], [-175, 220], [-95, 240],
      [-30, 235],
    ],
  },
};

export const TRACK_IDS = Object.freeze(Object.keys(TRACKS));
