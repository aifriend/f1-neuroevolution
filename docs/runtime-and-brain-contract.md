# Runtime And Brain Contract

## Brain JSON Format

Loaded brain files must be strict JSON with this shape:

```json
{
  "w1": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "... 5 rows total"],
  "b1": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "w2": [[0.0, 0.0], "... 6 rows total"],
  "b2": [0.0, 0.0]
}
```

Validation rules:

- `w1`: 5x6 matrix (`NUM_SENSORS` x `HIDDEN_SIZE`)
- `b1`: 6 numbers
- `w2`: 6x2 matrix (`HIDDEN_SIZE` x output size)
- `b2`: 2 numbers
- All values must be finite numbers

If validation fails, the simulator rejects the payload and keeps running.

## Save/Load UX Behavior

- `Save Brain` shows a status message.
- Save is blocked until a best car exists.
- `Load Brain` validates structure before accepting.
- A successfully loaded brain is applied to car `#0` in the next generation.

## Track Switching Resource Lifecycle

When `Apply & Restart` switches tracks:

- old track mesh is removed from scene
- all old track geometries, materials, and textures are disposed
- a fresh `Track` is created and added

This prevents GPU memory growth from repeated track changes.

Track geometry sampling is adaptive per layout:

- interpolation steps are computed from the longest control-point segment
- target centerline spacing stays below the collision-grid cell size
- this avoids sparse-grid holes that can cause false off-track deaths on long straights

## Browser vs Headless Parity Note

`train.js` and browser runtime use the same lap completion threshold:

- `LAP_COMPLETION_PROGRESS = 1.0`

When changing scoring/physics constants, update both paths together unless intentionally diverging.

Both paths also share adaptive mutation restart behavior:

- hard restart only triggers after prolonged stagnation (45+ gens) with a sustained weak-signal streak
- active finisher/progress signals favor mutation escalation instead of forced reset
- restart cooldown prevents immediate repeated resets
- all-time best brain is kept as an elite seed each generation to prevent regression from losing a prior breakthrough

## HUD Performance Behavior

- Neural-network visualization redraw is throttled to 10 Hz (`100ms` interval).
- Training stats and lap graph still render every frame.
- Short viewports clamp panel positions to stay visible.
