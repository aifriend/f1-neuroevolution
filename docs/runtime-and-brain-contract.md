# Runtime And Brain Contract

## Brain JSON Format

Loaded brain files must be strict JSON and match one of the supported formats.

### V2 (current, LoRA-aware)

```json
{
  "version": 2,
  "rank": 2,
  "currentLevel": 0,
  "base": {
    "w1": [[0.0, "... 16 cols"], "... 10 rows total"],
    "b1": ["... 16 values"],
    "w2": [[0.0, 0.0], "... 16 rows total"],
    "b2": [0.0, 0.0]
  },
  "adapters": {
    "1": {
      "A1": [[0.0, 0.0], "... 10 rows total"],
      "B1": [[0.0, "... 16 cols"], "... 2 rows total"],
      "A2": [[0.0, 0.0], "... 16 rows total"],
      "B2": [[0.0, 0.0], "... 2 rows total"]
    }
  }
}
```

Validation rules (from `js/brain.js` + `js/nn.js`):

- base network shape:
  - `base.w1`: `NUM_INPUTS x HIDDEN_SIZE` (`10x16`)
  - `base.b1`: `16` numbers
  - `base.w2`: `HIDDEN_SIZE x OUTPUT_SIZE` (`16x2`)
  - `base.b2`: `2` numbers
- adapter shape for each `adapters.<level>` with rank `r`:
  - `A1`: `10xr`
  - `B1`: `rx16`
  - `A2`: `16xr`
  - `B2`: `rx2`
- All numeric values must be finite numbers.

### V1 (legacy, base-only)

Legacy flat payloads are still accepted and treated as base weights:

```json
{
  "w1": [[0.0, "... 16 cols"], "... 10 rows total"],
  "b1": ["... 16 values"],
  "w2": [[0.0, 0.0], "... 16 rows total"],
  "b2": [0.0, 0.0]
}
```

If validation fails, the simulator rejects the payload and keeps running.

## Save/Load UX Behavior

- `Save Brain` shows a status message.
- Save is blocked until a best car exists.
- `Load Brain` validates structure before accepting.
- A successfully loaded brain is applied to car `#0` in the next generation.
- Runtime training state is auto-persisted to `localStorage` key `f1-neuroevo-state`
  (exported as `TRAINING_STATE_STORAGE_KEY` in `js/storage-keys.js`)
  on quit, tab hide, and page unload.
- Persisted state includes full population weights + generation counters + current
  settings so the next app start resumes training from the same snapshot (not only
  the best model).
- `Reset training on apply` checkbox forces a clean restart and clears the
  persisted snapshot.

## Visual Runtime URL Params

The visual simulator can be configured via query params on `/` (e.g. `http://localhost:8000/?track=monaco&cars=30`).

Supported params (all optional):

- `track`: one of the keys in `TRACK_DEFAULT_WIDTHS` (`js/evolution-core.js`)
- `cars`: population size, clamped to `5..100`
- `speed`: simulation speed multiplier, clamped to `0.5..3`
- `mutation`: base mutation rate, clamped to `0.02..0.3`
- `timeout`: `1` enables frame timeout, `0` disables
- `timeoutFrames`: frame timeout duration, clamped to `100..5000`
- `fresh`: `1` clears persisted training state (`localStorage` key `f1-neuroevo-state`) before starting; otherwise the runtime may restore saved state

### One-command Monaco Start

`scripts/run-visual-monaco.sh` starts a static server and opens a URL with default params:

- `track=monaco`
- `cars=30`
- `speed=1`
- `mutation=0.08`
- `timeout=1`
- `timeoutFrames=3500`
- `fresh=1`

## Track Switching Resource Lifecycle

When `Apply & Restart` switches tracks:

- old track mesh is removed from scene
- all old track geometries, materials, and textures are disposed
- a fresh `Track` is created and added

This prevents GPU memory growth from repeated track changes.

### Track Catalog Parity

Track definitions must stay in sync between browser runtime (`js/track.js`) and
headless trainer (`train.js`). This includes the newest technical layout:

- `stormfront_gp` (Stormfront GP), default width `20`
- `serpentine_bay` (Serpentine Bay), default width `20`
- `ironcliff` (Ironcliff), default width `18`

Track geometry sampling is adaptive per layout:

- interpolation steps are computed from the longest control-point segment
- target centerline spacing stays below the collision-grid cell size
- this avoids sparse-grid holes that can cause false off-track deaths on long straights

### Track Drivability Invariant

Every track must be completable by a simple lookahead oracle driver at stock
physics (min speed `2.5 u/f`, max `8.1 u/f`, steer rate `0.08 rad/f`,
corresponding vehicle min turning radius `31.25 u`).

This invariant is enforced by `tests/track-layout-parity.test.js`, which
interpolates each track, builds the collision grid, and simulates a full
lap with a lookahead+brake policy. The test guards against Catmull-Rom
cusps created by tightly clustered control points (neighbors < ~25 u apart
with sharp direction changes), which in earlier `serpentine_bay` geometry
produced centerline radii as tight as `0.4 u` — physically impossible to
drive regardless of neural-network quality.

The same cusp failure mode was also present in older `ironcliff` and
`stormfront_gp` SVG-derived traces; both were re-authored with lower-density,
monotonic control-point flow and are now included in the oracle drivability
suite.

Authoring guidelines for new control-point layouts:

- keep neighbor spacing `>= 30 u`
- avoid direction reversals across three consecutive points
- target centerline radius `>= 25 u` in every corner
- run the drivability test after editing any track definition

## Browser vs Headless Parity Note

`train.js` and browser runtime use the same lap completion threshold:

- `HEADLESS_LAP_COMPLETION_PROGRESS = 0.995` (`js/headless-track.js`)

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
