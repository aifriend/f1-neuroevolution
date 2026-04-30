# End-to-End Training Loop

Use the wrapper script to run a full headless training cycle with one command:

```bash
npm run train:e2e
```

What it does:

- Runs test preflight (`npm test`) by default.
- Runs `train.js` with validated arguments + production-tuned defaults.
- Verifies the output brain artifact exists and has required weight structure.
- Runs `cross-track-eval.js` against the saved brain so per-level retention
  appears in every run (skip with `--skipEval`).

## Production Defaults (post-v8 fix chain)

| Flag | Default | Rationale |
|------|---------|-----------|
| `--cars` | 80 | Population size for tournament selection. |
| `--gens` | 2000 | Empirical budget for full 10-level curriculum with clone-test gating: ~400 gens to find a lapping L0 brain, then ~150 gens per validated level escalation. |
| `--timeout` | 4000 frames | Comfortable headroom for ironcliff and serpentine_bay laps. |
| `--mutation` | 0.05 | Base mutation rate; adaptive escalation kicks in on plateau. |
| `--speed` / `--slow` | 1.0 / 0.5 | Two-phase: each level trains slow then fast. |
| `--rank` | 2 | LoRA rank; 88 params per level adapter. |
| `--minFinisherRate` | 0.10 | Robustness gate — blocks plateau escalation when only a fragile elite is lapping. |
| `--robustWindow` | 20 | Rolling window of (finisherRate, bestScore) for clone-test candidate pool. |
| `--cloneTestEvery` | 10 | Clone-test cadence in gens. |
| `--cloneTestCars` | 16 | Cars per clone-test re-spawn. |
| `--cloneTestK` | 3 | Top-K candidates from robust window to test each round. |

The clone-test gating + robustness gate combination is what closed the
"L2/L4 = 0/16 finishers" eval-vs-train gap. See
`docs/runtime-and-brain-contract.md` for mechanism details.

## Common Usage

Default full run (full 10-level curriculum, ~2000 gens):

```bash
npm run train:e2e
```

Custom run:

```bash
npm run train:e2e -- --track suzuka --cars 120 --gens 3000 --speed 1.2
```

Warm-start a dwell at a specific level:

```bash
npm run train:e2e -- --load models/best-brain-monaco.json --startLevel 2 --gens 1500
```

Skip preflight tests:

```bash
npm run train:e2e -- --skipTests
```

Skip post-training cross-track-eval:

```bash
npm run train:e2e -- --skipEval
```

Disable two-phase:

```bash
npm run train:e2e -- --noTwoPhase
```

Tighten clone-test (more validation cars, every 5 gens):

```bash
npm run train:e2e -- --cloneTestEvery 5 --cloneTestCars 32
```

Disable clone-test gating entirely (legacy fragile-elite behavior):

```bash
npm run train:e2e -- --cloneTestEvery 0 --minFinisherRate 0
```

Dry-run command preview:

```bash
npm run train:e2e -- --dryRun
```

## Supported Flags

- `--track` one of `monaco|suzuka|silverstone|spaghetti|serpentine|inferno|serpentine_bay|ironcliff|stormfront_gp`
- `--cars` integer in `[2, 1000]`
- `--gens` integer in `[1, 200000]`
- `--mutation` float in `[0, 1]`
- `--timeout` integer frames in `[200, 50000]`
- `--speed` float in `(0, 25]`
- `--slow` float in `(0, speed]`
- `--rank` integer in `[1, 64]`
- `--softFreeze` float in `[0, 1]`
- `--width` optional integer in `[4, 200]`
- `--output` output JSON path relative to project root
- `--load` warm-start brain JSON path relative to project root
- `--startLevel` integer in `[0, 30]` — explicit starting curriculum level (overrides loaded brain's currentLevel)
- `--minFinisherRate` float in `[0, 1]` — population finisher-rate gate for escalation (0 disables)
- `--robustWindow` integer in `[1, 1000]` — rolling-window length for robust-elite tracking
- `--cloneTestEvery` integer in `[0, 10000]` — clone-test cadence in gens (0 disables)
- `--cloneTestCars` integer in `[1, 200]` — cars per clone-test
- `--cloneTestK` integer in `[1, 50]` — top-K candidates per clone-test
- `--evalCars` integer in `[1, 200]` — cars per level in post-training eval
- `--evalTimeout` integer in `[200, 50000]` — eval-phase frame timeout
- `--noTwoPhase`, `--noLora`, `--skipTests`, `--skipEval`, `--dryRun`
