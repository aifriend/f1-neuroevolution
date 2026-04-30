# End-to-End Training Loop

Use the wrapper script to run a full headless training cycle with one command:

```bash
npm run train:e2e
```

What it does:

- Runs test preflight (`npm test`) by default.
- Runs `train.js` with validated arguments.
- Verifies the output brain artifact exists and has required weight structure.

## Common Usage

Default full run:

```bash
npm run train:e2e
```

Custom run:

```bash
npm run train:e2e -- --track suzuka --cars 120 --gens 2000 --speed 1.2
```

Skip tests:

```bash
npm run train:e2e -- --skipTests
```

Disable two-phase:

```bash
npm run train:e2e -- --noTwoPhase
```

Use warm-start:

```bash
npm run train:e2e -- --load models/best-brain-monaco.json
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
- `--noTwoPhase`, `--noLora`, `--skipTests`, `--dryRun`
