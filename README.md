# F1 Neuroevolution Sim

Browser-based F1-style racing simulation where cars learn to drive via neuroevolution, plus a fast headless trainer for generating brain weights.

## Requirements

- Node.js 18+ (tested with modern ESM support)
- npm
- A local static server for browser runtime

## Install

```bash
npm install
```

## Run Visual Simulator

Serve the project root, then open `index.html` through `http://localhost`:

```bash
python3 -m http.server 8000
```

Then open [http://localhost:8000](http://localhost:8000).

### One-command Monaco start (default params)

This starts a static server, opens the browser at Monaco, and forces a fresh run
ignoring any persisted localStorage training state.

```bash
bash scripts/run-visual-monaco.sh
```

Runtime controls in the UI:

- Track: Monaco / Suzuka / Silverstone / Spaghetti / Serpentine / Inferno / Serpentine Bay / Ironcliff
- Cars, speed multiplier, mutation rate
- Timeout toggle and frame limit
- Save Brain / Load Brain / Apply & Restart / Quit

## Run Headless Training

Fast CLI training is implemented in `train.js`.

```bash
node train.js
node train.js --track suzuka --cars 80 --gens 1000
node train.js --output my-brain.json
```

### Live Logs (Headless)

`train.js` prints progress to stdout while running.

```bash
node train.js --track suzuka --cars 80 --gens 1000
```

To both view logs live and save them:

```bash
node train.js --track suzuka --cars 80 --gens 1000 2>&1 | tee training-live.log
```

To follow the saved log in another terminal:

```bash
tail -f training-live.log
```

Available flags:

- `--track` (`monaco` | `suzuka` | `silverstone` | `spaghetti` | `serpentine` | `inferno` | `serpentine_bay` | `ironcliff`)
- `--cars` (population size)
- `--gens` (number of generations)
- `--mutation` (base mutation rate)
- `--timeout` (frame timeout)
- `--speed` (simulation speed multiplier)
- `--output` (output JSON file path; defaults to `models/best-brain-<track>.json`)

## Test

```bash
npm test
```

Uses Vitest with tests in `tests/` for shared evolution and NN behavior.

## Brain JSON Contract

Loaded brain JSON must match the expected shape and finite-number constraints documented in `docs/runtime-and-brain-contract.md`.

At runtime, invalid brain payloads are rejected and the simulator keeps running.

## Plateau-Based Curriculum Escalation

Difficulty no longer advances after a fixed generation count. Instead, escalation is
triggered when learning is statistically flat:

- EWMA slope of recent `avgProgress` is near zero
- Relative improvement between adjacent windows is small
- Progress variance is low (stable plateau, not noisy exploration)
- Finisher rate is not still climbing materially
- Plateau must be confirmed for multiple consecutive checks
- At least one valid lap must exist before escalation can occur

This policy is shared between browser runtime (`js/evolution.js`) and headless
training (`train.js`) through `js/evolution-core.js`, so both modes make
consistent level-jump decisions.

## Project Structure

- `index.html`: browser entrypoint and simulation controls
- `js/`: simulation runtime (car, track, evolution, NN, HUD)
- `train.js`: headless trainer using shared evolution logic
- `tests/`: Vitest unit tests
- `docs/runtime-and-brain-contract.md`: runtime and brain validation contract
