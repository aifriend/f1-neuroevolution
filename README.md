<h1 align="center">F1 Neuroevolution</h1>

<p align="center">
  <em>A population of AI cars learns F1-style racing from scratch through neuroevolution — across a 10-level curriculum of tracks, in your browser.</em>
</p>

<p align="center">
  <img alt="JavaScript" src="https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black">
  <img alt="Three.js"   src="https://img.shields.io/badge/Three.js-000000?style=flat&logo=threedotjs&logoColor=white">
  <img alt="Node.js"    src="https://img.shields.io/badge/Node.js-339933?style=flat&logo=nodedotjs&logoColor=white">
  <img alt="Vitest"     src="https://img.shields.io/badge/Vitest-6E9F18?style=flat&logo=vitest&logoColor=white">
  <img alt="License"    src="https://img.shields.io/badge/license-MIT-green.svg">
  <img alt="Status"     src="https://img.shields.io/badge/status-active-success.svg">
</p>

<p align="center">
  <img src="docs/hero.png" alt="Cars learning to race across generations" width="720">
</p>

---

## Overview

A browser-native simulation in which a population of simple neural-network agents learns to drive F1-style cars on hand-designed tracks. No hand-coded driving logic, no reinforcement learning library — just **mutation and selection**.

The system trains across a **10-level curriculum** over 9 tracks with narrowing widths. A **plateau-based escalation** rule advances to the next level only once fitness stops improving — and only if enough cars are actually finishing laps. Once level 0 is mastered, the base network is frozen and each new level learns a small **LoRA-style low-rank adapter** instead, so skills from earlier levels are retained by construction rather than re-learned.

Two ways to run it:

- **Browser** — serve the project root and open the page; watch the cars learn live with Three.js visualization.
- **Headless trainer** — `node train.js` runs generations far faster than real time, saves the best brain as JSON, and the browser can load that same file to watch it drive.

## Highlights

- 🧬 **Pure neuroevolution** — no gradient descent, no RL libraries. Tournament selection (k=3) + Gaussian mutation over a tiny 10→16→2 network (210 parameters).
- 🧠 **LoRA continual learning** — after level 0 the base net freezes; each curriculum level trains its own rank-2 adapter (88 params/level), a catastrophic-forgetting defense borrowed from LLM fine-tuning.
- 🏁 **10-level curriculum** with plateau detection and a robustness gate — a "lucky elite" plateau where only a handful of cars lap does not promote a fragile adapter.
- ⚡ **Headless trainer + eval harness** — `train.js` for training, `cross-track-eval.js` to measure per-level retention of any brain, `npm run train:e2e` for the full pipeline.
- 🎮 **Three.js visualization** — orbital camera, HUD with curriculum panel and live NN visualization, track switching mid-run.
- 🧪 **Vitest test suite** plus a zero-dependency integration suite (`node test-integration.js`).

## How it works

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Population of   │    │  Run all cars    │    │ Tournament       │
│  genomes         │───▶│  on track until  │───▶│ selection (k=3)  │
│  (NN weights)    │    │  lap or timeout  │    │ + Gaussian noise │
└──────────────────┘    └──────────────────┘    └────────┬─────────┘
                                                          │
                                                          ▼
                                                ┌──────────────────┐
                                                │  Next generation │
                                                └──────────────────┘
                                                          │
                          ┌───── plateau detector ────────┘
                          │  (fitness flat + enough finishers)
                          ▼
                  ┌──────────────────┐
                  │  Freeze base net │
                  │  + train next    │
                  │  level's adapter │
                  └──────────────────┘
```

Each car's network takes a fixed-length **ray-cast sensor array** (9 distances to track walls, plus normalized speed) and outputs `[steering, gas]`. Fitness rewards progress around the centerline; laps must pass every checkpoint in order.

After the population masters a level, escalation is gated on a plateau signal: EWMA slope near zero, small relative gain between windows, low variance, finisher rate not still climbing, at least one valid lap — and at least 10% of the latest generation finishing laps (the robustness gate). Both the browser app and the headless trainer share the same escalation logic via `js/evolution-core.js`, so they make consistent level-jump decisions.

## Getting started

### Prerequisites

- Node.js 20+ (CI runs 20.x and 22.x)
- A modern browser with WebGL

### Installation

```bash
git clone https://github.com/aifriend/f1-neuroevolution.git
cd f1-neuroevolution
npm install
```

### Live training in the browser

Serve the project root with any static server, then open the Monaco URL with default training params. The `fresh=1` param forces a new run, ignoring persisted localStorage state:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000/?track=monaco&cars=30&speed=1&mutation=0.08&timeout=1&timeoutFrames=3500&fresh=1>.

Runtime controls in the UI:

- Track: Monaco / Suzuka / Silverstone / Spaghetti / Serpentine / Inferno / Serpentine Bay / Ironcliff / Stormfront GP
- Cars, speed multiplier, mutation rate
- Timeout toggle and frame limit
- Save Brain / Load Brain / Apply & Restart / Quit

### Headless training

Fast CLI training is implemented in `train.js`:

```bash
node train.js                          # defaults: monaco, 30 cars, 500 gens
node train.js --track suzuka --cars 80 --gens 1000
node train.js --output my-brain.json
```

The full end-to-end pipeline (test preflight → 10-level curriculum training → artifact validation → retention eval):

```bash
npm run train:e2e
```

### Tests

```bash
npm test                      # vitest unit tests
node test-integration.js      # zero-dependency integration suite
npm run lint                  # eslint
```

## Project structure

```
.
├── js/
│   ├── nn.js                # 10→16→2 network, LoRA adapters, brain (de)serialization
│   ├── evolution-core.js    # Shared selection, scoring, plateau/escalation policy
│   ├── evolution.js         # Browser generation loop + localStorage persistence
│   ├── car.js               # Car physics + F1 car mesh (Three.js)
│   ├── track-data.js        # Single source of truth for all track layouts
│   ├── headless-track.js    # Shared headless physics (train/eval parity)
│   ├── headless-car.js      #   ↑ used by both train.js and the browser
│   └── hud.js, main.js, ... # Rendering, dashboard, scene setup
├── train.js                 # Headless CLI trainer
├── cross-track-eval.js      # Per-level retention eval for any brain JSON
├── test-integration.js      # Zero-dependency integration suite
├── scripts/run-full-training.js  # npm run train:e2e pipeline
├── models/                  # Trained brain JSONs (gitignored)
├── tests/                   # Vitest unit tests
└── docs/                    # Design notes, brain format contract, hero image
```

## Configuration

`train.js` flags (see `node train.js` header comments and `scripts/run-full-training.js --help` for the full list):

| Flag | Default | Meaning |
|---|---|---|
| `--track` | monaco | Track to train on |
| `--cars` | 30 | Population size per generation |
| `--gens` | 500 | Max generations |
| `--mutation` | — | Gaussian mutation scale |
| `--load` / `--output` | — | Brain JSON to start from / to save |
| `--width` | — | Track width override (e.g. `--track ironcliff --width 16` for level 9) |
| `--startLevel` | — | Force which adapter/level to train |
| `--rank` | 2 | LoRA adapter rank for fresh brains |
| `--noLora` | off | Disable adapters (full base training, stays on level 0) |
| `--minFinisherRate` | 0.10 | Robustness gate threshold for escalation |

## Results

The flagship artifact `models/best-brain-monaco-2026-08-27T09-59-42-356Z.json`
(level 9, adapters 1–9) completed the full 10-level curriculum in a single
`npm run train:e2e` run and retains every level at 1.0× speed. Verified with
`node cross-track-eval.js models/best-brain-monaco-2026-08-27T09-59-42-356Z.json --cars 12 --timeout 4000 --speed 1`:

| Lv | Track | Finishers | Avg score |
|----|-------|-----------|-----------|
| 0 | monaco | 10/12 | 13793.18 |
| 1 | suzuka | 12/12 | 5923.82 |
| 2 | silverstone | 12/12 | 7357.75 |
| 3 | spaghetti | 12/12 | 7854.43 |
| 4 | serpentine | 12/12 | 12459.24 |
| 5 | inferno | 12/12 | 8636.52 |
| 6 | serpentine_bay | 12/12 | 10886.18 |
| 7 | serpentine_bay (width 18) | 12/12 | 10857.30 |
| 8 | ironcliff | 12/12 | 11626.40 |
| 9 | ironcliff (width 16) | 12/12 | 11732.29 |

The previous flagship (`models/best-brain-monaco-2026-04-30T10-37-48-198Z.json`)
retained only 8 of 10 levels under the same eval (silverstone and serpentine
scored 0/12). The 2026-08-27 run is the first single brain that passes all ten.

## Roadmap

- [ ] Save and replay best genome alongside the live population
- [ ] Compare two genomes side-by-side on the same track
- [ ] Add an "NPCs" mode — load a previously-trained genome as a competitor
- [ ] Optional NEAT-style structural mutation (currently weights-only)
- [ ] WebGPU port for the training loop

## Why this project exists

I'm interested in **gradient-free learning** as a primitive — what circuits look like when they're shaped purely by selection pressure. F1-style racing is a clean test case: dense reward signal (distance), continuous control, easy to visualize, and the curriculum lets you watch competence emerge level by level.

## License

MIT — see [LICENSE](LICENSE).

## Author

**Jose Lopez** — AI engineer in Madrid, working on the intersection of biological and artificial intelligence.
