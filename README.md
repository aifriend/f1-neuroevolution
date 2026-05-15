<h1 align="center">F1 Neuroevolution</h1>

<p align="center">
  <em>80 AI cars learn F1-style racing from scratch through genetic algorithms — across a 7-level curriculum of tracks, in your browser.</em>
</p>

<p align="center">
  <img alt="JavaScript" src="https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black">
  <img alt="Three.js"   src="https://img.shields.io/badge/Three.js-000000?style=flat&logo=threedotjs&logoColor=white">
  <img alt="Node.js"    src="https://img.shields.io/badge/Node.js-339933?style=flat&logo=nodedotjs&logoColor=white">
  <img alt="Vitest"     src="https://img.shields.io/badge/Vitest-6E9F18?style=flat&logo=vitest&logoColor=white">
  <img alt="License"    src="https://img.shields.io/badge/license-MIT-green.svg">
  <img alt="Status"     src="https://img.shields.io/badge/status-active-success.svg">
</p>

<!--
TODO: Add a 2-3 second loop GIF showing cars improving across generations.
Save as docs/hero.gif. Even a still-frame grid (gen 0 / gen 50 / gen 250)
beats text. This is the single highest-impact upgrade.
-->
<p align="center">
  <img src="docs/hero.gif" alt="80 cars learning to race across generations" width="720">
</p>

---

## Overview

A browser-native simulation in which a population of 80 simple neural-network agents learns to drive F1-style cars on procedurally-generated tracks. No hand-coded driving logic, no reinforcement learning — just **mutation and selection**.

The system trains across a **7-level curriculum** of increasingly demanding tracks. Cars that fail to make meaningful progress on a level get culled; their genomes (network weights) are recombined and mutated to produce the next generation. A **plateau-based escalation** rule advances the whole population to the next track only once their best-of-generation fitness stops improving.

Two ways to run it:

- **Browser** — `npm run dev`, open the page, watch the cars learn live with Three.js visualization.
- **Headless trainer** — `npm run train` to run thousands of generations overnight, save the best genome, then load it back into the browser to watch it drive.

## Highlights

- 🧬 **Pure neuroevolution** — no gradient descent, no RL libraries. Genomes are real-valued vectors; selection is tournament-based; mutation is Gaussian.
- 🏁 **7-level curriculum** with plateau detection — population only advances when fitness flatlines.
- ⚡ **Headless trainer** — runs without rendering for fast iteration, then loads the saved genome into the visual mode.
- 🎮 **Three.js visualization** — orbital camera, real-time car colors keyed to fitness, track switching mid-run.
- 🧪 **Vitest test suite** for the simulation core (collision, ray-cast sensors, fitness, plateau detector).

## How it works

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Population of   │    │  Run all 80      │    │ Tournament       │
│  80 genomes      │───▶│  cars on track   │───▶│ selection +      │
│  (NN weights)    │    │  for N seconds   │    │ Gaussian mutation│
└──────────────────┘    └──────────────────┘    └────────┬─────────┘
                                                          │
                                                          ▼
                                                ┌──────────────────┐
                                                │  Next generation │
                                                └──────────────────┘
                                                          │
                          ┌───── plateau detector ────────┘
                          │  (best fitness flat for K gens)
                          ▼
                  ┌──────────────────┐
                  │  Advance to next │
                  │  track in the    │
                  │  curriculum      │
                  └──────────────────┘
```

Each car has a small feed-forward network that takes a fixed-length **ray-cast sensor array** (distances to track walls in N directions) as input and outputs `[steering, throttle]`. Fitness is a function of distance travelled and average speed, with a stiff penalty for collisions.

The 7 tracks are designed to expose distinct skills: straights (raw speed), banked corners (commit), chicanes (rapid reversal), hairpins (low-speed steering), and combinations.

## Getting started

### Prerequisites

- Node.js 18+
- A modern browser with WebGL (any Chrome/Firefox/Safari from the last few years)

### Installation

```bash
git clone https://github.com/aifriend/f1-neuroevolution.git
cd f1-neuroevolution
npm install
```

### Live training in the browser

```bash
npm run dev
```

Open the URL it prints. You'll see 80 cars on track 1. Watch them get less terrible.

### Headless trainer (faster)

```bash
npm run train -- --generations 500 --population 80 --save best.json
```

After training, load `best.json` in the browser:

```bash
npm run dev -- --load best.json
```

### Tests

```bash
npm test
```

## Project structure

```
.
├── src/
│   ├── genome.js         # NN weight vector, mutation, crossover
│   ├── network.js        # Tiny feed-forward NN evaluator
│   ├── car.js            # Physics, ray sensors, fitness
│   ├── tracks/           # 7 procedurally-generated tracks
│   ├── population.js     # Selection + reproduction
│   ├── curriculum.js     # Plateau detection + level advance
│   └── viz/              # Three.js scene, camera, HUD
├── trainer/
│   ├── headless.js       # Non-rendered training loop
│   └── benchmark.js
├── tests/                # Vitest unit tests
└── docs/                 # Hero image, screenshots
```

## Configuration

Tweakable from the CLI or `config.js`:

| Flag | Default | Meaning |
|---|---|---|
| `--population` | 80 | Cars per generation |
| `--generations` | 500 | Max generations per level |
| `--plateau-window` | 20 | Generations of flat fitness before advancing |
| `--mutation-sigma` | 0.1 | Gaussian noise added to weights |
| `--elitism` | 4 | Top genomes carried unchanged into next gen |
| `--tournament-size` | 5 | Candidates per selection round |

## Results

<!-- TODO: replace with actual numbers once you have a training run captured. -->

| Track | Generations to clear (median) | Best lap time |
|---|---:|---:|
| 1 — straight | ~5 | _to record_ |
| 2 — sweepers | ~30 | _to record_ |
| 3 — chicanes | ~80 | _to record_ |
| 4 — hairpins | ~140 | _to record_ |
| 5 — combo A | ~200 | _to record_ |
| 6 — combo B | ~270 | _to record_ |
| 7 — final | ~350 | _to record_ |

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

- GitHub: [@aifriend](https://github.com/aifriend)
- LinkedIn: [jafdl](https://www.linkedin.com/in/jafdl)
- Website: [auto-latam.com](https://auto-latam.com/en)

## Acknowledgments

- The classic NEAT line of work (Stanley & Miikkulainen) for the genetic-encoding intuitions
- Three.js for making in-browser 3D approachable
- Every "watch AI learn to do X" YouTube video that's made this style of demo a recognizable genre
