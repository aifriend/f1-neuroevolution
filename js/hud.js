// 2D canvas overlay: training monitor dashboard
// Top-left: stats | Bottom-left: neural network | Bottom-right: lap graph

import { NUM_SENSORS, NUM_INPUTS, HIDDEN_SIZE } from './nn.js?v=29';
import { DIFFICULTY_LADDER } from './evolution-core.js?v=29';
import { getViewportSize } from './main-utils.js?v=29';

export class Hud {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.nnCanvas = document.createElement('canvas');
    this.nnCtx = this.nnCanvas.getContext('2d');
    this.nnRefreshMs = 100;
    this._lastNNRender = 0;
    this.resize();
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const { width: w, height: h } = getViewportSize();
    this.canvas.width = w * dpr;
    this.canvas.height = h * dpr;
    this.canvas.style.width = w + 'px';
    this.canvas.style.height = h + 'px';
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    this.nnCanvas.width = this.canvas.width;
    this.nnCanvas.height = this.canvas.height;
    this.nnCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this._lastNNRender = 0;
  }

  render(state) {
    const ctx = this.ctx;
    const { width: w, height: h } = getViewportSize();
    ctx.clearRect(0, 0, w, h);

    this._drawStats(ctx, state, w);
    this._drawCurriculum(ctx, state, w, h);
    this._drawNNLayer(state, w, h);
    ctx.drawImage(this.nnCanvas, 0, 0, w, h);
    this._drawLapGraph(ctx, state.lapHistory, w, h);
  }

  // ─── Curriculum Progress (shows all levels, current one highlighted) ─
  _drawCurriculum(ctx, state, w, h) {
    const x = 16, y = 184, panelW = 280, panelH = 92;
    this._panel(ctx, x, y, panelW, panelH, 'Curriculum (LoRA adapters)');

    const currentLevel = state._difficultyLevel || 0;
    // Read adapter library off the elite brain to show what's been learned.
    const elite = state.bestCar || state.cars && state.cars[0];
    const adapters = (elite && elite.brain && elite.brain.adapters) || {};
    const rank = (elite && elite.brain && elite.brain.rank) || 2;

    const cellW = (panelW - 20) / DIFFICULTY_LADDER.length;
    const cellH = 28;
    const cellY = y + 30;

    // Draw a cell per level
    for (let i = 0; i < DIFFICULTY_LADDER.length; i++) {
      const cx = x + 10 + i * cellW;
      const isCurrent = i === currentLevel;
      const hasAdapter = i === 0 || adapters[i] != null;

      // Background fill
      let fill = 'rgba(40,50,70,0.55)';
      if (hasAdapter) fill = 'rgba(40,140,90,0.5)';      // mastered (or base)
      if (isCurrent) fill = 'rgba(255,144,48,0.65)';     // currently training
      ctx.fillStyle = fill;
      roundRect(ctx, cx + 1, cellY, cellW - 2, cellH, 3);
      ctx.fill();

      // Border for current
      if (isCurrent) {
        ctx.strokeStyle = 'rgba(255,200,120,0.95)';
        ctx.lineWidth = 1.5;
        roundRect(ctx, cx + 1, cellY, cellW - 2, cellH, 3);
        ctx.stroke();
      }

      // Level number
      ctx.fillStyle = isCurrent ? 'rgba(255,255,255,0.95)' : 'rgba(200,220,240,0.75)';
      ctx.font = 'bold 11px -apple-system, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(`L${i}`, cx + cellW / 2, cellY + 4);

      // Track abbrev
      const tname = DIFFICULTY_LADDER[i].track;
      const abbrev = tname === 'serpentine_bay' ? 'sBay'
        : tname === 'silverstone' ? 'Slv'
        : tname.slice(0, 3);
      ctx.font = '8px -apple-system, sans-serif';
      ctx.fillStyle = isCurrent ? 'rgba(255,255,255,0.85)' : 'rgba(150,180,210,0.6)';
      ctx.fillText(abbrev, cx + cellW / 2, cellY + 16);
    }

    // Footer line: rank info + adapter count
    const adapterCount = Object.keys(adapters).length;
    ctx.fillStyle = 'rgba(150,180,210,0.6)';
    ctx.font = '10px -apple-system, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(
      `rank ${rank}  ·  ${adapterCount} adapter${adapterCount === 1 ? '' : 's'} stored  ·  base frozen`,
      x + 12, y + panelH - 14
    );
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
  }

  _drawNNLayer(state, w, h) {
    const now = performance.now();
    if (now - this._lastNNRender < this.nnRefreshMs) return;
    this.nnCtx.clearRect(0, 0, w, h);
    this._drawNN(this.nnCtx, state, h);
    this._lastNNRender = now;
  }

  // ─── Top-left: Training Stats ──────────────────
  _drawStats(ctx, state, w) {
    const alive = state.cars.filter((c) => c.alive && !c.finished).length;
    const finished = state.cars.filter((c) => c.finished).length;
    const bestLap = state.bestLapTime < Infinity ? (state.bestLapTime / 60).toFixed(1) + 's' : '--';
    const genLap = state.genBestLap < Infinity ? (state.genBestLap / 60).toFixed(1) + 's' : '--';

    // Generation + track name
    ctx.fillStyle = 'rgba(255,255,255,0.92)';
    ctx.font = 'bold 22px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    const titleText = `Gen ${state.generation}  \u00b7  ${state.track.name}`;
    ctx.fillText(titleText, 20, 16);

    // Difficulty level badge
    const diffLevel = state._difficultyLevel || 0;
    if (diffLevel > 0) {
      const titleWidth = ctx.measureText(titleText).width;
      ctx.fillStyle = 'rgba(255,180,50,0.9)';
      ctx.font = 'bold 12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif';
      ctx.fillText(`Lv${diffLevel}`, 20 + titleWidth + 10, 20);
    }

    // Alive / finished
    ctx.font = '14px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif';
    ctx.fillStyle = 'rgba(180,220,255,0.85)';
    ctx.fillText(`Racing ${alive}/${state.settings.numCars}   Finished ${finished}`, 20, 46);

    // Lap times
    ctx.fillStyle = 'rgba(90,255,150,0.9)';
    ctx.fillText(`Best Lap ${bestLap}   Gen Lap ${genLap}`, 20, 66);

    // Camera hint
    ctx.fillStyle = 'rgba(150,170,200,0.55)';
    ctx.font = '11px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif';
    ctx.fillText(`C: camera (${state.cameraMode})   R: reset gen   +/-: speed`, 20, 88);

    // Progress bar (alive ratio)
    const ratio = alive / state.settings.numCars;
    const barX = 20, barY = 104, barW = 180, barH = 5;
    ctx.fillStyle = 'rgba(40,50,70,0.6)';
    roundRect(ctx, barX, barY, barW, barH, 2);
    ctx.fill();
    ctx.fillStyle = 'rgba(255,144,48,0.85)';
    if (ratio > 0) {
      roundRect(ctx, barX, barY, barW * ratio, barH, 2);
      ctx.fill();
    }

    // Leader team indicator
    const best = state.bestCar;
    if (best && best.alive) {
      const color = '#' + best.team.main.toString(16).padStart(6, '0');
      ctx.fillStyle = color;
      ctx.fillRect(20, 118, 12, 12);
      ctx.fillStyle = 'rgba(255,255,255,0.8)';
      ctx.font = '12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif';
      ctx.fillText(`${best.team.name} (Leader)  Score: ${best.score.toFixed(1)}`, 38, 128);
    }

    // All-time best
    ctx.fillStyle = 'rgba(255,200,80,0.7)';
    ctx.font = '11px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif';
    ctx.fillText(`All-Time Best: ${state.allTimeBest.toFixed(1)}`, 20, 146);

    // Curriculum progress (plateau-based escalation)
    const level = state._difficultyLevel || 0;
    if (level < DIFFICULTY_LADDER.length - 1) {
      const esc = state._escalationStatus;
      let status;
      if (!esc) {
        status = 'Next Lv: gathering plateau signal...';
      } else if (esc.reason === 'waiting_for_lap') {
        status = 'Next Lv: waiting for first valid lap';
      } else {
        const conf = Math.round((esc.confidence || 0) * 100);
        const checksLeft = Math.max(0, esc.remainingChecks || 0);
        status = `Next Lv: plateau ${conf}%  checks left ${checksLeft}`;
      }

      ctx.fillStyle = 'rgba(120,150,180,0.55)';
      ctx.font = '11px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif';
      ctx.fillText(status, 20, 162);
    }

    // FPS (top-right)
    ctx.fillStyle = 'rgba(100,120,140,0.5)';
    ctx.font = '11px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(`${state.fps || '--'} fps`, w - 16, 16);
    ctx.textAlign = 'left';
  }

  // ─── Bottom-left: Neural Network Visualization ─
  _drawNN(ctx, state, h) {
    const nnCar = state.bestCar || state.cars.find((c) => c.alive);
    if (!nnCar) return;

    const brain = nnCar.brain;
    const sensors = nnCar.sensors;
    const bx = 16, by = Math.max(12, h - 340), bw = 280, bh = 324;

    // Panel background — show LoRA status in title
    let title = 'Neural Network';
    if (brain.currentLevel > 0 && brain.adapters && brain.adapters[brain.currentLevel]) {
      title = `Neural Network · Lv${brain.currentLevel} adapter (rank ${brain.rank})`;
    } else if (brain.currentLevel === 0) {
      title = 'Neural Network · base only (Lv0)';
    }
    this._panel(ctx, bx, by, bw, bh, title);

    // LoRA-aware weight access: render the currently-active effective
    // weights (base + active adapter). NeuralCar always provides this.
    const { w1, w2 } = brain._effectiveWeights();

    const padX = 38, padY = 30;
    const layerX = [bx + padX, bx + bw / 2, bx + bw - padX];

    // Node positions — 10 inputs (9 sensors + speed), 16 hidden, 2 outputs
    const inputNodes = [];
    for (let i = 0; i < NUM_INPUTS; i++) {
      const y = by + padY + 6 + (i / (NUM_INPUTS - 1)) * (bh - padY * 2 - 12);
      inputNodes.push({ x: layerX[0], y });
    }
    const hiddenNodes = [];
    for (let i = 0; i < HIDDEN_SIZE; i++) {
      const y = by + padY + 6 + (i / (HIDDEN_SIZE - 1)) * (bh - padY * 2 - 12);
      hiddenNodes.push({ x: layerX[1], y });
    }
    const outputNodes = [];
    for (let i = 0; i < 2; i++) {
      const y = by + padY + 60 + i * (bh - padY * 2 - 120);
      outputNodes.push({ x: layerX[2], y });
    }

    // Draw connections: input → hidden (only show strong ones to reduce clutter)
    for (let i = 0; i < NUM_INPUTS; i++) {
      for (let j = 0; j < HIDDEN_SIZE; j++) {
        const w = w1[i][j];
        if (Math.abs(w) < 0.15) continue; // skip weak connections
        const alpha = Math.min(Math.abs(w) * 0.3, 0.7);
        ctx.strokeStyle = w > 0
          ? `rgba(80,255,140,${alpha})`
          : `rgba(255,80,80,${alpha})`;
        ctx.lineWidth = Math.min(Math.abs(w) * 0.5, 1.8);
        ctx.beginPath();
        ctx.moveTo(inputNodes[i].x, inputNodes[i].y);
        ctx.lineTo(hiddenNodes[j].x, hiddenNodes[j].y);
        ctx.stroke();
      }
    }

    // Draw connections: hidden → output
    for (let i = 0; i < HIDDEN_SIZE; i++) {
      for (let j = 0; j < 2; j++) {
        const w = w2[i][j];
        if (Math.abs(w) < 0.1) continue;
        const alpha = Math.min(Math.abs(w) * 0.35, 0.85);
        ctx.strokeStyle = w > 0
          ? `rgba(80,255,140,${alpha})`
          : `rgba(255,80,80,${alpha})`;
        ctx.lineWidth = Math.min(Math.abs(w) * 0.6, 2.0);
        ctx.beginPath();
        ctx.moveTo(hiddenNodes[i].x, hiddenNodes[i].y);
        ctx.lineTo(outputNodes[j].x, outputNodes[j].y);
        ctx.stroke();
      }
    }

    // Input nodes — 9 sensors + speed
    const inputLabels = ['L90', 'L68', 'L45', 'L23', 'Fwd', 'R23', 'R45', 'R68', 'R90', 'Spd'];
    for (let i = 0; i < NUM_INPUTS; i++) {
      const isSpeed = i === NUM_SENSORS;
      const v = isSpeed ? (nnCar.speed / 8.1) : (sensors[i] || 0);
      const brightness = Math.floor(80 + v * 175);
      ctx.fillStyle = isSpeed
        ? `rgb(50,${brightness},${brightness})` // cyan for speed
        : `rgb(${brightness},${Math.floor(brightness * 0.85)},50)`;
      ctx.beginPath();
      ctx.arc(inputNodes[i].x, inputNodes[i].y, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = 'rgba(180,200,220,0.6)';
      ctx.font = '7px sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(inputLabels[i], inputNodes[i].x - 8, inputNodes[i].y + 3);
    }

    // Hidden nodes
    const hv = brain._lastHidden || [];
    for (let i = 0; i < HIDDEN_SIZE; i++) {
      const v = hv[i] || 0;
      ctx.fillStyle = v > 0
        ? `rgba(${80 + v * 175},255,140,0.9)`
        : `rgba(255,${80 + (-v) * 175},80,0.9)`;
      ctx.beginPath();
      ctx.arc(hiddenNodes[i].x, hiddenNodes[i].y, 3.5, 0, Math.PI * 2);
      ctx.fill();
    }

    // Output nodes
    const ov = brain._lastOutput || [];
    const outputLabels = ['Steer', 'Gas'];
    for (let i = 0; i < 2; i++) {
      const v = ov[i] || 0;
      ctx.fillStyle = v > 0
        ? `rgba(${80 + v * 175},255,140,0.95)`
        : `rgba(255,${80 + (-v) * 175},80,0.95)`;
      ctx.beginPath();
      ctx.arc(outputNodes[i].x, outputNodes[i].y, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = 'rgba(180,200,220,0.8)';
      ctx.font = '9px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(outputLabels[i], outputNodes[i].x + 10, outputNodes[i].y + 4);
      ctx.fillStyle = 'rgba(255,255,255,0.7)';
      ctx.font = '8px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText((v >= 0 ? '+' : '') + v.toFixed(2), outputNodes[i].x, outputNodes[i].y + 16);
    }

    ctx.textAlign = 'left';
  }

  // ─── Lap Time Graph (left side, between curriculum and NN) ─────
  _drawLapGraph(ctx, history, w, h) {
    // Position between curriculum panel (ends ~y=276) and NN panel (starts y=h-340).
    // Dynamically compute available space to avoid overlap.
    const nnTop = h - 340;
    const gh = Math.min(120, nnTop - 290);
    const gw = 280;
    const gx = 16, gy = Math.max(284, nnTop - gh - 6);
    this._panel(ctx, gx, gy, gw, gh, 'Lap Time by Generation');

    if (history.length < 2) {
      ctx.fillStyle = 'rgba(120,150,180,0.5)';
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Waiting for data...', gx + gw / 2, gy + gh / 2);
      ctx.textAlign = 'left';
      return;
    }

    const padL = 38, padR = 14, padT = 30, padB = 24;
    const px = gx + padL;
    const py = gy + padT;
    const pw = gw - padL - padR;
    const ph = gh - padT - padB;

    // Axes
    ctx.strokeStyle = 'rgba(60,130,200,0.35)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(px, py + ph);
    ctx.lineTo(px + pw, py + ph);
    ctx.stroke();

    const entries = history.filter((e) => e.bestLap !== null);

    if (entries.length < 1) {
      // Show avg progress instead
      const maxProg = Math.max(...history.map((e) => e.avgProgress), 0.01);
      ctx.strokeStyle = 'rgba(255,160,50,0.85)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let i = 0; i < history.length; i++) {
        const x = px + (i / (history.length - 1)) * pw;
        const y = py + ph - (history[i].avgProgress / maxProg) * ph;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.fillStyle = 'rgba(255,160,50,0.7)';
      ctx.font = '9px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('Avg Progress', px + 4, py + 10);
      return;
    }

    // Lap time graph
    const minLap = Math.min(...entries.map((e) => e.bestLap));
    const maxLap = Math.max(...entries.map((e) => e.bestLap));
    const minPad = minLap * 0.92;
    const maxPad = maxLap * 1.08;
    const range = maxPad - minPad || 1;

    // Y-axis labels
    ctx.fillStyle = 'rgba(120,150,180,0.7)';
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText((maxPad / 60).toFixed(1) + 's', px - 4, py + 6);
    ctx.fillText((minPad / 60).toFixed(1) + 's', px - 4, py + ph - 2);

    // X-axis labels
    ctx.textAlign = 'center';
    ctx.fillText('Gen ' + history[0].gen, px, py + ph + 14);
    ctx.fillText('Gen ' + history[history.length - 1].gen, px + pw, py + ph + 14);

    // Line
    ctx.strokeStyle = 'rgba(80,255,140,0.9)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < history.length; i++) {
      if (history[i].bestLap === null) continue;
      const x = px + (i / (history.length - 1)) * pw;
      const y = py + ((history[i].bestLap - minPad) / range) * ph;
      if (!started) { ctx.moveTo(x, y); started = true; }
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Dots
    ctx.fillStyle = 'rgba(80,255,140,1)';
    for (let i = 0; i < history.length; i++) {
      if (history[i].bestLap === null) continue;
      const x = px + (i / (history.length - 1)) * pw;
      const y = py + ((history[i].bestLap - minPad) / range) * ph;
      ctx.beginPath();
      ctx.arc(x, y, 2.5, 0, Math.PI * 2);
      ctx.fill();
    }

    // Best time label
    ctx.fillStyle = 'rgba(80,255,140,0.85)';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('Best ' + (minLap / 60).toFixed(1) + 's', px + 4, py + 12);
  }

  // ─── Shared panel background ────────────────────
  _panel(ctx, x, y, w, h, title) {
    ctx.fillStyle = 'rgba(10,14,26,0.82)';
    ctx.strokeStyle = 'rgba(60,130,200,0.25)';
    ctx.lineWidth = 1;
    roundRect(ctx, x, y, w, h, 10);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = 'rgba(180,220,255,0.7)';
    ctx.font = '11px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(title, x + w / 2, y + 8);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
  }
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}
