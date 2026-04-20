import * as THREE from 'three';
import { Track } from './track.js?v=29';
import { initialCars, nextGeneration, restoreState, persistState } from './evolution.js?v=29';
import { Hud } from './hud.js?v=29';
import { createQuitController, isDebugMode, parseBrainJson, parseVisualRuntimeParams } from './main-utils.js?v=29';
import { TRACK_DEFAULT_WIDTHS } from './evolution-core.js?v=29';

const state = {
  renderer: null,
  scene: null,
  camera: null,
  track: null,
  cars: [],
  generation: 1,
  frameCounter: 0,
  bestScore: 0,
  allTimeBest: 0,
  bestLapTime: Infinity,
  genBestLap: Infinity,
  lapHistory: [],
  cameraMode: 'top',
  bestCar: null,
  fps: 0,
  _stagnantGens: 0,
  _currentMutation: 0.08,
  _bestWeights: null,
  _loadedWeights: null,
  _loadedPopulation: null,
  _difficultyLevel: 0,
  _bestLapStagnantGens: 0,
  _plateauConsecutiveChecks: 0,
  _plateauAvgHistory: [],
  _plateauFinishedRateHistory: [],
  _plateauStatus: null,
  _escalationStatus: null,
  _onEscalation: null,
  orbit: { azimuth: 0, elevation: 0.6, distance: 400, autoSpin: true },
  settings: {
    trackType: 'monaco',
    numCars: 30,
    speedMult: 1,
    mutationRate: 0.08,
    timeoutEnabled: true,
    timeoutDuration: 3500,
  },
};

export function main(canvas, hudCanvas) {
  if (isDebugMode(window.location.search)) {
    window.__state = state;
  }
  // Viewport may be 0×0 when loaded hidden (e.g., in a background iframe).
  // Fall back to a sane default so canvases can render; real resize handled later.
  const getW = () => window.innerWidth || document.documentElement.clientWidth || 1024;
  const getH = () => window.innerHeight || document.documentElement.clientHeight || 768;

  // ─── Renderer ──────────────────────────────────
  state.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  state.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  state.renderer.setSize(getW(), getH());
  state.renderer.shadowMap.enabled = true;
  state.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  state.renderer.toneMapping = THREE.ACESFilmicToneMapping;
  state.renderer.toneMappingExposure = 1.2;

  // ─── Scene ─────────────────────────────────────
  state.scene = new THREE.Scene();
  state.scene.background = new THREE.Color(0x0a0e1a);
  state.scene.fog = new THREE.Fog(0x0a0e1a, 500, 2000);

  // Sky dome
  const skyGeom = new THREE.SphereGeometry(1500, 32, 32);
  const skyMat = new THREE.MeshBasicMaterial({
    color: 0x1a2a4a, side: THREE.BackSide,
  });
  state.scene.add(new THREE.Mesh(skyGeom, skyMat));

  // ─── Lighting ──────────────────────────────────
  const hemiLight = new THREE.HemisphereLight(0xbcd8ff, 0x1a2438, 0.55);
  state.scene.add(hemiLight);

  const sunLight = new THREE.DirectionalLight(0xffffff, 1.2);
  sunLight.position.set(600, 900, 400);
  sunLight.castShadow = true;
  sunLight.shadow.mapSize.width = 2048;
  sunLight.shadow.mapSize.height = 2048;
  sunLight.shadow.camera.left = -500;
  sunLight.shadow.camera.right = 500;
  sunLight.shadow.camera.top = 500;
  sunLight.shadow.camera.bottom = -500;
  sunLight.shadow.camera.near = 1;
  sunLight.shadow.camera.far = 2000;
  state.scene.add(sunLight);

  // ─── Camera ────────────────────────────────────
  state.camera = new THREE.PerspectiveCamera(
    65, getW() / getH(), 0.1, 4000
  );
  // Start in top-down view centered on track
  state.camera.position.set(0, 620, 40);
  state.camera.lookAt(0, 0, 0);

  // ─── URL param defaults ────────────────────────
  const $ = (id) => document.getElementById(id);
  const syncSettingsControls = () => {
    const trackSelect = $('track-select');
    if (trackSelect) trackSelect.value = state.settings.trackType;
    const cars = $('num-cars');
    if (cars) cars.value = String(state.settings.numCars);
    const carsVal = $('num-cars-val');
    if (carsVal) carsVal.textContent = String(state.settings.numCars);
    const speed = $('speed-mult');
    if (speed) speed.value = String(state.settings.speedMult);
    const speedVal = $('speed-val');
    if (speedVal) speedVal.textContent = state.settings.speedMult.toFixed(1) + 'x';
    const mut = $('mutation-rate');
    if (mut) mut.value = String(state.settings.mutationRate);
    const mutVal = $('mutation-val');
    if (mutVal) mutVal.textContent = state.settings.mutationRate.toFixed(2);
    const toEn = $('timeout-enabled');
    if (toEn) toEn.checked = Boolean(state.settings.timeoutEnabled);
    const toDur = $('timeout-duration');
    if (toDur) toDur.value = String(state.settings.timeoutDuration);
    const resetCb = $('reset-training');
    if (resetCb) resetCb.checked = false;
  };
  const allowedTracks = Object.keys(TRACK_DEFAULT_WIDTHS);
  const params = parseVisualRuntimeParams(window.location.search, { allowedTracks });
  if (params.fresh) {
    try { localStorage.removeItem('f1-neuroevo-state'); } catch {}
  }
  if (params.trackType) state.settings.trackType = params.trackType;
  if (params.numCars != null) state.settings.numCars = params.numCars;
  if (params.speedMult != null) state.settings.speedMult = params.speedMult;
  if (params.mutationRate != null) state.settings.mutationRate = params.mutationRate;
  if (params.timeoutEnabled != null) state.settings.timeoutEnabled = params.timeoutEnabled;
  if (params.timeoutDuration != null) state.settings.timeoutDuration = params.timeoutDuration;

  // Sync UI controls to chosen defaults before any restoreState() happens.
  syncSettingsControls();

  // ─── Track ─────────────────────────────────────
  state.track = new Track(state.settings.trackType);
  state.scene.add(state.track.mesh);

  // ─── Restore saved training state (survives reloads) ───
  const restored = params.fresh ? false : restoreState(state);
  if (restored) {
    syncSettingsControls();
    console.log(`Restored training: Gen ${state.generation}, Lv${state._difficultyLevel || 0}, ${state.settings.trackType}`);
  }

  // ─── Cars ──────────────────────────────────────
  state.cars = initialCars(state);

  // ─── HUD ───────────────────────────────────────
  const hud = new Hud(hudCanvas);

  // ─── Input ─────────────────────────────────────
  const keys = {};
  window.addEventListener('keydown', (e) => {
    const k = e.key.toLowerCase();
    if (keys[k]) return; // debounce
    keys[k] = true;

    if (k === 'c') {
      const modes = ['chase', 'hero', 'top', 'orbit'];
      const idx = (modes.indexOf(state.cameraMode) + 1) % modes.length;
      state.cameraMode = modes[idx];
      document.getElementById('cam-mode').textContent = state.cameraMode;
    }
    if (k === 'r') nextGeneration(state);
    if (k === '+' || k === '=') {
      state.settings.speedMult = Math.min(3, state.settings.speedMult + 0.25);
      document.getElementById('speed-mult').value = state.settings.speedMult;
      document.getElementById('speed-val').textContent = state.settings.speedMult.toFixed(1) + 'x';
    }
    if (k === '-') {
      state.settings.speedMult = Math.max(0.5, state.settings.speedMult - 0.25);
      document.getElementById('speed-mult').value = state.settings.speedMult;
      document.getElementById('speed-val').textContent = state.settings.speedMult.toFixed(1) + 'x';
    }
  });
  window.addEventListener('keyup', (e) => { keys[e.key.toLowerCase()] = false; });

  // ─── Orbit mouse controls ─────────────────────
  let mouseDown = false;
  let autoSpinTimeout = null;
  canvas.addEventListener('mousedown', () => {
    mouseDown = true;
    state.orbit.autoSpin = false;
    clearTimeout(autoSpinTimeout);
  });
  canvas.addEventListener('mouseup', () => {
    mouseDown = false;
    autoSpinTimeout = setTimeout(() => { state.orbit.autoSpin = true; }, 3000);
  });
  canvas.addEventListener('mousemove', (e) => {
    if (!mouseDown || state.cameraMode !== 'orbit') return;
    state.orbit.azimuth -= e.movementX * 0.008;
    state.orbit.elevation = Math.max(0.1, Math.min(1.4,
      state.orbit.elevation - e.movementY * 0.006));
  });
  canvas.addEventListener('wheel', (e) => {
    if (state.cameraMode !== 'orbit') return;
    e.preventDefault();
    state.orbit.distance = Math.max(80, Math.min(1800,
      state.orbit.distance + e.deltaY * 0.8));
  }, { passive: false });

  // ─── Settings UI ───────────────────────────────
  const statusEl = $('status-msg');
  let statusTimer = null;

  function setStatus(message, type = 'info') {
    if (!statusEl) return;
    statusEl.textContent = message;
    statusEl.className = `status-msg status-${type}`;
    if (statusTimer) clearTimeout(statusTimer);
    statusTimer = setTimeout(() => {
      statusEl.textContent = '';
      statusEl.className = 'status-msg';
    }, 5000);
  }

  const quitController = createQuitController(() => {
    persistState(state);
    setStatus('Quit. Reload the page to resume.', 'warn');
  });

  // Curriculum escalation callback
  state._onEscalation = (step, level) => {
    const trackSelect = $('track-select');
    if (trackSelect) trackSelect.value = step.track;
    const w = step.widthDelta !== 0 ? ` (width ${TRACK_DEFAULT_WIDTHS[step.track] + step.widthDelta})` : '';
    setStatus(`Difficulty escalated! Level ${level}: ${step.track}${w}`, 'warn');
  };

  $('num-cars').addEventListener('input', (e) => {
    $('num-cars-val').textContent = e.target.value;
  });
  $('speed-mult').addEventListener('input', (e) => {
    $('speed-val').textContent = parseFloat(e.target.value).toFixed(1) + 'x';
  });
  $('mutation-rate').addEventListener('input', (e) => {
    $('mutation-val').textContent = parseFloat(e.target.value).toFixed(2);
  });

  $('apply-btn').addEventListener('click', () => {
    const newTrack = $('track-select').value;
    const resetTraining = $('reset-training').checked;
    const carryPopulation = !resetTraining
      ? state.cars.map((car) => car.brain.getWeights())
      : null;
    state.settings.numCars = parseInt($('num-cars').value);
    state.settings.speedMult = parseFloat($('speed-mult').value);
    state.settings.mutationRate = parseFloat($('mutation-rate').value);
    state.settings.timeoutEnabled = $('timeout-enabled').checked;
    state.settings.timeoutDuration = parseInt($('timeout-duration').value);
    state._currentMutation = state.settings.mutationRate;

    // Remove old track and cars
    for (const c of state.cars) {
      c.disposeSensorLines(state.scene);
      state.scene.remove(c.group);
    }

    if (newTrack !== state.settings.trackType) {
      state.track.dispose();
      state.scene.remove(state.track.mesh);
      state.settings.trackType = newTrack;
      state.track = new Track(newTrack);
      state.scene.add(state.track.mesh);
    }

    if (resetTraining) {
      // Clean reset: wipe all evolutionary progress + persisted snapshot.
      state.generation = 1;
      state.frameCounter = 0;
      state.bestScore = 0;
      state.allTimeBest = 0;
      state.bestLapTime = Infinity;
      state.genBestLap = Infinity;
      state.lapHistory = [];
      state._stagnantGens = 0;
      state._bestWeights = null;
      state._loadedWeights = null;
      state._loadedPopulation = null;
      state._difficultyLevel = 0;
      state._bestLapStagnantGens = 0;
      state._plateauConsecutiveChecks = 0;
      state._plateauAvgHistory = [];
      state._plateauFinishedRateHistory = [];
      state._plateauStatus = null;
      state._escalationStatus = null;
      try { localStorage.removeItem('f1-neuroevo-state'); } catch {}
      state.bestCar = null;
      state.cameraMode = 'top';
      document.getElementById('cam-mode').textContent = 'top';
    } else {
      // Restart sim from currently trained population with updated settings.
      state.frameCounter = 0;
      state.bestCar = null;
      state._loadedPopulation = carryPopulation;
      state._loadedWeights = null;
    }

    state.cars = initialCars(state);
    persistState(state);
    $('reset-training').checked = false;
    setStatus(
      resetTraining
        ? 'Applied. Training reset to clean start.'
        : 'Applied. Restarted from current training snapshot.',
      'ok'
    );
  });

  // Save / Load brain
  $('save-btn').addEventListener('click', () => {
    const best = state.bestCar;
    if (!best) {
      setStatus('No best brain available yet.', 'warn');
      return;
    }
    try {
      const data = JSON.stringify(best.brain.getWeights());
      const blob = new Blob([data], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `models/best-brain-gen${state.generation}.json`;
      a.click();
      URL.revokeObjectURL(url);
      setStatus('Brain saved.', 'ok');
    } catch {
      setStatus('Failed to save brain.', 'error');
    }
  });

  $('load-btn').addEventListener('click', () => $('load-file').click());
  $('load-file').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const parsed = parseBrainJson(reader.result);
      if (!parsed.ok) {
        setStatus(`Load failed: ${parsed.error}`, 'error');
      } else {
        state._loadedWeights = parsed.weights;
        setStatus('Brain loaded. Applies next generation.', 'ok');
      }
      e.target.value = '';
    };
    reader.onerror = () => {
      setStatus('Unable to read selected file.', 'error');
      e.target.value = '';
    };
    reader.readAsText(file);
  });

  $('quit-btn').addEventListener('click', () => {
    quitController.requestQuit();
  });

  // ─── Resize ────────────────────────────────────
  window.addEventListener('resize', () => {
    state.renderer.setSize(getW(), getH());
    state.camera.aspect = getW() / getH();
    state.camera.updateProjectionMatrix();
    hud.resize();
  });

  // Save state on reload/close so we don't lose progress
  window.addEventListener('beforeunload', () => { persistState(state); });
  // visibilitychange also fires when the tab is hidden (mobile/iframe quirks)
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) persistState(state);
  });

  // ─── FPS tracking ──────────────────────────────
  let frameCount = 0;
  let lastFpsTime = performance.now();

  // ─── Previous best car (for sensor cleanup) ────
  let prevBestCar = null;

  // ─── Physics step (single frame of simulation) ──
  function physicsStep() {
    for (const car of state.cars) car.update();

    // Find current best alive car
    state.bestCar = null;
    let bestScore = -1;
    for (const car of state.cars) {
      if (car.alive && car.score > bestScore) { bestScore = car.score; state.bestCar = car; }
    }
    if (!state.bestCar) {
      for (const car of state.cars) {
        if (car.score > bestScore) { bestScore = car.score; state.bestCar = car; }
      }
    }

    // Generation transition
    state.frameCounter++;
    const allDone = state.cars.every((c) => !c.alive || c.finished);
    const timedOut = state.settings.timeoutEnabled &&
      state.frameCounter > state.settings.timeoutDuration;
    if (allDone || timedOut) {
      if (timedOut) {
        for (const c of state.cars) {
          if (c.alive && !c.finished) {
            c.alive = false;
            if (!c.killReason || c.killReason === 'active') c.killReason = 'timeout';
          }
        }
      }
      prevBestCar = null;
      nextGeneration(state);
    }
  }

  // ─── Animation Loop ────────────────────────────
  // When tab is hidden, browsers throttle rAF/setTimeout. To keep training
  // progressing, run MANY physics steps per throttled tick and skip rendering.
  function tick() {
    if (quitController.isQuitRequested()) return;
    if (document.hidden) {
      // Background mode: batch physics, then render once so screenshots work
      for (let i = 0; i < 200; i++) physicsStep();
      updateCamera(state);
      state.renderer.render(state.scene, state.camera);
      hud.render(state);
      setTimeout(tick, 0);
      return;
    }

    // Foreground mode: single step + render at display rate
    requestAnimationFrame(tick);
    physicsStep();

    // Sensor beams (visual only)
    if (prevBestCar && prevBestCar !== state.bestCar) {
      prevBestCar.showSensors(false, state.scene);
    }
    if (state.bestCar) {
      state.bestCar.showSensors(true, state.scene);
      prevBestCar = state.bestCar;
    }

    updateCamera(state);
    state.renderer.render(state.scene, state.camera);
    hud.render(state);

    // FPS
    frameCount++;
    const now = performance.now();
    if (now - lastFpsTime > 500) {
      state.fps = Math.round(frameCount / ((now - lastFpsTime) / 1000));
      frameCount = 0;
      lastFpsTime = now;
    }
  }

  tick();
}

// ─── Camera System ─────────────────────────────────
function updateCamera(state) {
  const cam = state.camera;
  const best = state.bestCar;

  if (!best) return;

  const target = new THREE.Vector3(best.x, 0, best.z);

  switch (state.cameraMode) {
    case 'chase': {
      const behind = new THREE.Vector3(
        -Math.cos(best.angle) * 35,
        14,
        -Math.sin(best.angle) * 35
      );
      const desired = target.clone().add(behind);
      cam.position.lerp(desired, 0.06);
      const lookTarget = target.clone();
      lookTarget.y = 2;
      cam.lookAt(lookTarget);
      break;
    }

    case 'hero': {
      const side = new THREE.Vector3(
        -Math.cos(best.angle + 0.8) * 30,
        6,
        -Math.sin(best.angle + 0.8) * 30
      );
      const desired = target.clone().add(side);
      cam.position.lerp(desired, 0.04);
      const lookTarget = target.clone();
      lookTarget.y = 2;
      cam.lookAt(lookTarget);
      break;
    }

    case 'top': {
      const center = state.track.getTrackCenter();
      const desired = new THREE.Vector3(center.x, 620, center.z + 40);
      cam.position.lerp(desired, 0.04);
      cam.lookAt(new THREE.Vector3(center.x, 0, center.z));
      break;
    }

    case 'orbit': {
      if (state.orbit.autoSpin) {
        state.orbit.azimuth += 0.003;
      }
      const center = state.track.getTrackCenter();
      const d = state.orbit.distance;
      const el = state.orbit.elevation;
      const az = state.orbit.azimuth;
      const desired = new THREE.Vector3(
        center.x + d * Math.cos(el) * Math.sin(az),
        d * Math.sin(el),
        center.z + d * Math.cos(el) * Math.cos(az)
      );
      cam.position.lerp(desired, 0.08);
      cam.lookAt(new THREE.Vector3(center.x, 0, center.z));
      break;
    }
  }
}
