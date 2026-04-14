import * as THREE from 'three';
import { Track } from './track.js?v=5';
import { initialCars, nextGeneration, restoreState } from './evolution.js?v=5';
import { Hud } from './hud.js?v=5';
import { isDebugMode, parseBrainJson } from './main-utils.js?v=5';
import { TRACK_DEFAULT_WIDTHS } from './evolution-core.js?v=5';

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
  _difficultyLevel: 0,
  _bestLapStagnantGens: 0,
  _onEscalation: null,
  orbit: { azimuth: 0, elevation: 0.6, distance: 400, autoSpin: true },
  settings: {
    trackType: 'monaco',
    numCars: 80,
    speedMult: 1,
    mutationRate: 0.05,
    timeoutEnabled: true,
    timeoutDuration: 1200,
  },
};

export function main(canvas, hudCanvas) {
  if (isDebugMode(window.location.search)) {
    window.__state = state;
  }
  // ─── Renderer ──────────────────────────────────
  state.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  state.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  state.renderer.setSize(window.innerWidth, window.innerHeight);
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
    65, window.innerWidth / window.innerHeight, 0.1, 4000
  );
  // Start in top-down view centered on track
  state.camera.position.set(0, 620, 40);
  state.camera.lookAt(0, 0, 0);

  // ─── Track ─────────────────────────────────────
  state.track = new Track(state.settings.trackType);
  state.scene.add(state.track.mesh);

  // ─── Restore saved training state (survives reloads) ───
  const restored = restoreState(state);
  if (restored) {
    const trackSelect = document.getElementById('track-select');
    if (trackSelect) trackSelect.value = state.settings.trackType;
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
  const $ = (id) => document.getElementById(id);
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

    // Reset state
    state.generation = 1;
    state.bestScore = 0;
    state.allTimeBest = 0;
    state.bestLapTime = Infinity;
    state.genBestLap = Infinity;
    state.lapHistory = [];
    state._stagnantGens = 0;
    state._bestWeights = null;
    state._difficultyLevel = 0;
    state._bestLapStagnantGens = 0;
    try { localStorage.removeItem('f1-neuroevo-state'); } catch {}
    state.bestCar = null;
    state.cameraMode = 'top';
    document.getElementById('cam-mode').textContent = 'top';

    state.cars = initialCars(state);
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
      a.download = `best-brain-gen${state.generation}.json`;
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

  // ─── Resize ────────────────────────────────────
  window.addEventListener('resize', () => {
    state.renderer.setSize(window.innerWidth, window.innerHeight);
    state.camera.aspect = window.innerWidth / window.innerHeight;
    state.camera.updateProjectionMatrix();
    hud.resize();
  });

  // ─── FPS tracking ──────────────────────────────
  let frameCount = 0;
  let lastFpsTime = performance.now();

  // ─── Previous best car (for sensor cleanup) ────
  let prevBestCar = null;

  // ─── Animation Loop ────────────────────────────
  function tick() {
    requestAnimationFrame(tick);

    // Update all cars
    for (const car of state.cars) {
      car.update();
    }

    // Find current best alive car
    state.bestCar = null;
    let bestScore = -1;
    for (const car of state.cars) {
      if (car.alive && car.score > bestScore) {
        bestScore = car.score;
        state.bestCar = car;
      }
    }
    // If no alive car, use highest scoring overall
    if (!state.bestCar) {
      for (const car of state.cars) {
        if (car.score > bestScore) {
          bestScore = car.score;
          state.bestCar = car;
        }
      }
    }

    // Sensor beams: show only on best car
    if (prevBestCar && prevBestCar !== state.bestCar) {
      prevBestCar.showSensors(false, state.scene);
    }
    if (state.bestCar) {
      state.bestCar.showSensors(true, state.scene);
      prevBestCar = state.bestCar;
    }

    // Check if generation is done
    state.frameCounter++;
    const allDone = state.cars.every((c) => !c.alive || c.finished);
    const timedOut = state.settings.timeoutEnabled &&
      state.frameCounter > state.settings.timeoutDuration;

    if (allDone || timedOut) {
      // Kill remaining cars on timeout
      if (timedOut) {
        for (const c of state.cars) {
          if (c.alive && !c.finished) c.alive = false;
        }
      }
      prevBestCar = null;
      nextGeneration(state);
    }

    // Camera
    updateCamera(state);

    // Render
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
