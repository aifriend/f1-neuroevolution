import * as THREE from 'three';
import { NeuralCar, F1_TEAMS } from './nn.js?v=5';
import { computeScore } from './evolution-core.js?v=5';

// 9 sensors spanning -90° to +90° in 22.5° increments (full forward hemisphere)
const SENSOR_ANGLES = [
  -Math.PI / 2,      // -90° (hard left)
  -Math.PI * 3 / 8,  // -67.5°
  -Math.PI / 4,      // -45°
  -Math.PI / 8,      // -22.5°
  0,                  // 0° (straight ahead)
  Math.PI / 8,       // +22.5°
  Math.PI / 4,       // +45°
  Math.PI * 3 / 8,   // +67.5°
  Math.PI / 2,       // +90° (hard right)
];
const SENSOR_LENGTH = 220;
const CAR_LENGTH = 14;
const STUCK_LIMIT = 120;
const LAP_COMPLETION_PROGRESS = 1.0;

export class Car {
  constructor(track, brain = null, teamIdx = 0, speedMult = 1) {
    this.track = track;
    this.brain = brain || new NeuralCar();
    this.teamIdx = teamIdx % 10;
    this.team = F1_TEAMS[this.teamIdx];
    this.speedMult = speedMult;

    // Starting position with grid stagger
    const start = track.getStartPos();
    const t = track.tangents[0];
    const nx = -t[1]; // perpendicular to track direction
    const nz = t[0];
    const row = Math.floor(teamIdx / 2);
    const col = (teamIdx % 2) - 0.5;

    this.x = start.x - t[0] * row * 4 + nx * col * 8;
    this.z = start.z - t[1] * row * 4 + nz * col * 8;
    this.angle = start.angle;
    this.speed = 0;

    this.sensors = new Array(SENSOR_ANGLES.length).fill(0);
    this.lapTime = 0;
    this.totalProgress = 0;
    // Use global search for initial position, then local search during driving
    // to prevent aliasing on figure-8 tracks (Suzuka)
    const initProg = track.getProgress(this.x, this.z);
    this.initialProgress = initProg;
    this.lastProgress = initProg;
    this.lastProgressIdx = Math.round(initProg * track.points.length);
    this.progressAccum = 0; // accumulated forward progress
    this.score = 0;
    this.alive = true;
    this.finished = false;
    this.frameCounter = 0;
    this.stuckFrames = 0;
    this.reverseAccum = 0; // sums negative progress deltas; kills car at -0.05

    // 3D
    this.group = new THREE.Group();
    this.carMesh = this._buildCarMesh();
    this.group.add(this.carMesh);
    this._syncMesh();

    // Sensor beam lines (created but hidden initially)
    this.sensorLines = [];
    this._showingSensors = false;
  }

  update() {
    if (!this.alive || this.finished) return;

    // Cast sensor rays
    for (let i = 0; i < SENSOR_ANGLES.length; i++) {
      const a = this.angle + SENSOR_ANGLES[i];
      this.sensors[i] = this.track.castRay(this.x, this.z, a) / SENSOR_LENGTH;
    }

    // Neural network decision — pass sensors + normalized speed
    const inputs = [...this.sensors, this.speed / 8.1]; // normalize speed to 0-1
    const decision = this.brain.think(inputs);

    // Physics
    this.angle += decision.steer * 0.08;
    this.speed = (2.5 + (decision.gas + 1) * 2.8) * this.speedMult;

    const cosA = Math.cos(this.angle);
    const sinA = Math.sin(this.angle);
    const newX = this.x + cosA * this.speed;
    const newZ = this.z + sinA * this.speed;

    // Collision check — also check midpoint to prevent wall tunneling
    // at high speeds (8.1 u/f can jump over 5-unit grid walls)
    const midX = (this.x + newX) * 0.5;
    const midZ = (this.z + newZ) * 0.5;
    if (!this.track.isOnTrack(midX, midZ) || !this.track.isOnTrack(newX, newZ)) {
      this.alive = false;
      this._syncMesh();
      return;
    }

    this.x = newX;
    this.z = newZ;

    // Progress tracking — local search prevents aliasing on figure-8 tracks
    const { progress, idx } = this.track.getProgressLocal(this.x, this.z, this.lastProgressIdx);
    this.lastProgressIdx = idx;
    let delta = progress - this.lastProgress;

    // Handle wraparound (e.g., 0.99 → 0.01 = forward +0.02, not backward -0.98)
    if (delta < -0.5) delta += 1.0;
    if (delta > 0.5) delta -= 1.0;

    // Cap delta to prevent progress aliasing on tracks with overlapping
    // geometry (e.g., Suzuka's figure-8). Max speed 8.1 u/f on a 1400+ unit
    // track = ~0.006 progress/frame. Cap at 0.05 for generous safety margin.
    if (delta > 0.05) delta = 0.05;
    if (delta < -0.05) delta = -0.05;

    // Only accumulate forward movement
    if (delta > 0) {
      this.progressAccum += delta;
      this.stuckFrames = 0;
      this.reverseAccum = 0;
    } else {
      this.stuckFrames++;
      this.reverseAccum += delta;
    }
    this.lastProgress = progress;
    this.totalProgress = this.progressAccum;

    // Stuck detection
    if (this.stuckFrames > STUCK_LIMIT) {
      this.alive = false;
      this._syncMesh();
      return;
    }

    // Reverse driving detection — kill within ~10 frames of wrong-way driving
    if (this.reverseAccum < -0.05) {
      this.alive = false;
      this._syncMesh();
      return;
    }

    // Scaled loitering detection: require minimum progress rate.
    // Expects ~0.03% progress per frame (a car making zero learning
    // progress is killed early instead of wasting the entire timeout).
    if (this.frameCounter > 200 && this.progressAccum < this.frameCounter * 0.00015) {
      this.alive = false;
      this._syncMesh();
      return;
    }

    this.frameCounter++;

    // Lap completion
    if (this.progressAccum >= LAP_COMPLETION_PROGRESS) {
      this.finished = true;
      this.lapTime = this.frameCounter;
    }

    this.score = computeScore(this);

    this._syncMesh();
  }

  showSensors(visible, scene) {
    if (visible && !this._showingSensors && this.sensorLines.length === 0) {
      // Create sensor lines
      for (let i = 0; i < SENSOR_ANGLES.length; i++) {
        const points = [new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, 1)];
        const geo = new THREE.BufferGeometry().setFromPoints(points);
        const mat = new THREE.LineBasicMaterial({
          color: 0x00ff88, transparent: true, opacity: 0.5,
        });
        const line = new THREE.Line(geo, mat);
        scene.add(line);
        this.sensorLines.push(line);
      }
    }
    this._showingSensors = visible;

    for (let i = 0; i < this.sensorLines.length; i++) {
      const line = this.sensorLines[i];
      line.visible = visible && this.alive;
      if (!visible || !this.alive) continue;

      const a = this.angle + SENSOR_ANGLES[i];
      const d = this.sensors[i] * SENSOR_LENGTH;
      const positions = line.geometry.attributes.position.array;
      positions[0] = this.x;
      positions[1] = 1.5;
      positions[2] = this.z;
      positions[3] = this.x + Math.cos(a) * d;
      positions[4] = 1.5;
      positions[5] = this.z + Math.sin(a) * d;
      line.geometry.attributes.position.needsUpdate = true;
    }
  }

  disposeSensorLines(scene) {
    for (const line of this.sensorLines) {
      scene.remove(line);
      line.geometry.dispose();
      line.material.dispose();
    }
    this.sensorLines = [];
  }

  _syncMesh() {
    this.group.position.set(this.x, 0, this.z);
    this.group.rotation.y = -this.angle + Math.PI / 2;

    if (!this.alive) {
      this.carMesh.traverse((node) => {
        if (node.material) {
          node.material.transparent = true;
          node.material.opacity = 0.12;
        }
      });
    }
  }

  _buildCarMesh() {
    const group = new THREE.Group();
    const scale = CAR_LENGTH / 14;
    const mainColor = this.team.main;
    const accentColor = this.team.accent;

    const mainMat = new THREE.MeshStandardMaterial({
      color: mainColor, metalness: 0.7, roughness: 0.2,
    });
    const accentMat = new THREE.MeshStandardMaterial({
      color: accentColor, metalness: 0.5, roughness: 0.3,
    });
    const darkMat = new THREE.MeshStandardMaterial({
      color: 0x222222, metalness: 0.5, roughness: 0.5,
    });

    // Chassis tub
    const chassis = new THREE.Mesh(
      new THREE.BoxGeometry(4.5 * scale, 1.3 * scale, 12 * scale), mainMat
    );
    chassis.castShadow = true;
    chassis.receiveShadow = true;
    chassis.position.y = 0.8 * scale;
    group.add(chassis);

    // Nose cone
    const nose = new THREE.Mesh(
      new THREE.BoxGeometry(1.8 * scale, 0.8 * scale, 5 * scale), accentMat
    );
    nose.position.set(0, 0.5 * scale, 7.5 * scale);
    nose.castShadow = true;
    group.add(nose);

    // Sidepods
    for (let side = -1; side <= 1; side += 2) {
      const pod = new THREE.Mesh(
        new THREE.BoxGeometry(1.6 * scale, 1.3 * scale, 7 * scale), mainMat
      );
      pod.position.set(side * 2.8 * scale, 0.8 * scale, -0.5 * scale);
      pod.castShadow = true;
      group.add(pod);
    }

    // Engine cover
    const engine = new THREE.Mesh(
      new THREE.BoxGeometry(2.2 * scale, 1.6 * scale, 5 * scale), mainMat
    );
    engine.position.set(0, 1.2 * scale, -3.5 * scale);
    engine.castShadow = true;
    group.add(engine);

    // Airbox
    const airbox = new THREE.Mesh(
      new THREE.BoxGeometry(1.4 * scale, 1.1 * scale, 1.8 * scale), accentMat
    );
    airbox.position.set(0, 2.2 * scale, -0.5 * scale);
    group.add(airbox);

    // Cockpit
    const cockpit = new THREE.Mesh(
      new THREE.BoxGeometry(1.6 * scale, 0.9 * scale, 2.2 * scale), darkMat
    );
    cockpit.position.set(0, 1.8 * scale, 1.5 * scale);
    group.add(cockpit);

    // Helmet
    const helmet = new THREE.Mesh(
      new THREE.SphereGeometry(0.55 * scale, 12, 10), accentMat
    );
    helmet.position.set(0, 2.2 * scale, 1.5 * scale);
    group.add(helmet);

    // Halo
    const halo = new THREE.Mesh(
      new THREE.TorusGeometry(0.95 * scale, 0.08 * scale, 8, 24, Math.PI),
      darkMat
    );
    halo.position.set(0, 2.0 * scale, 1.5 * scale);
    halo.rotation.x = -Math.PI / 2;
    halo.rotation.z = Math.PI;
    group.add(halo);

    // Front wing
    const fwing = new THREE.Mesh(
      new THREE.BoxGeometry(5.5 * scale, 0.15 * scale, 1.5 * scale), accentMat
    );
    fwing.position.set(0, 0.25 * scale, 9.5 * scale);
    fwing.castShadow = true;
    group.add(fwing);

    // Rear wing
    const rwing = new THREE.Mesh(
      new THREE.BoxGeometry(4.5 * scale, 1.2 * scale, 0.2 * scale), mainMat
    );
    rwing.position.set(0, 2.5 * scale, -6.5 * scale);
    rwing.castShadow = true;
    group.add(rwing);

    // Wheels
    const wheelPositions = [
      [-2.4, 0.55, 5],   // front-left
      [2.4, 0.55, 5],    // front-right
      [-2.6, 0.55, -4.5],// rear-left
      [2.6, 0.55, -4.5], // rear-right
    ];
    for (const [wx, wy, wz] of wheelPositions) {
      const wheel = new THREE.Mesh(
        new THREE.CylinderGeometry(1.1 * scale, 1.1 * scale, 1.3 * scale, 20),
        darkMat
      );
      wheel.rotation.z = Math.PI / 2;
      wheel.position.set(wx * scale, wy * scale, wz * scale);
      wheel.castShadow = true;
      group.add(wheel);
    }

    return group;
  }
}
