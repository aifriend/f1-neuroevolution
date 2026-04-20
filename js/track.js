import * as THREE from 'three';
import { computeInterpolationSteps } from './track-geometry.js?v=29';
import { TRACKS } from './track-data.js?v=29';

const SENSOR_LENGTH = 220;

export class Track {
  constructor(type = 'monaco', widthOverride = null) {
    const cfg = TRACKS[type] || TRACKS.monaco;
    this.name = cfg.name;
    this.trackWidth = widthOverride !== null ? widthOverride : cfg.width;
    this.controlPoints = cfg.points;
    this.gridSize = 5;

    // Adaptive interpolation keeps centerline spacing dense enough that the
    // grid rasterization remains contiguous on long segments.
    const stepsPerSeg = computeInterpolationSteps(cfg.points, this.gridSize * 0.8);
    this.points = this._interpolate(cfg.points, stepsPerSeg);

    // Compute tangents at each point
    this.tangents = this._computeTangents();

    // Start position
    const sp = this.points[0];
    const st = this.tangents[0];
    this.startX = sp[0];
    this.startZ = sp[1];
    this.startAngle = Math.atan2(st[1], st[0]);

    // Build collision grid
    this.grid = {};
    this._buildGrid();

    // Build 3D mesh
    this.mesh = new THREE.Group();
    this._buildMesh();
  }

  _interpolate(pts, steps) {
    const result = [];
    const n = pts.length;
    for (let i = 0; i < n; i++) {
      const p0 = pts[(i - 1 + n) % n];
      const p1 = pts[i];
      const p2 = pts[(i + 1) % n];
      const p3 = pts[(i + 2) % n];

      for (let s = 0; s < steps; s++) {
        const t = s / steps;
        const t2 = t * t;
        const t3 = t2 * t;
        const c0 = -0.5 * t3 + t2 - 0.5 * t;
        const c1 = 1.5 * t3 - 2.5 * t2 + 1;
        const c2 = -1.5 * t3 + 2 * t2 + 0.5 * t;
        const c3 = 0.5 * t3 - 0.5 * t2;
        const x = c0 * p0[0] + c1 * p1[0] + c2 * p2[0] + c3 * p3[0];
        const z = c0 * p0[1] + c1 * p1[1] + c2 * p2[1] + c3 * p3[1];
        result.push([x, z]);
      }
    }
    return result;
  }

  _computeTangents() {
    const n = this.points.length;
    const tangents = [];
    for (let i = 0; i < n; i++) {
      const next = this.points[(i + 1) % n];
      const curr = this.points[i];
      const dx = next[0] - curr[0];
      const dz = next[1] - curr[1];
      const len = Math.sqrt(dx * dx + dz * dz) || 1;
      tangents.push([dx / len, dz / len]);
    }
    return tangents;
  }

  _buildGrid() {
    const n = this.points.length;
    const gs = this.gridSize;
    const hw = this.trackWidth;

    for (let i = 0; i < n; i++) {
      const px = this.points[i][0];
      const pz = this.points[i][1];
      const t = this.tangents[i];
      const nx = -t[1]; // perpendicular
      const nz = t[0];

      // Fill grid cells along the track width.
      // Use step of 1 unit (not gs*0.5) for dense coverage — prevents
      // holes on curved sections where perpendicular offsets skip cells.
      for (let w = -hw; w <= hw; w += 1) {
        const gx = Math.floor((px + nx * w) / gs);
        const gz = Math.floor((pz + nz * w) / gs);
        this.grid[gx + ',' + gz] = true;
      }
    }
  }

  isOnTrack(x, z) {
    const gx = Math.floor(x / this.gridSize);
    const gz = Math.floor(z / this.gridSize);
    return this.grid[gx + ',' + gz] === true;
  }

  castRay(x, z, angle) {
    const cosA = Math.cos(angle);
    const sinA = Math.sin(angle);
    for (let d = 2; d < SENSOR_LENGTH; d += 2) {
      if (!this.isOnTrack(x + cosA * d, z + sinA * d)) return d;
    }
    return SENSOR_LENGTH;
  }

  // Global nearest-point search (used for initial position only)
  getProgress(x, z) {
    let minDist = Infinity;
    let bestIdx = 0;
    const n = this.points.length;
    const points = this.points;
    for (let i = 0; i < n; i += 4) {
      const dx = x - points[i][0];
      const dz = z - points[i][1];
      const dist = dx * dx + dz * dz;
      if (dist < minDist) { minDist = dist; bestIdx = i; }
    }
    for (let offset = -6; offset <= 6; offset++) {
      const i = ((bestIdx + offset) % n + n) % n;
      const dx = x - points[i][0];
      const dz = z - points[i][1];
      const dist = dx * dx + dz * dz;
      if (dist < minDist) { minDist = dist; bestIdx = i; }
    }
    return bestIdx / n;
  }

  // Local search around a hint index — prevents aliasing on figure-8 tracks
  // like Suzuka where track sections pass close to each other. Searches only
  // within ±searchRadius of the hinted point (with wrap-around).
  getProgressLocal(x, z, hintIdx, searchRadius = 20) {
    let minDist = Infinity;
    let bestIdx = hintIdx;
    const n = this.points.length;
    const points = this.points;
    for (let offset = -searchRadius; offset <= searchRadius; offset++) {
      const i = ((hintIdx + offset) % n + n) % n;
      const dx = x - points[i][0];
      const dz = z - points[i][1];
      const dist = dx * dx + dz * dz;
      if (dist < minDist) { minDist = dist; bestIdx = i; }
    }
    return { progress: bestIdx / n, idx: bestIdx };
  }

  getStartPos() {
    return {
      x: this.startX,
      z: this.startZ,
      angle: this.startAngle,
    };
  }

  getTrackCenter() {
    let cx = 0, cz = 0;
    for (const p of this.points) { cx += p[0]; cz += p[1]; }
    cx /= this.points.length;
    cz /= this.points.length;
    return { x: cx, z: cz };
  }

  dispose() {
    if (!this.mesh) return;

    const geometries = new Set();
    const materials = new Set();
    const textures = new Set();
    const textureKeys = [
      'map', 'alphaMap', 'aoMap', 'bumpMap', 'normalMap',
      'roughnessMap', 'metalnessMap', 'emissiveMap',
    ];

    this.mesh.traverse((node) => {
      if (node.geometry) geometries.add(node.geometry);
      const mats = Array.isArray(node.material) ? node.material : [node.material];
      for (const mat of mats) {
        if (!mat) continue;
        materials.add(mat);
        for (const key of textureKeys) {
          if (mat[key]) textures.add(mat[key]);
        }
      }
    });

    for (const tex of textures) tex.dispose();
    for (const mat of materials) mat.dispose();
    for (const geom of geometries) geom.dispose();
  }

  _buildMesh() {
    const n = this.points.length;
    const hw = this.trackWidth;

    // Road surface
    const roadVerts = [];
    const roadIndices = [];
    for (let i = 0; i < n; i++) {
      const px = this.points[i][0];
      const pz = this.points[i][1];
      const t = this.tangents[i];
      const nx = -t[1] * hw;
      const nz = t[0] * hw;
      roadVerts.push(px - nx, 0.01, pz - nz);
      roadVerts.push(px + nx, 0.01, pz + nz);

      const next = (i + 1) % n;
      const v0 = i * 2, v1 = i * 2 + 1;
      const v2 = next * 2, v3 = next * 2 + 1;
      roadIndices.push(v0, v2, v1, v1, v2, v3);
    }
    const roadGeom = new THREE.BufferGeometry();
    roadGeom.setAttribute('position', new THREE.Float32BufferAttribute(roadVerts, 3));
    roadGeom.setIndex(roadIndices);
    roadGeom.computeVertexNormals();
    const roadMat = new THREE.MeshStandardMaterial({
      color: 0x222222, metalness: 0.05, roughness: 0.85,
    });
    const road = new THREE.Mesh(roadGeom, roadMat);
    road.receiveShadow = true;
    this.mesh.add(road);

    // Centerline (dashed)
    const clPts = [];
    for (let i = 0; i < n; i += 2) {
      clPts.push(new THREE.Vector3(this.points[i][0], 0.05, this.points[i][1]));
    }
    clPts.push(clPts[0].clone());
    const clGeom = new THREE.BufferGeometry().setFromPoints(clPts);
    const clMat = new THREE.LineDashedMaterial({
      color: 0xffffff, dashSize: 3, gapSize: 5,
      transparent: true, opacity: 0.25,
    });
    const clLine = new THREE.Line(clGeom, clMat);
    clLine.computeLineDistances();
    this.mesh.add(clLine);

    // Kerbs (red/white alternating using instanced mesh)
    const kerbCount = Math.floor(n / 4);
    const kerbGeom = new THREE.BoxGeometry(2.5, 0.15, 1.2);
    const kerbMatRed = new THREE.MeshStandardMaterial({ color: 0xdd2222 });
    const kerbMatWhite = new THREE.MeshStandardMaterial({ color: 0xeeeeee });

    for (let side = -1; side <= 1; side += 2) {
      for (let k = 0; k < kerbCount; k++) {
        const i = (k * 4) % n;
        const px = this.points[i][0];
        const pz = this.points[i][1];
        const t = this.tangents[i];
        const nx = -t[1] * (hw + 1.5) * side;
        const nz = t[0] * (hw + 1.5) * side;

        const mat = k % 2 === 0 ? kerbMatRed : kerbMatWhite;
        const kerb = new THREE.Mesh(kerbGeom, mat);
        kerb.position.set(px + nx, 0.08, pz + nz);
        kerb.rotation.y = Math.atan2(t[0], t[1]);
        this.mesh.add(kerb);
      }
    }

    // Track boundary lines — solid white edges, clearly visible
    for (let side = -1; side <= 1; side += 2) {
      const bPts = [];
      for (let i = 0; i < n; i++) {
        const px = this.points[i][0];
        const pz = this.points[i][1];
        const t = this.tangents[i];
        const nx = -t[1] * hw * side;
        const nz = t[0] * hw * side;
        bPts.push(new THREE.Vector3(px + nx, 0.08, pz + nz));
      }
      bPts.push(bPts[0].clone());
      const bGeom = new THREE.BufferGeometry().setFromPoints(bPts);
      const bMat = new THREE.LineBasicMaterial({
        color: 0xffffff, transparent: true, opacity: 0.6,
      });
      const bLine = new THREE.Line(bGeom, bMat);
      this.mesh.add(bLine);
    }

    // Start/finish line (checkered pattern)
    const startCanvas = document.createElement('canvas');
    startCanvas.width = 64;
    startCanvas.height = 64;
    const sctx = startCanvas.getContext('2d');
    const cs = 8;
    for (let y = 0; y < 64; y += cs) {
      for (let x = 0; x < 64; x += cs) {
        sctx.fillStyle = ((x + y) / cs) % 2 === 0 ? '#ffffff' : '#111111';
        sctx.fillRect(x, y, cs, cs);
      }
    }
    const checkerTex = new THREE.CanvasTexture(startCanvas);
    checkerTex.wrapS = THREE.RepeatWrapping;
    checkerTex.wrapT = THREE.RepeatWrapping;
    checkerTex.repeat.set(4, 1);

    const startGeo = new THREE.PlaneGeometry(hw * 2, 4);
    const startMat = new THREE.MeshBasicMaterial({ map: checkerTex });
    const startLine = new THREE.Mesh(startGeo, startMat);
    startLine.rotation.x = -Math.PI / 2;
    startLine.position.set(this.startX, 0.06, this.startZ);
    const st = this.tangents[0];
    startLine.rotation.z = -Math.atan2(st[1], st[0]) + Math.PI / 2;
    this.mesh.add(startLine);

    // Grass ground plane
    const grassGeom = new THREE.PlaneGeometry(8000, 8000);
    const grassMat = new THREE.MeshStandardMaterial({
      color: 0x2a5a2a, roughness: 0.9,
    });
    const grass = new THREE.Mesh(grassGeom, grassMat);
    grass.rotation.x = -Math.PI / 2;
    grass.position.y = -0.05;
    grass.receiveShadow = true;
    this.mesh.add(grass);
  }
}
