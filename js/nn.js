// Neural network for self-driving F1 cars
// Architecture: 9 sensors + 1 speed → 16 hidden (tanh) → 2 outputs (steer, gas)
// 194 trainable parameters (was 50 with 5→6→2)

export const NUM_SENSORS = 9;
export const NUM_INPUTS = NUM_SENSORS + 1; // sensors + current speed
export const HIDDEN_SIZE = 16;

function randomGaussian(mean = 0, std = 1) {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return mean + std * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

export class NeuralCar {
  constructor(weights) {
    if (!weights) {
      // Larger init scale for diverse random behaviors — helps first-gen exploration
      const scale1 = 0.8;
      const scale2 = 0.6;
      this.w1 = this.randomMatrix(NUM_INPUTS, HIDDEN_SIZE, scale1);
      this.b1 = new Array(HIDDEN_SIZE).fill(0);
      this.w2 = this.randomMatrix(HIDDEN_SIZE, 2, scale2);
      this.b2 = new Array(2).fill(0);
    } else {
      this.w1 = weights.w1.map((row) => [...row]);
      this.b1 = [...(weights.b1 || new Array(HIDDEN_SIZE).fill(0))];
      this.w2 = weights.w2.map((row) => [...row]);
      this.b2 = [...(weights.b2 || new Array(2).fill(0))];
    }
    this._lastHidden = new Array(HIDDEN_SIZE).fill(0);
    this._lastOutput = [0, 0];
  }

  randomMatrix(rows, cols, scale = 1) {
    const m = [];
    for (let i = 0; i < rows; i++) {
      const row = [];
      for (let j = 0; j < cols; j++) row.push(randomGaussian(0, scale));
      m.push(row);
    }
    return m;
  }

  randomVector(size) {
    return Array.from({ length: size }, () => randomGaussian(0, 0.5));
  }

  think(inputs) {
    // inputs = [...9 sensor values, speed]
    const hidden = [];
    for (let j = 0; j < HIDDEN_SIZE; j++) {
      let sum = this.b1[j];
      for (let i = 0; i < inputs.length; i++) sum += inputs[i] * this.w1[i][j];
      hidden.push(Math.tanh(sum));
    }

    const output = [];
    for (let j = 0; j < 2; j++) {
      let sum = this.b2[j];
      for (let i = 0; i < HIDDEN_SIZE; i++) sum += hidden[i] * this.w2[i][j];
      output.push(Math.tanh(sum));
    }

    this._lastHidden = hidden;
    this._lastOutput = output;
    return { steer: output[0], gas: output[1] };
  }

  mutate(rate) {
    for (let i = 0; i < this.w1.length; i++)
      for (let j = 0; j < this.w1[i].length; j++)
        this.w1[i][j] += randomGaussian(0, 1) * rate;
    for (let i = 0; i < this.b1.length; i++)
      this.b1[i] += randomGaussian(0, 1) * rate;
    for (let i = 0; i < this.w2.length; i++)
      for (let j = 0; j < this.w2[i].length; j++)
        this.w2[i][j] += randomGaussian(0, 1) * rate;
    for (let i = 0; i < this.b2.length; i++)
      this.b2[i] += randomGaussian(0, 1) * rate;
  }

  getWeights() {
    return {
      w1: this.w1.map((row) => [...row]),
      b1: [...this.b1],
      w2: this.w2.map((row) => [...row]),
      b2: [...this.b2],
    };
  }
}

export const F1_TEAMS = [
  { name: 'Red Bull',     main: 0x3671c6, accent: 0xffcd00 },
  { name: 'Ferrari',      main: 0xe8002d, accent: 0xffffff },
  { name: 'McLaren',      main: 0xff8000, accent: 0x000000 },
  { name: 'Mercedes',     main: 0x27f4d2, accent: 0x000000 },
  { name: 'Aston Martin', main: 0x006f62, accent: 0xcedc00 },
  { name: 'Alpine',       main: 0x0093cc, accent: 0xff57a4 },
  { name: 'Williams',     main: 0x64c4ff, accent: 0x003778 },
  { name: 'RB',           main: 0x6692ff, accent: 0xff0000 },
  { name: 'Kick Sauber',  main: 0x52e252, accent: 0x000000 },
  { name: 'Haas',         main: 0xb6babd, accent: 0xe8002d },
];
