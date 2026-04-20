const SNAPSHOT_VERSION = 2;

export function buildPersistPayload(state, { full = false } = {}) {
  const save = {
    version: SNAPSHOT_VERSION,
    bestWeights: state._bestWeights,
    difficultyLevel: state._difficultyLevel || 0,
    generation: state.generation,
    frameCounter: state.frameCounter || 0,
    bestLapTime: state.bestLapTime,
    allTimeBest: state.allTimeBest,
    bestScore: state.bestScore || 0,
    genBestLap: state.genBestLap,
    trackType: state.settings.trackType,
    currentMutation: state._currentMutation || state.settings.mutationRate,
    settings: {
      trackType: state.settings.trackType,
      numCars: state.settings.numCars,
      speedMult: state.settings.speedMult,
      mutationRate: state.settings.mutationRate,
      timeoutEnabled: state.settings.timeoutEnabled,
      timeoutDuration: state.settings.timeoutDuration,
    },
  };

  if (full) {
    save.lapHistory = Array.isArray(state.lapHistory) ? state.lapHistory : [];
    save.populationWeights = Array.isArray(state.cars)
      ? state.cars.map((c) => c?.brain?.getWeights?.()).filter(Boolean)
      : [];
  }
  return save;
}
