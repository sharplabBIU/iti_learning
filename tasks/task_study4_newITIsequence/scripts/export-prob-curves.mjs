import { writeFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

const TOTAL_TRIALS = 100;
const PRACTICE_TRIALS = 6;
const RW_SD = 0.06;
const RW_LOW = 0.1;
const RW_HIGH = 0.9;
const RW_SEED_REAL = 20263944;
const RW_SEED_PRACTICE = 20260106;

function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6D2B79F5;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

function randn(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function reflectToBounds(x, low, high) {
  if (x < low) x = low + (low - x);
  if (x > high) x = high - (x - high);
  if (x < low) x = low;
  if (x > high) x = high;
  return x;
}

function generateRandomWalkTrajectories(nTrials, startPs, sd, low, high, seed) {
  const rng = mulberry32(seed);
  return startPs.map((p0) => {
    const p = new Array(nTrials);
    p[0] = reflectToBounds(p0, low, high);
    for (let t = 1; t < nTrials; t++) {
      p[t] = reflectToBounds(p[t - 1] + sd * randn(rng), low, high);
    }
    return p;
  });
}

function argmax3(a, b, c) {
  if (a >= b && a >= c) return 0;
  if (b >= a && b >= c) return 1;
  return 2;
}

function countSwitches(bestIdxArr) {
  let switches = 0;
  for (let t = 1; t < bestIdxArr.length; t++) {
    if (bestIdxArr[t] !== bestIdxArr[t - 1]) switches++;
  }
  return switches;
}

function switchPositions(bestIdxArr) {
  const pos = [];
  for (let t = 1; t < bestIdxArr.length; t++) {
    if (bestIdxArr[t] !== bestIdxArr[t - 1]) pos.push(t);
  }
  return pos;
}

function switchesAreSpread(bestIdxArr, minGap) {
  const pos = switchPositions(bestIdxArr);
  for (let i = 1; i < pos.length; i++) {
    if (pos[i] - pos[i - 1] < minGap) return false;
  }
  return true;
}

function eachBanditBestAtLeastOnce(bestIdxArr) {
  const seen = new Set(bestIdxArr);
  return seen.has(0) && seen.has(1) && seen.has(2);
}

function generateTrajectoriesWithExactlyKSwitches(nTrials, startPs, sd, low, high, seed, targetSwitches, minGap = 10, maxAttempts = 5000) {
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const traj = generateRandomWalkTrajectories(nTrials, startPs, sd, low, high, seed + attempt);
    const best = new Array(nTrials);
    for (let t = 0; t < nTrials; t++) {
      best[t] = argmax3(traj[0][t], traj[1][t], traj[2][t]);
    }
    const switches = countSwitches(best);
    if (switches === targetSwitches && eachBanditBestAtLeastOnce(best) && switchesAreSpread(best, minGap)) {
      return { traj, best, switches, attemptUsed: attempt };
    }
  }
  throw new Error('Could not find trajectories meeting constraints.');
}

const REAL = generateTrajectoriesWithExactlyKSwitches(
  TOTAL_TRIALS, [0.25, 0.50, 0.75], RW_SD, RW_LOW, RW_HIGH, RW_SEED_REAL, 6, 10
);
const PRAC = generateRandomWalkTrajectories(
  PRACTICE_TRIALS, [0.75, 0.50, 0.25], RW_SD, RW_LOW, RW_HIGH, RW_SEED_PRACTICE
);

const round = (arr) => arr.map((x) => Math.round(x * 10000) / 10000);

const data = {
  params: { RW_SD, RW_LOW, RW_HIGH, RW_SEED_REAL, RW_SEED_PRACTICE, TOTAL_TRIALS, PRACTICE_TRIALS },
  real: {
    start: [0.25, 0.50, 0.75],
    switches: REAL.switches,
    machine1: round(REAL.traj[0]),
    machine2: round(REAL.traj[1]),
    machine3: round(REAL.traj[2]),
    bestMachine: REAL.best.map((b) => b + 1),
  },
  practice: {
    start: [0.75, 0.50, 0.25],
    machine1: round(PRAC[0]),
    machine2: round(PRAC[1]),
    machine3: round(PRAC[2]),
  },
};

const out = join(__dirname, '..', 'analysis', 'probability_curves.json');
writeFileSync(out, JSON.stringify(data, null, 2));
console.log('Wrote', out);
