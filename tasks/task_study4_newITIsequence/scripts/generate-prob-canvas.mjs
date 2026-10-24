import { readFileSync, writeFileSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const d = JSON.parse(readFileSync(join(__dirname, '..', 'analysis', 'probability_curves.json'), 'utf8'));
const sw = [];
for (let t = 1; t < d.real.bestMachine.length; t++) {
  if (d.real.bestMachine[t] !== d.real.bestMachine[t - 1]) sw.push(t + 1);
}

const content = `import {
  LineChart,
  H1,
  H2,
  Text,
  Stack,
  Card,
  CardBody,
  Stat,
  Grid,
} from 'cursor/canvas';

const REAL = ${JSON.stringify(d.real)};
const PRACTICE = ${JSON.stringify(d.practice)};
const SWITCH_TRIALS = ${JSON.stringify(sw)};

export default function BanditProbabilityCurves() {
  const realCategories = Array.from({ length: 100 }, (_, i) => String(i + 1));
  const practiceCategories = Array.from({ length: 6 }, (_, i) => String(i + 1));

  return (
    <Stack gap={24}>
      <Stack gap={8}>
        <H1>Bandit reward probability curves</H1>
        <Text>
          Reflecting random walks (SD=0.06, bounds 0.1 to 0.9). Identical for all participants.
          Q = Machine 1 (left), T = Machine 2 (middle), P = Machine 3 (right).
        </Text>
      </Stack>

      <Grid columns={3} gap={12}>
        <Stat label="Real task best-arm switches" value={String(REAL.switches)} tone="info" />
        <Stat label="Switch at trials" value={SWITCH_TRIALS.join(', ')} tone="neutral" />
        <Stat label="Real start M1, M2, M3" value={REAL.start.join(', ')} tone="neutral" />
      </Grid>

      <Card>
        <CardBody>
          <Stack gap={12}>
            <H2>Real experiment (100 trials)</H2>
            <Text>
              Start: M1=0.25, M2=0.50, M3=0.75 (Machine 3 best at trial 1). Six best-arm switches.
            </Text>
            <LineChart
              height={320}
              categories={realCategories}
              series={[
                { name: 'Machine 1 (Q)', data: REAL.machine1, tone: 'info' },
                { name: 'Machine 2 (T)', data: REAL.machine2, tone: 'warning' },
                { name: 'Machine 3 (P)', data: REAL.machine3, tone: 'success' },
              ]}
            />
            <Text>Source: task index.html, seed 20263944. Y-axis: win probability per trial.</Text>
          </Stack>
        </CardBody>
      </Card>

      <Card>
        <CardBody>
          <Stack gap={12}>
            <H2>Practice (6 trials)</H2>
            <Text>Start: M1=0.75, M2=0.50, M3=0.25 (Machine 1 best). Different seed from real task.</Text>
            <LineChart
              height={240}
              categories={practiceCategories}
              series={[
                { name: 'Machine 1 (Q)', data: PRACTICE.machine1, tone: 'info' },
                { name: 'Machine 2 (T)', data: PRACTICE.machine2, tone: 'warning' },
                { name: 'Machine 3 (P)', data: PRACTICE.machine3, tone: 'success' },
              ]}
            />
            <Text>Source: task index.html, seed 20260106.</Text>
          </Stack>
        </CardBody>
      </Card>
    </Stack>
  );
}
`;

const canvasDir = join(
  process.env.USERPROFILE || '',
  '.cursor',
  'projects',
  'c-Users-user-Downloads-task-v2-task-v2-1',
  'canvases'
);
mkdirSync(canvasDir, { recursive: true });
const out = join(canvasDir, 'bandit-probability-curves.canvas.tsx');
writeFileSync(out, content);
console.log('Wrote', out);
