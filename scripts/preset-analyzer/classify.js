const fs = require('fs');
const path = require('path');

const RESULTS_FILE = process.argv[2] || '/tmp/preset_results.json';
const results = JSON.parse(fs.readFileSync(RESULTS_FILE, 'utf8'));

// Thresholds for "dead/dull" classification:
//  - solid black/white screen: extreme mean, near-zero contrast, near-zero motion
//  - generally dull/static: low contrast AND low motion (barely animates)
const DEAD_MOTION = 4;
const DEAD_STD = 8;
const DULL_MOTION = 8;
const DULL_STD = 18;

const dead = [];
const dull = [];
const errored = [];
const good = [];

for (const r of results) {
  if (r.error) {
    errored.push(r);
    continue;
  }
  const { mean2, std2, motion, file } = r;

  if (std2 <= DEAD_STD && motion <= DEAD_MOTION) {
    dead.push(r);
  } else if (std2 <= DULL_STD && motion <= DULL_MOTION) {
    dull.push(r);
  } else {
    good.push(r);
  }
}

function sortByDeadness(a, b) {
  return (a.std2 + a.motion) - (b.std2 + b.motion);
}
dead.sort(sortByDeadness);
dull.sort(sortByDeadness);

console.log(`Total:   ${results.length}`);
console.log(`Good:    ${good.length}`);
console.log(`Dull:    ${dull.length}`);
console.log(`Dead:    ${dead.length}`);
console.log(`Errors:  ${errored.length}`);

const outDir = path.dirname(RESULTS_FILE);
fs.writeFileSync(path.join(outDir, 'presets_dead.txt'),
  dead.map(r => `${r.file}\t(mean=${r.mean2.toFixed(1)} std=${r.std2.toFixed(1)} motion=${r.motion.toFixed(1)})`).join('\n'));
fs.writeFileSync(path.join(outDir, 'presets_dull.txt'),
  dull.map(r => `${r.file}\t(mean=${r.mean2.toFixed(1)} std=${r.std2.toFixed(1)} motion=${r.motion.toFixed(1)})`).join('\n'));
fs.writeFileSync(path.join(outDir, 'presets_good.txt'),
  good.map(r => r.file).join('\n'));
fs.writeFileSync(path.join(outDir, 'presets_errored.txt'),
  errored.map(r => `${r.file}\t${r.error}`).join('\n'));

console.log(`\nWrote presets_dead.txt, presets_dull.txt, presets_good.txt, presets_errored.txt to ${outDir}`);
