const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const PRESETS_DIR = path.join(__dirname, '../../Sources/ButterchurnVisualizer/Resources/presets');
const FRAMES = Number(process.argv[3] || 90); // ~1.5s at 60fps equivalent
const LIMIT = process.argv[2] ? Number(process.argv[2]) : null;
const OUT_FILE = process.argv[4] || null;

async function main() {
  const files = fs.readdirSync(PRESETS_DIR).filter(f => f.endsWith('.json'));
  files.sort();
  const targets = LIMIT ? files.slice(0, LIMIT) : files;

  const browser = await chromium.launch({
    executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
    args: ['--use-gl=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist'],
  });
  const page = await browser.newPage();
  await page.goto('file://' + path.join(__dirname, 'index.html'));
  await page.evaluate(() => window.__init());

  const results = [];
  const start = Date.now();

  for (let i = 0; i < targets.length; i++) {
    const file = targets[i];
    const presetPath = path.join(PRESETS_DIR, file);
    let preset;
    try {
      preset = JSON.parse(fs.readFileSync(presetPath, 'utf8'));
    } catch (e) {
      results.push({ file, error: 'json parse: ' + e.message });
      continue;
    }

    const seed = i * 0.618;
    let res;
    try {
      res = await page.evaluate(
        ({ preset, frames, seed }) => window.__analyzePreset(preset, frames, seed),
        { preset, frames: FRAMES, seed }
      );
    } catch (e) {
      res = { error: 'evaluate: ' + e.message };
    }
    res.file = file;
    results.push(res);

    if ((i + 1) % 20 === 0 || i === targets.length - 1) {
      const elapsed = (Date.now() - start) / 1000;
      console.error(`[${i + 1}/${targets.length}] ${elapsed.toFixed(1)}s elapsed, ~${(elapsed / (i + 1)).toFixed(2)}s/preset`);
    }
  }

  await browser.close();
  const out = JSON.stringify(results, null, 2);
  if (OUT_FILE) {
    fs.writeFileSync(OUT_FILE, out);
    console.error(`Wrote ${results.length} results to ${OUT_FILE}`);
  } else {
    console.log(out);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
