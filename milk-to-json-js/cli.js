#!/usr/bin/env node
/**
 * cli.js  —  Node.js command-line tool for .milk ↔ Butterchurn JSON
 *
 * USAGE (no install required — Node.js only):
 *
 *   node cli.js  MyPreset.milk                    → MyPreset.json
 *   node cli.js  MyPreset.milk  out.json          → out.json
 *   node cli.js  /path/to/presets/               → batch, writes .json next to each .milk
 *   node cli.js  MyPreset.json  --reverse         → MyPreset.milk
 *   node cli.js  MyPreset.milk  --dry-run         → print JSON, no file written
 *   node cli.js  --help
 *
 * Node.js 18+ recommended (uses native fs/path, no npm packages needed).
 */

import { readFileSync, writeFileSync, readdirSync, statSync } from 'fs';
import { resolve, extname, basename, dirname, join } from 'path';
import { milkToJson, jsonToMilk } from './milk-to-json.js';

// ─── parse argv ──────────────────────────────────────────────────────────────

const args    = process.argv.slice(2);
const flags   = new Set(args.filter(a => a.startsWith('--')));
const inputs  = args.filter(a => !a.startsWith('--'));

if (flags.has('--help') || inputs.length === 0) {
  console.log(`
milk-to-json — .milk ↔ Butterchurn JSON converter

  node cli.js  <input.milk>           → <input>.json
  node cli.js  <input.milk>  out.json → out.json
  node cli.js  <folder/>              → batch convert all .milk files
  node cli.js  <input.json> --reverse → <input>.milk
  node cli.js  <input.milk> --dry-run → print JSON to stdout
`);
  process.exit(0);
}

const dryRun  = flags.has('--dry-run');
const reverse = flags.has('--reverse');
const inputPath = resolve(inputs[0]);
const outputArg = inputs[1] ? resolve(inputs[1]) : null;

// ─── batch mode ──────────────────────────────────────────────────────────────

const stat = statSync(inputPath, { throwIfNoEntry: false });

if (stat?.isDirectory()) {
  if (reverse) { console.error('--reverse batch mode not supported'); process.exit(1); }

  const files = readdirSync(inputPath)
    .filter(f => extname(f).toLowerCase() === '.milk')
    .sort();

  if (files.length === 0) {
    console.error('No .milk files found in', inputPath);
    process.exit(1);
  }

  console.log(`Converting ${files.length} preset(s) in ${basename(inputPath)}/\n`);
  let ok = 0, fail = 0;

  for (const file of files) {
    const inFile  = join(inputPath, file);
    const outFile = join(inputPath, file.replace(/\.milk$/i, '.json'));
    try {
      const name = basename(file, '.milk');
      const text = readFileSync(inFile, 'utf8');
      const json = JSON.stringify(milkToJson(text, name), null, 2);
      if (!dryRun) writeFileSync(outFile, json, 'utf8');
      console.log(`  ✓  ${file}  →  ${basename(outFile)}`);
      ok++;
    } catch (e) {
      console.error(`  ✗  ${file}  →  ${e.message}`);
      fail++;
    }
  }
  console.log(`\nDone: ${ok} converted, ${fail} failed${dryRun ? ' (dry-run)' : ''}.`);
  process.exit(fail > 0 ? 1 : 0);
}

// ─── single file mode ─────────────────────────────────────────────────────────

const ext = extname(inputPath).toLowerCase();

if (reverse) {
  // JSON → .milk
  if (ext !== '.json') { console.error('--reverse expects a .json file'); process.exit(1); }
  const preset = JSON.parse(readFileSync(inputPath, 'utf8'));
  const milk   = jsonToMilk(preset);
  const out    = outputArg ?? inputPath.replace(/\.json$/i, '.milk');
  if (dryRun) { console.log(milk); }
  else        { writeFileSync(out, milk, 'utf8'); console.log('Written:', out); }

} else {
  // .milk → JSON
  if (ext !== '.milk') { console.error('Expected a .milk file'); process.exit(1); }
  const name = basename(inputPath, '.milk');
  const text = readFileSync(inputPath, 'utf8');
  const json = JSON.stringify(milkToJson(text, name), null, 2);
  const out  = outputArg ?? inputPath.replace(/\.milk$/i, '.json');
  if (dryRun) { console.log(json); }
  else        { writeFileSync(out, json, 'utf8'); console.log('Written:', out); }
}
