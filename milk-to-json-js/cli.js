#!/usr/bin/env node
/**
 * cli.js  —  Node.js command-line tool for .milk ↔ Butterchurn JSON
 *
 * Walks ALL subfolders recursively — handles the standard Milkdrop preset
 * library layout (categories/sub-categories/presets + textures mixed in).
 * Writes each .json next to its .milk, preserving the full folder structure.
 * Textures (.jpg .png .bmp .tga) are listed in the summary but left in place.
 *
 * USAGE:
 *   node cli.js  MyPreset.milk                 → MyPreset.json (single file)
 *   node cli.js  ~/milkdrop/presets/           → recurse entire tree
 *   node cli.js  ~/milkdrop/presets/ --dry-run → report only, no files written
 *   node cli.js  MyPreset.json  --reverse      → MyPreset.milk
 *   node cli.js  --help
 *
 * Node.js 18+, no npm packages needed.
 */

import { readFileSync, writeFileSync, readdirSync, statSync } from 'fs';
import { resolve, extname, basename, join, relative } from 'path';
import { milkToJson, jsonToMilk } from './milk-to-json.js';

// ─── argv ─────────────────────────────────────────────────────────────────────

const args      = process.argv.slice(2);
const flags     = new Set(args.filter(a => a.startsWith('--')));
const inputs    = args.filter(a => !a.startsWith('--'));
const dryRun    = flags.has('--dry-run');
const reverse   = flags.has('--reverse');
const inputPath = inputs[0] ? resolve(inputs[0]) : null;
const outputArg = inputs[1] ? resolve(inputs[1]) : null;

const TEXTURE_EXTS = new Set(['.jpg','.jpeg','.png','.bmp','.tga','.dds','.avi','.gif']);
const MILK_EXT     = '.milk';
const JSON_EXT     = '.json';

if (flags.has('--help') || !inputPath) {
  console.log(`
milk-to-json — .milk ↔ Butterchurn JSON  (recurses all subfolders)

  node cli.js  <file.milk>              → file.json
  node cli.js  <folder/>               → converts every .milk found recursively
  node cli.js  <folder/> --dry-run     → report without writing files
  node cli.js  <file.json> --reverse   → file.milk
`);
  process.exit(0);
}

// ─── recursive file walker ────────────────────────────────────────────────────

/**
 * Walk a directory tree and collect all file paths matching a predicate.
 * @param {string}   dir
 * @param {Function} pred  (fullPath, ext) => boolean
 * @returns {string[]}
 */
function walk(dir, pred) {
  const found = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      found.push(...walk(full, pred));
    } else if (entry.isFile()) {
      const ext = extname(entry.name).toLowerCase();
      if (pred(full, ext)) found.push(full);
    }
  }
  return found;
}

// ─── batch directory mode ─────────────────────────────────────────────────────

const stat = statSync(inputPath, { throwIfNoEntry: false });

if (stat?.isDirectory()) {
  if (reverse) {
    console.error('--reverse batch mode not supported (run on individual .json files).');
    process.exit(1);
  }

  // Collect all .milk files in the entire tree.
  const milkFiles    = walk(inputPath, (_, ext) => ext === MILK_EXT).sort();
  const textureFiles = walk(inputPath, (_, ext) => TEXTURE_EXTS.has(ext));

  if (milkFiles.length === 0) {
    console.error('No .milk files found under', inputPath);
    process.exit(1);
  }

  // Build a folder breakdown for the summary header.
  const folderCounts = {};
  for (const f of milkFiles) {
    const rel = relative(inputPath, f);
    const parts = rel.split('/');
    const category = parts.length > 1 ? parts[0] : '(root)';
    folderCounts[category] = (folderCounts[category] ?? 0) + 1;
  }

  console.log(`\nMilkdrop preset tree: ${inputPath}`);
  console.log(`  ${milkFiles.length} presets across ${Object.keys(folderCounts).length} folder(s)`);
  console.log(`  ${textureFiles.length} texture file(s) found (left in place)\n`);

  for (const [cat, count] of Object.entries(folderCounts).sort()) {
    console.log(`  📁 ${cat}  (${count})`);
  }
  console.log();

  let ok = 0, fail = 0;

  for (const inFile of milkFiles) {
    const rel     = relative(inputPath, inFile);       // e.g. "Flexi/foo.milk"
    const outFile = inFile.replace(/\.milk$/i, JSON_EXT);
    const name    = basename(inFile, MILK_EXT);

    try {
      const text = readFileSync(inFile, 'utf8');
      const json = JSON.stringify(milkToJson(text, name), null, 2);
      if (!dryRun) writeFileSync(outFile, json, 'utf8');
      console.log(`  ✓  ${rel}`);
      ok++;
    } catch (e) {
      console.error(`  ✗  ${rel}  →  ${e.message}`);
      fail++;
    }
  }

  const dryNote = dryRun ? '  (dry-run — no files written)' : '';
  console.log(`\nDone: ${ok} converted, ${fail} failed.${dryNote}`);

  if (textureFiles.length > 0) {
    console.log(`\nTextures (${textureFiles.length}) — copy these alongside your JSON files`);
    console.log(`if your Butterchurn build supports texture loading:\n`);
    for (const t of textureFiles) {
      console.log(`  🖼  ${relative(inputPath, t)}`);
    }
  }

  process.exit(fail > 0 ? 1 : 0);
}

// ─── single file mode ─────────────────────────────────────────────────────────

const ext = extname(inputPath).toLowerCase();

if (reverse) {
  if (ext !== JSON_EXT) { console.error('--reverse expects a .json file'); process.exit(1); }
  const preset = JSON.parse(readFileSync(inputPath, 'utf8'));
  const milk   = jsonToMilk(preset);
  const out    = outputArg ?? inputPath.replace(/\.json$/i, MILK_EXT);
  if (dryRun) { console.log(milk); }
  else        { writeFileSync(out, milk, 'utf8'); console.log('Written:', out); }
} else {
  if (ext !== MILK_EXT) { console.error('Expected a .milk file'); process.exit(1); }
  const name = basename(inputPath, MILK_EXT);
  const text = readFileSync(inputPath, 'utf8');
  const json = JSON.stringify(milkToJson(text, name), null, 2);
  const out  = outputArg ?? inputPath.replace(/\.milk$/i, JSON_EXT);
  if (dryRun) { console.log(json); }
  else        { writeFileSync(out, json, 'utf8'); console.log('Written:', out); }
}
