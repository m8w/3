/**
 * milk-parser.js
 * Parses a Milkdrop .milk preset file into a structured JavaScript object.
 *
 * Works in Node.js AND in the browser (no dependencies, pure ES module).
 *
 * Usage:
 *   import { parseMilk } from './milk-parser.js';
 *   const preset = parseMilk(milkFileText);
 */

/**
 * Parse the raw text of a .milk file.
 *
 * @param {string} text  Full contents of a .milk file.
 * @returns {MilkPreset} Structured preset object.
 *
 * @typedef {Object} MilkPreset
 * @property {Object.<string,string>} baseVals   - All scalar key=value pairs.
 * @property {string[]}  initEqs      - per_frame_init_N lines, in order.
 * @property {string[]}  perFrameEqs  - per_frame_N lines, in order.
 * @property {string[]}  perPixelEqs  - per_pixel_N lines, in order.
 * @property {Object.<number, string[]>} waveFrameEqs  - wave_K_per_frame_N grouped by K.
 * @property {Object.<number, string[]>} wavePointEqs  - wave_K_per_point_N grouped by K.
 * @property {Object.<number, string[]>} shapeFrameEqs - shape_K_per_frame_N grouped by K.
 * @property {Object.<number, string[]>} shapePointEqs - shape_K_per_point_N grouped by K.
 */
export function parseMilk(text) {
  const preset = {
    baseVals:     {},
    initEqs:      [],   // will be sorted after parsing
    perFrameEqs:  [],
    perPixelEqs:  [],
    waveFrameEqs: {},   // { waveIndex: [ eq, eq, … ] }
    wavePointEqs: {},
    shapeFrameEqs:{},
    shapePointEqs:{},
  };

  // Temporary buckets: { index -> equation } so we can sort before storing.
  const initBucket      = {};
  const frameBucket     = {};
  const pixelBucket     = {};
  const waveFrameBucket = {};  // { waveIndex: { lineIndex: eq } }
  const wavePointBucket = {};
  const shapeFrameBucket= {};
  const shapePointBucket= {};

  let inPresetSection = false;

  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();

    // Skip blanks and comments.
    if (!line || line.startsWith(';') || line.startsWith('#')) continue;

    // Section header.
    if (line.startsWith('[')) {
      inPresetSection = /^\[preset/i.test(line);
      continue;
    }
    if (!inPresetSection) continue;

    // Split on the FIRST '=' only.
    const eqIdx = line.indexOf('=');
    if (eqIdx === -1) continue;

    const key = line.slice(0, eqIdx).trim();
    const val = line.slice(eqIdx + 1).trim();
    const kl  = key.toLowerCase();

    // ── per_frame_init_N ────────────────────────────────────────────────────
    {
      const m = kl.match(/^per_frame_init_(\d+)$/);
      if (m) { initBucket[+m[1]] = val; continue; }
    }
    // ── per_frame_N ─────────────────────────────────────────────────────────
    {
      const m = kl.match(/^per_frame_(\d+)$/);
      if (m) { frameBucket[+m[1]] = val; continue; }
    }
    // ── per_pixel_N ─────────────────────────────────────────────────────────
    {
      const m = kl.match(/^per_pixel_(\d+)$/);
      if (m) { pixelBucket[+m[1]] = val; continue; }
    }
    // ── wave_K_per_frame_N ───────────────────────────────────────────────────
    {
      const m = kl.match(/^wave_(\d+)_per_frame_(\d+)$/);
      if (m) {
        const [, k, n] = m.map(Number);
        (waveFrameBucket[k] ??= {})[n] = val;
        continue;
      }
    }
    // ── wave_K_per_point_N ───────────────────────────────────────────────────
    {
      const m = kl.match(/^wave_(\d+)_per_point_(\d+)$/);
      if (m) {
        const [, k, n] = m.map(Number);
        (wavePointBucket[k] ??= {})[n] = val;
        continue;
      }
    }
    // ── shape_K_per_frame_N ──────────────────────────────────────────────────
    {
      const m = kl.match(/^shape_(\d+)_per_frame_(\d+)$/);
      if (m) {
        const [, k, n] = m.map(Number);
        (shapeFrameBucket[k] ??= {})[n] = val;
        continue;
      }
    }
    // ── shape_K_per_point_N ──────────────────────────────────────────────────
    {
      const m = kl.match(/^shape_(\d+)_per_point_(\d+)$/);
      if (m) {
        const [, k, n] = m.map(Number);
        (shapePointBucket[k] ??= {})[n] = val;
        continue;
      }
    }

    // Everything else → baseVals (preserves original key casing).
    preset.baseVals[key] = val;
  }

  // Sort each bucket by numeric index and flatten to arrays.
  const sortedVals = bucket =>
    Object.keys(bucket)
          .map(Number)
          .sort((a, b) => a - b)
          .map(k => bucket[k]);

  const sortedNested = bucketMap =>
    Object.fromEntries(
      Object.entries(bucketMap).map(([k, bucket]) => [k, sortedVals(bucket)])
    );

  preset.initEqs      = sortedVals(initBucket);
  preset.perFrameEqs  = sortedVals(frameBucket);
  preset.perPixelEqs  = sortedVals(pixelBucket);
  preset.waveFrameEqs = sortedNested(waveFrameBucket);
  preset.wavePointEqs = sortedNested(wavePointBucket);
  preset.shapeFrameEqs= sortedNested(shapeFrameBucket);
  preset.shapePointEqs= sortedNested(shapePointBucket);

  return preset;
}

/**
 * Serialise a MilkPreset back to .milk text (round-trip).
 *
 * @param {MilkPreset} preset
 * @param {string}     [name]  Ignored in output but useful for the caller.
 * @returns {string}
 */
export function serialiseMilk(preset, name = 'Converted') {
  const lines = ['[preset00]'];

  for (const [k, v] of Object.entries(preset.baseVals)) {
    lines.push(`${k}=${v}`);
  }

  const emitEqs = (pfx, eqs) =>
    eqs.forEach((eq, i) => lines.push(`${pfx}${i + 1}=${eq}`));

  emitEqs('per_frame_init_', preset.initEqs);
  emitEqs('per_frame_',      preset.perFrameEqs);
  emitEqs('per_pixel_',      preset.perPixelEqs);

  for (const k of Object.keys(preset.waveFrameEqs).map(Number).sort((a,b)=>a-b)) {
    emitEqs(`wave_${k}_per_frame_`, preset.waveFrameEqs[k] ?? []);
    emitEqs(`wave_${k}_per_point_`, preset.wavePointEqs[k] ?? []);
  }
  for (const k of Object.keys(preset.shapeFrameEqs).map(Number).sort((a,b)=>a-b)) {
    emitEqs(`shape_${k}_per_frame_`, preset.shapeFrameEqs[k] ?? []);
    emitEqs(`shape_${k}_per_point_`, preset.shapePointEqs[k] ?? []);
  }

  return lines.join('\n');
}
