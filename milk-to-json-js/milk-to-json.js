/**
 * milk-to-json.js
 * Converts a Milkdrop .milk preset into a Butterchurn-compatible JSON object.
 *
 * Works in Node.js AND in the browser (pure ES module, zero dependencies).
 *
 * Usage — convert text directly:
 *   import { milkToJson, jsonToMilk } from './milk-to-json.js';
 *
 *   const json    = milkToJson(milkFileText, 'MyPreset');
 *   const milkOut = jsonToMilk(json);                    // round-trip
 *
 * The returned object can be passed straight to Butterchurn:
 *   visualizer.loadPreset(milkToJson(text, name), 0);
 */

import { parseMilk, serialiseMilk } from './milk-parser.js';

// ─────────────────────────────────────────────────────────────────────────────
// Scalar field maps  (.milk key → Butterchurn baseVals key)
// .milk key matching is case-insensitive.
// ─────────────────────────────────────────────────────────────────────────────

const FLOAT_MAP = {
  // Core warp
  zoom:              'zoom',
  rot:               'rot',
  cx:                'cx',
  cy:                'cy',
  dx:                'dx',
  dy:                'dy',
  warp:              'warp',
  sx:                'sx',
  sy:                'sy',
  // fSizeFactorX / fSizeFactorY are aliases for sx/sy
  fsizefactorx:      'sx',
  fsizefactory:      'sy',

  // Decay / image
  fdecay:            'decay',
  fgammaadj:         'gammaAdj',
  fshader:           'fShader',

  // Video echo
  fvideoechozoom:    'echo_zoom',
  fvideoechoalpha:   'echo_alpha',

  // Wave
  fwavealpha:        'wave_a',
  fwavescale:        'wave_scale',
  fwavesmoothing:    'wave_smoothing',
  fwaveparam:        'wave_mystery',
  wave_r:            'wave_r',
  wave_g:            'wave_g',
  wave_b:            'wave_b',
  wave_x:            'wave_x',
  wave_y:            'wave_y',

  // Warp animation
  fwarpanimspeed:    'warpAnimSpeed',
  fwarpscale:        'warpScale',
  fzoomexponent:     'zoomExponent',

  // Borders
  ob_size:           'ob_size',
  ob_r:              'ob_r',  ob_g: 'ob_g', ob_b: 'ob_b', ob_a: 'ob_a',
  ib_size:           'ib_size',
  ib_r:              'ib_r',  ib_g: 'ib_g', ib_b: 'ib_b', ib_a: 'ib_a',

  // Motion vectors
  nmotionvectorsx:   'mv_x',
  nmotionvectorsy:   'mv_y',
  mv_dx:             'mv_dx',
  mv_dy:             'mv_dy',
  mv_l:              'mv_l',
  mv_r:              'mv_r',
  mv_g:              'mv_g',
  mv_b:              'mv_b',
  mv_a:              'mv_a',

  // Volume modulation
  fmodwavealphastart:'modWaveAlphaStart',
  fmodwavealphaend:  'modWaveAlphaEnd',

  frating:           'fRating',
};

const INT_MAP = {
  nvideoechoorientation: 'echo_orient',
  nwavemode:             'wave_mode',
  badditivewaves:        'additiveWaves',
  bwavedots:             'waveDots',
  bwavethick:            'waveThick',
  btexwrap:              'bTexWrap',
  bdarkencenter:         'bDarkenCenter',
  bredbluestero:         'bRedBlueStereo',
  bredbluestero:         'bRedBlueStereo',
  bbrighten:             'bBrighten',
  bdarken:               'bDarken',
  bsolarize:             'bSolarize',
  binvert:               'bInvert',
  bmodwavealphabyvolume: 'modWaveAlphaByVol',
  bmaximizewavecolor:    'maximizeWaveColor',
};

// ─────────────────────────────────────────────────────────────────────────────
// Default Butterchurn baseVals
// ─────────────────────────────────────────────────────────────────────────────

function defaultBaseVals() {
  return {
    zoom: 1,     rot: 0,      cx: 0.5,  cy: 0.5,
    dx: 0,       dy: 0,       warp: 1,  sx: 1,  sy: 1,
    decay: 0.98,
    echo_zoom: 1, echo_alpha: 0, echo_orient: 0,
    wave_mode: 0,
    additiveWaves: 0, waveDots: 0, waveThick: 0,
    wave_r: 0.5,  wave_g: 0.5,  wave_b: 0.5,
    wave_x: 0.5,  wave_y: 0.5,
    wave_a: 0.8,  wave_scale: 1, wave_smoothing: 0.75, wave_mystery: 0,
    ob_size: 0.01, ob_r: 0, ob_g: 0, ob_b: 0, ob_a: 0,
    ib_size: 0.01, ib_r: 0, ib_g: 0, ib_b: 0, ib_a: 0,
    mv_x: 12, mv_y: 9, mv_l: 0.9,
    mv_r: 0, mv_g: 0, mv_b: 0, mv_a: 0, mv_dx: 0, mv_dy: 0,
    warpAnimSpeed: 1, warpScale: 1.33, zoomExponent: 1,
    gammaAdj: 1.7, fShader: 0,
    bTexWrap: 1, bDarkenCenter: 0, bRedBlueStereo: 0,
    bBrighten: 0, bDarken: 0, bSolarize: 0, bInvert: 0,
    modWaveAlphaByVol: 0, modWaveAlphaStart: 0.75, modWaveAlphaEnd: 0.95,
    maximizeWaveColor: 0, fRating: 3,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Join an array of equation strings into one newline-separated block. */
function joinEqs(eqs) {
  return eqs
    .map(e => e.trim())
    .filter(Boolean)
    .join('\n');
}

/** Build a Butterchurn wave object from .milk parsed data. */
function buildWave(index, milk) {
  const wave = {
    enabled: 1, bSpectrum: 0, bUseDots: 0, bDrawThick: 0, bAdditive: 0,
    samples: 512, sep: 0, smoothing: 0.5, scaling: 1,
    r: 1, g: 1, b: 1, a: 1,
    per_frame_init_eqs_compat: '',
    per_frame_eqs_compat:      joinEqs(milk.waveFrameEqs[index] ?? []),
    per_point_eqs_compat:      joinEqs(milk.wavePointEqs[index] ?? []),
  };

  // Pull scalar wave_K_* fields from baseVals.
  const prefix = `wave_${index}_`;
  for (const [rawKey, val] of Object.entries(milk.baseVals)) {
    if (!rawKey.toLowerCase().startsWith(prefix)) continue;
    const sub = rawKey.slice(prefix.length).toLowerCase();
    switch (sub) {
      case 'enabled':    wave.enabled    = +val; break;
      case 'bspectrum':  wave.bSpectrum  = +val; break;
      case 'busedots':   wave.bUseDots   = +val; break;
      case 'bdrawthick': wave.bDrawThick = +val; break;
      case 'badditive':  wave.bAdditive  = +val; break;
      case 'samples':    wave.samples    = +val; break;
      case 'sep':        wave.sep        = +val; break;
      case 'smoothing':  wave.smoothing  = +val; break;
      case 'scaling':    wave.scaling    = +val; break;
      case 'r':          wave.r          = +val; break;
      case 'g':          wave.g          = +val; break;
      case 'b':          wave.b          = +val; break;
      case 'a':          wave.a          = +val; break;
    }
  }
  return wave;
}

/** Build a Butterchurn shape object from .milk parsed data. */
function buildShape(index, milk) {
  const shape = {
    enabled: 1, sides: 4, additive: 0, thickOutline: 0,
    textured: 0, num_inst: 1,
    x: 0.5, y: 0.5, rad: 0.1, ang: 0,
    r: 1, g: 0, b: 0, a: 1,
    r2: 1, g2: 1, b2: 0, a2: 0,
    border_r: 1, border_g: 1, border_b: 1, border_a: 0.1,
    per_frame_init_eqs_compat: '',
    per_frame_eqs_compat: joinEqs(milk.shapeFrameEqs[index] ?? []),
    per_point_eqs_compat: joinEqs(milk.shapePointEqs[index] ?? []),
  };

  const prefix = `shape_${index}_`;
  for (const [rawKey, val] of Object.entries(milk.baseVals)) {
    if (!rawKey.toLowerCase().startsWith(prefix)) continue;
    const sub = rawKey.slice(prefix.length).toLowerCase();
    switch (sub) {
      case 'enabled':      shape.enabled      = +val; break;
      case 'sides':        shape.sides        = +val; break;
      case 'additive':     shape.additive     = +val; break;
      case 'thickoutline': shape.thickOutline = +val; break;
      case 'textured':     shape.textured     = +val; break;
      case 'num_inst':     shape.num_inst     = +val; break;
      case 'x':            shape.x            = +val; break;
      case 'y':            shape.y            = +val; break;
      case 'rad':          shape.rad          = +val; break;
      case 'ang':          shape.ang          = +val; break;
      case 'r':            shape.r            = +val; break;
      case 'g':            shape.g            = +val; break;
      case 'b':            shape.b            = +val; break;
      case 'a':            shape.a            = +val; break;
      case 'r2':           shape.r2           = +val; break;
      case 'g2':           shape.g2           = +val; break;
      case 'b2':           shape.b2           = +val; break;
      case 'a2':           shape.a2           = +val; break;
      case 'border_r':     shape.border_r     = +val; break;
      case 'border_g':     shape.border_g     = +val; break;
      case 'border_b':     shape.border_b     = +val; break;
      case 'border_a':     shape.border_a     = +val; break;
    }
  }
  return shape;
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Convert .milk file text to a Butterchurn preset object.
 *
 * @param {string} milkText   Raw text of the .milk file.
 * @param {string} [name]     Display name for the preset.
 * @returns {Object}          Butterchurn-compatible preset object.
 *                            Pass directly to visualizer.loadPreset().
 */
export function milkToJson(milkText, name = 'Converted') {
  const milk = parseMilk(milkText);
  const bv   = defaultBaseVals();

  // ── Map scalar base values ──────────────────────────────────────────────────
  for (const [rawKey, val] of Object.entries(milk.baseVals)) {
    const kl = rawKey.toLowerCase();

    const floatKey = FLOAT_MAP[kl];
    if (floatKey !== undefined) {
      const n = parseFloat(val);
      if (!isNaN(n)) bv[floatKey] = n;
      continue;
    }

    const intKey = INT_MAP[kl];
    if (intKey !== undefined) {
      const n = parseInt(val, 10);
      if (!isNaN(n)) bv[intKey] = n;
    }
  }

  // ── Equation blocks ─────────────────────────────────────────────────────────
  const initEQs     = joinEqs(milk.initEqs);
  const perFrameEQs = joinEqs(milk.perFrameEqs);
  const perPixelEQs = joinEqs(milk.perPixelEqs);

  // ── Waves ────────────────────────────────────────────────────────────────────
  // Collect all wave indices from both frame and point equation maps + baseVals.
  const waveIndices = new Set([
    ...Object.keys(milk.waveFrameEqs).map(Number),
    ...Object.keys(milk.wavePointEqs).map(Number),
  ]);
  // Also detect wave indices from wave_K_enabled keys in baseVals.
  for (const key of Object.keys(milk.baseVals)) {
    const m = key.toLowerCase().match(/^wave_(\d+)_/);
    if (m) waveIndices.add(+m[1]);
  }
  const waves = [...waveIndices].sort((a, b) => a - b)
                                .map(i => buildWave(i, milk));

  // ── Shapes ────────────────────────────────────────────────────────────────────
  const shapeIndices = new Set([
    ...Object.keys(milk.shapeFrameEqs).map(Number),
    ...Object.keys(milk.shapePointEqs).map(Number),
  ]);
  for (const key of Object.keys(milk.baseVals)) {
    const m = key.toLowerCase().match(/^shape_(\d+)_/);
    if (m) shapeIndices.add(+m[1]);
  }
  const shapes = [...shapeIndices].sort((a, b) => a - b)
                                  .map(i => buildShape(i, milk));

  return { name, baseVals: bv, initEQs, perFrameEQs, perPixelEQs, waves, shapes };
}

/**
 * Convert a Butterchurn preset object back to .milk text (round-trip).
 *
 * @param {Object} preset   A Butterchurn preset object (from milkToJson or loaded JSON).
 * @returns {string}        .milk file text.
 */
export function jsonToMilk(preset) {
  // Build a MilkPreset structure and call serialiseMilk.
  const bv = preset.baseVals ?? {};

  // Reverse the float/int maps.
  const baseVals = {};
  const reverseFloat = Object.fromEntries(Object.entries(FLOAT_MAP).map(([k,v])=>[v,k]));
  const reverseInt   = Object.fromEntries(Object.entries(INT_MAP  ).map(([k,v])=>[v,k]));

  for (const [bcKey, milkKey] of Object.entries(reverseFloat)) {
    if (bv[bcKey] !== undefined) baseVals[milkKey] = String(bv[bcKey]);
  }
  for (const [bcKey, milkKey] of Object.entries(reverseInt)) {
    if (bv[bcKey] !== undefined) baseVals[milkKey] = String(bv[bcKey]);
  }

  // Waves → baseVals + waveFrame/wavePoint eq maps.
  const waveFrameEqs = {}, wavePointEqs = {};
  for (const [i, w] of (preset.waves ?? []).entries()) {
    const k = i + 1;
    baseVals[`wave_${k}_enabled`]    = String(w.enabled   ?? 1);
    baseVals[`wave_${k}_bSpectrum`]  = String(w.bSpectrum ?? 0);
    baseVals[`wave_${k}_bUseDots`]   = String(w.bUseDots  ?? 0);
    baseVals[`wave_${k}_bDrawThick`] = String(w.bDrawThick?? 0);
    baseVals[`wave_${k}_bAdditive`]  = String(w.bAdditive ?? 0);
    baseVals[`wave_${k}_samples`]    = String(w.samples   ?? 512);
    baseVals[`wave_${k}_sep`]        = String(w.sep       ?? 0);
    baseVals[`wave_${k}_smoothing`]  = String(w.smoothing ?? 0.5);
    baseVals[`wave_${k}_scaling`]    = String(w.scaling   ?? 1);
    baseVals[`wave_${k}_r`]          = String(w.r ?? 1);
    baseVals[`wave_${k}_g`]          = String(w.g ?? 1);
    baseVals[`wave_${k}_b`]          = String(w.b ?? 1);
    baseVals[`wave_${k}_a`]          = String(w.a ?? 1);
    waveFrameEqs[k] = splitEqs(w.per_frame_eqs_compat ?? '');
    wavePointEqs[k] = splitEqs(w.per_point_eqs_compat ?? '');
  }

  // Shapes → baseVals.
  const shapeFrameEqs = {}, shapePointEqs = {};
  for (const [i, s] of (preset.shapes ?? []).entries()) {
    const k = i + 1;
    for (const [field, val] of Object.entries(s)) {
      if (field === 'per_frame_eqs_compat' || field === 'per_frame_init_eqs_compat') {
        shapeFrameEqs[k] = splitEqs(val);
      } else if (field === 'per_point_eqs_compat') {
        shapePointEqs[k] = splitEqs(val);
      } else {
        baseVals[`shape_${k}_${field}`] = String(val);
      }
    }
  }

  const milkPreset = {
    baseVals,
    initEqs:      splitEqs(preset.initEQs     ?? ''),
    perFrameEqs:  splitEqs(preset.perFrameEQs ?? ''),
    perPixelEqs:  splitEqs(preset.perPixelEQs ?? ''),
    waveFrameEqs, wavePointEqs,
    shapeFrameEqs, shapePointEqs,
  };

  return serialiseMilk(milkPreset, preset.name ?? 'Preset');
}

/** Split an equation block string back into an array of statements. */
function splitEqs(block) {
  return block
    .split(/[;\n]/)
    .map(s => s.trim())
    .filter(s => s && !s.startsWith(';') && !s.startsWith('//'));
}
