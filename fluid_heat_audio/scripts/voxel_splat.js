// voxel_splat.js
// Injects SDF primitives into a 3D scalar jit.matrix at positions defined by
// 8 voices, with amplitudes read from an 8-bin audio list.
//
// Inlets (right-to-left in Max):
//   0 : jit_matrix (current field, 1 plane float32, dim W H D)  - modified in place
//   0 : bins       -> sets AMPS   (list of 8 floats 0..4)
//   0 : voices     -> sets VOICES (list of 8 * 5 floats: [x y z shape r] each in 0..1 for xyz)
//   0 : shape      -> "shape <i> <int>"  set individual voice shape
//   0 : voice      -> "voice <i> <x> <y> <z> <shape> <r>"
//
// Shape codes:
//   0 = sphere       1 = box         2 = torus
//   3 = octahedron   4 = capsule (y) 5 = gyroid slice
//
// The matrix is passed through unchanged in structure - only voxel values are
// modified. Output is the same jit.matrix name that came in.

autowatch = 1;
inlets  = 1;
outlets = 1;

// Persistent config, updated by messages
var AMPS   = [0,0,0,0,0,0,0,0];
var VOICES = [
	// x, y, z, shape, radius   (defaults spread across a unit cube)
	0.30, 0.20, 0.30, 0, 0.18,
	0.50, 0.20, 0.50, 0, 0.20,
	0.70, 0.20, 0.70, 0, 0.18,
	0.20, 0.50, 0.50, 2, 0.16,
	0.80, 0.50, 0.50, 2, 0.16,
	0.30, 0.80, 0.30, 1, 0.14,
	0.70, 0.80, 0.70, 1, 0.14,
	0.50, 0.55, 0.50, 5, 0.28
];

function bins()   { AMPS   = arrayFromArgs(arguments); }
function voices() { VOICES = arrayFromArgs(arguments); }

function voice() {
	// voice <i> <x> <y> <z> <shape> <r>
	var a = arrayFromArgs(arguments);
	if (a.length < 6) return;
	var i = parseInt(a[0], 10);
	if (i < 0 || i > 7) return;
	var off = i * 5;
	VOICES[off  ] = parseFloat(a[1]);
	VOICES[off+1] = parseFloat(a[2]);
	VOICES[off+2] = parseFloat(a[3]);
	VOICES[off+3] = parseFloat(a[4]);
	VOICES[off+4] = parseFloat(a[5]);
}

function shape() {
	// shape <i> <int>
	var a = arrayFromArgs(arguments);
	if (a.length < 2) return;
	var i = parseInt(a[0], 10);
	if (i < 0 || i > 7) return;
	VOICES[i * 5 + 3] = parseInt(a[1], 10);
}

function arrayFromArgs(args) {
	var a = new Array(args.length);
	for (var i = 0; i < args.length; ++i) a[i] = args[i];
	return a;
}

// SDF evaluators - return an *additive* mass value in [0..1].
// (We're not doing signed-distance rendering; we're building a positive
//  scalar field for marching cubes at threshold ~0.5.)

function splat_sphere(dx, dy, dz, r) {
	var d2 = dx*dx + dy*dy + dz*dz;
	var r2 = r * r;
	if (d2 >= r2) return 0.0;
	var t = 1.0 - d2 / r2;
	return t * t;                      // smooth falloff
}

function splat_box(dx, dy, dz, r) {
	dx = Math.abs(dx); dy = Math.abs(dy); dz = Math.abs(dz);
	if (dx >= r || dy >= r || dz >= r) return 0.0;
	var m = Math.max(dx, dy, dz);
	var t = 1.0 - m / r;
	return t * t;
}

function splat_torus(dx, dy, dz, r) {
	// major radius R, minor radius r/3
	var R = r * 0.7;
	var mr = r * 0.35;
	var q = Math.sqrt(dx*dx + dz*dz) - R;
	var d = Math.sqrt(q*q + dy*dy);
	if (d >= mr) return 0.0;
	var t = 1.0 - d / mr;
	return t * t;
}

function splat_octa(dx, dy, dz, r) {
	var s = Math.abs(dx) + Math.abs(dy) + Math.abs(dz);
	if (s >= r) return 0.0;
	var t = 1.0 - s / r;
	return t * t;
}

function splat_capsule(dx, dy, dz, r) {
	var h = r * 1.4;
	var cy = dy;
	if (cy > h) cy -= h;
	else if (cy < -h) cy += h;
	else cy = 0;
	return splat_sphere(dx, cy, dz, r * 0.6);
}

function splat_gyroid(dx, dy, dz, r) {
	// Gyroid iso-surface, thickened; only add mass inside a bounding sphere r.
	var d2 = dx*dx + dy*dy + dz*dz;
	if (d2 >= r*r) return 0.0;
	var s = 6.0;
	var g = Math.sin(dx*s) * Math.cos(dy*s)
	      + Math.sin(dy*s) * Math.cos(dz*s)
	      + Math.sin(dz*s) * Math.cos(dx*s);
	var v = 1.0 - Math.abs(g) * 0.5;
	var falloff = 1.0 - Math.sqrt(d2) / r;
	return Math.max(0.0, v) * falloff * falloff;
}

function splat_dispatch(shape, dx, dy, dz, r) {
	switch (shape | 0) {
		case 0: return splat_sphere (dx, dy, dz, r);
		case 1: return splat_box    (dx, dy, dz, r);
		case 2: return splat_torus  (dx, dy, dz, r);
		case 3: return splat_octa   (dx, dy, dz, r);
		case 4: return splat_capsule(dx, dy, dz, r);
		case 5: return splat_gyroid (dx, dy, dz, r);
	}
	return splat_sphere(dx, dy, dz, r);
}

function jit_matrix(name) {
	var mat = new JitterMatrix(name);
	var dim = mat.dim;
	if (!dim || dim.length < 3) {
		outlet(0, "jit_matrix", name);
		return;
	}
	var W = dim[0], H = dim[1], D = dim[2];

	for (var i = 0; i < 8; ++i) {
		var amp = AMPS[i] || 0;
		if (amp <= 1e-3) continue;
		var vo = i * 5;
		var vx = VOICES[vo  ];
		var vy = VOICES[vo+1];
		var vz = VOICES[vo+2];
		var sh = VOICES[vo+3];
		var rr = VOICES[vo+4];

		// Convert normalized [0,1] radius to voxel-space distance
		var vr = rr * Math.min(W, H, D);
		var cx = vx * W;
		var cy = vy * H;
		var cz = vz * D;
		var vri = Math.ceil(vr) + 1;

		var x0 = Math.max(0, Math.floor(cx - vri));
		var x1 = Math.min(W - 1, Math.floor(cx + vri));
		var y0 = Math.max(0, Math.floor(cy - vri));
		var y1 = Math.min(H - 1, Math.floor(cy + vri));
		var z0 = Math.max(0, Math.floor(cz - vri));
		var z1 = Math.min(D - 1, Math.floor(cz + vri));

		var gain = amp * 0.9;

		for (var z = z0; z <= z1; ++z) {
			var dz = z - cz;
			for (var y = y0; y <= y1; ++y) {
				var dy = y - cy;
				for (var x = x0; x <= x1; ++x) {
					var dx = x - cx;
					var v = splat_dispatch(sh, dx, dy, dz, vr);
					if (v <= 0) continue;
					var prev = mat.getcell(x, y, z);
					mat.setcell(x, y, z, "val",
						Math.min(1.5, prev[0] + v * gain));
				}
			}
		}
	}

	outlet(0, "jit_matrix", name);
}
