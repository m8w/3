// mesh_export.js
// Write a triangle mesh (position + normal jit.matrix pair from marching_cubes.js)
// to disk as OBJ, STL (binary), or PLY. One-click export from Max.
//
// Messages:
//   position <jit_matrix-name>     latch the position matrix (3 planes float32)
//   normal   <jit_matrix-name>     latch the normal matrix
//   dir      <folder-path>         set output directory (default ~/ExternalRadio/fh_meshes)
//   prefix   <str>                 filename prefix (default 'fh_mesh')
//   obj                            write .obj to <dir>/<prefix>_<ts>.obj
//   stl                            write .stl (binary) to <dir>/<prefix>_<ts>.stl
//   ply                            write .ply (ascii) to <dir>/<prefix>_<ts>.ply
//   all                            write all three
//
// Outlets:
//   0: last written file path
//   1: status ("written <path>" | "error <msg>")

autowatch = 1;
inlets  = 1;
outlets = 2;

var POS_NAME = "";
var NRM_NAME = "";
var DIR      = null;
var PREFIX   = "fh_mesh";

function _defaultDir() {
	var home = File.applicationpath;              // may not be $HOME
	var candidates = [
		Max.env["HOME"] ? (Max.env["HOME"] + "/ExternalRadio/fh_meshes") : null,
		"~/ExternalRadio/fh_meshes"
	];
	for (var i = 0; i < candidates.length; ++i) {
		if (candidates[i]) return candidates[i];
	}
	return "./fh_meshes";
}

function position(n) { POS_NAME = String(n); }
function normal(n)   { NRM_NAME = String(n); }
function dir(p)      { DIR = String(p); }
function prefix(p)   { PREFIX = String(p); }

function _stamp() {
	var d = new Date();
	function p(n) { return (n < 10 ? "0" : "") + n; }
	return d.getFullYear() +
	       p(d.getMonth() + 1) + p(d.getDate()) + "_" +
	       p(d.getHours())    + p(d.getMinutes()) + p(d.getSeconds());
}

function _prepare() {
	if (!POS_NAME) { outlet(1, "error", "no position matrix"); return null; }
	var pos = new JitterMatrix(POS_NAME);
	var nrm = NRM_NAME ? new JitterMatrix(NRM_NAME) : null;
	var d = pos.dim;
	if (!d || d.length < 1 || d[0] < 3) {
		outlet(1, "error", "position matrix too small");
		return null;
	}
	var nVerts = d[0];
	if (nVerts % 3 !== 0) {
		outlet(1, "error", "position count not divisible by 3");
		return null;
	}
	if (!DIR) DIR = _defaultDir();

	// Read all vertex/normal data up front
	var P = new Array(nVerts);
	var N = nrm ? new Array(nVerts) : null;
	for (var i = 0; i < nVerts; ++i) {
		P[i] = pos.getcell(i, 0);
		if (N) N[i] = nrm.getcell(i, 0);
	}
	return { P: P, N: N, nTri: nVerts / 3, dir: DIR };
}

function _openFile(path) {
	var f = new File(path, "write");
	if (!f.isopen) {
		outlet(1, "error", "cannot open " + path);
		return null;
	}
	f.filetype = "TEXT";
	return f;
}

function obj() {
	var m = _prepare(); if (!m) return;
	var path = m.dir + "/" + PREFIX + "_" + _stamp() + ".obj";
	var f = _openFile(path); if (!f) return;
	try {
		f.writeline("# fluid_heat_audio - fh.mesh_synth export");
		f.writeline("# " + m.nTri + " triangles, " + (m.P.length) + " vertices");
		for (var i = 0; i < m.P.length; ++i) {
			var v = m.P[i];
			f.writeline("v " + v[0].toFixed(5) + " " +
			                    v[1].toFixed(5) + " " +
			                    v[2].toFixed(5));
		}
		if (m.N) {
			for (var j = 0; j < m.N.length; ++j) {
				var n = m.N[j];
				f.writeline("vn " + n[0].toFixed(5) + " " +
				                    n[1].toFixed(5) + " " +
				                    n[2].toFixed(5));
			}
		}
		for (var t = 0; t < m.nTri; ++t) {
			var a = t*3 + 1, b = t*3 + 2, c = t*3 + 3;
			if (m.N) {
				f.writeline("f " + a + "//" + a + " " +
				                    b + "//" + b + " " +
				                    c + "//" + c);
			} else {
				f.writeline("f " + a + " " + b + " " + c);
			}
		}
	} finally {
		f.close();
	}
	outlet(0, path);
	outlet(1, "written", path);
}

function ply() {
	var m = _prepare(); if (!m) return;
	var path = m.dir + "/" + PREFIX + "_" + _stamp() + ".ply";
	var f = _openFile(path); if (!f) return;
	try {
		var nVerts = m.P.length;
		f.writeline("ply");
		f.writeline("format ascii 1.0");
		f.writeline("comment fluid_heat_audio - fh.mesh_synth export");
		f.writeline("element vertex " + nVerts);
		f.writeline("property float x");
		f.writeline("property float y");
		f.writeline("property float z");
		if (m.N) {
			f.writeline("property float nx");
			f.writeline("property float ny");
			f.writeline("property float nz");
		}
		f.writeline("element face " + m.nTri);
		f.writeline("property list uchar int vertex_indices");
		f.writeline("end_header");
		for (var i = 0; i < nVerts; ++i) {
			var v = m.P[i];
			var line = v[0].toFixed(5) + " " + v[1].toFixed(5) + " " + v[2].toFixed(5);
			if (m.N) {
				var n = m.N[i];
				line += " " + n[0].toFixed(5) + " " + n[1].toFixed(5) + " " + n[2].toFixed(5);
			}
			f.writeline(line);
		}
		for (var t = 0; t < m.nTri; ++t) {
			f.writeline("3 " + (t*3) + " " + (t*3+1) + " " + (t*3+2));
		}
	} finally {
		f.close();
	}
	outlet(0, path);
	outlet(1, "written", path);
}

// Binary STL - most reliable format for import into Blender/Cinema4D/etc.
function stl() {
	var m = _prepare(); if (!m) return;
	var path = m.dir + "/" + PREFIX + "_" + _stamp() + ".stl";
	var f = new File(path, "write");
	if (!f.isopen) { outlet(1, "error", "cannot open " + path); return; }
	f.filetype = "BINA";
	try {
		// 80-byte header
		var header = "fluid_heat_audio - fh.mesh_synth export";
		while (header.length < 80) header += " ";
		for (var h = 0; h < 80; ++h) f.writechar(header.charCodeAt(h));
		// 4-byte little-endian triangle count
		_writeUint32LE(f, m.nTri);
		// per-triangle: 12 floats + 2 pad bytes = 50 bytes
		for (var t = 0; t < m.nTri; ++t) {
			var v0 = m.P[t*3], v1 = m.P[t*3+1], v2 = m.P[t*3+2];
			// STL normal = average of vertex normals (fallback = computed face normal)
			var nx, ny, nz;
			if (m.N) {
				var n0 = m.N[t*3], n1 = m.N[t*3+1], n2 = m.N[t*3+2];
				nx = (n0[0] + n1[0] + n2[0]) / 3;
				ny = (n0[1] + n1[1] + n2[1]) / 3;
				nz = (n0[2] + n1[2] + n2[2]) / 3;
			} else {
				var ax = v1[0]-v0[0], ay = v1[1]-v0[1], az = v1[2]-v0[2];
				var bx = v2[0]-v0[0], by = v2[1]-v0[1], bz = v2[2]-v0[2];
				nx = ay*bz - az*by;
				ny = az*bx - ax*bz;
				nz = ax*by - ay*bx;
			}
			var nl = Math.sqrt(nx*nx + ny*ny + nz*nz) + 1e-8;
			_writeFloat32LE(f, nx/nl);
			_writeFloat32LE(f, ny/nl);
			_writeFloat32LE(f, nz/nl);
			for (var k = 0; k < 3; ++k) {
				var v = [v0, v1, v2][k];
				_writeFloat32LE(f, v[0]);
				_writeFloat32LE(f, v[1]);
				_writeFloat32LE(f, v[2]);
			}
			f.writechar(0); f.writechar(0);       // attribute byte count
		}
	} finally {
		f.close();
	}
	outlet(0, path);
	outlet(1, "written", path);
}

function all() { obj(); ply(); stl(); }

// --- little helpers for binary write (Max's File doesn't have writeuint32 etc)

function _writeUint32LE(f, n) {
	f.writechar(n & 0xff);
	f.writechar((n >>> 8) & 0xff);
	f.writechar((n >>> 16) & 0xff);
	f.writechar((n >>> 24) & 0xff);
}

function _writeFloat32LE(f, x) {
	// pack float32 via a scratch buffer
	var buf = new ArrayBuffer(4);
	var dv  = new DataView(buf);
	dv.setFloat32(0, x, true);
	f.writechar(dv.getUint8(0));
	f.writechar(dv.getUint8(1));
	f.writechar(dv.getUint8(2));
	f.writechar(dv.getUint8(3));
}
