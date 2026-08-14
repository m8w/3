{
	"patcher" : 	{
		"fileversion" : 1,
		"appversion" : 		{ "major" : 9, "minor" : 1, "revision" : 0, "architecture" : "x64", "modernui" : 1 },
		"classnamespace" : "box",
		"rect" : [ 60.0, 60.0, 1080.0, 700.0 ],
		"gridonopen" : 1,
		"gridsize" : [ 15.0, 15.0 ],
		"boxes" : [
			{ "box" : { "id" : "in-bang", "maxclass" : "inlet", "numinlets":0, "numoutlets":1, "outlettype":["bang"],
				"patching_rect" : [ 30.0, 20.0, 30.0, 30.0 ], "comment" : "frame bang (from qmetro)" } },
			{ "box" : { "id" : "in-bins", "maxclass" : "inlet", "numinlets":0, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 80.0, 20.0, 30.0, 30.0 ], "comment" : "8-bin list (post organic-mod)" } },
			{ "box" : { "id" : "in-voices", "maxclass" : "inlet", "numinlets":0, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 130.0, 20.0, 30.0, 30.0 ], "comment" : "voice config: list of 8 [x y z shape r] (each 5 floats)" } },
			{ "box" : { "id" : "in-decay", "maxclass" : "inlet", "numinlets":0, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 180.0, 20.0, 30.0, 30.0 ], "comment" : "decay (0..1)  default 0.94" } },
			{ "box" : { "id" : "in-blur", "maxclass" : "inlet", "numinlets":0, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 230.0, 20.0, 30.0, 30.0 ], "comment" : "blur amount (0..1)  default 0.25" } },

			{ "box" : { "id" : "out-mat", "maxclass" : "outlet", "numinlets":1, "numoutlets":0,
				"patching_rect" : [ 30.0, 640.0, 30.0, 30.0 ], "comment" : "jit_matrix (scalar 3D field, float32)" } },
			{ "box" : { "id" : "out-info", "maxclass" : "outlet", "numinlets":1, "numoutlets":0,
				"patching_rect" : [ 80.0, 640.0, 30.0, 30.0 ], "comment" : "info messages" } },

			{ "box" : { "id" : "title", "maxclass" : "comment",
				"patching_rect" : [ 30.0, 60.0, 1000.0, 22.0 ], "fontsize" : 14.0,
				"text" : "fh.voxel_field : 3D scalar field (float32, dim 48^3) evolved per frame - decay + blur + SDF splats from 8 audio voices.",
				"numinlets":1, "numoutlets":0 } },

			{ "box" : { "id" : "field-a", "maxclass" : "newobj",
				"patching_rect" : [ 30.0, 100.0, 340.0, 22.0 ],
				"text" : "jit.matrix fh_field_a 1 float32 48 48 48",
				"numinlets":1, "numoutlets":1, "outlettype":["jit_matrix"] } },

			{ "box" : { "id" : "field-b", "maxclass" : "newobj",
				"patching_rect" : [ 30.0, 130.0, 340.0, 22.0 ],
				"text" : "jit.matrix fh_field_b 1 float32 48 48 48",
				"numinlets":1, "numoutlets":1, "outlettype":["jit_matrix"] } },

			{ "box" : { "id" : "clear-load", "maxclass" : "newobj", "numinlets":1, "numoutlets":1, "outlettype":["bang"],
				"patching_rect" : [ 380.0, 100.0, 60.0, 22.0 ], "text" : "loadbang" } },
			{ "box" : { "id" : "clear-msg", "maxclass" : "message", "numinlets":2, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 450.0, 100.0, 80.0, 22.0 ], "text" : "clear" } },

			{ "box" : { "id" : "decay-store", "maxclass" : "newobj", "numinlets":2, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 180.0, 55.0, 80.0, 22.0 ], "text" : "f 0.94" } },
			{ "box" : { "id" : "blur-store", "maxclass" : "newobj", "numinlets":2, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 265.0, 55.0, 80.0, 22.0 ], "text" : "f 0.25" } },
			{ "box" : { "id" : "voices-store", "maxclass" : "newobj", "numinlets":2, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 130.0, 55.0, 40.0, 22.0 ], "text" : "zl.reg" } },
			{ "box" : { "id" : "bins-store", "maxclass" : "newobj", "numinlets":2, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 85.0, 55.0, 40.0, 22.0 ], "text" : "zl.reg" } },

			{ "box" : { "id" : "step-1-decay", "maxclass" : "newobj",
				"patching_rect" : [ 30.0, 170.0, 340.0, 22.0 ],
				"text" : "jit.expr @expr \"in[0].p[0] * in[1].p[0]\"",
				"numinlets":2, "numoutlets":1, "outlettype":["jit_matrix"],
				"comment" : "field *= decay" } },

			{ "box" : { "id" : "decay-mat", "maxclass" : "newobj",
				"patching_rect" : [ 380.0, 170.0, 300.0, 22.0 ],
				"text" : "jit.matrix decay_scalar 1 float32 1 1 1",
				"numinlets":1, "numoutlets":1, "outlettype":["jit_matrix"] } },

			{ "box" : { "id" : "step-2-splat", "maxclass" : "newobj", "numinlets":1, "numoutlets":1, "outlettype":["jit_matrix"],
				"patching_rect" : [ 30.0, 210.0, 340.0, 22.0 ],
				"text" : "js voxel_splat.js" } },

			{ "box" : { "id" : "step-3-blur", "maxclass" : "newobj",
				"patching_rect" : [ 30.0, 250.0, 340.0, 22.0 ],
				"text" : "jit.convolve @mode boundless @dim 3 3 3 @kernel 0.02 0.05 0.02 0.05 0.10 0.05 0.02 0.05 0.02 0.05 0.10 0.05 0.10 0.20 0.10 0.05 0.10 0.05 0.02 0.05 0.02 0.05 0.10 0.05 0.02 0.05 0.02",
				"numinlets":1, "numoutlets":1, "outlettype":["jit_matrix"] } },

			{ "box" : { "id" : "step-3-mix", "maxclass" : "newobj",
				"patching_rect" : [ 30.0, 280.0, 340.0, 22.0 ],
				"text" : "jit.op @op * @val 0.85",
				"numinlets":2, "numoutlets":1, "outlettype":["jit_matrix"] } },

			{ "box" : { "id" : "step-4-clip", "maxclass" : "newobj",
				"patching_rect" : [ 30.0, 310.0, 340.0, 22.0 ],
				"text" : "jit.op @op max @val 0.",
				"numinlets":2, "numoutlets":1, "outlettype":["jit_matrix"] } },

			{ "box" : { "id" : "dispatch", "maxclass" : "newobj", "numinlets":1, "numoutlets":4,
				"outlettype":["bang","bang","bang","bang"],
				"patching_rect" : [ 30.0, 100.0, 60.0, 22.0 ], "text" : "t b b b b" } },

			{ "box" : { "id" : "commit-to-a", "maxclass" : "newobj",
				"patching_rect" : [ 30.0, 340.0, 200.0, 22.0 ],
				"text" : "jit.matrix @name fh_field_a",
				"numinlets":1, "numoutlets":1, "outlettype":["jit_matrix"] } },

			{ "box" : { "id" : "cmt-flow", "maxclass" : "comment",
				"patching_rect" : [ 30.0, 380.0, 1020.0, 130.0 ],
				"text" : "Per-frame chain (bang from left inlet):\n  1. decay:  field_a  *  decay_scalar  ->  field_b   (jit.expr)\n  2. splat:  add SDF primitive of shape voice[i].shape at (vx,vy,vz) with amplitude bin[i] * voice[i].r   (js voxel_splat.js)\n  3. blur:   3x3x3 Gaussian convolution                (jit.convolve)\n  4. clip:   max(0)                                    (jit.op)\n  5. commit: write back into fh_field_a for the next frame\nOutput: fh_field_a as jit_matrix, ready for marching_cubes.js.",
				"numinlets":1, "numoutlets":0 } }
		],
		"lines" : [
			{ "patchline" : { "source" : [ "in-bins", 0 ], "destination" : [ "bins-store", 0 ] } },
			{ "patchline" : { "source" : [ "in-voices", 0 ], "destination" : [ "voices-store", 0 ] } },
			{ "patchline" : { "source" : [ "in-decay", 0 ], "destination" : [ "decay-store", 0 ] } },
			{ "patchline" : { "source" : [ "in-blur", 0 ], "destination" : [ "blur-store", 0 ] } },

			{ "patchline" : { "source" : [ "clear-load", 0 ], "destination" : [ "clear-msg", 0 ] } },
			{ "patchline" : { "source" : [ "clear-msg", 0 ], "destination" : [ "field-a", 0 ] } },

			{ "patchline" : { "source" : [ "in-bang", 0 ], "destination" : [ "dispatch", 0 ] } },
			{ "patchline" : { "source" : [ "dispatch", 0 ], "destination" : [ "decay-store", 0 ] } },
			{ "patchline" : { "source" : [ "decay-store", 0 ], "destination" : [ "decay-mat", 0 ] } },
			{ "patchline" : { "source" : [ "dispatch", 1 ], "destination" : [ "field-a", 0 ] } },
			{ "patchline" : { "source" : [ "field-a", 0 ], "destination" : [ "step-1-decay", 0 ] } },
			{ "patchline" : { "source" : [ "decay-mat", 0 ], "destination" : [ "step-1-decay", 1 ] } },
			{ "patchline" : { "source" : [ "step-1-decay", 0 ], "destination" : [ "step-2-splat", 0 ] } },
			{ "patchline" : { "source" : [ "bins-store", 0 ], "destination" : [ "step-2-splat", 0 ] } },
			{ "patchline" : { "source" : [ "voices-store", 0 ], "destination" : [ "step-2-splat", 0 ] } },
			{ "patchline" : { "source" : [ "step-2-splat", 0 ], "destination" : [ "step-3-blur", 0 ] } },
			{ "patchline" : { "source" : [ "step-3-blur", 0 ], "destination" : [ "step-3-mix", 0 ] } },
			{ "patchline" : { "source" : [ "step-3-mix", 0 ], "destination" : [ "step-4-clip", 0 ] } },
			{ "patchline" : { "source" : [ "step-4-clip", 0 ], "destination" : [ "commit-to-a", 0 ] } },
			{ "patchline" : { "source" : [ "commit-to-a", 0 ], "destination" : [ "out-mat", 0 ] } }
		]
	}
}
