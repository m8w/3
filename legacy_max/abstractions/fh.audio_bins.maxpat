{
	"patcher" : 	{
		"fileversion" : 1,
		"appversion" : 		{
			"major" : 9,
			"minor" : 1,
			"revision" : 0,
			"architecture" : "x64",
			"modernui" : 1
		}
,
		"classnamespace" : "box",
		"rect" : [ 80.0, 80.0, 920.0, 620.0 ],
		"gridonopen" : 1,
		"gridsize" : [ 15.0, 15.0 ],
		"boxes" : [
			{
				"box" : {
					"id" : "in-audio",
					"maxclass" : "inlet",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "signal" ],
					"patching_rect" : [ 40.0, 20.0, 30.0, 30.0 ],
					"comment" : "audio in (signal) - from adc~ or sfplay~"
				}
			},
			{
				"box" : {
					"id" : "in-gain",
					"maxclass" : "inlet",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 90.0, 20.0, 30.0, 30.0 ],
					"comment" : "gain (float 0..4)"
				}
			},
			{
				"box" : {
					"id" : "out-bins",
					"maxclass" : "outlet",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 40.0, 560.0, 30.0, 30.0 ],
					"comment" : "list of 8 bin amplitudes (0..1+)"
				}
			},
			{
				"box" : {
					"id" : "out-peak",
					"maxclass" : "outlet",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 120.0, 560.0, 30.0, 30.0 ],
					"comment" : "global peak amplitude (float)"
				}
			},
			{
				"box" : {
					"id" : "out-centroid",
					"maxclass" : "outlet",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 200.0, 560.0, 30.0, 30.0 ],
					"comment" : "spectral centroid 0..1"
				}
			},
			{
				"box" : {
					"id" : "comment-title",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 40.0, 60.0, 580.0, 20.0 ],
					"text" : "fh.audio_bins : 8-bin FFT analyzer for fluid + heat injection",
					"fontsize" : 14.0
				}
			},
			{
				"box" : {
					"id" : "pfft",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 9,
					"outlettype" : [ "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "" ],
					"patching_rect" : [ 40.0, 100.0, 520.0, 22.0 ],
					"text" : "zl.nth 1 2 3 4 5 6 7 8",
					"comment" : "placeholder; see fh.audio_bins_fft sub"
				}
			},
			{
				"box" : {
					"id" : "analyzer",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 3,
					"outlettype" : [ "list", "float", "float" ],
					"patching_rect" : [ 40.0, 150.0, 420.0, 22.0 ],
					"text" : "zsa.bands~ 8 60. 18000. @scale log"
				}
			},
			{
				"box" : {
					"id" : "fallback-analyzer",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 3,
					"outlettype" : [ "list", "list", "list" ],
					"patching_rect" : [ 40.0, 180.0, 420.0, 22.0 ],
					"text" : "analyzer~ @bands 8 @numpeaks 0 @clock 40 ms"
				}
			},
			{
				"box" : {
					"id" : "fcomment",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 40.0, 200.0, 620.0, 20.0 ],
					"text" : "( use whichever analyzer package you have: zsa.bands~ OR analyzer~ - route list to scaler )"
				}
			},
			{
				"box" : {
					"id" : "snap-peak",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "signal" ],
					"patching_rect" : [ 600.0, 100.0, 110.0, 22.0 ],
					"text" : "peakamp~ 40"
				}
			},
			{
				"box" : {
					"id" : "snap",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 600.0, 130.0, 110.0, 22.0 ],
					"text" : "snapshot~ 25"
				}
			},
			{
				"box" : {
					"id" : "clock",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "bang" ],
					"patching_rect" : [ 720.0, 100.0, 60.0, 22.0 ],
					"text" : "metro 25"
				}
			},
			{
				"box" : {
					"id" : "togglerun",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"patching_rect" : [ 720.0, 70.0, 24.0, 24.0 ],
					"parameter_enable" : 0
				}
			},
			{
				"box" : {
					"id" : "bin-scale",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 40.0, 240.0, 500.0, 22.0 ],
					"text" : "vexpr (($f1 * $f9) ^ 0.6) * 2.0"
				}
			},
			{
				"box" : {
					"id" : "bin-attack",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 40.0, 275.0, 500.0, 22.0 ],
					"text" : "slide~list 1 6 @length 8"
				}
			},
			{
				"box" : {
					"id" : "bin-clip",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 40.0, 310.0, 500.0, 22.0 ],
					"text" : "vexpr clip($f1, 0., 4.)"
				}
			},
			{
				"box" : {
					"id" : "gain-store",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 90.0, 60.0, 90.0, 22.0 ],
					"text" : "f 1.5"
				}
			},
			{
				"box" : {
					"id" : "pk-smooth",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 600.0, 170.0, 120.0, 22.0 ],
					"text" : "slide 1 8"
				}
			},
			{
				"box" : {
					"id" : "centroid",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 40.0, 400.0, 260.0, 22.0 ],
					"text" : "expr (0.*$f1 + 0.14*$f2 + 0.28*$f3 + 0.43*$f4 + 0.57*$f5 + 0.71*$f6 + 0.86*$f7 + 1.0*$f8) / (0.001 + $f1+$f2+$f3+$f4+$f5+$f6+$f7+$f8)"
				}
			},
			{
				"box" : {
					"id" : "spread",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 8,
					"outlettype" : [ "", "", "", "", "", "", "", "" ],
					"patching_rect" : [ 40.0, 360.0, 500.0, 22.0 ],
					"text" : "unpack 0. 0. 0. 0. 0. 0. 0. 0."
				}
			},
			{
				"box" : {
					"id" : "repack",
					"maxclass" : "newobj",
					"numinlets" : 8,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 40.0, 500.0, 500.0, 22.0 ],
					"text" : "pak 0. 0. 0. 0. 0. 0. 0. 0."
				}
			},
			{
				"box" : {
					"id" : "help",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 40.0, 430.0, 720.0, 50.0 ],
					"text" : "Outputs: [bins 0..3 = sub, bass, low-mid, mid] [bins 4..7 = upper-mid, presence, brilliance, air]. The shader 'fh.inject' interprets these as 8 spatial injection jets. Pipe list to fluid pipeline 'bins' uniform as two vec4."
				}
			}
		],
		"lines" : [
			{ "patchline" : { "source" : [ "in-audio", 0 ], "destination" : [ "analyzer", 0 ] } },
			{ "patchline" : { "source" : [ "in-audio", 0 ], "destination" : [ "snap-peak", 0 ] } },
			{ "patchline" : { "source" : [ "in-gain", 0 ], "destination" : [ "gain-store", 0 ] } },
			{ "patchline" : { "source" : [ "togglerun", 0 ], "destination" : [ "clock", 0 ] } },
			{ "patchline" : { "source" : [ "clock", 0 ], "destination" : [ "snap", 0 ] } },
			{ "patchline" : { "source" : [ "snap-peak", 0 ], "destination" : [ "snap", 0 ] } },
			{ "patchline" : { "source" : [ "snap", 0 ], "destination" : [ "pk-smooth", 0 ] } },
			{ "patchline" : { "source" : [ "pk-smooth", 0 ], "destination" : [ "out-peak", 0 ] } },
			{ "patchline" : { "source" : [ "analyzer", 0 ], "destination" : [ "bin-scale", 0 ] } },
			{ "patchline" : { "source" : [ "gain-store", 0 ], "destination" : [ "bin-scale", 1 ] } },
			{ "patchline" : { "source" : [ "bin-scale", 0 ], "destination" : [ "bin-attack", 0 ] } },
			{ "patchline" : { "source" : [ "bin-attack", 0 ], "destination" : [ "bin-clip", 0 ] } },
			{ "patchline" : { "source" : [ "bin-clip", 0 ], "destination" : [ "spread", 0 ] } },
			{ "patchline" : { "source" : [ "bin-clip", 0 ], "destination" : [ "out-bins", 0 ] } },
			{ "patchline" : { "source" : [ "spread", 0 ], "destination" : [ "centroid", 0 ] } },
			{ "patchline" : { "source" : [ "spread", 1 ], "destination" : [ "centroid", 1 ] } },
			{ "patchline" : { "source" : [ "spread", 2 ], "destination" : [ "centroid", 2 ] } },
			{ "patchline" : { "source" : [ "spread", 3 ], "destination" : [ "centroid", 3 ] } },
			{ "patchline" : { "source" : [ "spread", 4 ], "destination" : [ "centroid", 4 ] } },
			{ "patchline" : { "source" : [ "spread", 5 ], "destination" : [ "centroid", 5 ] } },
			{ "patchline" : { "source" : [ "spread", 6 ], "destination" : [ "centroid", 6 ] } },
			{ "patchline" : { "source" : [ "spread", 7 ], "destination" : [ "centroid", 7 ] } },
			{ "patchline" : { "source" : [ "centroid", 0 ], "destination" : [ "out-centroid", 0 ] } }
		]
	}
}
