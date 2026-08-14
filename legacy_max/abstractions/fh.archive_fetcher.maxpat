{
	"patcher" : 	{
		"fileversion" : 1,
		"appversion" : 		{ "major" : 9, "minor" : 1, "revision" : 0, "architecture" : "x64", "modernui" : 1 },
		"classnamespace" : "box",
		"rect" : [ 80.0, 80.0, 900.0, 600.0 ],
		"gridonopen" : 1,
		"gridsize" : [ 15.0, 15.0 ],
		"boxes" : [
			{ "box" : { "id" : "in-trigger", "maxclass" : "inlet", "numinlets" : 0, "numoutlets" : 1, "outlettype" : [ "" ],
				"patching_rect" : [ 30.0, 20.0, 30.0, 30.0 ], "comment" : "trigger (bang = next / float 0..1 = heat-select / int 0..4 = bucket)" } },
			{ "box" : { "id" : "in-db", "maxclass" : "inlet", "numinlets" : 0, "numoutlets" : 1, "outlettype" : [ "" ],
				"patching_rect" : [ 80.0, 20.0, 30.0, 30.0 ], "comment" : "open <path-to-sqlite>" } },
			{ "box" : { "id" : "out-tex", "maxclass" : "outlet", "numinlets" : 1, "numoutlets" : 0,
				"patching_rect" : [ 30.0, 550.0, 30.0, 30.0 ], "comment" : "jit_gl_texture (current archive frame)" } },
			{ "box" : { "id" : "out-tex-ghost", "maxclass" : "outlet", "numinlets" : 1, "numoutlets" : 0,
				"patching_rect" : [ 80.0, 550.0, 30.0, 30.0 ], "comment" : "jit_gl_texture (previous frame - for ghosting)" } },
			{ "box" : { "id" : "out-info", "maxclass" : "outlet", "numinlets" : 1, "numoutlets" : 0,
				"patching_rect" : [ 130.0, 550.0, 30.0, 30.0 ], "comment" : "info messages (path, stats, error)" } },

			{ "box" : { "id" : "title", "maxclass" : "comment",
				"patching_rect" : [ 30.0, 60.0, 760.0, 22.0 ], "fontsize" : 14.0,
				"text" : "fh.archive_fetcher : heat-aware random selection from your 50k SQLite archive",
				"numinlets":1, "numoutlets":0 } },

			{ "box" : { "id" : "js", "maxclass" : "newobj", "numinlets":1, "numoutlets":2, "outlettype":["",""],
				"patching_rect" : [ 30.0, 100.0, 280.0, 22.0 ], "text" : "js archive_fetcher.js" } },

			{ "box" : { "id" : "route", "maxclass" : "newobj", "numinlets":1, "numoutlets":6,
				"outlettype" : [ "", "", "", "", "", "" ],
				"patching_rect" : [ 30.0, 140.0, 480.0, 22.0 ], "text" : "route path resolve prefetched count stats error" } },

			{ "box" : { "id" : "resolver", "maxclass" : "newobj", "numinlets":3, "numoutlets":4,
				"outlettype" : [ "", "", "", "" ],
				"patching_rect" : [ 380.0, 175.0, 280.0, 22.0 ], "text" : "fh.resolver_bridge" } },

			{ "box" : { "id" : "resolve-prepend", "maxclass" : "newobj", "numinlets":1, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 380.0, 145.0, 200.0, 22.0 ], "text" : "prepend resolve" } },

			{ "box" : { "id" : "out-thumb", "maxclass" : "outlet", "numinlets":1, "numoutlets":0,
				"patching_rect" : [ 700.0, 175.0, 30.0, 30.0 ], "comment" : "thumbnail jpg path (instant fallback while clip downloads)" } },

			{ "box" : { "id" : "path-sprintf", "maxclass" : "newobj", "numinlets":2, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 30.0, 175.0, 220.0, 22.0 ], "text" : "sprintf read %s" } },

			{ "box" : { "id" : "mov-a", "maxclass" : "newobj", "numinlets":1, "numoutlets":2, "outlettype":["jit_gl_texture", ""],
				"patching_rect" : [ 30.0, 210.0, 320.0, 22.0 ],
				"text" : "jit.movie @autostart 1 @output_texture 1 @colormode uyvy @dim 1280 720" } },

			{ "box" : { "id" : "mov-b", "maxclass" : "newobj", "numinlets":1, "numoutlets":2, "outlettype":["jit_gl_texture", ""],
				"patching_rect" : [ 370.0, 210.0, 320.0, 22.0 ],
				"text" : "jit.movie @autostart 1 @output_texture 1 @colormode uyvy @dim 1280 720" } },

			{ "box" : { "id" : "toggle-slot", "maxclass" : "newobj", "numinlets":1, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 30.0, 245.0, 80.0, 22.0 ], "text" : "change" } },

			{ "box" : { "id" : "flip", "maxclass" : "newobj", "numinlets":1, "numoutlets":2, "outlettype":["",""],
				"patching_rect" : [ 30.0, 275.0, 80.0, 22.0 ], "text" : "bucket 0 1" } },

			{ "box" : { "id" : "gate-a", "maxclass" : "newobj", "numinlets":2, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 30.0, 310.0, 60.0, 22.0 ], "text" : "gate 2 1" } },
			{ "box" : { "id" : "gate-b", "maxclass" : "newobj", "numinlets":2, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 100.0, 310.0, 60.0, 22.0 ], "text" : "gate 2 1" } },

			{ "box" : { "id" : "rate-msg", "maxclass" : "message", "numinlets":2, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 170.0, 310.0, 140.0, 22.0 ], "text" : "rate $1" } },
			{ "box" : { "id" : "rate-scale", "maxclass" : "newobj", "numinlets":2, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 170.0, 280.0, 140.0, 22.0 ], "text" : "scale 0. 1. 0.25 2.0" } },
			{ "box" : { "id" : "in-pulse", "maxclass" : "inlet", "numinlets":0, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 170.0, 250.0, 30.0, 30.0 ], "comment" : "biological pulse 0..1 - modulates playback rate" } },

			{ "box" : { "id" : "xfade", "maxclass" : "newobj", "numinlets":2, "numoutlets":1, "outlettype":["jit_gl_texture"],
				"patching_rect" : [ 30.0, 420.0, 360.0, 22.0 ], "text" : "jit.gl.slab fh @file shaders/fh.crossfade.jxs" } },
			{ "box" : { "id" : "xfade-line", "maxclass" : "newobj", "numinlets":1, "numoutlets":2, "outlettype":["",""],
				"patching_rect" : [ 30.0, 390.0, 120.0, 22.0 ], "text" : "line 0. 500" } },
			{ "box" : { "id" : "xfade-msg", "maxclass" : "newobj", "numinlets":2, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 160.0, 390.0, 140.0, 22.0 ], "text" : "prepend t" } },

			{ "box" : { "id" : "open-prepend", "maxclass" : "newobj", "numinlets":2, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 80.0, 60.0, 160.0, 22.0 ], "text" : "prepend open" } },

			{ "box" : { "id" : "heat-filter-f", "maxclass" : "newobj", "numinlets":1, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 30.0, 80.0, 80.0, 22.0 ], "text" : "prepend heat" } },
			{ "box" : { "id" : "heat-filter-i", "maxclass" : "newobj", "numinlets":1, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 120.0, 80.0, 80.0, 22.0 ], "text" : "prepend bucket" } },
			{ "box" : { "id" : "heat-route", "maxclass" : "newobj", "numinlets":1, "numoutlets":3,
				"outlettype" : [ "bang", "int", "float" ],
				"patching_rect" : [ 30.0, 50.0, 280.0, 22.0 ], "text" : "route bang int float" } },
			{ "box" : { "id" : "next-msg", "maxclass" : "message", "numinlets":2, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 220.0, 80.0, 60.0, 22.0 ], "text" : "next" } },

			{ "box" : { "id" : "cmt-flow", "maxclass" : "comment",
				"patching_rect" : [ 400.0, 100.0, 500.0, 200.0 ],
				"text" : "Flow:\n  1. open <path>                \u2192 loads SQLite\n  2. heat 0.73                  \u2192 js picks matching bucket\n  3. js outputs 'path <file>'   \u2192 jit.movie (A or B slot alternates)\n  4. slot-change triggers crossfade shader over 500 ms\n  5. rate-scale modulates playback rate with biological pulse\n\nOutlet 0  = foreground texture (active movie)\nOutlet 1  = ghost texture (previous movie, lingering after crossfade)\nOutlet 2  = info  messages",
				"numinlets":1, "numoutlets":0 } }
		],
		"lines" : [
			{ "patchline" : { "source" : [ "in-db", 0 ], "destination" : [ "open-prepend", 0 ] } },
			{ "patchline" : { "source" : [ "open-prepend", 0 ], "destination" : [ "js", 0 ] } },

			{ "patchline" : { "source" : [ "in-trigger", 0 ], "destination" : [ "heat-route", 0 ] } },
			{ "patchline" : { "source" : [ "heat-route", 0 ], "destination" : [ "next-msg", 0 ] } },
			{ "patchline" : { "source" : [ "heat-route", 1 ], "destination" : [ "heat-filter-i", 0 ] } },
			{ "patchline" : { "source" : [ "heat-route", 2 ], "destination" : [ "heat-filter-f", 0 ] } },
			{ "patchline" : { "source" : [ "heat-filter-f", 0 ], "destination" : [ "js", 0 ] } },
			{ "patchline" : { "source" : [ "heat-filter-i", 0 ], "destination" : [ "js", 0 ] } },
			{ "patchline" : { "source" : [ "next-msg", 0 ], "destination" : [ "js", 0 ] } },

			{ "patchline" : { "source" : [ "js", 0 ], "destination" : [ "route", 0 ] } },
			{ "patchline" : { "source" : [ "route", 0 ], "destination" : [ "path-sprintf", 0 ] } },
			{ "patchline" : { "source" : [ "route", 1 ], "destination" : [ "resolve-prepend", 0 ] } },
			{ "patchline" : { "source" : [ "resolve-prepend", 0 ], "destination" : [ "resolver", 0 ] } },
			{ "patchline" : { "source" : [ "resolver", 0 ], "destination" : [ "path-sprintf", 0 ] } },
			{ "patchline" : { "source" : [ "resolver", 2 ], "destination" : [ "out-thumb", 0 ] } },
			{ "patchline" : { "source" : [ "resolver", 3 ], "destination" : [ "out-info", 0 ] } },
			{ "patchline" : { "source" : [ "route", 2 ], "destination" : [ "out-info", 0 ] } },
			{ "patchline" : { "source" : [ "route", 3 ], "destination" : [ "out-info", 0 ] } },
			{ "patchline" : { "source" : [ "route", 4 ], "destination" : [ "out-info", 0 ] } },
			{ "patchline" : { "source" : [ "route", 5 ], "destination" : [ "out-info", 0 ] } },

			{ "patchline" : { "source" : [ "path-sprintf", 0 ], "destination" : [ "flip", 0 ] } },
			{ "patchline" : { "source" : [ "flip", 0 ], "destination" : [ "gate-a", 0 ] } },
			{ "patchline" : { "source" : [ "flip", 1 ], "destination" : [ "gate-b", 0 ] } },
			{ "patchline" : { "source" : [ "gate-a", 0 ], "destination" : [ "mov-a", 0 ] } },
			{ "patchline" : { "source" : [ "gate-b", 0 ], "destination" : [ "mov-b", 0 ] } },

			{ "patchline" : { "source" : [ "flip", 0 ], "destination" : [ "toggle-slot", 0 ] } },
			{ "patchline" : { "source" : [ "toggle-slot", 0 ], "destination" : [ "xfade-line", 0 ] } },
			{ "patchline" : { "source" : [ "xfade-line", 0 ], "destination" : [ "xfade-msg", 0 ] } },
			{ "patchline" : { "source" : [ "xfade-msg", 0 ], "destination" : [ "xfade", 0 ] } },

			{ "patchline" : { "source" : [ "mov-a", 0 ], "destination" : [ "xfade", 0 ] } },
			{ "patchline" : { "source" : [ "mov-b", 0 ], "destination" : [ "xfade", 1 ] } },
			{ "patchline" : { "source" : [ "xfade", 0 ], "destination" : [ "out-tex", 0 ] } },
			{ "patchline" : { "source" : [ "mov-a", 0 ], "destination" : [ "out-tex-ghost", 0 ] } },

			{ "patchline" : { "source" : [ "in-pulse", 0 ], "destination" : [ "rate-scale", 0 ] } },
			{ "patchline" : { "source" : [ "rate-scale", 0 ], "destination" : [ "rate-msg", 0 ] } },
			{ "patchline" : { "source" : [ "rate-msg", 0 ], "destination" : [ "mov-a", 0 ] } },
			{ "patchline" : { "source" : [ "rate-msg", 0 ], "destination" : [ "mov-b", 0 ] } }
		]
	}
}
