{
	"patcher" : 	{
		"fileversion" : 1,
		"appversion" : 		{ "major" : 9, "minor" : 1, "revision" : 0, "architecture" : "x64", "modernui" : 1 },
		"classnamespace" : "box",
		"rect" : [ 80.0, 80.0, 1040.0, 680.0 ],
		"gridonopen" : 1,
		"gridsize" : [ 15.0, 15.0 ],
		"boxes" : [
			{ "box" : { "id" : "in-db", "maxclass" : "inlet", "numinlets":0, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 30.0, 20.0, 30.0, 30.0 ], "comment" : "open <sqlite-path>" } },
			{ "box" : { "id" : "in-heat", "maxclass" : "inlet", "numinlets":0, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 80.0, 20.0, 30.0, 30.0 ], "comment" : "heat 0..1 (SQL match)" } },
			{ "box" : { "id" : "in-energy", "maxclass" : "inlet", "numinlets":0, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 130.0, 20.0, 30.0, 30.0 ], "comment" : "energy 0..1" } },
			{ "box" : { "id" : "in-visc", "maxclass" : "inlet", "numinlets":0, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 180.0, 20.0, 30.0, 30.0 ], "comment" : "viscosity 0..1" } },
			{ "box" : { "id" : "in-pulse", "maxclass" : "inlet", "numinlets":0, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 230.0, 20.0, 30.0, 30.0 ], "comment" : "biological pulse 0..1" } },

			{ "box" : { "id" : "out-texture", "maxclass" : "outlet", "numinlets":1, "numoutlets":0,
				"patching_rect" : [ 30.0, 630.0, 30.0, 30.0 ], "comment" : "Channel A skin / texture  (jit_gl_texture)" } },
			{ "box" : { "id" : "out-velocity", "maxclass" : "outlet", "numinlets":1, "numoutlets":0,
				"patching_rect" : [ 80.0, 630.0, 30.0, 30.0 ], "comment" : "Channel B nerves / velocity (jit_gl_texture)" } },
			{ "box" : { "id" : "out-info", "maxclass" : "outlet", "numinlets":1, "numoutlets":0,
				"patching_rect" : [ 130.0, 630.0, 30.0, 30.0 ], "comment" : "info messages" } },

			{ "box" : { "id" : "title", "maxclass" : "comment",
				"patching_rect" : [ 30.0, 50.0, 900.0, 22.0 ], "fontsize" : 14.0,
				"text" : "fh.archive_pair : parallel 53k/10k fetcher. Channel A \u2192 skin, Channel B \u2192 velocity. Audio parameters drive SQL match.",
				"numinlets":1, "numoutlets":0 } },

			{ "box" : { "id" : "lbl-a", "maxclass" : "comment",
				"patching_rect" : [ 30.0, 80.0, 480.0, 22.0 ], "fontface":1,
				"text" : "Channel A  (primary 53k - texture / skin / density)", "numinlets":1, "numoutlets":0 } },
			{ "box" : { "id" : "lbl-b", "maxclass" : "comment",
				"patching_rect" : [ 530.0, 80.0, 480.0, 22.0 ], "fontface":1,
				"text" : "Channel B  (secondary 10k - velocity / nerves)", "numinlets":1, "numoutlets":0 } },

			{ "box" : { "id" : "fetch-a", "maxclass" : "newobj",
				"numinlets":2, "numoutlets":3, "outlettype":["jit_gl_texture","jit_gl_texture",""],
				"patching_rect" : [ 30.0, 360.0, 440.0, 22.0 ], "text" : "fh.archive_fetcher" } },
			{ "box" : { "id" : "fetch-b", "maxclass" : "newobj",
				"numinlets":2, "numoutlets":3, "outlettype":["jit_gl_texture","jit_gl_texture",""],
				"patching_rect" : [ 530.0, 360.0, 440.0, 22.0 ], "text" : "fh.archive_fetcher" } },

			{ "box" : { "id" : "cfg-a-role", "maxclass" : "message", "numinlets":2, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 30.0, 110.0, 220.0, 22.0 ], "text" : "role texture" } },
			{ "box" : { "id" : "cfg-a-chan", "maxclass" : "message", "numinlets":2, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 30.0, 135.0, 220.0, 22.0 ], "text" : "channel A" } },
			{ "box" : { "id" : "cfg-a-mindur", "maxclass" : "message", "numinlets":2, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 30.0, 160.0, 220.0, 22.0 ], "text" : "min_duration 2.0" } },

			{ "box" : { "id" : "cfg-b-role", "maxclass" : "message", "numinlets":2, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 530.0, 110.0, 220.0, 22.0 ], "text" : "role velocity" } },
			{ "box" : { "id" : "cfg-b-chan", "maxclass" : "message", "numinlets":2, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 530.0, 135.0, 220.0, 22.0 ], "text" : "channel B" } },
			{ "box" : { "id" : "cfg-b-mindur", "maxclass" : "message", "numinlets":2, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 530.0, 160.0, 220.0, 22.0 ], "text" : "min_duration 1.0" } },

			{ "box" : { "id" : "load", "maxclass" : "newobj", "numinlets":1, "numoutlets":1, "outlettype":["bang"],
				"patching_rect" : [ 260.0, 110.0, 80.0, 22.0 ], "text" : "loadbang" } },

			{ "box" : { "id" : "pack-match-a", "maxclass" : "newobj", "numinlets":4, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 30.0, 240.0, 260.0, 22.0 ], "text" : "pak match 0.5 0.5 0.5" } },
			{ "box" : { "id" : "pack-match-b", "maxclass" : "newobj", "numinlets":4, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 530.0, 240.0, 260.0, 22.0 ], "text" : "pak match 0.5 0.5 0.5" } },

			{ "box" : { "id" : "match-rate-a", "maxclass" : "newobj", "numinlets":2, "numoutlets":1, "outlettype":["bang"],
				"patching_rect" : [ 30.0, 270.0, 100.0, 22.0 ], "text" : "speedlim 1500" } },
			{ "box" : { "id" : "match-rate-b", "maxclass" : "newobj", "numinlets":2, "numoutlets":1, "outlettype":["bang"],
				"patching_rect" : [ 530.0, 270.0, 100.0, 22.0 ], "text" : "speedlim 900" } },

			{ "box" : { "id" : "swap-a", "maxclass" : "newobj", "numinlets":2, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 30.0, 300.0, 280.0, 22.0 ], "text" : "r --- fh_match_a" } },
			{ "box" : { "id" : "send-a", "maxclass" : "newobj", "numinlets":1, "numoutlets":0,
				"patching_rect" : [ 30.0, 330.0, 280.0, 22.0 ], "text" : "s --- fh_match_a" } },
			{ "box" : { "id" : "swap-b", "maxclass" : "newobj", "numinlets":2, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 530.0, 300.0, 280.0, 22.0 ], "text" : "r --- fh_match_b" } },
			{ "box" : { "id" : "send-b", "maxclass" : "newobj", "numinlets":1, "numoutlets":0,
				"patching_rect" : [ 530.0, 330.0, 280.0, 22.0 ], "text" : "s --- fh_match_b" } },

			{ "box" : { "id" : "params-store", "maxclass" : "newobj", "numinlets":3, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 30.0, 210.0, 260.0, 22.0 ], "text" : "pak 0.5 0.5 0.5" } },
			{ "box" : { "id" : "params-store-b", "maxclass" : "newobj", "numinlets":3, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 530.0, 210.0, 260.0, 22.0 ], "text" : "pak 0.5 0.5 0.5" } },

			{ "box" : { "id" : "b-invert", "maxclass" : "newobj", "numinlets":1, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 230.0, 180.0, 60.0, 22.0 ], "text" : "!- 1." } },

			{ "box" : { "id" : "unpack-hev", "maxclass" : "newobj", "numinlets":1, "numoutlets":3, "outlettype":["","",""],
				"patching_rect" : [ 30.0, 180.0, 260.0, 22.0 ], "text" : "unpack 0. 0. 0." } },

			{ "box" : { "id" : "cmt-flow", "maxclass" : "comment",
				"patching_rect" : [ 30.0, 400.0, 960.0, 70.0 ],
				"text" : "Every incoming (heat, energy, viscosity) update is debounced (speedlim) then sent to both fetchers as a 'match' SQL order-by. A searches its texture bucket; B searches an inverted-heat bucket (so audio that heats A cools B, producing counter-flow nerves).",
				"numinlets":1, "numoutlets":0 } }
		],
		"lines" : [
			{ "patchline" : { "source" : [ "in-db", 0 ], "destination" : [ "fetch-a", 1 ] } },
			{ "patchline" : { "source" : [ "in-db", 0 ], "destination" : [ "fetch-b", 1 ] } },

			{ "patchline" : { "source" : [ "load", 0 ], "destination" : [ "cfg-a-role", 0 ] } },
			{ "patchline" : { "source" : [ "load", 0 ], "destination" : [ "cfg-a-chan", 0 ] } },
			{ "patchline" : { "source" : [ "load", 0 ], "destination" : [ "cfg-a-mindur", 0 ] } },
			{ "patchline" : { "source" : [ "load", 0 ], "destination" : [ "cfg-b-role", 0 ] } },
			{ "patchline" : { "source" : [ "load", 0 ], "destination" : [ "cfg-b-chan", 0 ] } },
			{ "patchline" : { "source" : [ "load", 0 ], "destination" : [ "cfg-b-mindur", 0 ] } },

			{ "patchline" : { "source" : [ "cfg-a-role", 0 ], "destination" : [ "fetch-a", 0 ] } },
			{ "patchline" : { "source" : [ "cfg-a-chan", 0 ], "destination" : [ "fetch-a", 0 ] } },
			{ "patchline" : { "source" : [ "cfg-a-mindur", 0 ], "destination" : [ "fetch-a", 0 ] } },
			{ "patchline" : { "source" : [ "cfg-b-role", 0 ], "destination" : [ "fetch-b", 0 ] } },
			{ "patchline" : { "source" : [ "cfg-b-chan", 0 ], "destination" : [ "fetch-b", 0 ] } },
			{ "patchline" : { "source" : [ "cfg-b-mindur", 0 ], "destination" : [ "fetch-b", 0 ] } },

			{ "patchline" : { "source" : [ "in-heat", 0 ], "destination" : [ "params-store", 0 ] } },
			{ "patchline" : { "source" : [ "in-energy", 0 ], "destination" : [ "params-store", 1 ] } },
			{ "patchline" : { "source" : [ "in-visc", 0 ], "destination" : [ "params-store", 2 ] } },

			{ "patchline" : { "source" : [ "params-store", 0 ], "destination" : [ "unpack-hev", 0 ] } },
			{ "patchline" : { "source" : [ "unpack-hev", 0 ], "destination" : [ "pack-match-a", 1 ] } },
			{ "patchline" : { "source" : [ "unpack-hev", 1 ], "destination" : [ "pack-match-a", 2 ] } },
			{ "patchline" : { "source" : [ "unpack-hev", 2 ], "destination" : [ "pack-match-a", 3 ] } },

			{ "patchline" : { "source" : [ "unpack-hev", 0 ], "destination" : [ "b-invert", 0 ] } },
			{ "patchline" : { "source" : [ "b-invert", 0 ], "destination" : [ "pack-match-b", 1 ] } },
			{ "patchline" : { "source" : [ "unpack-hev", 1 ], "destination" : [ "pack-match-b", 2 ] } },
			{ "patchline" : { "source" : [ "unpack-hev", 2 ], "destination" : [ "pack-match-b", 3 ] } },

			{ "patchline" : { "source" : [ "pack-match-a", 0 ], "destination" : [ "match-rate-a", 0 ] } },
			{ "patchline" : { "source" : [ "pack-match-b", 0 ], "destination" : [ "match-rate-b", 0 ] } },
			{ "patchline" : { "source" : [ "match-rate-a", 0 ], "destination" : [ "send-a", 0 ] } },
			{ "patchline" : { "source" : [ "match-rate-b", 0 ], "destination" : [ "send-b", 0 ] } },
			{ "patchline" : { "source" : [ "swap-a", 0 ], "destination" : [ "fetch-a", 0 ] } },
			{ "patchline" : { "source" : [ "swap-b", 0 ], "destination" : [ "fetch-b", 0 ] } },

			{ "patchline" : { "source" : [ "in-pulse", 0 ], "destination" : [ "fetch-a", 0 ] } },
			{ "patchline" : { "source" : [ "in-pulse", 0 ], "destination" : [ "fetch-b", 0 ] } },

			{ "patchline" : { "source" : [ "fetch-a", 0 ], "destination" : [ "out-texture", 0 ] } },
			{ "patchline" : { "source" : [ "fetch-b", 0 ], "destination" : [ "out-velocity", 0 ] } },
			{ "patchline" : { "source" : [ "fetch-a", 2 ], "destination" : [ "out-info", 0 ] } },
			{ "patchline" : { "source" : [ "fetch-b", 2 ], "destination" : [ "out-info", 0 ] } }
		]
	}
}
