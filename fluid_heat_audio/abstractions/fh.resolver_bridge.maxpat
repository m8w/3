{
	"patcher" : 	{
		"fileversion" : 1,
		"appversion" : 		{ "major" : 9, "minor" : 1, "revision" : 0, "architecture" : "x64", "modernui" : 1 },
		"classnamespace" : "box",
		"rect" : [ 80.0, 80.0, 1000.0, 540.0 ],
		"gridonopen" : 1,
		"gridsize" : [ 15.0, 15.0 ],
		"boxes" : [
			{ "box" : { "id" : "in-resolve", "maxclass" : "inlet", "numinlets":0, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 30.0, 20.0, 30.0, 30.0 ], "comment" : "resolve <url> | prefetch <url>... | thumb <url> | status" } },
			{ "box" : { "id" : "in-host", "maxclass" : "inlet", "numinlets":0, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 80.0, 20.0, 30.0, 30.0 ], "comment" : "host (default 127.0.0.1)" } },
			{ "box" : { "id" : "in-ports", "maxclass" : "inlet", "numinlets":0, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 130.0, 20.0, 30.0, 30.0 ], "comment" : "send-port recv-port (default 7401 7402)" } },

			{ "box" : { "id" : "out-path", "maxclass" : "outlet", "numinlets":1, "numoutlets":0,
				"patching_rect" : [ 30.0, 490.0, 30.0, 30.0 ], "comment" : "local cache path (str)" } },
			{ "box" : { "id" : "out-stream", "maxclass" : "outlet", "numinlets":1, "numoutlets":0,
				"patching_rect" : [ 80.0, 490.0, 30.0, 30.0 ], "comment" : "direct stream URL fallback" } },
			{ "box" : { "id" : "out-thumb", "maxclass" : "outlet", "numinlets":1, "numoutlets":0,
				"patching_rect" : [ 130.0, 490.0, 30.0, 30.0 ], "comment" : "thumbnail jpg path (instant)" } },
			{ "box" : { "id" : "out-status", "maxclass" : "outlet", "numinlets":1, "numoutlets":0,
				"patching_rect" : [ 180.0, 490.0, 30.0, 30.0 ], "comment" : "status / error messages" } },

			{ "box" : { "id" : "title", "maxclass" : "comment",
				"patching_rect" : [ 30.0, 60.0, 880.0, 22.0 ], "fontsize" : 14.0,
				"text" : "fh.resolver_bridge : OSC bridge to scripts/archive_resolver.py (yt-dlp + LRU cache)",
				"numinlets":1, "numoutlets":0 } },

			{ "box" : { "id" : "in-route", "maxclass" : "newobj", "numinlets":1, "numoutlets":5,
				"outlettype":["","","","",""],
				"patching_rect" : [ 30.0, 100.0, 480.0, 22.0 ],
				"text" : "route resolve resolve_stream prefetch thumb status" } },

			{ "box" : { "id" : "pre-resolve", "maxclass" : "newobj", "numinlets":1, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 30.0, 130.0, 160.0, 22.0 ], "text" : "prepend /resolve" } },
			{ "box" : { "id" : "pre-resolve-s", "maxclass" : "newobj", "numinlets":1, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 200.0, 130.0, 160.0, 22.0 ], "text" : "prepend /resolve_stream" } },
			{ "box" : { "id" : "pre-prefetch", "maxclass" : "newobj", "numinlets":1, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 370.0, 130.0, 120.0, 22.0 ], "text" : "prepend /prefetch" } },
			{ "box" : { "id" : "pre-thumb", "maxclass" : "newobj", "numinlets":1, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 500.0, 130.0, 120.0, 22.0 ], "text" : "prepend /thumb" } },
			{ "box" : { "id" : "pre-status", "maxclass" : "message", "numinlets":2, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 630.0, 130.0, 80.0, 22.0 ], "text" : "/status" } },

			{ "box" : { "id" : "udp-send", "maxclass" : "newobj", "numinlets":1, "numoutlets":0,
				"patching_rect" : [ 30.0, 170.0, 280.0, 22.0 ], "text" : "udpsend 127.0.0.1 7401" } },

			{ "box" : { "id" : "udp-recv", "maxclass" : "newobj", "numinlets":0, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 30.0, 220.0, 200.0, 22.0 ], "text" : "udpreceive 7402" } },

			{ "box" : { "id" : "out-route", "maxclass" : "newobj", "numinlets":1, "numoutlets":7,
				"outlettype":["","","","","","",""],
				"patching_rect" : [ 30.0, 250.0, 720.0, 22.0 ],
				"text" : "route /path /stream /thumb_path /status /error /prefetched /size_limit_set" } },

			{ "box" : { "id" : "path-fmt", "maxclass" : "newobj", "numinlets":2, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 30.0, 285.0, 220.0, 22.0 ], "text" : "sprintf %s" } },

			{ "box" : { "id" : "stream-fmt", "maxclass" : "newobj", "numinlets":2, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 260.0, 285.0, 220.0, 22.0 ], "text" : "sprintf %s" } },

			{ "box" : { "id" : "thumb-fmt", "maxclass" : "newobj", "numinlets":2, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 490.0, 285.0, 220.0, 22.0 ], "text" : "sprintf %s" } },

			{ "box" : { "id" : "host-msg", "maxclass" : "newobj", "numinlets":2, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 80.0, 60.0, 200.0, 22.0 ], "text" : "prepend host" } },
			{ "box" : { "id" : "ports-msg", "maxclass" : "newobj", "numinlets":2, "numoutlets":1, "outlettype":[""],
				"patching_rect" : [ 290.0, 60.0, 200.0, 22.0 ], "text" : "prepend port" } },

			{ "box" : { "id" : "cmt-flow", "maxclass" : "comment",
				"patching_rect" : [ 30.0, 320.0, 920.0, 140.0 ],
				"text" : "Wiring:\n  in-resolve <- resolve <url>            -> /resolve over UDP -> resolver downloads + caches -> /path <local> back\n  in-resolve <- prefetch <urls>          -> /prefetch                  -> background warmup\n  in-resolve <- thumb <url>              -> /thumb                     -> /thumb_path back instantly\n  in-resolve <- status                   -> /status                    -> /status n bytes jobs back\n\nThe resolver answers on UDP 7402 (default). Loopback only - no traffic leaves the machine\nexcept yt-dlp's calls to YouTube. Configure quota and cache size with --cache-gb.",
				"numinlets":1, "numoutlets":0 } }
		],
		"lines" : [
			{ "patchline" : { "source" : [ "in-resolve", 0 ], "destination" : [ "in-route", 0 ] } },
			{ "patchline" : { "source" : [ "in-route", 0 ], "destination" : [ "pre-resolve", 0 ] } },
			{ "patchline" : { "source" : [ "in-route", 1 ], "destination" : [ "pre-resolve-s", 0 ] } },
			{ "patchline" : { "source" : [ "in-route", 2 ], "destination" : [ "pre-prefetch", 0 ] } },
			{ "patchline" : { "source" : [ "in-route", 3 ], "destination" : [ "pre-thumb", 0 ] } },
			{ "patchline" : { "source" : [ "in-route", 4 ], "destination" : [ "pre-status", 0 ] } },

			{ "patchline" : { "source" : [ "pre-resolve", 0 ], "destination" : [ "udp-send", 0 ] } },
			{ "patchline" : { "source" : [ "pre-resolve-s", 0 ], "destination" : [ "udp-send", 0 ] } },
			{ "patchline" : { "source" : [ "pre-prefetch", 0 ], "destination" : [ "udp-send", 0 ] } },
			{ "patchline" : { "source" : [ "pre-thumb", 0 ], "destination" : [ "udp-send", 0 ] } },
			{ "patchline" : { "source" : [ "pre-status", 0 ], "destination" : [ "udp-send", 0 ] } },

			{ "patchline" : { "source" : [ "in-host", 0 ], "destination" : [ "host-msg", 0 ] } },
			{ "patchline" : { "source" : [ "host-msg", 0 ], "destination" : [ "udp-send", 0 ] } },
			{ "patchline" : { "source" : [ "in-ports", 0 ], "destination" : [ "ports-msg", 0 ] } },
			{ "patchline" : { "source" : [ "ports-msg", 0 ], "destination" : [ "udp-send", 0 ] } },

			{ "patchline" : { "source" : [ "udp-recv", 0 ], "destination" : [ "out-route", 0 ] } },
			{ "patchline" : { "source" : [ "out-route", 0 ], "destination" : [ "path-fmt", 0 ] } },
			{ "patchline" : { "source" : [ "path-fmt", 0 ], "destination" : [ "out-path", 0 ] } },
			{ "patchline" : { "source" : [ "out-route", 1 ], "destination" : [ "stream-fmt", 0 ] } },
			{ "patchline" : { "source" : [ "stream-fmt", 0 ], "destination" : [ "out-stream", 0 ] } },
			{ "patchline" : { "source" : [ "out-route", 2 ], "destination" : [ "thumb-fmt", 0 ] } },
			{ "patchline" : { "source" : [ "thumb-fmt", 0 ], "destination" : [ "out-thumb", 0 ] } },
			{ "patchline" : { "source" : [ "out-route", 3 ], "destination" : [ "out-status", 0 ] } },
			{ "patchline" : { "source" : [ "out-route", 4 ], "destination" : [ "out-status", 0 ] } },
			{ "patchline" : { "source" : [ "out-route", 5 ], "destination" : [ "out-status", 0 ] } },
			{ "patchline" : { "source" : [ "out-route", 6 ], "destination" : [ "out-status", 0 ] } }
		]
	}
}
