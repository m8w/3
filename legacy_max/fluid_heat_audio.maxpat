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
		"rect" : [ 80.0, 80.0, 1440.0, 900.0 ],
		"gridonopen" : 1,
		"gridsize" : [ 15.0, 15.0 ],
		"bglocked" : 0,
		"openinpresentation" : 0,
		"default_fontsize" : 12.0,
		"default_fontface" : 0,
		"default_fontname" : "Arial",
		"gridsnaponopen" : 1,
		"objectsnaponopen" : 1,
		"statusbarvisible" : 2,
		"toolbarvisible" : 1,
		"boxes" : [

			{ "box" : { "id" : "title", "maxclass" : "comment", "patching_rect" : [ 30.0, 10.0, 1100.0, 22.0 ],
				"text" : "fluid_heat_audio.maxpat  --  3D fluid + heat coupled solver driven by sound.  Stable Fluids (Stam) + Boussinesq + Blackbody LUT + Volumetric raymarch.",
				"fontsize" : 14.0, "numinlets":1, "numoutlets":0 } },

			{ "box" : { "id" : "lbl-audio", "maxclass" : "comment", "patching_rect" : [ 30.0, 50.0, 220.0, 20.0 ],
				"text" : "AUDIO  adc~ / sfplay~  \u2192  fh.audio_bins", "fontface":1, "numinlets":1, "numoutlets":0 } },

			{ "box" : { "id" : "adc", "maxclass" : "newobj",
				"patching_rect" : [ 30.0, 80.0, 80.0, 22.0 ],
				"text" : "adc~ 1 2",
				"numinlets":2, "numoutlets":3, "outlettype":["signal","signal",""] } },

			{ "box" : { "id" : "mix", "maxclass" : "newobj",
				"patching_rect" : [ 30.0, 110.0, 60.0, 22.0 ],
				"text" : "+~",
				"numinlets":2, "numoutlets":1, "outlettype":["signal"] } },

			{ "box" : { "id" : "scale-in", "maxclass" : "newobj",
				"patching_rect" : [ 30.0, 140.0, 60.0, 22.0 ],
				"text" : "*~ 0.5",
				"numinlets":2, "numoutlets":1, "outlettype":["signal"] } },

			{ "box" : { "id" : "dspstate", "maxclass" : "newobj",
				"patching_rect" : [ 140.0, 80.0, 80.0, 22.0 ],
				"text" : "dac~",
				"numinlets":2, "numoutlets":0 } },

			{ "box" : { "id" : "gain-msg", "maxclass" : "flonum",
				"patching_rect" : [ 230.0, 80.0, 60.0, 22.0 ],
				"minimum" : 0.0, "maximum" : 4.0,
				"numinlets":1, "numoutlets":2, "outlettype":["","bang"] } },

			{ "box" : { "id" : "gain-lbl", "maxclass" : "comment",
				"patching_rect" : [ 295.0, 82.0, 100.0, 20.0 ],
				"text" : "audio gain", "numinlets":1, "numoutlets":0 } },

			{ "box" : { "id" : "audio-bins", "maxclass" : "newobj",
				"patching_rect" : [ 30.0, 180.0, 260.0, 22.0 ],
				"text" : "fh.audio_bins",
				"numinlets":2, "numoutlets":3, "outlettype":["","",""] } },

			{ "box" : { "id" : "organic-mod", "maxclass" : "newobj",
				"patching_rect" : [ 30.0, 208.0, 260.0, 22.0 ],
				"text" : "fh.organic_mod",
				"numinlets":3, "numoutlets":1, "outlettype":[""] } },

			{ "box" : { "id" : "audio-tap", "maxclass" : "message",
				"patching_rect" : [ 300.0, 180.0, 180.0, 22.0 ],
				"text" : "(out 1: 8 bins list - out 2: peak - out 3: centroid)",
				"numinlets":2, "numoutlets":1, "outlettype":[""] } },

			{ "box" : { "id" : "bins-split", "maxclass" : "newobj",
				"patching_rect" : [ 30.0, 215.0, 260.0, 22.0 ],
				"text" : "zl.slice 4",
				"numinlets":2, "numoutlets":2, "outlettype":["",""] } },

			{ "box" : { "id" : "bins-lo-pk", "maxclass" : "newobj",
				"patching_rect" : [ 30.0, 245.0, 120.0, 22.0 ],
				"text" : "pak bins 0. 0. 0. 0.",
				"numinlets":5, "numoutlets":1, "outlettype":[""] } },

			{ "box" : { "id" : "bins-hi-pk", "maxclass" : "newobj",
				"patching_rect" : [ 160.0, 245.0, 130.0, 22.0 ],
				"text" : "pak bins 0. 0. 0. 0.",
				"numinlets":5, "numoutlets":1, "outlettype":[""] } },

			{ "box" : { "id" : "bins-lo-unpack", "maxclass" : "newobj",
				"patching_rect" : [ 30.0, 270.0, 120.0, 22.0 ],
				"text" : "unpack 0. 0. 0. 0.",
				"numinlets":1, "numoutlets":4, "outlettype":["","","",""] } },

			{ "box" : { "id" : "bins-hi-unpack", "maxclass" : "newobj",
				"patching_rect" : [ 160.0, 270.0, 120.0, 22.0 ],
				"text" : "unpack 0. 0. 0. 0.",
				"numinlets":1, "numoutlets":4, "outlettype":["","","",""] } },

			{ "box" : { "id" : "lbl-world", "maxclass" : "comment", "patching_rect" : [ 520.0, 50.0, 300.0, 20.0 ],
				"text" : "RENDER CONTEXT  jit.world + simulation grid", "fontface":1, "numinlets":1, "numoutlets":0 } },

			{ "box" : { "id" : "world", "maxclass" : "newobj",
				"patching_rect" : [ 520.0, 80.0, 330.0, 22.0 ],
				"text" : "jit.world fh @fsaa 1 @floating 1 @size 1280 720 @fps 60",
				"numinlets":1, "numoutlets":1, "outlettype":["bang"] } },

			{ "box" : { "id" : "world-start", "maxclass" : "toggle",
				"patching_rect" : [ 500.0, 80.0, 20.0, 20.0 ],
				"numinlets":1, "numoutlets":1, "outlettype":["int"] } },

			{ "box" : { "id" : "world-erase", "maxclass" : "message",
				"patching_rect" : [ 860.0, 80.0, 60.0, 22.0 ],
				"text" : "erase",
				"numinlets":2, "numoutlets":1, "outlettype":[""] } },

			{ "box" : { "id" : "dim-msg", "maxclass" : "message",
				"patching_rect" : [ 520.0, 110.0, 140.0, 22.0 ],
				"text" : "dim 512 288",
				"numinlets":2, "numoutlets":1, "outlettype":[""] } },

			{ "box" : { "id" : "dim-init", "maxclass" : "newobj",
				"patching_rect" : [ 670.0, 110.0, 100.0, 22.0 ],
				"text" : "loadbang",
				"numinlets":1, "numoutlets":1, "outlettype":["bang"] } },

			{ "box" : { "id" : "lbl-tex", "maxclass" : "comment", "patching_rect" : [ 30.0, 320.0, 500.0, 20.0 ],
				"text" : "STATE TEXTURES  ping-pong RGBA32f : R=u, G=v, B=T, A=D", "fontface":1, "numinlets":1, "numoutlets":0 } },

			{ "box" : { "id" : "tex-state-a", "maxclass" : "newobj",
				"patching_rect" : [ 30.0, 350.0, 340.0, 22.0 ],
				"text" : "jit.gl.texture fh @name fh_state_a @type float32 @dim 512 288",
				"numinlets":1, "numoutlets":1, "outlettype":["jit_gl_texture"] } },

			{ "box" : { "id" : "tex-state-b", "maxclass" : "newobj",
				"patching_rect" : [ 30.0, 380.0, 340.0, 22.0 ],
				"text" : "jit.gl.texture fh @name fh_state_b @type float32 @dim 512 288",
				"numinlets":1, "numoutlets":1, "outlettype":["jit_gl_texture"] } },

			{ "box" : { "id" : "tex-div", "maxclass" : "newobj",
				"patching_rect" : [ 30.0, 410.0, 340.0, 22.0 ],
				"text" : "jit.gl.texture fh @name fh_div @type float32 @dim 512 288",
				"numinlets":1, "numoutlets":1, "outlettype":["jit_gl_texture"] } },

			{ "box" : { "id" : "tex-pres-a", "maxclass" : "newobj",
				"patching_rect" : [ 30.0, 440.0, 340.0, 22.0 ],
				"text" : "jit.gl.texture fh @name fh_pres_a @type float32 @dim 512 288",
				"numinlets":1, "numoutlets":1, "outlettype":["jit_gl_texture"] } },

			{ "box" : { "id" : "tex-pres-b", "maxclass" : "newobj",
				"patching_rect" : [ 30.0, 470.0, 340.0, 22.0 ],
				"text" : "jit.gl.texture fh @name fh_pres_b @type float32 @dim 512 288",
				"numinlets":1, "numoutlets":1, "outlettype":["jit_gl_texture"] } },

			{ "box" : { "id" : "tex-color", "maxclass" : "newobj",
				"patching_rect" : [ 30.0, 500.0, 340.0, 22.0 ],
				"text" : "jit.gl.texture fh @name fh_color @dim 1280 720",
				"numinlets":1, "numoutlets":1, "outlettype":["jit_gl_texture"] } },

			{ "box" : { "id" : "tex-final", "maxclass" : "newobj",
				"patching_rect" : [ 30.0, 530.0, 340.0, 22.0 ],
				"text" : "jit.gl.texture fh @name fh_final @dim 1280 720",
				"numinlets":1, "numoutlets":1, "outlettype":["jit_gl_texture"] } },

			{ "box" : { "id" : "lbl-solver", "maxclass" : "comment", "patching_rect" : [ 560.0, 180.0, 500.0, 20.0 ],
				"text" : "SOLVER PIPELINE  (one frame = inject \u2192 buoyancy \u2192 advect \u2192 diffuse \u2192 vort \u2192 divergence \u2192 jacobi\u00d720 \u2192 gradient)", "fontface":1, "numinlets":1, "numoutlets":0 } },

			{ "box" : { "id" : "slab-inject", "maxclass" : "newobj",
				"patching_rect" : [ 560.0, 220.0, 500.0, 22.0 ],
				"text" : "jit.gl.slab fh @file shaders/fh.inject.jxs @out_name fh_state_b",
				"numinlets":2, "numoutlets":1, "outlettype":["jit_gl_texture"] } },

			{ "box" : { "id" : "slab-buoyancy", "maxclass" : "newobj",
				"patching_rect" : [ 560.0, 250.0, 500.0, 22.0 ],
				"text" : "jit.gl.slab fh @file shaders/fh.buoyancy.jxs",
				"numinlets":1, "numoutlets":1, "outlettype":["jit_gl_texture"] } },

			{ "box" : { "id" : "slab-advect", "maxclass" : "newobj",
				"patching_rect" : [ 560.0, 280.0, 500.0, 22.0 ],
				"text" : "jit.gl.slab fh @file shaders/fh.advect.jxs",
				"numinlets":2, "numoutlets":1, "outlettype":["jit_gl_texture"] } },

			{ "box" : { "id" : "slab-diffuse", "maxclass" : "newobj",
				"patching_rect" : [ 560.0, 310.0, 500.0, 22.0 ],
				"text" : "jit.gl.slab fh @file shaders/fh.diffuse.jxs",
				"numinlets":1, "numoutlets":1, "outlettype":["jit_gl_texture"] } },

			{ "box" : { "id" : "slab-vort", "maxclass" : "newobj",
				"patching_rect" : [ 560.0, 340.0, 500.0, 22.0 ],
				"text" : "jit.gl.slab fh @file shaders/fh.vorticity.jxs",
				"numinlets":1, "numoutlets":1, "outlettype":["jit_gl_texture"] } },

			{ "box" : { "id" : "slab-div", "maxclass" : "newobj",
				"patching_rect" : [ 560.0, 370.0, 500.0, 22.0 ],
				"text" : "jit.gl.slab fh @file shaders/fh.divergence.jxs @out_name fh_div",
				"numinlets":1, "numoutlets":1, "outlettype":["jit_gl_texture"] } },

			{ "box" : { "id" : "slab-jacobi", "maxclass" : "newobj",
				"patching_rect" : [ 560.0, 400.0, 500.0, 22.0 ],
				"text" : "poly~ fh.jacobi_iter 1 @iterations 20",
				"numinlets":3, "numoutlets":1, "outlettype":["jit_gl_texture"] } },

			{ "box" : { "id" : "slab-jacobi-real", "maxclass" : "newobj",
				"patching_rect" : [ 560.0, 430.0, 500.0, 22.0 ],
				"text" : "jit.gl.slab fh @file shaders/fh.jacobi.jxs",
				"numinlets":2, "numoutlets":1, "outlettype":["jit_gl_texture"] } },

			{ "box" : { "id" : "slab-jacobi-iter", "maxclass" : "newobj",
				"patching_rect" : [ 560.0, 460.0, 500.0, 22.0 ],
				"text" : "uzi 20",
				"numinlets":3, "numoutlets":3, "outlettype":["int","int","bang"] } },

			{ "box" : { "id" : "slab-gradient", "maxclass" : "newobj",
				"patching_rect" : [ 560.0, 490.0, 500.0, 22.0 ],
				"text" : "jit.gl.slab fh @file shaders/fh.gradient.jxs",
				"numinlets":2, "numoutlets":1, "outlettype":["jit_gl_texture"] } },

			{ "box" : { "id" : "lbl-render", "maxclass" : "comment", "patching_rect" : [ 560.0, 530.0, 500.0, 20.0 ],
				"text" : "RENDER  heat LUT + asemic + raymarch + display quad", "fontface":1, "numinlets":1, "numoutlets":0 } },

			{ "box" : { "id" : "tex-asemic", "maxclass" : "newobj",
				"patching_rect" : [ 560.0, 560.0, 500.0, 22.0 ],
				"text" : "jit.gl.texture fh @name fh_asemic @file assets/asemic.png @filter linear linear",
				"numinlets":1, "numoutlets":1, "outlettype":["jit_gl_texture"] } },

			{ "box" : { "id" : "slab-blackbody", "maxclass" : "newobj",
				"patching_rect" : [ 560.0, 590.0, 500.0, 22.0 ],
				"text" : "jit.gl.slab fh @file shaders/fh.blackbody.jxs @out_name fh_color",
				"numinlets":2, "numoutlets":1, "outlettype":["jit_gl_texture"] } },

			{ "box" : { "id" : "slab-volume", "maxclass" : "newobj",
				"patching_rect" : [ 560.0, 620.0, 500.0, 22.0 ],
				"text" : "jit.gl.slab fh @file shaders/fh.volume.jxs @out_name fh_final",
				"numinlets":1, "numoutlets":1, "outlettype":["jit_gl_texture"] } },

			{ "box" : { "id" : "lbl-organic", "maxclass" : "comment",
				"patching_rect" : [ 560.0, 700.0, 500.0, 20.0 ],
				"text" : "LIVING EXTENSIONS  (drop-in alternates - see docs/ORGANIC.md)",
				"fontface":1, "numinlets":1, "numoutlets":0 } },

			{ "box" : { "id" : "slab-video-displace", "maxclass" : "newobj",
				"patching_rect" : [ 560.0, 730.0, 500.0, 22.0 ],
				"text" : "jit.gl.slab fh @file shaders/fh.video_displace.jxs",
				"numinlets":2, "numoutlets":1, "outlettype":["jit_gl_texture"] } },

			{ "box" : { "id" : "slab-viscosity", "maxclass" : "newobj",
				"patching_rect" : [ 560.0, 760.0, 500.0, 22.0 ],
				"text" : "jit.gl.slab fh @file shaders/fh.viscosity.jxs",
				"numinlets":1, "numoutlets":1, "outlettype":["jit_gl_texture"] } },

			{ "box" : { "id" : "slab-reaction", "maxclass" : "newobj",
				"patching_rect" : [ 560.0, 790.0, 500.0, 22.0 ],
				"text" : "jit.gl.slab fh @file shaders/fh.reaction.jxs @out_name fh_rd",
				"numinlets":2, "numoutlets":1, "outlettype":["jit_gl_texture"] } },

			{ "box" : { "id" : "slab-organic-lut", "maxclass" : "newobj",
				"patching_rect" : [ 560.0, 820.0, 500.0, 22.0 ],
				"text" : "jit.gl.slab fh @file shaders/fh.organic_lut.jxs @out_name fh_color",
				"numinlets":3, "numoutlets":1, "outlettype":["jit_gl_texture"] } },

			{ "box" : { "id" : "tex-rd", "maxclass" : "newobj",
				"patching_rect" : [ 30.0, 560.0, 340.0, 22.0 ],
				"text" : "jit.gl.texture fh @name fh_rd @type float32 @dim 512 288",
				"numinlets":1, "numoutlets":1, "outlettype":["jit_gl_texture"] } },

			{ "box" : { "id" : "archive", "maxclass" : "newobj",
				"patching_rect" : [ 30.0, 740.0, 340.0, 22.0 ],
				"text" : "fh.archive_fetcher",
				"numinlets":2, "numoutlets":3, "outlettype":["jit_gl_texture","jit_gl_texture",""] } },

			{ "box" : { "id" : "archive-db", "maxclass" : "message",
				"patching_rect" : [ 30.0, 710.0, 340.0, 22.0 ],
				"text" : "../videos.sqlite",
				"numinlets":2, "numoutlets":1, "outlettype":[""] } },

			{ "box" : { "id" : "archive-note", "maxclass" : "comment",
				"patching_rect" : [ 30.0, 770.0, 500.0, 40.0 ],
				"text" : "Set the db path, click the message, then peak-amp \u2192 archive:heat triggers clip-of-the-moment. Output texture goes into slab-video-displace (tex1) and/or slab-organic-lut (tex1).",
				"numinlets":1, "numoutlets":0 } },

			{ "box" : { "id" : "display-quad", "maxclass" : "newobj",
				"patching_rect" : [ 560.0, 660.0, 500.0, 22.0 ],
				"text" : "jit.gl.videoplane fh @scale 1.778 1 1 @texture fh_final @blend_enable 0",
				"numinlets":1, "numoutlets":1, "outlettype":[""] } },

			{ "box" : { "id" : "lbl-params", "maxclass" : "comment", "patching_rect" : [ 1090.0, 50.0, 300.0, 20.0 ],
				"text" : "PARAMETERS  (sent as messages to slabs)", "fontface":1, "numinlets":1, "numoutlets":0 } },

			{ "box" : { "id" : "p-heat-gain", "maxclass" : "live.slider",
				"patching_rect" : [ 1090.0, 80.0, 40.0, 120.0 ],
				"parameter_enable":1, "numinlets":1, "numoutlets":1, "outlettype":["float"],
				"saved_attribute_attributes" : { "valueof" : { "parameter_longname" : "heat_gain", "parameter_shortname" : "heat", "parameter_initial" : [ 1.0 ], "parameter_range" : [ 0.0, 4.0 ] } } } },

			{ "box" : { "id" : "p-alpha", "maxclass" : "live.slider",
				"patching_rect" : [ 1140.0, 80.0, 40.0, 120.0 ],
				"parameter_enable":1, "numinlets":1, "numoutlets":1, "outlettype":["float"],
				"saved_attribute_attributes" : { "valueof" : { "parameter_longname" : "alpha", "parameter_shortname" : "alpha", "parameter_initial" : [ 1.8 ], "parameter_range" : [ 0.0, 5.0 ] } } } },

			{ "box" : { "id" : "p-beta", "maxclass" : "live.slider",
				"patching_rect" : [ 1190.0, 80.0, 40.0, 120.0 ],
				"parameter_enable":1, "numinlets":1, "numoutlets":1, "outlettype":["float"],
				"saved_attribute_attributes" : { "valueof" : { "parameter_longname" : "beta", "parameter_shortname" : "beta", "parameter_initial" : [ 0.25 ], "parameter_range" : [ 0.0, 2.0 ] } } } },

			{ "box" : { "id" : "p-vort", "maxclass" : "live.slider",
				"patching_rect" : [ 1240.0, 80.0, 40.0, 120.0 ],
				"parameter_enable":1, "numinlets":1, "numoutlets":1, "outlettype":["float"],
				"saved_attribute_attributes" : { "valueof" : { "parameter_longname" : "epsilon", "parameter_shortname" : "vort", "parameter_initial" : [ 0.35 ], "parameter_range" : [ 0.0, 2.0 ] } } } },

			{ "box" : { "id" : "p-asemic", "maxclass" : "live.slider",
				"patching_rect" : [ 1290.0, 80.0, 40.0, 120.0 ],
				"parameter_enable":1, "numinlets":1, "numoutlets":1, "outlettype":["float"],
				"saved_attribute_attributes" : { "valueof" : { "parameter_longname" : "asemic_mix", "parameter_shortname" : "ase", "parameter_initial" : [ 0.65 ], "parameter_range" : [ 0.0, 1.0 ] } } } },

			{ "box" : { "id" : "p-exposure", "maxclass" : "live.slider",
				"patching_rect" : [ 1340.0, 80.0, 40.0, 120.0 ],
				"parameter_enable":1, "numinlets":1, "numoutlets":1, "outlettype":["float"],
				"saved_attribute_attributes" : { "valueof" : { "parameter_longname" : "exposure", "parameter_shortname" : "exp", "parameter_initial" : [ 1.1 ], "parameter_range" : [ 0.1 , 3.0 ] } } } },

			{ "box" : { "id" : "prefix-heat", "maxclass" : "newobj",
				"patching_rect" : [ 1090.0, 210.0, 160.0, 22.0 ],
				"text" : "prepend heat_gain",
				"numinlets":2, "numoutlets":1, "outlettype":[""] } },

			{ "box" : { "id" : "prefix-alpha", "maxclass" : "newobj",
				"patching_rect" : [ 1090.0, 235.0, 160.0, 22.0 ],
				"text" : "prepend alpha",
				"numinlets":2, "numoutlets":1, "outlettype":[""] } },

			{ "box" : { "id" : "prefix-beta", "maxclass" : "newobj",
				"patching_rect" : [ 1090.0, 260.0, 160.0, 22.0 ],
				"text" : "prepend beta",
				"numinlets":2, "numoutlets":1, "outlettype":[""] } },

			{ "box" : { "id" : "prefix-vort", "maxclass" : "newobj",
				"patching_rect" : [ 1090.0, 285.0, 160.0, 22.0 ],
				"text" : "prepend epsilon",
				"numinlets":2, "numoutlets":1, "outlettype":[""] } },

			{ "box" : { "id" : "prefix-asemic", "maxclass" : "newobj",
				"patching_rect" : [ 1090.0, 310.0, 160.0, 22.0 ],
				"text" : "prepend asemic_mix",
				"numinlets":2, "numoutlets":1, "outlettype":[""] } },

			{ "box" : { "id" : "prefix-exposure", "maxclass" : "newobj",
				"patching_rect" : [ 1090.0, 335.0, 160.0, 22.0 ],
				"text" : "prepend exposure",
				"numinlets":2, "numoutlets":1, "outlettype":[""] } },

			{ "box" : { "id" : "lbl-frame", "maxclass" : "comment", "patching_rect" : [ 1090.0, 380.0, 300.0, 20.0 ],
				"text" : "FRAME CLOCK drives the solver on each bang", "fontface":1, "numinlets":1, "numoutlets":0 } },

			{ "box" : { "id" : "metro-frame", "maxclass" : "newobj",
				"patching_rect" : [ 1090.0, 410.0, 80.0, 22.0 ],
				"text" : "qmetro 16",
				"numinlets":2, "numoutlets":1, "outlettype":["bang"] } },

			{ "box" : { "id" : "frame-tgl", "maxclass" : "toggle",
				"patching_rect" : [ 1060.0, 410.0, 24.0, 24.0 ],
				"numinlets":1, "numoutlets":1, "outlettype":["int"] } },

			{ "box" : { "id" : "bang-solver", "maxclass" : "newobj",
				"patching_rect" : [ 1090.0, 440.0, 240.0, 22.0 ],
				"text" : "t b b b b b b b b b",
				"numinlets":1, "numoutlets":9, "outlettype":["bang","bang","bang","bang","bang","bang","bang","bang","bang"] } },

			{ "box" : { "id" : "cmt-flow", "maxclass" : "comment",
				"patching_rect" : [ 1090.0, 470.0, 320.0, 140.0 ],
				"text" : "Per-frame order:\n  1. inject    (state_a, bins)  \u2192 state_b\n  2. buoyancy  (state_b)         \u2192 state_b\n  3. advect    (state_b, state_b)\u2192 state_a\n  4. diffuse   (state_a)         \u2192 state_a\n  5. vorticity (state_a)         \u2192 state_a\n  6. divergence(state_a)         \u2192 fh_div\n  7. jacobi x20 (pres, div)      \u2192 pres\n  8. gradient  (state_a, pres)   \u2192 state_a\n  9. blackbody (state_a, asemic) \u2192 fh_color\n 10. volume    (fh_color)        \u2192 fh_final",
				"numinlets":1, "numoutlets":0 } },

			{ "box" : { "id" : "note-assets", "maxclass" : "comment",
				"patching_rect" : [ 30.0, 600.0, 500.0, 100.0 ],
				"text" : "Drop a PNG or JPG at assets/asemic.png to become the distorted light-language layer. It is carried by the velocity field and tinted by local heat color.\n\nFeed 3rd-party JSON (FFT / amplitude) into the left inlet of fh.audio_bins as a list of 8 floats, bypassing the built-in analyzer.",
				"numinlets":1, "numoutlets":0 } },

			{ "box" : { "id" : "init-clear", "maxclass" : "message",
				"patching_rect" : [ 380.0, 350.0, 140.0, 22.0 ],
				"text" : "clear",
				"numinlets":2, "numoutlets":1, "outlettype":[""] } },

			{ "box" : { "id" : "init-bang", "maxclass" : "newobj",
				"patching_rect" : [ 380.0, 320.0, 60.0, 22.0 ],
				"text" : "loadbang",
				"numinlets":1, "numoutlets":1, "outlettype":["bang"] } }
		],
		"lines" : [
			{ "patchline" : { "source" : [ "adc", 0 ], "destination" : [ "mix", 0 ] } },
			{ "patchline" : { "source" : [ "adc", 1 ], "destination" : [ "mix", 1 ] } },
			{ "patchline" : { "source" : [ "mix", 0 ], "destination" : [ "scale-in", 0 ] } },
			{ "patchline" : { "source" : [ "scale-in", 0 ], "destination" : [ "audio-bins", 0 ] } },
			{ "patchline" : { "source" : [ "gain-msg", 0 ], "destination" : [ "audio-bins", 1 ] } },

			{ "patchline" : { "source" : [ "audio-bins", 0 ], "destination" : [ "organic-mod", 0 ] } },
			{ "patchline" : { "source" : [ "organic-mod", 0 ], "destination" : [ "bins-split", 0 ] } },
			{ "patchline" : { "source" : [ "bins-split", 0 ], "destination" : [ "bins-lo-unpack", 0 ] } },
			{ "patchline" : { "source" : [ "bins-split", 1 ], "destination" : [ "bins-hi-unpack", 0 ] } },
			{ "patchline" : { "source" : [ "bins-lo-unpack", 0 ], "destination" : [ "bins-lo-pk", 1 ] } },
			{ "patchline" : { "source" : [ "bins-lo-unpack", 1 ], "destination" : [ "bins-lo-pk", 2 ] } },
			{ "patchline" : { "source" : [ "bins-lo-unpack", 2 ], "destination" : [ "bins-lo-pk", 3 ] } },
			{ "patchline" : { "source" : [ "bins-lo-unpack", 3 ], "destination" : [ "bins-lo-pk", 4 ] } },
			{ "patchline" : { "source" : [ "bins-hi-unpack", 0 ], "destination" : [ "bins-hi-pk", 1 ] } },
			{ "patchline" : { "source" : [ "bins-hi-unpack", 1 ], "destination" : [ "bins-hi-pk", 2 ] } },
			{ "patchline" : { "source" : [ "bins-hi-unpack", 2 ], "destination" : [ "bins-hi-pk", 3 ] } },
			{ "patchline" : { "source" : [ "bins-hi-unpack", 3 ], "destination" : [ "bins-hi-pk", 4 ] } },
			{ "patchline" : { "source" : [ "bins-lo-pk", 0 ], "destination" : [ "slab-inject", 0 ] } },
			{ "patchline" : { "source" : [ "bins-hi-pk", 0 ], "destination" : [ "slab-inject", 0 ] } },

			{ "patchline" : { "source" : [ "world-start", 0 ], "destination" : [ "world", 0 ] } },
			{ "patchline" : { "source" : [ "world-erase", 0 ], "destination" : [ "world", 0 ] } },
			{ "patchline" : { "source" : [ "dim-init", 0 ], "destination" : [ "dim-msg", 0 ] } },
			{ "patchline" : { "source" : [ "dim-msg", 0 ], "destination" : [ "tex-state-a", 0 ] } },
			{ "patchline" : { "source" : [ "dim-msg", 0 ], "destination" : [ "tex-state-b", 0 ] } },
			{ "patchline" : { "source" : [ "dim-msg", 0 ], "destination" : [ "tex-div", 0 ] } },
			{ "patchline" : { "source" : [ "dim-msg", 0 ], "destination" : [ "tex-pres-a", 0 ] } },
			{ "patchline" : { "source" : [ "dim-msg", 0 ], "destination" : [ "tex-pres-b", 0 ] } },

			{ "patchline" : { "source" : [ "init-bang", 0 ], "destination" : [ "init-clear", 0 ] } },
			{ "patchline" : { "source" : [ "init-clear", 0 ], "destination" : [ "tex-state-a", 0 ] } },
			{ "patchline" : { "source" : [ "init-clear", 0 ], "destination" : [ "tex-state-b", 0 ] } },

			{ "patchline" : { "source" : [ "frame-tgl", 0 ], "destination" : [ "metro-frame", 0 ] } },
			{ "patchline" : { "source" : [ "metro-frame", 0 ], "destination" : [ "bang-solver", 0 ] } },

			{ "patchline" : { "source" : [ "bang-solver", 0 ], "destination" : [ "tex-state-a", 0 ] } },
			{ "patchline" : { "source" : [ "bang-solver", 1 ], "destination" : [ "slab-inject", 0 ] } },
			{ "patchline" : { "source" : [ "bang-solver", 2 ], "destination" : [ "slab-buoyancy", 0 ] } },
			{ "patchline" : { "source" : [ "bang-solver", 3 ], "destination" : [ "slab-advect", 0 ] } },
			{ "patchline" : { "source" : [ "bang-solver", 4 ], "destination" : [ "slab-diffuse", 0 ] } },
			{ "patchline" : { "source" : [ "bang-solver", 5 ], "destination" : [ "slab-vort", 0 ] } },
			{ "patchline" : { "source" : [ "bang-solver", 6 ], "destination" : [ "slab-div", 0 ] } },
			{ "patchline" : { "source" : [ "bang-solver", 7 ], "destination" : [ "slab-jacobi-iter", 0 ] } },
			{ "patchline" : { "source" : [ "bang-solver", 8 ], "destination" : [ "slab-gradient", 0 ] } },

			{ "patchline" : { "source" : [ "tex-state-a", 0 ], "destination" : [ "slab-inject", 0 ] } },
			{ "patchline" : { "source" : [ "slab-inject", 0 ], "destination" : [ "slab-buoyancy", 0 ] } },
			{ "patchline" : { "source" : [ "slab-buoyancy", 0 ], "destination" : [ "slab-advect", 0 ] } },
			{ "patchline" : { "source" : [ "slab-buoyancy", 0 ], "destination" : [ "slab-advect", 1 ] } },
			{ "patchline" : { "source" : [ "slab-advect", 0 ], "destination" : [ "slab-diffuse", 0 ] } },
			{ "patchline" : { "source" : [ "slab-diffuse", 0 ], "destination" : [ "slab-vort", 0 ] } },
			{ "patchline" : { "source" : [ "slab-vort", 0 ], "destination" : [ "slab-div", 0 ] } },
			{ "patchline" : { "source" : [ "slab-div", 0 ], "destination" : [ "slab-jacobi-real", 1 ] } },
			{ "patchline" : { "source" : [ "slab-jacobi-iter", 0 ], "destination" : [ "slab-jacobi-real", 0 ] } },
			{ "patchline" : { "source" : [ "slab-jacobi-real", 0 ], "destination" : [ "slab-jacobi-real", 0 ] } },
			{ "patchline" : { "source" : [ "slab-jacobi-real", 0 ], "destination" : [ "slab-gradient", 1 ] } },
			{ "patchline" : { "source" : [ "slab-vort", 0 ], "destination" : [ "slab-gradient", 0 ] } },

			{ "patchline" : { "source" : [ "slab-gradient", 0 ], "destination" : [ "slab-blackbody", 0 ] } },
			{ "patchline" : { "source" : [ "tex-asemic", 0 ], "destination" : [ "slab-blackbody", 1 ] } },
			{ "patchline" : { "source" : [ "slab-blackbody", 0 ], "destination" : [ "slab-volume", 0 ] } },
			{ "patchline" : { "source" : [ "slab-volume", 0 ], "destination" : [ "display-quad", 0 ] } },

			{ "patchline" : { "source" : [ "p-heat-gain", 0 ], "destination" : [ "prefix-heat", 0 ] } },
			{ "patchline" : { "source" : [ "prefix-heat", 0 ], "destination" : [ "slab-blackbody", 0 ] } },
			{ "patchline" : { "source" : [ "p-alpha", 0 ], "destination" : [ "prefix-alpha", 0 ] } },
			{ "patchline" : { "source" : [ "prefix-alpha", 0 ], "destination" : [ "slab-buoyancy", 0 ] } },
			{ "patchline" : { "source" : [ "p-beta", 0 ], "destination" : [ "prefix-beta", 0 ] } },
			{ "patchline" : { "source" : [ "prefix-beta", 0 ], "destination" : [ "slab-buoyancy", 0 ] } },
			{ "patchline" : { "source" : [ "p-vort", 0 ], "destination" : [ "prefix-vort", 0 ] } },
			{ "patchline" : { "source" : [ "prefix-vort", 0 ], "destination" : [ "slab-vort", 0 ] } },
			{ "patchline" : { "source" : [ "p-asemic", 0 ], "destination" : [ "prefix-asemic", 0 ] } },
			{ "patchline" : { "source" : [ "prefix-asemic", 0 ], "destination" : [ "slab-blackbody", 0 ] } },
			{ "patchline" : { "source" : [ "p-exposure", 0 ], "destination" : [ "prefix-exposure", 0 ] } },
			{ "patchline" : { "source" : [ "prefix-exposure", 0 ], "destination" : [ "slab-blackbody", 0 ] } }
		]
	}
}
