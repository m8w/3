import Foundation

// MARK: - Butterchurn JSON field-name mapping
//
// Milkdrop .milk files use prefixed names (fDecay, bAdditiveWaves, nWaveMode…).
// Butterchurn's JSON preset format uses lowercase snake_case without prefixes.
// This map covers every base-value key that Butterchurn's renderer reads.

private let milkToButterchurnKey: [String: String] = [
    // Motion
    "zoom":               "zoom",
    "rot":                "rot",
    "cx":                 "cx",
    "cy":                 "cy",
    "dx":                 "dx",
    "dy":                 "dy",
    "warp":               "warp",
    "sx":                 "sx",
    "sy":                 "sy",
    // Feedback
    "fdecay":             "decay",
    // Brightness / post-processing
    "fgammaadj":          "gammaadj",
    "fbrighten":          "brighten",      // treated as bool 0/1
    "bdarken":            "darken",
    "bsolarize":          "solarize",
    "binvert":            "invert",
    "bredblue" :          "red_blue",
    "bdarken":            "darken",
    "bbrighten":          "brighten",
    // Echo / feedback overlay
    "fvideoechozoom":     "echo_zoom",
    "fvideoechoalpha":    "echo_alpha",
    "nvideoechoorientation": "echo_orient",
    // Waveform style
    "nwavemode":          "wave_mode",
    "badditivewaves":     "additivewave",
    "bwavedots":          "wave_dots",
    "bwavethick":         "wave_thick",
    "bmodwavealphabyvolume": "modwavealphabyvolume",
    "bmaximizewavecolor": "maximizewavecolor",
    // Waveform geometry / colour
    "fwavealpha":         "wave_alpha",
    "fwavescale":         "wave_scale",
    "fwavesmoothing":     "wave_smoothing",
    "fwaveparam":         "wave_param",
    "wave_r":             "wave_r",
    "wave_g":             "wave_g",
    "wave_b":             "wave_b",
    "wave_x":             "wave_x",
    "wave_y":             "wave_y",
    // Texture / warp
    "btexwrap":           "texwrap",
    "bdarkencenter":      "darken_center",
    "fwarpanimspeed":     "fWarpAnimSpeed",  // butterchurn reads this camelCase
    "fwarpscale":         "fWarpScale",
    "fzoomexponent":      "fZoomExponent",
    "fshader":            "fshader",
    // Borders
    "ob_size":            "ob_size",
    "ib_size":            "ib_size",
    // Motion vectors
    "nmotionvectorsx":    "nMotionVectorsX",
    "nmotionvectorsy":    "nMotionVectorsY",
    "mv_l":               "mv_l",
    // Rating (informational only, Butterchurn ignores it but doesn't reject it)
    "frating":            "fRating",
]

// MARK: - Converter

public enum MilkPresetConverter {

    /// Convert a `MilkPreset` (parsed by `PresetParser`) to a Dictionary
    /// that matches the Butterchurn JSON preset schema and can be serialised
    /// with `JSONSerialization.data(withJSONObject:)` then injected via
    /// `window._addPresets({ "Name": <result> })`.
    public static func toButterchurnDict(_ preset: MilkPreset) -> [String: Any] {
        var baseVals: [String: Double] = [:]

        for (rawKey, rawValue) in preset.baseVals {
            let lookupKey = rawKey.lowercased().replacingOccurrences(of: "_", with: "")
            // Try direct match first, then the no-underscore lookup.
            let bcKey = milkToButterchurnKey[rawKey.lowercased()]
                     ?? milkToButterchurnKey[lookupKey]
                     ?? rawKey   // fall back to the raw key

            if let d = Double(rawValue) {
                baseVals[bcKey] = d
            }
        }

        // Join the numbered equation lines into single EEL strings.
        let initEQs     = preset.initEqs.joined(separator: " ")
        let perFrameEQs = preset.perFrameEqs.joined(separator: " ")
        let perPixelEQs = preset.perPixelEqs.joined(separator: " ")

        // Build per-wave objects.
        var waves: [[String: Any]] = []
        let waveKeys = Set(preset.waveFrameEqs.keys).union(Set(preset.wavePointEqs.keys))
        for k in waveKeys.sorted() {
            let frameEqs = (preset.waveFrameEqs[k] ?? []).joined(separator: " ")
            let pointEqs = (preset.wavePointEqs[k] ?? []).joined(separator: " ")
            waves.append([
                "per_frame_eqs_eel": frameEqs,
                "per_point_eqs_eel": pointEqs,
                "enabled": 1,
                "samples": 512,
                "sep": 0,
                "bSpectrum": 0,
                "bOscillate": 0,
                "bWireframe": 0,
                "bAdditive": 0,
                "bUseDots": 0,
                "scaling": 1.0,
                "smoothing": 0.5,
                "r": 1.0, "g": 1.0, "b": 1.0, "a": 0.8,
                "x": 0.5, "y": 0.5,
            ])
        }

        return [
            "baseVals":    baseVals,
            "initEQs":     initEQs,
            "perFrameEQs": perFrameEQs,
            "perPixelEQs": perPixelEQs,
            "waves":       waves,
            "shapes":      [] as [[String: Any]],
        ]
    }
}
