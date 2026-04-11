// MilkConverter — command-line tool
//
// USAGE (run from the repo root):
//
//   Convert a single .milk file to .json:
//     swift run MilkConverter  MyPreset.milk
//     swift run MilkConverter  MyPreset.milk  output.json
//
//   Convert every .milk in a folder (writes .json next to each file):
//     swift run MilkConverter  /path/to/presets/
//
//   Convert .json back to .milk (round-trip):
//     swift run MilkConverter  MyPreset.json  --reverse
//
//   Validate a conversion without writing files:
//     swift run MilkConverter  MyPreset.milk  --dry-run
//
// The output JSON is ready to paste into Butterchurn via:
//   visualizer.loadPreset(JSON.parse(fs.readFileSync('MyPreset.json')), 0)
// or drop the file into any Butterchurn preset loader.

import Foundation

// Bring in the converter — it lives in the ButterchurnVisualizer library target.
// (Xcode resolves this because both targets are in the same package.)

// ─────────────────────────────────────────────────────────────────────────────
// Argument parsing
// ─────────────────────────────────────────────────────────────────────────────

let args = CommandLine.arguments.dropFirst()   // skip executable name

guard !args.isEmpty else {
    print("""
    MilkConverter — .milk ↔ Butterchurn JSON

    Usage:
      swift run MilkConverter  <input.milk>  [output.json]
      swift run MilkConverter  <input.json>  --reverse  [output.milk]
      swift run MilkConverter  <folder/>               (batch .milk → .json)
      swift run MilkConverter  <input.milk>  --dry-run  (print JSON, no write)
    """)
    exit(0)
}

let dryRun  = args.contains("--dry-run")
let reverse = args.contains("--reverse")
let inputs  = args.filter { !$0.hasPrefix("--") }

guard let inputPath = inputs.first else {
    fputs("Error: no input path supplied.\n", stderr)
    exit(1)
}

let inputURL = URL(fileURLWithPath: inputPath).standardizedFileURL

// ─────────────────────────────────────────────────────────────────────────────
// Batch mode — convert entire folder
// ─────────────────────────────────────────────────────────────────────────────

var isDir: ObjCBool = false
FileManager.default.fileExists(atPath: inputURL.path, isDirectory: &isDir)

if isDir.boolValue {
    guard !reverse else {
        fputs("Error: --reverse batch mode not supported.\n", stderr)
        exit(1)
    }
    let fm    = FileManager.default
    let items = (try? fm.contentsOfDirectory(at: inputURL,
                                              includingPropertiesForKeys: nil)) ?? []
    let milks = items.filter { $0.pathExtension.lowercased() == "milk" }

    guard !milks.isEmpty else {
        fputs("No .milk files found in \(inputURL.path)\n", stderr)
        exit(1)
    }

    print("Converting \(milks.count) preset(s) in \(inputURL.lastPathComponent)/")
    var ok = 0, fail = 0

    for url in milks.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
        do {
            let json    = try MilkToJSON.convertFile(at: url)
            let outURL  = url.deletingPathExtension().appendingPathExtension("json")
            if !dryRun {
                try json.write(to: outURL, atomically: true, encoding: .utf8)
            }
            print("  ✓  \(url.lastPathComponent)  →  \(outURL.lastPathComponent)")
            ok += 1
        } catch {
            print("  ✗  \(url.lastPathComponent)  →  \(error.localizedDescription)")
            fail += 1
        }
    }
    print("\nDone: \(ok) converted, \(fail) failed\(dryRun ? " (dry-run)" : "").")
    exit(fail > 0 ? 1 : 0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Single-file mode
// ─────────────────────────────────────────────────────────────────────────────

let ext = inputURL.pathExtension.lowercased()

if reverse {
    // ── JSON → .milk ─────────────────────────────────────────────────────────
    guard ext == "json" else {
        fputs("Error: --reverse expects a .json input file.\n", stderr)
        exit(1)
    }
    do {
        let data    = try Data(contentsOf: inputURL)
        let milk    = try JSONToMilk.convert(data)
        let outPath = inputs.dropFirst().first
        let outURL  = outPath.map { URL(fileURLWithPath: $0) }
                   ?? inputURL.deletingPathExtension().appendingPathExtension("milk")
        if dryRun {
            print(milk)
        } else {
            try milk.write(to: outURL, atomically: true, encoding: .utf8)
            print("Written: \(outURL.path)")
        }
    } catch {
        fputs("Error: \(error.localizedDescription)\n", stderr)
        exit(1)
    }

} else {
    // ── .milk → JSON ──────────────────────────────────────────────────────────
    guard ext == "milk" else {
        fputs("Error: expected a .milk input file (got .\(ext)).\n", stderr)
        exit(1)
    }
    do {
        let json   = try MilkToJSON.convertFile(at: inputURL)
        let outPath = inputs.dropFirst().first
        let outURL  = outPath.map { URL(fileURLWithPath: $0) }
                   ?? inputURL.deletingPathExtension().appendingPathExtension("json")
        if dryRun {
            print(json)
        } else {
            try json.write(to: outURL, atomically: true, encoding: .utf8)
            print("Written: \(outURL.path)")
        }
    } catch {
        fputs("Error: \(error.localizedDescription)\n", stderr)
        exit(1)
    }
}
