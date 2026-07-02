import SwiftUI
import Foundation

// MARK: - Broadcaster (FFmpeg backend)
//
// Live RTMP broadcast to YouTube / Restream. We shell out to ffmpeg, which
// captures the screen + BlackHole audio via avfoundation, encodes H.264
// (VideoToolbox) / AAC, and pushes RTMP. ffmpeg's RTMP video is rock-solid —
// unlike HaishinKit, which accepted our video frames but never encoded them.
//
// ── REQUIREMENTS ──────────────────────────────────────────────────────────────
//  • ffmpeg installed (you already have it — radiot.py uses ffplay):
//        brew install ffmpeg
//  • Run the app from Terminal so the Screen-Recording grant applies:
//        ./ButterchurnVisualizer.app/Contents/MacOS/ButterchurnVisualizer
//  • Fullscreen the mix (press F) so the captured display is just the visuals.
//
//  Capture devices default to "Capture screen 0" + "BlackHole 2ch". Override via
//  env CAPTURE_VIDEO_DEVICE / CAPTURE_AUDIO_DEVICE if your indices/names differ.
//  If a name is wrong, ffmpeg prints the available device list to the console.
// ─────────────────────────────────────────────────────────────────────────────

@MainActor
final class Broadcaster: ObservableObject {

    @Published var isLive = false
    @Published var status = "Idle"

    private var process: Process?

    // Resolve avfoundation device indices by parsing ffmpeg's device list, so we
    // don't depend on names (which ffmpeg matches unreliably) or a fixed index
    // (which shifts with however many cameras are attached).
    nonisolated private static func resolveDevices(ffmpeg: String) -> (video: String, audio: String) {
        let env = ProcessInfo.processInfo.environment
        if let v = env["CAPTURE_VIDEO_DEVICE"], let a = env["CAPTURE_AUDIO_DEVICE"] { return (v, a) }

        let p = Process()
        p.executableURL = URL(fileURLWithPath: ffmpeg)
        p.arguments = ["-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""]
        let pipe = Pipe()
        p.standardError = pipe
        p.standardOutput = pipe
        try? p.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit()
        let text = String(data: data, encoding: .utf8) ?? ""

        var section = ""
        var video = env["CAPTURE_VIDEO_DEVICE"] ?? "3"
        var audio = env["CAPTURE_AUDIO_DEVICE"] ?? "0"
        var foundVideo = false, foundAudio = false
        for raw in text.split(whereSeparator: \.isNewline) {
            let l = String(raw)
            if l.contains("video devices:") { section = "video"; continue }
            if l.contains("audio devices:") { section = "audio"; continue }
            guard let br = l.range(of: "] [") else { continue }
            let after = l[br.upperBound...]
            guard let close = after.firstIndex(of: "]") else { continue }
            let idx = String(after[..<close])
            guard Int(idx) != nil else { continue }
            let name = String(after[after.index(after: close)...]).trimmingCharacters(in: .whitespaces)
            if section == "video", !foundVideo, name.localizedCaseInsensitiveContains("Capture screen") {
                video = idx; foundVideo = true
            }
            if section == "audio", !foundAudio, name == "BlackHole 2ch" {
                audio = idx; foundAudio = true
            }
        }
        return (video, audio)
    }

    // MARK: Control

    func goLive(url: String, key: String) {
        guard !isLive else { return }
        let u = url.trimmingCharacters(in: .whitespacesAndNewlines)
        let k = key.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !u.isEmpty, !k.isEmpty else { status = "Enter RTMP URL and stream key"; return }
        let target = u.hasSuffix("/") ? u + k : u + "/" + k

        isLive = true
        status = "Connecting…"

        // ffmpeg lookup + device probe block (they run sub-processes), so do all of
        // it — and the launch — off the main thread to keep the UI responsive.
        Task.detached(priority: .userInitiated) { [weak self] in
            guard let ffmpeg = Broadcaster.findFFmpeg() else {
                await MainActor.run { self?.status = "ffmpeg not found — run: brew install ffmpeg"; self?.isLive = false }
                return
            }
            let dev = Broadcaster.resolveDevices(ffmpeg: ffmpeg)
            let env = ProcessInfo.processInfo.environment
            let fps = env["FPS"] ?? "30"
            let vbr = env["VBITRATE"] ?? "8M"
            let gop = String((Int(fps) ?? 30) * 2)
            // Known-good combined capture + the single safe audio fix:
            // -max_interleave_delta 0 lets the muxer flush audio immediately instead
            // of holding it to interleave with the bursty screen video (that hold is
            // the stutter). Do NOT add -use_wallclock_as_timestamps / -fps_mode cfr —
            // avfoundation's epoch-scale timestamps send it into a frame-dup spiral.
            let args = [
                "-hide_banner",
                "-thread_queue_size", "1024",
                "-f", "avfoundation", "-capture_cursor", "0", "-framerate", fps, "-i", "\(dev.video):\(dev.audio)",
                "-c:v", "h264_videotoolbox", "-realtime", "1",
                "-b:v", vbr, "-maxrate", vbr, "-bufsize", vbr,
                "-pix_fmt", "yuv420p", "-g", gop,
                "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
                "-max_interleave_delta", "0",
                "-f", "flv", target,
            ]
            print("[Broadcast] avfoundation devices → video:\(dev.video) audio:\(dev.audio)")
            print("[Broadcast] ffmpeg \(args.joined(separator: " "))")

            let p = Process()
            p.executableURL = URL(fileURLWithPath: ffmpeg)
            p.arguments = args
            let errPipe = Pipe()
            p.standardError = errPipe
            p.standardOutput = errPipe
            errPipe.fileHandleForReading.readabilityHandler = { [weak self] h in
                let chunk = h.availableData
                guard !chunk.isEmpty, let s = String(data: chunk, encoding: .utf8) else { return }
                Task { @MainActor in self?.handleOutput(s) }
            }
            p.terminationHandler = { [weak self] proc in
                Task { @MainActor in
                    self?.isLive = false
                    if self?.status == "● LIVE" || self?.status == "Connecting…" {
                        self?.status = proc.terminationStatus == 0 ? "Stopped" : "ffmpeg exited (\(proc.terminationStatus))"
                    }
                }
            }
            do {
                try p.run()
                await MainActor.run { self?.process = p }
            } catch {
                await MainActor.run { self?.status = "Failed to launch ffmpeg: \(error.localizedDescription)"; self?.isLive = false }
            }
        }
    }

    func stop() {
        if let p = process, p.isRunning {
            // ffmpeg flushes cleanly on 'q'/SIGINT.
            p.interrupt()
        }
        process = nil
        isLive = false
        status = "Idle"
    }

    // MARK: ffmpeg output → status

    private func handleOutput(_ s: String) {
        print("[ffmpeg] \(s)", terminator: "")
        if s.contains("frame=") {
            if status != "● LIVE" { status = "● LIVE" }
        } else {
            let lower = s.lowercased()
            if lower.contains("error") || lower.contains("failed") ||
               lower.contains("could not") || lower.contains("unable to") ||
               lower.contains("connection refused") || lower.contains("no such") {
                let line = s.split(whereSeparator: \.isNewline).last.map(String.init) ?? s
                status = "⚠︎ " + String(line.prefix(90))
            }
        }
    }

    // MARK: ffmpeg lookup

    nonisolated private static func findFFmpeg() -> String? {
        let candidates = ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg", "/opt/local/bin/ffmpeg"]
        for c in candidates where FileManager.default.isExecutableFile(atPath: c) { return c }
        // Fall back to `which ffmpeg` via a login shell (picks up PATH).
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/zsh")
        p.arguments = ["-lc", "command -v ffmpeg"]
        let pipe = Pipe()
        p.standardOutput = pipe
        try? p.run()
        p.waitUntilExit()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let path = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
        if let path, !path.isEmpty, FileManager.default.isExecutableFile(atPath: path) { return path }
        return nil
    }
}
