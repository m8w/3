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

    private var videoDevice: String {
        ProcessInfo.processInfo.environment["CAPTURE_VIDEO_DEVICE"] ?? "Capture screen 0"
    }
    private var audioDevice: String {
        ProcessInfo.processInfo.environment["CAPTURE_AUDIO_DEVICE"] ?? "BlackHole 2ch"
    }

    // MARK: Control

    func goLive(url: String, key: String) {
        guard !isLive else { return }
        let u = url.trimmingCharacters(in: .whitespacesAndNewlines)
        let k = key.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !u.isEmpty, !k.isEmpty else { status = "Enter RTMP URL and stream key"; return }
        guard let ffmpeg = Self.findFFmpeg() else {
            status = "ffmpeg not found — run: brew install ffmpeg"; return
        }
        let target = u.hasSuffix("/") ? u + k : u + "/" + k

        let args = [
            "-hide_banner",
            "-f", "avfoundation",
            "-capture_cursor", "0",
            "-framerate", "60",
            "-i", "\(videoDevice):\(audioDevice)",
            "-c:v", "h264_videotoolbox",
            "-realtime", "1",
            "-b:v", "16M", "-maxrate", "16M", "-bufsize", "32M",
            "-pix_fmt", "yuv420p",
            "-g", "120",
            "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
            "-f", "flv", target,
        ]
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
            process = p
            isLive = true
            status = "Connecting…"
        } catch {
            status = "Failed to launch ffmpeg: \(error.localizedDescription)"
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

    private static func findFFmpeg() -> String? {
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
