import SwiftUI
import ScreenCaptureKit
import AVFoundation
import CoreGraphics
import HaishinKit

// MARK: - Broadcaster
//
// Live RTMP broadcast of the mixer to YouTube / Restream.
//
//   • Video — ScreenCaptureKit captures the visualizer window (just the mix,
//     no operator UI), 1080p30.
//   • Audio — ScreenCaptureKit captures the display's system audio (the music),
//     so whatever is playing goes out with the visuals.
//   • Encode + RTMP — HaishinKit encodes H.264 / AAC and publishes to the
//     RTMP URL + stream key you enter in the menu-bar Broadcast panel.
//
// ── ON-MAC NOTES ──────────────────────────────────────────────────────────────
//  • Screen Recording permission: first Go Live triggers the macOS prompt.
//    ScreenCaptureKit ties that permission to an app bundle, so run from Xcode
//    (or a built/signed .app) rather than a bare `swift run` binary, otherwise
//    the permission may not stick.
//  • HaishinKit's settings API shifts between versions. The codec-settings block
//    in `configureEncoder()` is the only version-sensitive spot — if it fails to
//    compile, comment it out (defaults still stream) and report the error.
//
//  YouTube ingest URL:  rtmp://a.rtmp.youtube.com/live2
//  (stream key from YouTube Studio → Go Live → Stream settings)
// ─────────────────────────────────────────────────────────────────────────────

@MainActor
final class Broadcaster: NSObject, ObservableObject, SCStreamOutput, SCStreamDelegate {

    @Published var isLive = false
    @Published var status = "Idle"

    // Title of the window we capture for video (the main visualizer window).
    static let windowTitle = "Butterchurn — Microtonal Visualizer"

    private let connection = RTMPConnection()
    private lazy var stream = RTMPStream(connection: connection)
    private var videoStream: SCStream?
    private var audioStream: SCStream?
    private var streamKey = ""

    override init() {
        super.init()
        connection.addEventListener(.rtmpStatus, selector: #selector(onRTMPStatus), observer: self)
    }

    // MARK: Public control

    func goLive(url: String, key: String) {
        guard !isLive else { return }
        let trimmedURL = url.trimmingCharacters(in: .whitespacesAndNewlines)
        streamKey = key.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedURL.isEmpty, !streamKey.isEmpty else { status = "Enter RTMP URL and stream key"; return }

        configureEncoder()
        status = "Starting capture…"
        Task {
            do {
                try await startCapture()
                status = "Connecting…"
                connection.connect(trimmedURL)
            } catch {
                status = "Capture failed: \(error.localizedDescription)"
                await stopCapture()
            }
        }
    }

    func stop() {
        Task { await stopCapture() }
        stream.close()
        connection.close()
        isLive = false
        status = "Idle"
    }

    // MARK: HaishinKit encoder config (version-sensitive — see header note)

    private func configureEncoder() {
        stream.videoSettings.videoSize = .init(width: 1920, height: 1080)
        stream.videoSettings.bitRate   = 16_000_000   // 16 Mbps
        stream.videoSettings.maxKeyFrameIntervalDuration = 2   // keyframe every 2s (RTMP needs this)
        stream.audioSettings.bitRate   = 160_000
    }

    @objc private func onRTMPStatus(_ notification: Notification) {
        let e = Event.from(notification)
        guard let data = e.data as? ASObject, let code = data["code"] as? String else { return }
        Task { @MainActor in
            switch code {
            case RTMPConnection.Code.connectSuccess.rawValue:
                stream.publish(streamKey)
                isLive = true
                status = "● LIVE"
            case RTMPConnection.Code.connectFailed.rawValue,
                 RTMPConnection.Code.connectClosed.rawValue:
                isLive = false
                status = "Disconnected (\(code))"
            default:
                status = code
            }
        }
    }

    // MARK: ScreenCaptureKit capture

    private func startCapture() async throws {
        // NOTE: we deliberately do NOT hard-gate on CGPreflightScreenCaptureAccess()
        // — it can return false negatives. Try the capture first; only if we can't
        // see our own window do we diagnose the permission.
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)

        // Find our visualizer window by our own process ID (a SwiftPM binary has no
        // bundle id), preferring the titled window over the menu-bar panel.
        let myPID = ProcessInfo.processInfo.processIdentifier
        let mine = content.windows.filter { $0.owningApplication?.processID == myPID }
        let byTitle = { (w: SCWindow) in (w.title ?? "").contains("Butterchurn") }
        let win = mine.first(where: byTitle)
            ?? mine.max(by: { ($0.frame.width * $0.frame.height) < ($1.frame.width * $1.frame.height) })
            ?? content.windows.first(where: byTitle)
        guard let win else {
            let hasPerm = CGPreflightScreenCaptureAccess()
            if !hasPerm { CGRequestScreenCaptureAccess() }
            let hint = hasPerm
                ? "Visualizer window not found — make sure its window is open and on screen (not minimized)."
                : "Screen Recording permission isn't active for this build. Run the app as a built .app (scripts/build-app.sh → open ButterchurnVisualizer.app), NOT through Xcode — the debugger blocks the grant. Then allow it in System Settings ▸ Privacy & Security ▸ Screen Recording."
            throw NSError(domain: "Broadcaster", code: 1, userInfo: [NSLocalizedDescriptionKey: hint])
        }
        guard let display = content.displays.first(where: { NSPointInRect(CGPoint(x: win.frame.midX, y: win.frame.midY), $0.frame) })
                         ?? content.displays.first else {
            throw NSError(domain: "Broadcaster", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "No display available to capture."])
        }

        let videoCfg = SCStreamConfiguration()
        videoCfg.width = 1920
        videoCfg.height = 1080
        videoCfg.minimumFrameInterval = CMTime(value: 1, timescale: 60)   // 60 fps
        videoCfg.pixelFormat = kCVPixelFormatType_32BGRA
        videoCfg.queueDepth = 6
        videoCfg.showsCursor = false

        // Capture the whole display — what's actually on screen — which is the most
        // reliable way to grab WebGL/GPU-composited content (window capture can come
        // back black for it). Run the mixer fullscreen (press F) so the broadcast is
        // just the mix.
        let vStream = SCStream(filter: SCContentFilter(display: display, excludingWindows: []),
                               configuration: videoCfg, delegate: self)
        try vStream.addStreamOutput(self, type: .screen, sampleHandlerQueue: .global(qos: .userInitiated))
        videoStream = vStream

        // Audio: the display's system audio (the music).
        let audioCfg = SCStreamConfiguration()
        audioCfg.capturesAudio = true
        audioCfg.sampleRate = 48_000
        audioCfg.channelCount = 2
        audioCfg.width = 2; audioCfg.height = 2     // audio-only; video frames ignored
        let aStream = SCStream(filter: SCContentFilter(display: display, excludingWindows: []),
                               configuration: audioCfg, delegate: self)
        try aStream.addStreamOutput(self, type: .audio, sampleHandlerQueue: .global(qos: .userInitiated))
        audioStream = aStream

        try await videoStream?.startCapture()
        try await audioStream?.startCapture()
    }

    private func stopCapture() async {
        try? await videoStream?.stopCapture()
        try? await audioStream?.stopCapture()
        videoStream = nil
        audioStream = nil
    }

    // MARK: SCStreamOutput — forward sample buffers to the encoder

    nonisolated func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard sampleBuffer.isValid else { return }
        // Video frames must carry a pixel buffer; audio passes straight through.
        // (No SCStreamFrameInfo attachment parsing — that cast is fragile and was
        // silently dropping every video frame.)
        if type == .screen, CMSampleBufferGetImageBuffer(sampleBuffer) == nil { return }
        Task { @MainActor in self.stream.append(sampleBuffer) }
    }

    nonisolated func stream(_ stream: SCStream, didStopWithError error: Error) {
        Task { @MainActor in
            self.status = "Capture stopped: \(error.localizedDescription)"
            self.isLive = false
        }
    }
}
