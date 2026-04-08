import AVFoundation

// MARK: - AudioEngine
//
// Captures live audio via AVAudioEngine and publishes 32 microtonal
// magnitudes (q1–q32) to the WebView on every audio callback.
//
// ── SOURCE OPTIONS ────────────────────────────────────────────────────────────
//  A. MICROPHONE (default, zero setup):
//     Works immediately after granting microphone permission.
//
//  B. SYSTEM AUDIO via BlackHole (free, recommended for music playback):
//     1. Install BlackHole 2ch from existential.audio/blackhole
//     2. Open Audio MIDI Setup → "+" → Create Multi-Output Device
//        → check both "BlackHole 2ch" and your speakers/headphones
//     3. Set System Preferences → Sound → Output → that Multi-Output Device
//     4. Set System Preferences → Sound → Input  → BlackHole 2ch
//     AVAudioEngine then receives the system mix as its input. No code changes.
//
//  C. SYSTEM AUDIO via ScreenCaptureKit (macOS 12.3+, no virtual device):
//     Requires adding com.apple.security.screen-recording entitlement.
//     Swap the body of start() with an SCStream configuration.
//     See: developer.apple.com/documentation/screencapturekit
// ─────────────────────────────────────────────────────────────────────────────

final class AudioEngine: ObservableObject {

    // Called on the main thread with 32 normalised [0,1] magnitudes.
    var onQ: (([Float]) -> Void)?

    // ── Private state ─────────────────────────────────────────────────────────

    private let engine      = AVAudioEngine()
    private var fft:        MicrotonalFFT?
    private var normaliser  = AdaptiveNormaliser()

    // 31-EDO probe frequencies: baseHz=55, stride=3, 32 bins.
    // Swap out for any scale using makeEDOFrequencies(edo:baseHz:stride:).
    private let targetHz: [Float] = makeEDOFrequencies(
        edo:       31,
        baseHz:    55.0,
        startStep: 0,
        stride:    3
    )

    // MARK: - Lifecycle

    func start() {
        let inputNode = engine.inputNode
        let fmt       = inputNode.outputFormat(forBus: 0)

        // Use the hardware sample rate; fall back to 44100 if not yet available.
        let sr: Float = fmt.sampleRate > 0 ? Float(fmt.sampleRate) : 44100

        // Buffer size: 4096 samples → ~93 ms at 44100 Hz.
        // Frequency resolution = sr / 4096 ≈ 10.8 Hz per FFT bin.
        // Halving to 2048 gives ~21 Hz/bin but halves latency if preferred.
        let bufferSize: AVAudioFrameCount = 4096

        fft = MicrotonalFFT(bufferSize: Int(bufferSize), sampleRate: sr)

        inputNode.installTap(onBus: 0, bufferSize: bufferSize, format: fmt) {
            [weak self] buffer, _ in
            guard let self,
                  let fft  = self.fft,
                  let data = buffer.floatChannelData?[0]
            else { return }

            let count = Int(buffer.frameLength)
            var q = fft.magnitudes(samples: data,
                                   count:   count,
                                   targetHz: self.targetHz)
            self.normaliser.normalise(&q)

            // Deliver on the main thread so WebView injection is safe.
            DispatchQueue.main.async { self.onQ?(q) }
        }

        do {
            try engine.start()
            print("[AudioEngine] started — sampleRate: \(sr) Hz")
        } catch {
            print("[AudioEngine] start failed: \(error)")
        }
    }

    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        print("[AudioEngine] stopped")
    }
}
