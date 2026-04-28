import AVFoundation
import CoreAudio

// MARK: - AudioEngine
//
// Captures audio from BlackHole 2ch via AVAudioEngine and publishes 32
// microtonal magnitudes (q1–q32) to the WebView on every audio callback.
//
// ── SETUP (one-time) ──────────────────────────────────────────────────────────
//  1. Open Audio MIDI Setup → "+" → Create Multi-Output Device
//       → check "BlackHole 2ch" + your headphones / speakers
//  2. System Settings → Sound → Output  → that Multi-Output Device
//     System Settings → Sound → Input   → BlackHole 2ch
//  3. The visualiser now receives your full system mix. No microphone needed.
//
//  To change the device name (e.g. "BlackHole 16ch"), edit `inputDeviceName`.
// ─────────────────────────────────────────────────────────────────────────────

final class AudioEngine: ObservableObject {

    // Called on the main thread with 32 normalised [0,1] magnitudes.
    var onQ: (([Float]) -> Void)?

    // Name of the Core Audio input device to use.
    // Must exactly match the name shown in Audio MIDI Setup.
    static let inputDeviceName = "BlackHole 2ch"

    // ── Private state ─────────────────────────────────────────────────────────

    private let engine     = AVAudioEngine()
    private var fft:       MicrotonalFFT?
    private var normaliser = AdaptiveNormaliser()

    // 31-EDO probe frequencies: baseHz=55, stride=3, 32 bins.
    private let targetHz: [Float] = makeEDOFrequencies(
        edo:       31,
        baseHz:    55.0,
        startStep: 0,
        stride:    3
    )

    // MARK: - Lifecycle

    func start() {
        // Route the input node to BlackHole 2ch BEFORE reading its format;
        // the format is device-specific (sample rate, channel count).
        let inputNode = engine.inputNode

        if let deviceID = coreAudioDeviceID(named: Self.inputDeviceName) {
            if routeInputNode(to: deviceID) {
                print("[AudioEngine] Input → \(Self.inputDeviceName) (id \(deviceID))")
            } else {
                print("[AudioEngine] Could not route to \(Self.inputDeviceName) — using system default")
            }
        } else {
            print("[AudioEngine] '\(Self.inputDeviceName)' not found.")
            print("[AudioEngine] Available input devices: \(inputDeviceNames())")
            print("[AudioEngine] Falling back to system default input.")
        }

        // Read format after device is selected so sample rate is correct.
        let fmt = inputNode.outputFormat(forBus: 0)
        let sr: Float = fmt.sampleRate > 0 ? Float(fmt.sampleRate) : 44100

        // 4096 samples → ~93 ms latency at 44100 Hz, ~10.8 Hz/bin resolution.
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

            DispatchQueue.main.async { self.onQ?(q) }
        }

        do {
            try engine.start()
            print("[AudioEngine] started — device: \(Self.inputDeviceName), sampleRate: \(sr) Hz")
        } catch {
            print("[AudioEngine] start failed: \(error)")
        }
    }

    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        print("[AudioEngine] stopped")
    }

    // MARK: - Core Audio device helpers

    /// Set `engine.inputNode`'s underlying AUHAL to a specific device.
    @discardableResult
    private func routeInputNode(to deviceID: AudioDeviceID) -> Bool {
        guard let au = engine.inputNode.audioUnit else { return false }
        var id = deviceID
        return AudioUnitSetProperty(
            au,
            kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global,
            0,
            &id,
            UInt32(MemoryLayout<AudioDeviceID>.size)
        ) == noErr
    }

    /// Find the `AudioDeviceID` for a device whose name exactly matches `name`.
    private func coreAudioDeviceID(named name: String) -> AudioDeviceID? {
        for id in allDeviceIDs() {
            if deviceName(for: id) == name { return id }
        }
        return nil
    }

    /// Names of all devices that have at least one input stream.
    private func inputDeviceNames() -> [String] {
        allDeviceIDs().compactMap { id -> String? in
            guard hasInputStream(id) else { return nil }
            return deviceName(for: id)
        }
    }

    // MARK: - Low-level Core Audio queries

    private func allDeviceIDs() -> [AudioDeviceID] {
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope:    kAudioObjectPropertyScopeGlobal,
            mElement:  kAudioObjectPropertyElementMain
        )
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(
            AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size
        ) == noErr else { return [] }

        var ids = [AudioDeviceID](repeating: 0, count: Int(size) / MemoryLayout<AudioDeviceID>.size)
        guard AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &ids
        ) == noErr else { return [] }
        return ids
    }

    private func deviceName(for id: AudioDeviceID) -> String? {
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioObjectPropertyName,
            mScope:    kAudioObjectPropertyScopeGlobal,
            mElement:  kAudioObjectPropertyElementMain
        )
        var cfName = "" as CFString
        var size   = UInt32(MemoryLayout<CFString>.size)
        guard AudioObjectGetPropertyData(id, &addr, 0, nil, &size, &cfName) == noErr else { return nil }
        return cfName as String
    }

    private func hasInputStream(_ id: AudioDeviceID) -> Bool {
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreams,
            mScope:    kAudioObjectPropertyScopeInput,
            mElement:  kAudioObjectPropertyElementMain
        )
        var size: UInt32 = 0
        return AudioObjectGetPropertyDataSize(id, &addr, 0, nil, &size) == noErr && size > 0
    }
}
