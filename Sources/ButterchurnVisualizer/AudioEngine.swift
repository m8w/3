import AVFoundation
import CoreAudio

// MARK: - AudioEngine
//
// Captures audio from BlackHole 2ch (the system default input — set once in
// System Settings → Sound → Input → BlackHole 2ch).
//
// No AUHAL device-routing is performed here; AVAudioEngine uses whatever
// Core Audio reports as the default input device, which the user has already
// configured. Attempting to force a device via kAudioOutputUnitProperty_CurrentDevice
// after the engine has cached the format causes a channel/rate mismatch.
//
// ── SETUP (one-time) ──────────────────────────────────────────────────────────
//  1. Open Audio MIDI Setup → "+" → Create Multi-Output Device
//       → check "BlackHole 2ch" + your headphones / speakers
//  2. System Settings → Sound → Output  → that Multi-Output Device
//     System Settings → Sound → Input   → BlackHole 2ch
//  3. The visualiser now receives your full system mix. No microphone needed.
// ─────────────────────────────────────────────────────────────────────────────

final class AudioEngine: ObservableObject {

    // Called on the main thread with 32 normalised [0,1] magnitudes.
    var onQ: (([Float]) -> Void)?

    static let inputDeviceName = "BlackHole 2ch"

    private let engine     = AVAudioEngine()
    private var fft:       MicrotonalFFT?
    private var normaliser = AdaptiveNormaliser()

    private let targetHz: [Float] = makeEDOFrequencies(
        edo:       31,
        baseHz:    55.0,
        startStep: 0,
        stride:    3
    )

    // MARK: - Lifecycle

    func start() {
        let inputNode = engine.inputNode

        // Log the current system default input so the user can verify routing.
        if let id = systemDefaultInputDeviceID(), let name = deviceName(for: id) {
            print("[AudioEngine] System default input: \(name) (id \(id))")
            if name != Self.inputDeviceName {
                print("[AudioEngine] WARNING: expected '\(Self.inputDeviceName)'.")
                print("[AudioEngine] Go to System Settings → Sound → Input and select '\(Self.inputDeviceName)'.")
                print("[AudioEngine] Available inputs: \(inputDeviceNames())")
            }
        } else {
            print("[AudioEngine] Could not read system default input device.")
        }

        // Use AVAudioEngine's already-resolved format for the current default
        // input device. Reading this *before* any routing attempt keeps the
        // format cache consistent with the tap format we pass below.
        let fmt = inputNode.outputFormat(forBus: 0)
        let sr: Float = fmt.sampleRate > 0 ? Float(fmt.sampleRate) : 44100
        print("[AudioEngine] Input format: \(fmt.channelCount) ch, \(fmt.sampleRate) Hz")

        let bufferSize: AVAudioFrameCount = 4096
        fft = MicrotonalFFT(bufferSize: Int(bufferSize), sampleRate: sr)

        // Pass `fmt` explicitly so the tap and the engine agree on channel
        // count and sample rate, avoiding "config change pending" errors.
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

    // MARK: - Core Audio device helpers

    private func systemDefaultInputDeviceID() -> AudioDeviceID? {
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope:    kAudioObjectPropertyScopeGlobal,
            mElement:  kAudioObjectPropertyElementMain
        )
        var id   = AudioDeviceID(0)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        guard AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &id
        ) == noErr, id != 0 else { return nil }
        return id
    }

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
