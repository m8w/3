import AVFoundation
import CoreAudio

// MARK: - AudioEngine
//
// Captures audio from BlackHole 2ch. On every call to start() the engine
// switches the macOS system default input to BlackHole 2ch — this must happen
// BEFORE the first access to engine.inputNode, because AVAudioEngine caches
// the input-device format lazily on that first access. Changing the system
// default afterwards leaves the cache stale.
//
// The user never needs to open System Settings; the app owns this input slot.
//
// ── ONE-TIME SETUP ────────────────────────────────────────────────────────────
//  Audio MIDI Setup → "+" → Multi-Output Device
//    → check "BlackHole 2ch" + headphones / speakers
//  System Settings → Sound → Output → that Multi-Output Device
//  (Input will be managed automatically by this engine.)
// ─────────────────────────────────────────────────────────────────────────────

final class AudioEngine: ObservableObject {

    /// Called on the main thread with 32 normalised [0,1] magnitudes.
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
        // ── Step 1: switch system default input to BlackHole 2ch ──────────────
        // Must happen before the first engine.inputNode access.
        if let id = coreAudioDeviceID(named: Self.inputDeviceName) {
            if setSystemDefaultInput(to: id) {
                print("[AudioEngine] System input → \(Self.inputDeviceName) (id \(id))")
            } else {
                print("[AudioEngine] Could not switch to \(Self.inputDeviceName) — continuing with current default")
            }
        } else {
            print("[AudioEngine] '\(Self.inputDeviceName)' not found.")
            print("[AudioEngine] Available inputs: \(inputDeviceNames())")
        }

        // ── Step 2: read the (now-correct) hardware format ────────────────────
        let inputNode = engine.inputNode
        let fmt = inputNode.outputFormat(forBus: 0)
        let sr: Float = fmt.sampleRate > 0 ? Float(fmt.sampleRate) : 48000
        print("[AudioEngine] Input format: \(fmt.channelCount) ch, \(fmt.sampleRate) Hz")

        // ── Step 3: initialise FFT ────────────────────────────────────────────
        let bufferSize: AVAudioFrameCount = 4096
        fft = MicrotonalFFT(bufferSize: Int(bufferSize), sampleRate: sr)

        // ── Step 4: install tap ───────────────────────────────────────────────
        // Pass the live fmt explicitly — avoids the "config change pending" error
        // that occurs when the tap format and the cached hardware format disagree.
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

        // ── Step 5: start ─────────────────────────────────────────────────────
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

    @discardableResult
    private func setSystemDefaultInput(to deviceID: AudioDeviceID) -> Bool {
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope:    kAudioObjectPropertyScopeGlobal,
            mElement:  kAudioObjectPropertyElementMain
        )
        var id = deviceID
        return AudioObjectSetPropertyData(
            AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil,
            UInt32(MemoryLayout<AudioDeviceID>.size), &id
        ) == noErr
    }

    private func coreAudioDeviceID(named name: String) -> AudioDeviceID? {
        for id in allDeviceIDs() {
            if deviceName(for: id) == name { return id }
        }
        return nil
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
        var cfName: CFString = "" as CFString
        var size = UInt32(MemoryLayout<CFString>.size)
        let status = withUnsafeMutablePointer(to: &cfName) {
            AudioObjectGetPropertyData(id, &addr, 0, nil, &size, $0)
        }
        guard status == noErr else { return nil }
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
