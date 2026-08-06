import CoreAudio
import AudioToolbox

// MARK: - AudioEngine
//
// Captures audio directly from BlackHole 2ch using AudioQueue.
//
// AudioQueue lets us target a specific input device by its UID (via
// kAudioQueueProperty_CurrentDevice) without changing the macOS system default
// input. This avoids the aggregate-device rebuild failures that occur when
// the default input is switched programmatically while aggregate devices include
// BlackHole 2ch as a sub-device.
//
// ── SETUP (one-time) ──────────────────────────────────────────────────────────
//  1. Audio MIDI Setup → "+" → Multi-Output Device
//       → check "BlackHole 2ch" + headphones / speakers
//  2. System Settings → Sound → Output → that Multi-Output Device
//  3. No input-device change needed — the app routes directly by UID.
// ─────────────────────────────────────────────────────────────────────────────

final class AudioEngine: ObservableObject {

    /// Called on the main thread with 32 normalised [0,1] magnitudes (mono).
    var onQ: (([Float]) -> Void)?
    /// Called on the main thread with per-channel magnitudes (left, right) when
    /// stereo-reactive mode is on.
    var onStereoQ: (([Float], [Float]) -> Void)?

    // Which input device drives the visuals. Defaults to BlackHole 2ch; set the
    // AUDIO_DEVICE env var to any input's exact name (e.g. your SN2 interface) to
    // react to that instead. If the name isn't found, the console prints every
    // available input so you can copy the right one.
    static let inputDeviceName = ProcessInfo.processInfo.environment["AUDIO_DEVICE"] ?? "BlackHole 2ch"

    // Stereo-reactive mode: capture L/R separately and drive the 3 screens with
    // left / center / right. Off by default (mono, unchanged). STEREO_REACTIVE=1.
    static let stereo = ProcessInfo.processInfo.environment["STEREO_REACTIVE"] == "1"

    private var audioQueue: AudioQueueRef?
    private var fft:        MicrotonalFFT?
    private var normaliser  = AdaptiveNormaliser()

    private let targetHz: [Float] = makeEDOFrequencies(
        edo:       31,
        baseHz:    55.0,
        startStep: 0,
        stride:    3
    )

    // MARK: - Lifecycle

    func start() {
        // 1. Locate BlackHole 2ch by name.
        guard let deviceID = coreAudioDeviceID(named: Self.inputDeviceName) else {
            print("[AudioEngine] '\(Self.inputDeviceName)' not found.")
            print("[AudioEngine] Available inputs: \(inputDeviceNames())")
            return
        }

        // 2. Read its native sample rate directly from Core Audio.
        //    BlackHole follows the system output rate — typically 44100 or 48000 Hz.
        let sr: Double = coreAudioNominalSampleRate(for: deviceID) ?? 48_000
        print("[AudioEngine] \(Self.inputDeviceName) (id \(deviceID)) — sr: \(sr) Hz")

        // 3. Get the device UID. AudioQueue uses UIDs (stable CFStrings) to target
        //    a specific device independently of the system default input.
        guard let uid = coreAudioDeviceUID(for: deviceID) else {
            print("[AudioEngine] Failed to get \(Self.inputDeviceName) UID")
            return
        }

        // 4. Request Float32 PCM at the device's native rate. Mono (down-mixed by
        //    AudioQueue) normally; interleaved stereo when stereo-reactive is on.
        let chans: UInt32 = Self.stereo ? 2 : 1
        var fmt = AudioStreamBasicDescription(
            mSampleRate:       sr,
            mFormatID:         kAudioFormatLinearPCM,
            mFormatFlags:      kLinearPCMFormatFlagIsFloat | kLinearPCMFormatFlagIsPacked,
            mBytesPerPacket:   4 * chans,
            mFramesPerPacket:  1,
            mBytesPerFrame:    4 * chans,
            mChannelsPerFrame: chans,
            mBitsPerChannel:   32,
            mReserved:         0
        )

        fft = MicrotonalFFT(bufferSize: 4096, sampleRate: Float(sr))

        // 5. Create the input queue.
        var qRef: AudioQueueRef?
        let createStatus = AudioQueueNewInput(
            &fmt,
            Self.queueCallback,
            Unmanaged.passUnretained(self).toOpaque(),
            nil, nil, 0,
            &qRef
        )
        guard createStatus == noErr, let q = qRef else {
            print("[AudioEngine] AudioQueueNewInput failed: \(createStatus)")
            return
        }

        // 6. Route to BlackHole 2ch by UID — no system default change.
        var cfUID = uid as CFString
        let routeStatus = withUnsafeMutablePointer(to: &cfUID) { ptr in
            AudioQueueSetProperty(
                q,
                kAudioQueueProperty_CurrentDevice,
                ptr,
                UInt32(MemoryLayout<CFString>.size)
            )
        }
        if routeStatus != noErr {
            print("[AudioEngine] Route to \(Self.inputDeviceName) failed: \(routeStatus)")
        }

        // 7. Allocate and prime 3 ring buffers (4096 frames each; ×channels).
        let bufBytes = UInt32(4096 * Int(chans) * MemoryLayout<Float>.size)
        for _ in 0..<3 {
            var buf: AudioQueueBufferRef?
            if AudioQueueAllocateBuffer(q, bufBytes, &buf) == noErr, let b = buf {
                AudioQueueEnqueueBuffer(q, b, 0, nil)
            }
        }

        audioQueue = q
        let startStatus = AudioQueueStart(q, nil)
        if startStatus == noErr {
            print("[AudioEngine] started — AudioQueue → \(Self.inputDeviceName), sr: \(sr) Hz")
        } else {
            print("[AudioEngine] AudioQueueStart failed: \(startStatus)")
        }
    }

    func stop() {
        if let q = audioQueue {
            AudioQueueStop(q, true)
            AudioQueueDispose(q, true)
            audioQueue = nil
        }
        print("[AudioEngine] stopped")
    }

    // MARK: - AudioQueue callback (C convention, HAL thread)

    private static let queueCallback: AudioQueueInputCallback = {
        userData, queue, buffer, _, _, _ in

        // Re-enqueue immediately so the ring never stalls.
        defer { AudioQueueEnqueueBuffer(queue, buffer, 0, nil) }

        guard let ptr = userData else { return }
        let engine = Unmanaged<AudioEngine>.fromOpaque(ptr).takeUnretainedValue()
        guard let fft = engine.fft else { return }

        let floatCount = Int(buffer.pointee.mAudioDataByteSize) / MemoryLayout<Float>.size
        guard floatCount > 0 else { return }
        let samples = buffer.pointee.mAudioData.assumingMemoryBound(to: Float.self)

        if AudioEngine.stereo {
            // De-interleave L/R, FFT each, normalise with one shared gain.
            let frames = floatCount / 2
            guard frames > 0 else { return }
            var left  = [Float](repeating: 0, count: frames)
            var right = [Float](repeating: 0, count: frames)
            for i in 0..<frames { left[i] = samples[2*i]; right[i] = samples[2*i + 1] }
            var qL = left.withUnsafeBufferPointer  { fft.magnitudes(samples: $0.baseAddress!, count: frames, targetHz: engine.targetHz) }
            var qR = right.withUnsafeBufferPointer { fft.magnitudes(samples: $0.baseAddress!, count: frames, targetHz: engine.targetHz) }
            engine.normaliser.normaliseStereo(&qL, &qR)
            DispatchQueue.main.async { engine.onStereoQ?(qL, qR) }
        } else {
            var q = fft.magnitudes(samples: samples, count: floatCount, targetHz: engine.targetHz)
            engine.normaliser.normalise(&q)
            DispatchQueue.main.async { engine.onQ?(q) }
        }
    }

    // MARK: - Core Audio helpers

    private func coreAudioDeviceID(named name: String) -> AudioDeviceID? {
        allDeviceIDs().first { deviceName(for: $0) == name }
    }

    private func coreAudioNominalSampleRate(for id: AudioDeviceID) -> Double? {
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyNominalSampleRate,
            mScope:    kAudioObjectPropertyScopeGlobal,
            mElement:  kAudioObjectPropertyElementMain
        )
        var rate = Float64(0)
        var size = UInt32(MemoryLayout<Float64>.size)
        guard AudioObjectGetPropertyData(id, &addr, 0, nil, &size, &rate) == noErr,
              rate > 1000 else { return nil }
        return rate
    }

    private func coreAudioDeviceUID(for id: AudioDeviceID) -> String? {
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyDeviceUID,
            mScope:    kAudioObjectPropertyScopeGlobal,
            mElement:  kAudioObjectPropertyElementMain
        )
        var cfUID: CFString = "" as CFString
        var size = UInt32(MemoryLayout<CFString>.size)
        let status = withUnsafeMutablePointer(to: &cfUID) {
            AudioObjectGetPropertyData(id, &addr, 0, nil, &size, $0)
        }
        guard status == noErr else { return nil }
        return cfUID as String
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
