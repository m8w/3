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

    /// Called on the main thread with 32 normalised [0,1] magnitudes.
    var onQ: (([Float]) -> Void)?

    static let inputDeviceName = "BlackHole 2ch"

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

        // 4. Request Float32 mono PCM at BlackHole's native rate.
        //    AudioQueue de-interleaves / down-mixes from the 2ch hardware.
        var fmt = AudioStreamBasicDescription(
            mSampleRate:       sr,
            mFormatID:         kAudioFormatLinearPCM,
            mFormatFlags:      kLinearPCMFormatFlagIsFloat | kLinearPCMFormatFlagIsPacked,
            mBytesPerPacket:   4,
            mFramesPerPacket:  1,
            mBytesPerFrame:    4,
            mChannelsPerFrame: 1,
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

        // 7. Allocate and prime 3 ring buffers (~93 ms each at 44100 Hz).
        let bufBytes = UInt32(4096 * MemoryLayout<Float>.size)
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

        // For Float32 mono PCM each frame = 4 bytes.
        let frameCount = Int(buffer.pointee.mAudioDataByteSize) / MemoryLayout<Float>.size
        guard frameCount > 0 else { return }

        let samples = buffer.pointee.mAudioData.assumingMemoryBound(to: Float.self)
        var q = fft.magnitudes(samples: samples, count: frameCount, targetHz: engine.targetHz)
        engine.normaliser.normalise(&q)
        DispatchQueue.main.async { engine.onQ?(q) }
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
