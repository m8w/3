// AudioEngine.swift — Captures audio and produces FFT + waveform data for the visualizer

import Foundation
@preconcurrency import AVFoundation
import Accelerate
import Combine

// MARK: - Audio data packet sent each frame

struct AudioData {
    var waveform:  [Float]   // Raw PCM samples (512 samples)
    var spectrum:  [Float]   // FFT magnitude bins (256 bins, 0–22kHz)
    var bass:      Float     // 20–200 Hz RMS
    var mid:       Float     // 200–2000 Hz RMS
    var treble:    Float     // 2–20 kHz RMS
    var rms:       Float     // Overall RMS level
    var bassLevel: Float     // Beat detection band (normalized 0–1)
    var bassAttn:  Float     // Attenuated bass for beat detection

    static let silence = AudioData(
        waveform:  Array(repeating: 0, count: 512),
        spectrum:  Array(repeating: 0, count: 256),
        bass: 0, mid: 0, treble: 0,
        rms: 0, bassLevel: 0, bassAttn: 0
    )
}

// MARK: - Audio engine

@MainActor
class AudioEngine: ObservableObject {
    @Published var audioData: AudioData = .silence
    @Published var isRunning: Bool = false
    @Published var availableDevices: [AVCaptureDevice] = []

    // FFT configuration — marked nonisolated(unsafe) so they can be read from the
    // background audioQueue without hopping to the main actor. Only ever written
    // during init (fftSetup, window) or configureSession (formatConverter), so
    // there is no concurrent write hazard.
    private let fftSize  = 1024
    private let hopSize  = 512
    nonisolated(unsafe) private var fftSetup: FFTSetup?
    nonisolated(unsafe) private var window:   [Float] = []

    // Ring buffer — only touched from audioQueue (serial), safe as nonisolated(unsafe)
    nonisolated(unsafe) private var ringBuffer: [Float]
    nonisolated(unsafe) private var ringWrite:  Int = 0

    // AVAudioEngine pipeline
    private let engine = AVAudioEngine()
    private var inputNode: AVAudioInputNode { engine.inputNode }
    nonisolated(unsafe) private var formatConverter: AVAudioConverter?

    // Serial background queue — all heavy audio work runs here, never on main
    private let audioQueue = DispatchQueue(label: "milkdrop.audio", qos: .userInteractive)

    // Smoothing — only touched from audioQueue
    nonisolated(unsafe) private var bassSmooth:   Float = 0
    nonisolated(unsafe) private var midSmooth:    Float = 0
    nonisolated(unsafe) private var trebleSmooth: Float = 0
    nonisolated(unsafe) private var rmsSmooth:    Float = 0
    private let smoothFactor: Float = 0.7

    init() {
        ringBuffer = Array(repeating: 0, count: fftSize * 4)
        setupFFT()
        buildHannWindow()
        fetchDevices()
    }

    // MARK: - Setup

    private func setupFFT() {
        let log2n = vDSP_Length(log2(Double(fftSize)))
        fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(FFT_RADIX2))
    }

    private func buildHannWindow() {
        window = (0..<fftSize).map { i in
            let n = Float(i)
            let N = Float(fftSize)
            return 0.5 * (1 - cos(2 * .pi * n / (N - 1)))
        }
    }

    private func fetchDevices() {
        var types: [AVCaptureDevice.DeviceType] = [.microphone]
        if #available(macOS 14.0, *) {
            types.append(.external)
        } else {
            types.append(.externalUnknown)
        }
        availableDevices = AVCaptureDevice.DiscoverySession(
            deviceTypes: types,
            mediaType: .audio,
            position: .unspecified
        ).devices
    }

    // MARK: - Start / Stop

    func start(source: AudioSource = .systemDefault) {
        guard !isRunning else { return }
        do {
            try configureSession(source: source)
            try engine.start()
            isRunning = true
        } catch {
            print("[AudioEngine] Failed to start: \(error)")
        }
    }

    func stop() {
        engine.stop()
        inputNode.removeTap(onBus: 0)
        isRunning = false
    }

    private func configureSession(source: AudioSource) throws {
        inputNode.removeTap(onBus: 0)

        // Must tap at the input node's NATIVE format — AVAudioEngine does not auto-convert.
        // AVAudioConverter handles the resample/channel-mix to mono float32 44100 Hz.
        let inputFormat = inputNode.inputFormat(forBus: 0)
        let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 44100,
            channels: 1,
            interleaved: false
        )!
        formatConverter = AVAudioConverter(from: inputFormat, to: targetFormat)

        // The tap fires on the audio render thread. Dispatch to audioQueue so
        // conversion + FFT run on a background thread, never on the main thread.
        inputNode.installTap(onBus: 0, bufferSize: 512, format: inputFormat) { [weak self] buffer, _ in
            guard let self else { return }
            self.audioQueue.async {
                self.processTap(buffer: buffer, targetFormat: targetFormat)
            }
        }

        engine.prepare()
    }

    // MARK: - Audio processing (runs on audioQueue — NOT main actor)

    // nonisolated so the method body executes on the calling queue (audioQueue)
    // rather than hopping to the @MainActor. State accessed here is either
    // nonisolated(unsafe) or local.
    nonisolated private func processTap(buffer: AVAudioPCMBuffer, targetFormat: AVAudioFormat) {
        guard let converter = formatConverter else { return }

        let frameCount = AVAudioFrameCount(buffer.frameLength)
        guard let converted = AVAudioPCMBuffer(pcmFormat: targetFormat,
                                               frameCapacity: max(frameCount * 2, 1024)),
              let channelData = converted.floatChannelData else { return }

        var error: NSError?
        converter.convert(to: converted, error: &error) { _, outStatus in
            outStatus.pointee = .haveData
            return buffer
        }
        guard error == nil else { return }

        let sampleCount = Int(converted.frameLength)
        let samples = Array(UnsafeBufferPointer(start: channelData[0], count: sampleCount))

        for sample in samples {
            ringBuffer[ringWrite % ringBuffer.count] = sample
            ringWrite += 1
        }

        guard ringWrite >= hopSize else { return }

        let data = computeAudioData()

        // Publish on main using async — defers to the next run loop iteration so
        // we never publish inside a SwiftUI view-update pass.
        DispatchQueue.main.async { [weak self] in
            self?.audioData = data
        }
    }

    nonisolated private func computeAudioData() -> AudioData {
        let start = (ringWrite - fftSize + ringBuffer.count) % ringBuffer.count
        var samples = [Float](repeating: 0, count: fftSize)
        for i in 0..<fftSize {
            samples[i] = ringBuffer[(start + i) % ringBuffer.count]
        }

        let waveform = Array(samples.prefix(512))

        var windowed = [Float](repeating: 0, count: fftSize)
        vDSP_vmul(samples, 1, window, 1, &windowed, 1, vDSP_Length(fftSize))

        let halfN = fftSize / 2
        var realPart  = [Float](repeating: 0, count: halfN)
        var imagPart  = [Float](repeating: 0, count: halfN)
        var magnitudes = [Float](repeating: 0, count: halfN)
        realPart.withUnsafeMutableBufferPointer { realBuf in
            imagPart.withUnsafeMutableBufferPointer { imagBuf in
                var splitComplex = DSPSplitComplex(realp: realBuf.baseAddress!,
                                                   imagp: imagBuf.baseAddress!)
                windowed.withUnsafeBytes { ptr in
                    let complexPtr = ptr.bindMemory(to: DSPComplex.self)
                    vDSP_ctoz(complexPtr.baseAddress!, 2, &splitComplex, 1, vDSP_Length(halfN))
                }
                let log2n = vDSP_Length(log2(Double(fftSize)))
                vDSP_fft_zrip(fftSetup!, &splitComplex, 1, log2n, FFTDirection(FFT_FORWARD))
                vDSP_zvabs(&splitComplex, 1, &magnitudes, 1, vDSP_Length(halfN))
            }
        }

        var scale: Float = 2.0 / Float(fftSize)
        vDSP_vsmul(magnitudes, 1, &scale, &magnitudes, 1, vDSP_Length(halfN))

        let spectrum = reduceSpectrum(magnitudes, outputBins: 256)

        let binHz   = 44100.0 / Double(fftSize)
        let bassEnd = Int(200.0  / binHz)
        let midEnd  = Int(2000.0 / binHz)

        let bassRaw   = rmsOf(magnitudes[1..<min(bassEnd, halfN)])
        let midRaw    = rmsOf(magnitudes[bassEnd..<min(midEnd, halfN)])
        let trebleRaw = rmsOf(magnitudes[midEnd..<halfN])
        let rmsRaw    = rmsOf(samples)

        bassSmooth   = bassSmooth   * smoothFactor + bassRaw   * (1 - smoothFactor)
        midSmooth    = midSmooth    * smoothFactor + midRaw    * (1 - smoothFactor)
        trebleSmooth = trebleSmooth * smoothFactor + trebleRaw * (1 - smoothFactor)
        rmsSmooth    = rmsSmooth    * smoothFactor + rmsRaw    * (1 - smoothFactor)

        let bassLevel = min(bassSmooth * 3.0, 1.0)
        let bassAttn  = max(bassLevel - 0.4, 0) / 0.6

        return AudioData(
            waveform:  waveform,
            spectrum:  spectrum,
            bass:      bassSmooth,
            mid:       midSmooth,
            treble:    trebleSmooth,
            rms:       rmsSmooth,
            bassLevel: bassLevel,
            bassAttn:  bassAttn
        )
    }

    nonisolated private func reduceSpectrum(_ input: [Float], outputBins: Int) -> [Float] {
        var output = [Float](repeating: 0, count: outputBins)
        let n = input.count
        let logMin = log2(1.0)
        let logMax = log2(Double(n))
        for i in 0..<outputBins {
            let lo = pow(2.0, logMin + (logMax - logMin) * Double(i)     / Double(outputBins))
            let hi = pow(2.0, logMin + (logMax - logMin) * Double(i + 1) / Double(outputBins))
            let startIdx = max(1, Int(lo))
            let endIdx   = min(n, Int(ceil(hi)))
            guard endIdx > startIdx else { output[i] = input[startIdx]; continue }
            var sum: Float = 0
            for j in startIdx..<endIdx { sum += input[j] }
            output[i] = sum / Float(endIdx - startIdx)
        }
        return output
    }

    nonisolated private func rmsOf<S: Collection>(_ samples: S) -> Float where S.Element == Float {
        guard !samples.isEmpty else { return 0 }
        var sumSq: Float = 0
        let arr = Array(samples)
        vDSP_svesq(arr, 1, &sumSq, vDSP_Length(arr.count))
        return sqrt(sumSq / Float(arr.count))
    }

    deinit {
        fftSetup.map { vDSP_destroy_fftsetup($0) }
    }
}
