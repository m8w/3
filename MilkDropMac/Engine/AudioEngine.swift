// AudioEngine.swift — Captures audio and produces FFT + waveform data for the visualizer

import Foundation
import AVFoundation
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

    // FFT configuration
    private let fftSize   = 1024
    private let hopSize   = 512
    private var fftSetup: FFTSetup?
    private var window:   [Float] = []

    // Ring buffer for audio samples
    private var ringBuffer: [Float]
    private var ringWrite: Int = 0

    // AVAudioEngine pipeline
    private let engine = AVAudioEngine()
    private var inputNode: AVAudioInputNode { engine.inputNode }
    private var formatConverter: AVAudioConverter?

    // Processing queues
    private let audioQueue = DispatchQueue(label: "milkdrop.audio", qos: .userInteractive)

    // Smoothing
    private var bassSmooth:   Float = 0
    private var midSmooth:    Float = 0
    private var trebleSmooth: Float = 0
    private var rmsSmooth:    Float = 0
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
        availableDevices = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.microphone, .externalUnknown],
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
        // Remove existing tap
        inputNode.removeTap(onBus: 0)

        let inputFormat = inputNode.inputFormat(forBus: 0)

        // Target format: mono, 44100 Hz, float32
        let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 44100,
            channels: 1,
            interleaved: false
        )!

        formatConverter = AVAudioConverter(from: inputFormat, to: targetFormat)

        inputNode.installTap(onBus: 0, bufferSize: 512, format: inputFormat) { [weak self] buffer, _ in
            guard let self else { return }
            let frameLength = buffer.frameLength
            let frameCapacity = buffer.frameCapacity
            let format = buffer.format
            // Copy PCM data to avoid capturing non-Sendable AVAudioPCMBuffer
            guard let copy = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCapacity),
                  let srcData = buffer.floatChannelData,
                  let dstData = copy.floatChannelData else { return }
            copy.frameLength = frameLength
            let channelCount = Int(format.channelCount)
            for ch in 0..<channelCount {
                memcpy(dstData[ch], srcData[ch], Int(frameLength) * MemoryLayout<Float>.size)
            }
            self.audioQueue.async { [weak self] in
                Task { @MainActor [weak self] in
                    self?.processTap(buffer: copy, targetFormat: targetFormat)
                }
            }
        }

        engine.prepare()
    }

    // MARK: - Audio processing

    private func processTap(buffer: AVAudioPCMBuffer, targetFormat: AVAudioFormat) {
        guard let converter = formatConverter else { return }

        // Convert to mono float32
        let frameCount = AVAudioFrameCount(buffer.frameLength)
        guard let converted = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: frameCount),
              let channelData = converted.floatChannelData
        else { return }

        var error: NSError?
        converter.convert(to: converted, error: &error) { _, outStatus in
            outStatus.pointee = .haveData
            return buffer
        }
        guard error == nil else { return }

        converted.frameLength = frameCount
        let samples = Array(UnsafeBufferPointer(start: channelData[0], count: Int(frameCount)))

        // Write to ring buffer
        for sample in samples {
            ringBuffer[ringWrite % ringBuffer.count] = sample
            ringWrite += 1
        }

        // Compute FFT when we have enough new data
        if ringWrite >= hopSize {
            let data = computeAudioData()
            Task { @MainActor in
                self.audioData = data
            }
        }
    }

    private func computeAudioData() -> AudioData {
        // Extract fftSize samples from ring buffer
        let start = (ringWrite - fftSize + ringBuffer.count) % ringBuffer.count
        var samples = [Float](repeating: 0, count: fftSize)
        for i in 0..<fftSize {
            samples[i] = ringBuffer[(start + i) % ringBuffer.count]
        }

        // Waveform (first 512 samples, normalized)
        let waveform = Array(samples.prefix(512))

        // Apply Hann window
        var windowed = [Float](repeating: 0, count: fftSize)
        vDSP_vmul(samples, 1, window, 1, &windowed, 1, vDSP_Length(fftSize))

        // Forward FFT
        let halfN = fftSize / 2
        var realPart = [Float](repeating: 0, count: halfN)
        var imagPart = [Float](repeating: 0, count: halfN)
        var magnitudes = [Float](repeating: 0, count: halfN)
        realPart.withUnsafeMutableBufferPointer { realBuf in
            imagPart.withUnsafeMutableBufferPointer { imagBuf in
                var splitComplex = DSPSplitComplex(realp: realBuf.baseAddress!, imagp: imagBuf.baseAddress!)

                windowed.withUnsafeBytes { ptr in
                    let complexPtr = ptr.bindMemory(to: DSPComplex.self)
                    vDSP_ctoz(complexPtr.baseAddress!, 2, &splitComplex, 1, vDSP_Length(halfN))
                }

                let log2n = vDSP_Length(log2(Double(fftSize)))
                vDSP_fft_zrip(fftSetup!, &splitComplex, 1, log2n, FFTDirection(FFT_FORWARD))

                vDSP_zvabs(&splitComplex, 1, &magnitudes, 1, vDSP_Length(halfN))
            }
        }

        // Scale
        var scale: Float = 2.0 / Float(fftSize)
        vDSP_vsmul(magnitudes, 1, &scale, &magnitudes, 1, vDSP_Length(halfN))

        // Reduce to 256 bins (logarithmic grouping for perceptual scaling)
        let spectrum = reduceSpectrum(magnitudes, outputBins: 256)

        // Frequency bands (44100 Hz, halfN bins → sampleRate/2 Hz at bin halfN)
        let binHz = 44100.0 / Double(fftSize)
        let bassEnd   = Int(200.0  / binHz)
        let midEnd    = Int(2000.0 / binHz)

        let bassRaw   = rms(magnitudes[1..<min(bassEnd, halfN)])
        let midRaw    = rms(magnitudes[bassEnd..<min(midEnd, halfN)])
        let trebleRaw = rms(magnitudes[midEnd..<halfN])
        let rmsRaw    = rms(samples)

        // Smooth
        bassSmooth   = bassSmooth   * smoothFactor + bassRaw   * (1 - smoothFactor)
        midSmooth    = midSmooth    * smoothFactor + midRaw    * (1 - smoothFactor)
        trebleSmooth = trebleSmooth * smoothFactor + trebleRaw * (1 - smoothFactor)
        rmsSmooth    = rmsSmooth    * smoothFactor + rmsRaw    * (1 - smoothFactor)

        // Attenuated bass for beat detection (slower attack, fast decay like MilkDrop)
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

    private func reduceSpectrum(_ input: [Float], outputBins: Int) -> [Float] {
        var output = [Float](repeating: 0, count: outputBins)
        let n = input.count
        // Skip DC bin (index 0); map [1..n] → log scale across outputBins
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

    private func rms<S: Collection>(_ samples: S) -> Float where S.Element == Float {
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
