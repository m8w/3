import Accelerate
import Foundation

// MARK: - Frequency set helpers

/// Build 32 Hz values spread logarithmically across [minHz, maxHz].
/// Use this as a starting point, then tweak for your specific microtonal tuning.
public func makeLogFrequencies(minHz: Float = 30, maxHz: Float = 16_000, count: Int = 32) -> [Float] {
    let logMin = log2(minHz)
    let logMax = log2(maxHz)
    return (0..<count).map { i in
        pow(2, logMin + Float(i) / Float(count - 1) * (logMax - logMin))
    }
}

/// Build 32 Hz values from an EDO (equal-division-of-the-octave) tuning.
/// - Parameters:
///   - edo:        Divisions per octave (e.g. 31, 41, 53, 72).
///   - baseHz:     Reference pitch for step 0 (e.g. 55.0 for A1).
///   - startStep:  First step index relative to baseHz.
///   - stride:     How many EDO steps between each of the 32 bins.
///                 stride = 1 → chromatic; stride = 2 → every other step, etc.
public func makeEDOFrequencies(
    edo: Int,
    baseHz: Float = 55.0,
    startStep: Int = 0,
    stride: Int = 3,
    count: Int = 32
) -> [Float] {
    let ratio = pow(2.0, 1.0 / Float(edo))
    return (0..<count).map { i in
        baseHz * pow(ratio, Float(startStep + i * stride))
    }
}

// MARK: - Core FFT processor

/// Extracts 32 magnitudes at arbitrary Hz targets from a raw mono PCM buffer.
/// Designed for Milkdrop-style visualizers using microtonal tuning systems.
///
/// Usage:
/// ```swift
/// let fft = MicrotonalFFT(bufferSize: 4096, sampleRate: 44100)
/// let q = fft.magnitudes(samples: bufferPointer, count: 4096, targetHz: myFreqs)
/// // q[0]…q[31] → pass as q1…q32 uniforms
/// ```
public final class MicrotonalFFT {

    // MARK: - Public constants

    /// Default 32 frequencies covering the audible range logarithmically.
    /// Swap these out for your EDO scale with makeEDOFrequencies(…).
    public static let defaultFrequencies: [Float] = makeLogFrequencies()

    // MARK: - Private state

    private let bufferSize: Int
    private let halfSize:   Int
    private let log2n:      vDSP_Length
    private let sampleRate: Float
    private let fftSetup:   FFTSetup

    /// Hann window coefficients — reduces spectral leakage.
    private var window:      [Float]
    /// Intermediate real/imag split-complex buffers.
    private var realBuf:     [Float]
    private var imagBuf:     [Float]
    /// Power spectrum (magnitude²) output.
    private var powerBuf:    [Float]

    // MARK: - Init / deinit

    /// - Parameters:
    ///   - bufferSize: Must be a power of two (e.g. 2048, 4096).
    ///                 Larger → finer frequency resolution; smaller → lower latency.
    ///                 Frequency resolution = sampleRate / bufferSize.
    ///   - sampleRate: Audio sample rate in Hz (typically 44100 or 48000).
    public init(bufferSize: Int, sampleRate: Float) {
        precondition(bufferSize.nonzeroBitCount == 1, "bufferSize must be a power of two")

        self.bufferSize = bufferSize
        self.halfSize   = bufferSize / 2
        self.log2n      = vDSP_Length(log2(Double(bufferSize)))
        self.sampleRate = sampleRate

        guard let setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            fatalError("vDSP_create_fftsetup failed — check bufferSize is a power of two")
        }
        self.fftSetup = setup

        // Normalised Hann window: sum(w²) = 1, preserves RMS amplitude.
        window   = [Float](repeating: 0, count: bufferSize)
        vDSP_hann_window(&window, vDSP_Length(bufferSize), Int32(vDSP_HANN_NORM))

        realBuf  = [Float](repeating: 0, count: halfSize)
        imagBuf  = [Float](repeating: 0, count: halfSize)
        powerBuf = [Float](repeating: 0, count: halfSize)
    }

    deinit {
        vDSP_destroy_fftsetup(fftSetup)
    }

    // MARK: - Public API

    /// Compute 32 normalised magnitudes at `targetHz` from a raw mono PCM buffer.
    ///
    /// - Parameters:
    ///   - samples:   Pointer to interleaved Float32 mono samples.
    ///   - count:     Number of valid samples (≤ bufferSize).
    ///   - targetHz:  Exactly 32 target frequencies in Hz. Defaults to a
    ///                log-spaced set. Pass makeEDOFrequencies(…) for microtonal use.
    ///   - peakHold:  When > 0, each bin decays by this factor per call instead of
    ///                being replaced. Values 0.7–0.9 give smooth Milkdrop-style hold.
    /// - Returns:     32 Floats in [0, 1] ready to bind as q1…q32 shader uniforms.
    @discardableResult
    public func magnitudes(
        samples:   UnsafePointer<Float>,
        count:     Int,
        targetHz:  [Float] = MicrotonalFFT.defaultFrequencies,
        peakHold:  Float   = 0.0
    ) -> [Float] {
        precondition(targetHz.count == 32, "targetHz must contain exactly 32 values")

        // ── 1. Window ──────────────────────────────────────────────────────────
        var windowed = [Float](repeating: 0, count: bufferSize)
        let n = min(count, bufferSize)
        // Multiply input by window; zero-pad the tail if count < bufferSize.
        vDSP_vmul(samples, 1, window, 1, &windowed, 1, vDSP_Length(n))

        // ── 2. Pack into split-complex and forward FFT ─────────────────────────
        // vDSP_fft_zrip operates on a split-complex view of the real input:
        //   realp[k] ← even-indexed samples, imagp[k] ← odd-indexed samples
        windowed.withUnsafeBufferPointer { wPtr in
            realBuf.withUnsafeMutableBufferPointer { rPtr in
                imagBuf.withUnsafeMutableBufferPointer { iPtr in
                    var split = DSPSplitComplex(realp: rPtr.baseAddress!,
                                                imagp: iPtr.baseAddress!)
                    // Reinterpret pairs of floats as complex numbers.
                    wPtr.baseAddress!.withMemoryRebound(
                        to: DSPComplex.self,
                        capacity: halfSize
                    ) { cPtr in
                        vDSP_ctoz(cPtr, 2, &split, 1, vDSP_Length(halfSize))
                    }
                    vDSP_fft_zrip(fftSetup, &split, 1, log2n, FFTDirection(FFT_FORWARD))
                }
            }
        }

        // ── 3. Power spectrum ─────────────────────────────────────────────────
        // vDSP_zvmags writes |Z|² = re² + im² for each bin.
        realBuf.withUnsafeBufferPointer { rPtr in
            imagBuf.withUnsafeBufferPointer { iPtr in
                var split = DSPSplitComplex(
                    realp: UnsafeMutablePointer(mutating: rPtr.baseAddress!),
                    imagp: UnsafeMutablePointer(mutating: iPtr.baseAddress!)
                )
                vDSP_zvmags(&split, 1, &powerBuf, 1, vDSP_Length(halfSize))
            }
        }

        // ── 4. Scale: compensate for Hann window and one-sided FFT ────────────
        // Factor = (2 / N)² so that amplitude ≈ 1 for a full-scale sine.
        // (The ² lives inside powerBuf since we haven't taken √ yet.)
        var scale = pow(2.0 / Float(bufferSize), 2.0)
        vDSP_vsmul(powerBuf, 1, &scale, &powerBuf, 1, vDSP_Length(halfSize))

        // Fix DC (bin 0) and Nyquist (packed into imagBuf[0] by vDSP_fft_zrip).
        // These bins are not doubled in the one-sided spectrum.
        powerBuf[0] *= 0.25

        // ── 5. Extract amplitude at each target frequency ─────────────────────
        let freqRes = sampleRate / Float(bufferSize)
        let nyquist = sampleRate * 0.5

        return targetHz.map { hz in
            guard hz > 0, hz < nyquist else { return 0 }

            // Fractional bin index for this frequency.
            let exactBin = hz / freqRes
            let lo = Int(exactBin)                         // floor bin
            let hi = min(lo + 1, halfSize - 1)             // ceil bin
            let frac = exactBin - Float(lo)

            // Amplitude = √(power).  Linearly interpolate between adjacent bins
            // to avoid aliasing artifacts at non-integer bin positions.
            let ampLo = sqrt(powerBuf[lo])
            let ampHi = sqrt(powerBuf[hi])
            let amp   = ampLo + frac * (ampHi - ampLo)

            return max(0, amp)
        }
    }
}

// MARK: - Running normaliser (optional)

/// Keeps a rolling maximum so magnitudes are auto-scaled to [0, 1]
/// without knowing the signal level in advance.  Feed the raw output of
/// MicrotonalFFT.magnitudes(…) through this before binding to q1…q32.
public final class AdaptiveNormaliser {

    private var runningMax: Float
    private let decayRate:  Float   // per-call decay (e.g. 0.999)
    private let floor:      Float   // minimum ceiling to avoid div/zero on silence

    public init(decayRate: Float = 0.999, floor: Float = 1e-4) {
        self.runningMax = floor
        self.decayRate  = decayRate
        self.floor      = floor
    }

    /// Normalise `values` in-place and return them.
    @discardableResult
    public func normalise(_ values: inout [Float]) -> [Float] {
        // Find this frame's peak.
        var frameMax: Float = 0
        vDSP_maxv(values, 1, &frameMax, vDSP_Length(values.count))

        // Let the running max rise instantly, decay slowly.
        runningMax = max(runningMax * decayRate, max(frameMax, floor))

        // Scale all values into [0, 1].
        var inv = 1.0 / runningMax
        vDSP_vsmul(values, 1, &inv, &values, 1, vDSP_Length(values.count))

        return values
    }
}
