// BeatDetector.swift — MilkDrop 3-style beat detection for auto preset switching

import Foundation
import Combine

@MainActor
class BeatDetector: ObservableObject {
    // Published signals
    @Published var hardcutDetected: Bool = false
    @Published var softbeatDetected: Bool = false
    @Published var beatStrength: Double = 0
    @Published var bpm: Double = 0

    // Configuration (mirrors MilkDrop 3 settings)
    var hardcutLowThreshold: Double  = 0.8   // Bass level to trigger hardcut
    var hardcutHighThreshold: Double = 0.5   // Treble level to trigger hardcut
    var hardcutMinDelay: Double      = 3.0   // Minimum seconds between hardcuts
    var hardcutMode: HardcutMode     = .bassAndTreble

    // Internal state
    private var lastHardcut: Date = .distantPast
    private var bassHistory:    RingBuffer<Float>
    private var trebleHistory:  RingBuffer<Float>
    private var beatHistory:    RingBuffer<Date>
    private var prevBass:       Float = 0
    private var bassAttnSmooth: Float = 0

    // BPM tracking
    private var beatTimestamps: [Date] = []

    enum HardcutMode: String, CaseIterable {
        case bass           = "Bass Only"
        case treble         = "Treble Only"
        case bassAndTreble  = "Bass + Treble"
        case bassOrTreble   = "Bass or Treble"
    }

    init() {
        bassHistory   = RingBuffer(capacity: 60)
        trebleHistory = RingBuffer(capacity: 60)
        beatHistory   = RingBuffer(capacity: 8)
    }

    // MARK: - Process each audio frame

    func process(_ data: AudioData) {
        bassHistory.push(data.bass)
        trebleHistory.push(data.treble)

        // Normalized beat strength (0–1)
        let adaptive = adaptiveThreshold(history: bassHistory)
        let strength = adaptive > 0 ? Double(data.bass / adaptive) : 0
        beatStrength = min(strength, 1.0)

        // BPM estimation via beat onset detection
        detectBeat(bass: data.bass, strength: Float(strength))

        // Hardcut evaluation
        checkHardcut(data: data)
    }

    // MARK: - Hardcut detection (MilkDrop 3 style)

    private func checkHardcut(data: AudioData) {
        let now = Date()
        guard now.timeIntervalSince(lastHardcut) >= hardcutMinDelay else { return }

        let bassNorm   = Double(data.bassAttn)
        let trebleNorm = Double(data.treble * 3.0).clamped(to: 0...1)

        let triggered: Bool
        switch hardcutMode {
        case .bass:
            triggered = bassNorm > hardcutLowThreshold
        case .treble:
            triggered = trebleNorm > hardcutHighThreshold
        case .bassAndTreble:
            triggered = bassNorm > hardcutLowThreshold && trebleNorm > hardcutHighThreshold
        case .bassOrTreble:
            triggered = bassNorm > hardcutLowThreshold || trebleNorm > hardcutHighThreshold
        }

        if triggered {
            lastHardcut = now
            hardcutDetected = true
            // Reset after one tick
            Task { @MainActor in
                try? await Task.sleep(for: .milliseconds(50))
                self.hardcutDetected = false
            }
        }
    }

    // MARK: - BPM estimation

    private func detectBeat(bass: Float, strength: Float) {
        guard strength > 1.2 && bass > prevBass * 1.3 else {
            prevBass = bass
            return
        }
        prevBass = bass

        let now = Date()
        beatTimestamps.append(now)
        // Keep last 8 beats
        if beatTimestamps.count > 8 {
            beatTimestamps.removeFirst()
        }

        // Compute BPM from inter-beat intervals
        if beatTimestamps.count >= 2 {
            let intervals = zip(beatTimestamps, beatTimestamps.dropFirst()).map { $1.timeIntervalSince($0) }
            let avgInterval = intervals.reduce(0, +) / Double(intervals.count)
            bpm = avgInterval > 0 ? 60.0 / avgInterval : 0
        }
    }

    // MARK: - Adaptive threshold

    private func adaptiveThreshold(history: RingBuffer<Float>) -> Float {
        let values = history.all()
        guard !values.isEmpty else { return 1 }
        return values.reduce(0, +) / Float(values.count) * 1.5
    }
}

// MARK: - Ring buffer utility

struct RingBuffer<T> {
    private var buffer: [T?]
    private var writeIdx: Int = 0
    let capacity: Int

    init(capacity: Int) {
        self.capacity = capacity
        self.buffer = Array(repeating: nil, count: capacity)
    }

    mutating func push(_ value: T) {
        buffer[writeIdx % capacity] = value
        writeIdx += 1
    }

    func all() -> [T] {
        buffer.compactMap { $0 }
    }

    func latest() -> T? {
        buffer[(writeIdx - 1 + capacity) % capacity]
    }
}

extension Comparable {
    func clamped(to range: ClosedRange<Self>) -> Self {
        min(max(self, range.lowerBound), range.upperBound)
    }
}
