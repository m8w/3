//
//  ConsistentMorphSettingsPanel.swift
//  metal_video_morpher
//
//  Drop-in SwiftUI panel for the spacetime-consistent morph pipeline.
//  Add it to your existing ContentView with:
//
//      ConsistentMorphSettingsPanel(processor: $processor)
//
//  where `processor` is a @StateObject ConsistentMorphProcessorVM (below).
//

import SwiftUI
import Combine

/// View model wrapping ConsistentMorphProcessor so SwiftUI can bind to it.
final class ConsistentMorphProcessorVM: ObservableObject {
    let processor: ConsistentMorphProcessor

    @Published var morphStrength: Float = 0.7   { didSet { processor.morphStrength = morphStrength } }
    @Published var flowWeight: Float    = 1.0   { didSet { processor.flowWeight    = flowWeight } }
    @Published var ffdWeight: Float     = 1.0   { didSet { processor.ffdWeight     = ffdWeight } }
    @Published var tetWeight: Float     = 1.0   { didSet { processor.tetWeight     = tetWeight } }
    @Published var useFlow: Bool        = true  { didSet { processor.useFlow       = useFlow } }
    @Published var keypointCount: Int   = 32
    @Published var latticePreset: LatticePreset = .identity {
        didSet {
            switch latticePreset {
            case .identity:  processor.bezierLattice = .identity()
            case .breathing: processor.bezierLattice = .breathing(amplitude: 0.05)
            }
        }
    }

    enum LatticePreset: String, CaseIterable, Identifiable {
        case identity, breathing
        var id: String { rawValue }
    }

    init() {
        self.processor = (try? ConsistentMorphProcessor())!
    }
}

struct ConsistentMorphSettingsPanel: View {
    @ObservedObject var vm: ConsistentMorphProcessorVM

    var body: some View {
        GroupBox("Spacetime-Consistent Morph") {
            VStack(alignment: .leading, spacing: 10) {

                Group {
                    LabeledSlider(label: "Morph Strength",
                                  value: $vm.morphStrength, range: 0...2)
                    LabeledSlider(label: "FFD Weight",
                                  value: $vm.ffdWeight, range: 0...2)
                    LabeledSlider(label: "Tetra Weight",
                                  value: $vm.tetWeight, range: 0...2)
                    LabeledSlider(label: "Flow Weight",
                                  value: $vm.flowWeight, range: 0...2)
                }

                Toggle("Use Optical Flow Advection", isOn: $vm.useFlow)

                Picker("Lattice Preset", selection: $vm.latticePreset) {
                    ForEach(ConsistentMorphProcessorVM.LatticePreset.allCases) { preset in
                        Text(preset.rawValue.capitalized).tag(preset)
                    }
                }
                .pickerStyle(.segmented)

                Stepper(value: $vm.keypointCount, in: 4...256, step: 4) {
                    Text("Spacetime Keypoints: \(vm.keypointCount)")
                }
            }
            .padding(8)
        }
    }
}

private struct LabeledSlider: View {
    let label: String
    @Binding var value: Float
    let range: ClosedRange<Float>

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(label).font(.caption)
                Spacer()
                Text(String(format: "%.2f", value)).font(.caption.monospaced())
            }
            Slider(value: Binding(
                get: { Double(value) },
                set: { value = Float($0) }
            ), in: Double(range.lowerBound)...Double(range.upperBound))
        }
    }
}
