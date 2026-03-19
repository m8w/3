// SyphonOutput.swift — Swift-friendly Syphon output manager
// Manages the Syphon server lifecycle and status

import Foundation
import Metal
import Combine

@MainActor
class SyphonOutput: ObservableObject {
    @Published var isRunning: Bool = false
    @Published var clientCount: Int = 0
    @Published var serverName: String = "MilkDropMac"

    private var bridge: SyphonBridgeWrapper?
    private var statusTimer: AnyCancellable?

    init(device: MTLDevice) {
        start(device: device)
    }

    func start(device: MTLDevice) {
        bridge = SyphonBridgeWrapper(device: device, name: serverName)
        isRunning = bridge?.isRunning ?? false

        // Poll client count
        statusTimer = Timer.publish(every: 2, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in
                guard let self, let b = self.bridge else { return }
                self.clientCount = Int(b.clientCount)
                self.isRunning   = b.isRunning
            }
    }

    func stop() {
        bridge?.stop()
        isRunning   = false
        clientCount = 0
        statusTimer?.cancel()
    }

    func publish(texture: MTLTexture, commandBuffer: MTLCommandBuffer) {
        bridge?.publishTexture(texture, commandBuffer: commandBuffer)
    }

    func updateName(_ name: String) {
        serverName = name
        bridge?.setName(name)
    }
}
