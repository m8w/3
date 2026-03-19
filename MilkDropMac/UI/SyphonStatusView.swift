// SyphonStatusView.swift — Syphon server status and configuration

import SwiftUI

struct SyphonStatusView: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                // Status
                ParameterSection(title: "Syphon Server") {
                    HStack {
                        Circle()
                            .fill(state.syphonEnabled ? Color.green : Color.red)
                            .frame(width: 10, height: 10)
                        Text(state.syphonEnabled ? "Active" : "Stopped")
                            .font(.system(size: 12))
                        Spacer()
                        Toggle("", isOn: $state.syphonEnabled)
                            .labelsHidden()
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 6)

                    // Server name
                    HStack {
                        Text("Server Name")
                            .font(.system(size: 11))
                            .foregroundColor(.secondary)
                            .frame(width: 90, alignment: .trailing)
                        Text("MilkDropMac")
                            .font(.system(size: 12, design: .monospaced))
                            .foregroundColor(.primary)
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                }

                // How to use
                ParameterSection(title: "Compatible Apps") {
                    VStack(alignment: .leading, spacing: 6) {
                        SyphonAppRow(name: "VDMX", description: "Professional VJ software")
                        SyphonAppRow(name: "Resolume", description: "VJ & media server")
                        SyphonAppRow(name: "TouchDesigner", description: "Interactive media")
                        SyphonAppRow(name: "MadMapper", description: "Video mapping")
                        SyphonAppRow(name: "Max/MSP (Jitter)", description: "Creative coding")
                        SyphonAppRow(name: "OBS Studio", description: "Streaming & recording")
                        SyphonAppRow(name: "Unreal Engine", description: "Game engine / XR")
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 6)
                }

                // Instructions
                ParameterSection(title: "Setup") {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("1. Ensure MilkDropMac is running with Syphon enabled")
                        Text("2. In your VJ app, add a Syphon input source")
                        Text("3. Select \"MilkDropMac\" from the server list")
                        Text("4. The visualizer will stream at full resolution with zero latency")
                    }
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 6)
                }

                // Framework note
                ParameterSection(title: "Framework") {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Syphon.framework required:")
                            .font(.system(size: 11, weight: .medium))
                        Link("Download from GitHub",
                             destination: URL(string: "https://github.com/Syphon/Syphon-Framework/releases")!)
                            .font(.system(size: 11))
                        Text("Add to Xcode → Frameworks, Libraries, Embedded Content → Embed & Sign")
                            .font(.system(size: 10))
                            .foregroundColor(.secondary)
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 6)
                }
            }
        }
        .background(Color(hex: "0d0d0d"))
    }
}

struct SyphonAppRow: View {
    let name: String
    let description: String

    var body: some View {
        HStack {
            Circle()
                .fill(Color.accentColor.opacity(0.7))
                .frame(width: 6, height: 6)
            Text(name)
                .font(.system(size: 12, weight: .medium))
            Text("— \(description)")
                .font(.system(size: 11))
                .foregroundColor(.secondary)
        }
    }
}
