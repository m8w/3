// MilkDropMac
// A native macOS MilkDrop 3 music visualizer with Syphon output and BeatDrop-style preset editor
// Architecture: SwiftUI + Metal + projectM + Syphon

import SwiftUI
import AVFoundation

@main
struct MilkDropMacApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @StateObject private var appState = AppState()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
                .frame(minWidth: 1100, minHeight: 720)
        }
        .windowStyle(.hiddenTitleBar)
        .commands {
            CommandMenu("Visualizer") {
                Button("Next Preset") {
                    appState.presetManager.nextPreset()
                }
                .keyboardShortcut(.rightArrow, modifiers: [])

                Button("Previous Preset") {
                    appState.presetManager.previousPreset()
                }
                .keyboardShortcut(.leftArrow, modifiers: [])

                Button("Random Preset") {
                    appState.presetManager.randomPreset()
                }
                .keyboardShortcut("r", modifiers: [])

                Divider()

                Button("Toggle Lock Preset") {
                    appState.isPresetLocked.toggle()
                }
                .keyboardShortcut("l", modifiers: [])

                Button("Toggle Beat Detection Auto-Switch") {
                    appState.beatDetectionEnabled.toggle()
                }
                .keyboardShortcut("b", modifiers: [.command])

                Divider()

                Button("New Double Preset (.milk2)") {
                    appState.createDoublePreset()
                }
                .keyboardShortcut("d", modifiers: [.command, .shift])
            }

            CommandMenu("Presets") {
                Button("Open Preset Editor") {
                    appState.showPresetEditor = true
                }
                .keyboardShortcut("e", modifiers: [.command])

                Button("Import Presets...") {
                    appState.importPresets()
                }
                .keyboardShortcut("i", modifiers: [.command])

                Button("Save Current Preset As...") {
                    appState.savePresetAs()
                }
                .keyboardShortcut("s", modifiers: [.command, .shift])

                Divider()

                Button("Open Preset Folder") {
                    appState.openPresetFolder()
                }
            }

            CommandMenu("Syphon") {
                Button("Start Syphon Output") {
                    appState.syphonEnabled = true
                }
                Button("Stop Syphon Output") {
                    appState.syphonEnabled = false
                }
            }
        }

        Settings {
            SettingsView()
                .environmentObject(appState)
        }
    }
}

class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSWindow.allowsAutomaticWindowTabbing = false
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}
