import SwiftUI

@main
struct ButterchurnVisualizerApp: App {

    var body: some Scene {
        WindowGroup("Butterchurn — Microtonal Visualizer") {
            ContentView()
                // Comfortable default on a Mac Mini driving an external display.
                .frame(minWidth: 960, minHeight: 540)
        }
        // Full-screen chromeless window — the WebView fills everything.
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentSize)
        .commands {
            // ⌘W closes; everything else is handled inside the WebView.
            CommandGroup(replacing: .newItem) {}
        }
    }
}
