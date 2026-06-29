import SwiftUI

@main
struct ButterchurnVisualizerApp: App {

    // Curate mode is active when CURATE_INPUT / CURATE=1 is in the environment.
    @StateObject private var curator = CuratorBox()
    @StateObject private var broadcaster = Broadcaster()

    var body: some Scene {
        WindowGroup("Butterchurn — Microtonal Visualizer") {
            Group {
                if let curator = curator.instance {
                    CurateView(curator: curator)
                } else {
                    ContentView()
                }
            }
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

        // Broadcast controls live in the menu bar so they're never captured into
        // the stream. URL/key + Go Live / Stop.
        MenuBarExtra("Broadcast",
                     systemImage: broadcaster.isLive ? "dot.radiowaves.left.and.right"
                                                     : "antenna.radiowaves.left.and.right") {
            BroadcastPanel(broadcaster: broadcaster)
        }
        .menuBarExtraStyle(.window)
    }
}
