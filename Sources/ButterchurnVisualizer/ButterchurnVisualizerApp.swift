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
            // Opens ~720p; freely resizable so you can size it for OBS window
            // capture and keep the rest of the Mac for other work.
            .frame(minWidth: 640, idealWidth: 1280, minHeight: 360, idealHeight: 720)
        }
        // A normal titled window: freely movable/resizable, and OBS can find it
        // by name ("Butterchurn — Microtonal Visualizer") in Window Capture.
        .windowResizability(.contentMinSize)
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
