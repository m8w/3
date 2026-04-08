import SwiftUI

struct ContentView: View {

    @StateObject private var audio = AudioEngine()

    var body: some View {
        WebViewContainer(audio: audio)
            .ignoresSafeArea()
            .background(Color.black)
            .onAppear  { audio.start() }
            .onDisappear { audio.stop() }
            // Toggle fullscreen with F (common Milkdrop convention).
            .onKeyPress(.init("f")) {
                NSApp.mainWindow?.toggleFullScreen(nil)
                return .handled
            }
    }
}

#Preview {
    ContentView()
        .frame(width: 960, height: 540)
}
