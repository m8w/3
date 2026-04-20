import SwiftUI
import AppKit

struct ContentView: View {

    @StateObject private var audio = AudioEngine()

    var body: some View {
        WebViewContainer(audio: audio)
            .ignoresSafeArea()
            .background(Color.black)
            .onAppear {
                audio.start()
                installKeyHandler()
            }
            .onDisappear { audio.stop() }
    }

    // onKeyPress requires macOS 14+; use NSEvent monitor for macOS 13 compat.
    private func installKeyHandler() {
        NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in
            if event.charactersIgnoringModifiers?.lowercased() == "f" {
                NSApp.mainWindow?.toggleFullScreen(nil)
                return nil   // consume event
            }
            return event
        }
    }
}

#Preview {
    ContentView()
        .frame(width: 960, height: 540)
}
