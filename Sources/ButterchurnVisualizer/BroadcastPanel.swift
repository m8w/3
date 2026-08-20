import SwiftUI

// MARK: - BroadcastPanel
//
// Lives in the menu bar (never inside the captured visualizer window). Enter the
// RTMP ingest URL + stream key, then Go Live. URL/key persist between launches.

struct BroadcastPanel: View {
    @ObservedObject var broadcaster: Broadcaster

    @AppStorage("rtmpURL") private var url = "rtmp://a.rtmp.youtube.com/live2"
    @AppStorage("rtmpKey") private var key = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Broadcast").font(.headline)

            VStack(alignment: .leading, spacing: 4) {
                Text("RTMP ingest URL").font(.caption).foregroundColor(.secondary)
                TextField("rtmp://…", text: $url)
                    .textFieldStyle(.roundedBorder)
                    .disabled(broadcaster.isLive)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Stream key").font(.caption).foregroundColor(.secondary)
                SecureField("xxxx-xxxx-xxxx-xxxx", text: $key)
                    .textFieldStyle(.roundedBorder)
                    .disabled(broadcaster.isLive)
            }

            HStack {
                Circle()
                    .fill(broadcaster.isLive ? Color.red : Color.secondary)
                    .frame(width: 9, height: 9)
                Text(broadcaster.status).font(.caption).foregroundColor(.secondary)
                    .lineLimit(1).truncationMode(.tail)
            }

            Button(broadcaster.isLive ? "Stop Broadcast" : "Go Live") {
                if broadcaster.isLive { broadcaster.stop() }
                else { broadcaster.goLive(url: url, key: key) }
            }
            .keyboardShortcut(.defaultAction)
            .tint(broadcaster.isLive ? .red : .accentColor)

            Text("Open the visualizer window and fullscreen it (press F) before going live.")
                .font(.caption2).foregroundColor(.secondary)

            Divider()
            Button("Quit") { NSApplication.shared.terminate(nil) }
        }
        .padding(14)
        .frame(width: 320)
    }
}
