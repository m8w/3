import SwiftUI
import WebKit

// MARK: - CurateView
//
// Shown instead of the normal visualizer when curate mode is active
// (CURATE_INPUT / CURATE=1 in the environment). Displays the live culling
// WebView (presets flash by as they're scored) with a progress overlay.

struct CurateView: View {
    @ObservedObject var curator: Curator

    var body: some View {
        ZStack(alignment: .bottom) {
            CurateWebView(webView: curator.webView)
                .ignoresSafeArea()

            VStack(alignment: .leading, spacing: 6) {
                Text(curator.finished ? "✅ Curation complete" : "🎛 Curating presets…")
                    .font(.system(size: 15, weight: .semibold, design: .monospaced))
                Text(curator.status)
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundColor(.white.opacity(0.85))
                if curator.total > 0 {
                    ProgressView(value: Double(curator.processed), total: Double(curator.total))
                        .tint(.white)
                    HStack(spacing: 16) {
                        stat("processed", "\(curator.processed)/\(curator.total)")
                        stat("kept", "\(curator.kept)")
                        stat("failed", "\(curator.failed)")
                    }
                }
                Text(curator.current)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(.white.opacity(0.55))
                    .lineLimit(1).truncationMode(.middle)
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(LinearGradient(colors: [.clear, .black.opacity(0.8)],
                                       startPoint: .top, endPoint: .bottom))
            .foregroundColor(.white)
        }
        .background(Color.black)
    }

    private func stat(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(label).font(.system(size: 9, design: .monospaced)).foregroundColor(.white.opacity(0.5))
            Text(value).font(.system(size: 13, weight: .bold, design: .monospaced))
        }
    }
}

// Hosts the Curator's own WKWebView in the SwiftUI hierarchy so WebGL actually
// renders (offscreen WebViews get throttled and read back black).
private struct CurateWebView: NSViewRepresentable {
    let webView: WKWebView
    func makeNSView(context: Context) -> WKWebView { webView }
    func updateNSView(_ nsView: WKWebView, context: Context) {}
}
