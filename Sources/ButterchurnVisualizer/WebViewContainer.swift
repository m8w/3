import SwiftUI
import WebKit

// MARK: - SwiftUI wrapper

struct WebViewContainer: NSViewRepresentable {

    @ObservedObject var audio: AudioEngine

    // MARK: NSViewRepresentable

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> WKWebView {
        let webView = buildWebView(coordinator: context.coordinator)
        context.coordinator.webView = webView
        loadHost(into: webView)

        // Wire audio → JS injection.
        // The AudioEngine fires onQ on the main thread; the coordinator
        // calls window._updateQ([...]) inside the WebView.
        audio.onQ = { [weak coord = context.coordinator] q in
            coord?.injectQ(q)
        }

        return webView
    }

    func updateNSView(_ nsView: WKWebView, context: Context) {
        // Re-wire if the audio engine instance changes.
        audio.onQ = { [weak coord = context.coordinator] q in
            coord?.injectQ(q)
        }
    }

    // MARK: - Private helpers

    private func buildWebView(coordinator: Coordinator) -> WKWebView {
        let config = WKWebViewConfiguration()

        // Allow Web Audio API and media playback without a user gesture.
        // Required so Butterchurn can create an AudioContext on load.
        config.mediaTypesRequiringUserActionForPlayback = []

        let prefs = WKWebpagePreferences()
        prefs.allowsContentJavaScript = true
        config.defaultWebpagePreferences = prefs

        // Message handler lets JS send status strings back to Swift (optional).
        config.userContentController.add(coordinator, name: "log")

        let wv = WKWebView(frame: .zero, configuration: config)
        wv.navigationDelegate = coordinator
        // isOpaque is read-only on WKWebView; use setValue to suppress white flash.
        wv.setValue(false, forKey: "drawsBackground")
        wv.layer?.backgroundColor = .black

        // Hide the scroll indicators that can flash during resize.
        wv.enclosingScrollView?.hasHorizontalScroller = false
        wv.enclosingScrollView?.hasVerticalScroller   = false

        return wv
    }

    private func loadHost(into webView: WKWebView) {
        guard let url  = Bundle.module.url(forResource: "butterchurn_host",
                                           withExtension: "html"),
              let html = try? String(contentsOf: url, encoding: .utf8)
        else {
            print("[WebViewContainer] butterchurn_host.html not found in bundle")
            return
        }
        // Use an HTTPS base URL so the ESM imports from esm.sh/unpkg work.
        // WKWebView blocks cross-origin requests from file:// origins,
        // but allows them from https:// origins (CDN servers set CORS headers).
        webView.loadHTMLString(html,
                               baseURL: URL(string: "https://milkdrop.local"))
    }

    // MARK: - Coordinator

    final class Coordinator: NSObject, WKNavigationDelegate, WKScriptMessageHandler {

        weak var webView: WKWebView?

        // Called by AudioEngine.onQ on the main thread.
        func injectQ(_ q: [Float]) {
            guard let wv = webView else { return }
            // Build compact JS: window._updateQ([0.12,0.34,...])
            let arr = q.map { String(format: "%.5f", $0) }.joined(separator: ",")
            let js  = "if(typeof window._updateQ==='function')window._updateQ([\(arr)]);"
            wv.evaluateJavaScript(js, completionHandler: nil)
        }

        // MARK: WKNavigationDelegate

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            print("[WebView] butterchurn_host loaded")
            PresetLoader.injectAll(into: webView)
        }

        func webView(_ webView: WKWebView,
                     didFail navigation: WKNavigation!,
                     withError error: Error) {
            print("[WebView] navigation error: \(error.localizedDescription)")
        }

        // MARK: WKScriptMessageHandler — receives window.webkit.messageHandlers.log.postMessage(…)

        func userContentController(_ ucc: WKUserContentController,
                                   didReceive message: WKScriptMessage) {
            if let body = message.body as? String {
                print("[JS] \(body)")
            }
        }
    }
}
