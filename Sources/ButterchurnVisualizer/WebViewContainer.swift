import SwiftUI
import WebKit

// MARK: - SwiftUI wrapper

struct WebViewContainer: NSViewRepresentable {

    @ObservedObject var audio: AudioEngine

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> WKWebView {
        let webView = buildWebView(coordinator: context.coordinator)
        context.coordinator.webView = webView
        loadHost(into: webView)

        audio.onQ = { [weak coord = context.coordinator] q in
            coord?.injectQ(q)
        }

        // Intercept key events at the app level so they always reach the
        // visualizer regardless of which view is first responder.
        context.coordinator.installKeyMonitor()

        return webView
    }

    func updateNSView(_ nsView: WKWebView, context: Context) {
        audio.onQ = { [weak coord = context.coordinator] q in
            coord?.injectQ(q)
        }
    }

    // MARK: - Private helpers

    private func buildWebView(coordinator: Coordinator) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.mediaTypesRequiringUserActionForPlayback = []

        let prefs = WKWebpagePreferences()
        prefs.allowsContentJavaScript = true
        config.defaultWebpagePreferences = prefs

        config.userContentController.add(coordinator, name: "log")

        let wv = WKWebView(frame: .zero, configuration: config)
        wv.navigationDelegate = coordinator
        wv.setValue(false, forKey: "drawsBackground")
        wv.layer?.backgroundColor = .black
        wv.enclosingScrollView?.hasHorizontalScroller = false
        wv.enclosingScrollView?.hasVerticalScroller   = false

        return wv
    }

    private func loadHost(into webView: WKWebView) {
        guard let htmlURL = Bundle.module.url(forResource: "butterchurn_host",
                                              withExtension: "html"),
              let html = try? String(contentsOf: htmlURL, encoding: .utf8)
        else {
            print("[WebViewContainer] butterchurn_host.html not found in bundle")
            return
        }

        // 1. Bundled copy (fastest — no network needed)
        if let bcURL    = Bundle.module.url(forResource: "butterchurn.min", withExtension: "js"),
           let bcScript = try? String(contentsOf: bcURL, encoding: .utf8),
           bcScript.count > 100_000 {
            print("[WebViewContainer] butterchurn.min.js loaded from bundle (\(bcScript.count) bytes)")
            injectAndLoad(bcScript, html: html, into: webView)
            return
        }

        // 2. Cached copy from a previous successful fetch
        let cacheURL = Self.butterchurnCacheURL
        if let cached = try? String(contentsOf: cacheURL, encoding: .utf8),
           cached.count > 100_000 {
            print("[WebViewContainer] butterchurn.min.js loaded from cache (\(cached.count) bytes)")
            injectAndLoad(cached, html: html, into: webView)
            return
        }

        // 3. Fetch via URLSession — the Swift app process has full outbound network
        //    access; the WKWebView content process is sandboxed and cannot reach CDNs
        //    on macOS 15/26.
        print("[WebViewContainer] Fetching butterchurn.min.js via URLSession…")
        let cdnURL = URL(string: "https://cdn.jsdelivr.net/npm/butterchurn@2.6.7/build/butterchurn.min.js")!
        URLSession.shared.dataTask(with: cdnURL) { data, _, error in
            guard let data, error == nil, data.count > 100_000,
                  let script = String(data: data, encoding: .utf8) else {
                print("[WebViewContainer] CDN fetch failed: \(error?.localizedDescription ?? "bad/empty response")")
                DispatchQueue.main.async {
                    webView.loadHTMLString(html, baseURL: URL(string: "https://milkdrop.local"))
                }
                return
            }
            try? data.write(to: cacheURL)
            print("[WebViewContainer] butterchurn.min.js fetched (\(data.count) bytes) and cached")
            DispatchQueue.main.async { self.injectAndLoad(script, html: html, into: webView) }
        }.resume()
    }

    private static var butterchurnCacheURL: URL {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("butterchurn.min.js")
    }

    private func injectAndLoad(_ script: String, html: String, into webView: WKWebView) {
        let userScript = WKUserScript(source: script,
                                      injectionTime: .atDocumentStart,
                                      forMainFrameOnly: true)
        webView.configuration.userContentController.addUserScript(userScript)
        webView.loadHTMLString(html, baseURL: URL(string: "https://milkdrop.local"))
    }

    // MARK: - Coordinator

    final class Coordinator: NSObject, WKNavigationDelegate, WKScriptMessageHandler {

        weak var webView: WKWebView?
        private var keyMonitor: Any?

        // MARK: Key handling
        // NSEvent monitor works regardless of first-responder state.
        func installKeyMonitor() {
            guard keyMonitor == nil else { return }
            keyMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
                self?.forwardKey(event)
                return nil   // consume — stop the event propagating further
            }
        }

        private func forwardKey(_ event: NSEvent) {
            let cmd: String
            switch event.keyCode {
            case 123: cmd = "prev"    // ←
            case 124: cmd = "next"    // →
            default:
                guard let ch = event.charactersIgnoringModifiers?.lowercased() else { return }
                switch ch {
                case "n":  cmd = "next"
                case "p":  cmd = "prev"
                case "b":  cmd = "cut"
                case " ":  cmd = "pause"
                case "h":  cmd = "hud"
                default:   return
                }
            }
            print("[Key] \(cmd)")
            webView?.evaluateJavaScript("window._keyCommand('\(cmd)');", completionHandler: nil)
        }

        deinit {
            if let m = keyMonitor { NSEvent.removeMonitor(m) }
        }

        // MARK: Audio injection

        func injectQ(_ q: [Float]) {
            guard let wv = webView else { return }
            let arr = q.map { String(format: "%.5f", $0) }.joined(separator: ",")
            let js  = "if(typeof window._updateQ==='function')window._updateQ([\(arr)]);"
            wv.evaluateJavaScript(js, completionHandler: nil)
        }

        // MARK: WKNavigationDelegate

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            print("[WebView] butterchurn_host loaded")
            // Make webView first responder so it can also receive events natively.
            DispatchQueue.main.async {
                webView.window?.makeFirstResponder(webView)
            }
            PresetLoader.injectAll(into: webView)
        }

        func webView(_ webView: WKWebView,
                     didFail navigation: WKNavigation!,
                     withError error: Error) {
            print("[WebView] navigation error: \(error.localizedDescription)")
        }

        // MARK: WKScriptMessageHandler

        func userContentController(_ ucc: WKUserContentController,
                                   didReceive message: WKScriptMessage) {
            if let body = message.body as? String {
                print("[JS] \(body)")
            }
        }
    }
}
