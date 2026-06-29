#!/bin/bash
# Build ButterchurnVisualizer.app — a real, signed app bundle.
#
# WHY: ScreenCaptureKit's Screen Recording permission does NOT work when the app
# is run through the Xcode debugger or as a bare `swift run` binary (no stable
# identity → the grant never sticks, and the debugger is the TCC "responsible
# process"). A proper .app, launched normally, fixes that.
#
# USAGE:
#   ./scripts/build-app.sh                 # ad-hoc signed (re-grant after each rebuild)
#   CODESIGN_IDENTITY="Apple Development: you@example.com (TEAMID)" ./scripts/build-app.sh
#                                          # stable identity → permission persists across rebuilds
#
# Find your signing identity with:  security find-identity -v -p codesigning
#
# Then:  open ButterchurnVisualizer.app   (double-click it; do NOT run via Xcode)
set -euo pipefail
cd "$(dirname "$0")/.."

APP_NAME="ButterchurnVisualizer"
BUNDLE_ID="com.butterchurn.visualizer"
CONFIG="release"
BUILD_DIR=".build/${CONFIG}"
APP="${APP_NAME}.app"
IDENTITY="${CODESIGN_IDENTITY:--}"     # default: ad-hoc (-)

echo "▶ Building (${CONFIG})… this can take a minute"
swift build -c "${CONFIG}"

echo "▶ Assembling ${APP}"
rm -rf "${APP}"
mkdir -p "${APP}/Contents/MacOS" "${APP}/Contents/Resources"
cp "${BUILD_DIR}/${APP_NAME}" "${APP}/Contents/MacOS/${APP_NAME}"

# SwiftPM resource bundle(s) → Contents/Resources so Bundle.module resolves them.
shopt -s nullglob
for b in "${BUILD_DIR}"/*.bundle; do
  cp -R "$b" "${APP}/Contents/Resources/"
done
shopt -u nullglob

cat > "${APP}/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>${APP_NAME}</string>
  <key>CFBundleDisplayName</key><string>Butterchurn Visualizer</string>
  <key>CFBundleIdentifier</key><string>${BUNDLE_ID}</string>
  <key>CFBundleExecutable</key><string>${APP_NAME}</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSMicrophoneUsageDescription</key><string>Captures the BlackHole audio device to drive the visualizer and broadcast.</string>
</dict>
</plist>
PLIST

echo "▶ Signing with identity: ${IDENTITY}"
codesign --force --deep --sign "${IDENTITY}" "${APP}"

echo ""
echo "✔ Built ${APP}"
echo ""
echo "  RUN IT:   open ${APP}"
echo "  (Launch it this way — NOT through Xcode — so Screen Recording permission applies.)"
echo ""
echo "  First launch will prompt for Screen Recording + Microphone. Grant both."
if [ "${IDENTITY}" = "-" ]; then
  echo "  NOTE: ad-hoc signed — you'll re-grant Screen Recording after each rebuild."
  echo "        To make it permanent, rebuild with CODESIGN_IDENTITY set to your"
  echo "        Apple Development identity (see header of this script)."
fi
