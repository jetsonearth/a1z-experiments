#!/bin/bash
# Builds ~/Applications/A1Z Console.app from launch.sh and make_icon.py. Rerun after editing either.
set -e;cd "$(dirname "$0")"
APP="$HOME/Applications/A1Z Console.app";rm -rf "$APP";mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp launch.sh "$APP/Contents/MacOS/A1Z Console";chmod +x "$APP/Contents/MacOS/A1Z Console"
T=$(mktemp -d);python3 make_icon.py "$T/icon.png";mkdir "$T/a.iconset"
for s in 16 32 128 256 512;do sips -z $s $s "$T/icon.png" --out "$T/a.iconset/icon_${s}x${s}.png" >/dev/null
  sips -z $((s*2)) $((s*2)) "$T/icon.png" --out "$T/a.iconset/icon_${s}x${s}@2x.png" >/dev/null;done
iconutil -c icns "$T/a.iconset" -o "$APP/Contents/Resources/AppIcon.icns";rm -rf "$T"
cat > "$APP/Contents/Info.plist" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleName</key><string>A1Z Console</string>
<key>CFBundleDisplayName</key><string>A1Z Console</string>
<key>CFBundleIdentifier</key><string>local.a1z.console-launcher</string>
<key>CFBundleExecutable</key><string>A1Z Console</string>
<key>CFBundleIconFile</key><string>AppIcon</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>1.0</string>
<key>LSMinimumSystemVersion</key><string>11.0</string>
<key>LSUIElement</key><true/>
</dict></plist>
PL
touch "$APP";echo "built $APP"
