#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  BUILD SCRIPT — Signal Simulation Suite                             ║
# ║  Compiles to .app/.dmg (macOS) or binary (Linux)                   ║
# ╚══════════════════════════════════════════════════════════════════════╝

set -e

VENV_DIR=".venv"
APP_NAME="SignalSimulationSuite"

echo "=========================================="
echo "  Signal Simulation Suite — Build Script  "
echo "=========================================="

# ── 1. Create virtual environment ─────────────────────────────────────
echo ""
echo "[1/5] Creating virtual environment in $VENV_DIR ..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
echo "      Python: $(python --version)"
echo "      Pip:    $(pip --version)"

# ── 2. Install dependencies inside venv ───────────────────────────────
echo ""
echo "[2/5] Installing dependencies into venv..."
pip install --upgrade pip --quiet
pip install PyQt6 matplotlib numpy scipy pyinstaller --quiet
echo "      Done."

# ─────────────────────────────────────────────────────────────────────
# macOS — .app + .dmg
# ─────────────────────────────────────────────────────────────────────
build_macos() {
    echo ""
    echo "[3/5] Building macOS .app with PyInstaller..."
    pyinstaller \
        --onedir \
        --windowed \
        --name "$APP_NAME" \
        --clean \
        --noconfirm \
        signal_simulator.py

    APP_PATH="dist/${APP_NAME}.app"

    echo ""
    echo "[4/5] Wrapping .app into .dmg ..."
    DMG_NAME="${APP_NAME}.dmg"

    rm -rf dmg_staging
    mkdir -p dmg_staging
    cp -r "$APP_PATH" dmg_staging/
    ln -s /Applications dmg_staging/Applications

    hdiutil create \
        -volname "Signal Simulation Suite" \
        -srcfolder dmg_staging \
        -ov -format UDZO \
        "$DMG_NAME" \
        -quiet

    rm -rf dmg_staging

    echo ""
    echo "  ✅  $DMG_NAME created successfully!"
    echo "  📁  dist/${APP_NAME}.app  (raw .app bundle)"
}

# ─────────────────────────────────────────────────────────────────────
# Linux — standalone binary
# ─────────────────────────────────────────────────────────────────────
build_linux() {
    echo ""
    echo "[3/5] Building Linux binary with PyInstaller..."
    pyinstaller \
        --onefile \
        --windowed \
        --name "$APP_NAME" \
        --clean \
        --noconfirm \
        signal_simulator.py

    echo ""
    echo "  ✅  dist/${APP_NAME} created successfully!"
}

# ── Detect OS ─────────────────────────────────────────────────────────
OS_TYPE="$(uname -s)"
echo ""
echo "[3/5] Detected OS: $OS_TYPE"

case "$OS_TYPE" in
    Darwin*)  build_macos  ;;
    Linux*)   build_linux  ;;
    *)
        echo "  [WARN] Unknown OS — attempting Linux build..."
        build_linux
        ;;
esac

deactivate 2>/dev/null || true

echo ""
echo "[5/5] Build complete!"
echo "=========================================="
echo "  Run directly:  .venv/bin/python signal_simulator.py"
echo "=========================================="
