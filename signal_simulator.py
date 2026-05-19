"""
╔══════════════════════════════════════════════════════════════════════╗
║   Signal Simulation App — Fourier & Z-Transform Analyzer            ║
║   Developed with AI-assisted engineering (Claude)                   ║
║   Mata Kuliah: Linier System / Digital Signal Processing            ║
╚══════════════════════════════════════════════════════════════════════╝

Dependencies:
    pip install PyQt6 matplotlib numpy scipy

Compile to .exe  : pyinstaller --onefile --windowed signal_simulator.py
Compile to .dmg  : pyinstaller --onefile --windowed signal_simulator.py
                   (then wrap the .app bundle with create-dmg or hdiutil)
"""

import sys
import numpy as np
from scipy import signal as sp_signal
from scipy.fft import fft, fftfreq, ifft
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle
import matplotlib.ticker as ticker
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QDoubleSpinBox, QSpinBox, QComboBox, QPushButton,
    QSlider, QGroupBox, QTextEdit, QSplitter, QFrame,
    QLineEdit, QCheckBox, QMessageBox, QStatusBar
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon


# ─────────────────────────────────────────────────────────────────────
#  GLOBAL STYLE
# ─────────────────────────────────────────────────────────────────────
DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #0d1117;
    color: #e6edf3;
    font-family: 'Segoe UI', 'SF Pro Display', Helvetica, Arial, sans-serif;
    font-size: 13px;
}
QTabWidget::pane {
    border: 1px solid #30363d;
    border-radius: 8px;
    background-color: #161b22;
}
QTabBar::tab {
    background-color: #21262d;
    color: #8b949e;
    border: 1px solid #30363d;
    border-bottom: none;
    padding: 10px 22px;
    border-radius: 6px 6px 0 0;
    margin-right: 3px;
    font-weight: 600;
    font-size: 12px;
    letter-spacing: 0.4px;
}
QTabBar::tab:selected {
    background-color: #161b22;
    color: #58a6ff;
    border-color: #388bfd;
    border-bottom: 2px solid #388bfd;
}
QTabBar::tab:hover:!selected {
    background-color: #30363d;
    color: #c9d1d9;
}
QGroupBox {
    border: 1px solid #30363d;
    border-radius: 8px;
    margin-top: 14px;
    padding: 12px 10px 10px 10px;
    background-color: #161b22;
    font-weight: 700;
    font-size: 11px;
    letter-spacing: 0.8px;
    color: #8b949e;
    text-transform: uppercase;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #58a6ff;
}
QPushButton {
    background-color: #238636;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 9px 20px;
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 0.3px;
}
QPushButton:hover { background-color: #2ea043; }
QPushButton:pressed { background-color: #1a6f2a; }
QPushButton#danger {
    background-color: #da3633;
}
QPushButton#danger:hover { background-color: #f85149; }
QPushButton#secondary {
    background-color: #21262d;
    border: 1px solid #388bfd;
    color: #58a6ff;
}
QPushButton#secondary:hover { background-color: #30363d; }
QPushButton#accent {
    background-color: #1f6feb;
}
QPushButton#accent:hover { background-color: #388bfd; }
QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit {
    background-color: #21262d;
    color: #e6edf3;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 7px 10px;
    font-size: 13px;
}
QDoubleSpinBox:focus, QSpinBox:focus,
QComboBox:focus, QLineEdit:focus {
    border-color: #388bfd;
    background-color: #0d1117;
}
QComboBox::drop-down { border: none; width: 24px; }
QComboBox::down-arrow { color: #8b949e; }
QComboBox QAbstractItemView {
    background-color: #21262d;
    border: 1px solid #30363d;
    color: #e6edf3;
    selection-background-color: #388bfd;
}
QSlider::groove:horizontal {
    border: none;
    height: 4px;
    background: #30363d;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #388bfd;
    border: none;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal { background: #388bfd; border-radius: 2px; }
QTextEdit {
    background-color: #0d1117;
    color: #7ee787;
    border: 1px solid #30363d;
    border-radius: 6px;
    font-family: 'Cascadia Code', 'Fira Code', 'Courier New', monospace;
    font-size: 12px;
    padding: 8px;
}
QLabel#heading {
    font-size: 22px;
    font-weight: 700;
    color: #58a6ff;
    letter-spacing: -0.5px;
}
QLabel#subheading {
    font-size: 12px;
    color: #8b949e;
}
QLabel#badge {
    background-color: #388bfd20;
    color: #58a6ff;
    border: 1px solid #388bfd50;
    border-radius: 4px;
    padding: 3px 8px;
    font-size: 11px;
    font-weight: 700;
}
QLabel#badge_green {
    background-color: #2ea04320;
    color: #3fb950;
    border: 1px solid #2ea04350;
    border-radius: 4px;
    padding: 3px 8px;
    font-size: 11px;
    font-weight: 700;
}
QLabel#badge_red {
    background-color: #da363320;
    color: #f85149;
    border: 1px solid #da363350;
    border-radius: 4px;
    padding: 3px 8px;
    font-size: 11px;
    font-weight: 700;
}
QSplitter::handle { background-color: #30363d; width: 2px; }
QStatusBar {
    background-color: #161b22;
    color: #8b949e;
    border-top: 1px solid #30363d;
    font-size: 11px;
}
QFrame#divider {
    background-color: #30363d;
    max-height: 1px;
}
QCheckBox { spacing: 8px; }
QCheckBox::indicator {
    width: 16px; height: 16px;
    border: 1px solid #30363d;
    border-radius: 4px;
    background-color: #21262d;
}
QCheckBox::indicator:checked {
    background-color: #388bfd;
    border-color: #388bfd;
}
"""

MPLSTYLE = {
    "figure.facecolor": "#161b22",
    "axes.facecolor": "#0d1117",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#8b949e",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "text.color": "#e6edf3",
    "lines.linewidth": 2.0,
    "figure.dpi": 100,
}
matplotlib.rcParams.update(MPLSTYLE)

COLORS = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff",
          "#ffa657", "#79c0ff", "#56d364", "#ff7b72"]


# ─────────────────────────────────────────────────────────────────────
#  MATPLOTLIB CANVAS
# ─────────────────────────────────────────────────────────────────────
class MplCanvas(FigureCanvas):
    def __init__(self, rows=1, cols=1, figsize=None):
        w, h = figsize or (10, 4)
        self.fig = Figure(figsize=(w, h), tight_layout=True)
        self.axes = []
        for i in range(rows * cols):
            ax = self.fig.add_subplot(rows, cols, i + 1)
            ax.grid(True, alpha=0.4)
            self.axes.append(ax)
        super().__init__(self.fig)
        self.setMinimumHeight(260)

    def ax(self, idx=0):
        return self.axes[idx]

    def clear_all(self):
        for ax in self.axes:
            ax.cla()
            ax.grid(True, alpha=0.4)

    def draw_safe(self):
        try:
            self.fig.tight_layout()
        except Exception:
            pass
        self.draw()


# ─────────────────────────────────────────────────────────────────────
#  TAB 1 — FOURIER SIGNAL ANALYZER
# ─────────────────────────────────────────────────────────────────────
class FourierTab(QWidget):
    """Generate composite signals + FFT analysis."""

    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(14)

        # ── Left Panel ──────────────────────────────────────────────
        left = QWidget(); left.setFixedWidth(300)
        lv = QVBoxLayout(left); lv.setSpacing(12)

        # Header
        h = QLabel("Fourier Signal Analyzer"); h.setObjectName("heading")
        sub = QLabel("Time & Frequency Domain Analysis"); sub.setObjectName("subheading")
        lv.addWidget(h); lv.addWidget(sub)
        lv.addWidget(self._divider())

        # Signal Components
        self.components = []
        comp_group = QGroupBox("Signal Components")
        cg_layout = QVBoxLayout(comp_group)

        for i in range(3):
            row = QHBoxLayout()
            chk = QCheckBox(f"S{i+1}")
            chk.setChecked(i == 0)
            freq_sb = QDoubleSpinBox()
            freq_sb.setRange(1, 2000); freq_sb.setValue([50, 120, 300][i])
            freq_sb.setSuffix(" Hz"); freq_sb.setDecimals(1)
            amp_sb = QDoubleSpinBox()
            amp_sb.setRange(0.01, 10); amp_sb.setValue([1.0, 0.5, 0.3][i])
            amp_sb.setSingleStep(0.1); amp_sb.setDecimals(2)
            row.addWidget(chk)
            row.addWidget(QLabel("f:")); row.addWidget(freq_sb)
            row.addWidget(QLabel("A:")); row.addWidget(amp_sb)
            cg_layout.addLayout(row)
            self.components.append((chk, freq_sb, amp_sb))
        lv.addWidget(comp_group)

        # Signal Type & Parameters
        param_group = QGroupBox("Parameters")
        pg = QFormLayout(param_group)
        self.sig_type = QComboBox()
        self.sig_type.addItems(["Sine", "Square", "Triangle", "Sawtooth", "Mixed Sine"])
        self.fs_spin = QDoubleSpinBox()
        self.fs_spin.setRange(1000, 100000); self.fs_spin.setValue(8000)
        self.fs_spin.setSuffix(" Hz"); self.fs_spin.setSingleStep(1000)
        self.dur_spin = QDoubleSpinBox()
        self.dur_spin.setRange(0.01, 5.0); self.dur_spin.setValue(0.5)
        self.dur_spin.setSuffix(" s"); self.dur_spin.setDecimals(3)
        self.noise_chk = QCheckBox("Add Gaussian Noise")
        self.noise_snr = QDoubleSpinBox()
        self.noise_snr.setRange(1, 100); self.noise_snr.setValue(20)
        self.noise_snr.setSuffix(" dB"); self.noise_snr.setDecimals(1)
        pg.addRow("Type:", self.sig_type)
        pg.addRow("Sample Rate:", self.fs_spin)
        pg.addRow("Duration:", self.dur_spin)
        pg.addRow("", self.noise_chk)
        pg.addRow("SNR:", self.noise_snr)
        lv.addWidget(param_group)

        # Window Function
        win_group = QGroupBox("FFT Window")
        wg = QFormLayout(win_group)
        self.window_cb = QComboBox()
        self.window_cb.addItems(["Rectangular", "Hanning", "Hamming",
                                  "Blackman", "Kaiser"])
        wg.addRow("Window:", self.window_cb)
        lv.addWidget(win_group)

        # Buttons
        btn_run = QPushButton("▶  Analyze Signal")
        btn_run.setObjectName("accent")
        btn_run.clicked.connect(self.analyze)
        btn_clr = QPushButton("Clear Plots")
        btn_clr.setObjectName("secondary")
        btn_clr.clicked.connect(self.clear_plots)
        lv.addWidget(btn_run)
        lv.addWidget(btn_clr)
        lv.addStretch()

        # Analysis Output
        self.info_box = QTextEdit()
        self.info_box.setMaximumHeight(140)
        self.info_box.setReadOnly(True)
        self.info_box.setPlaceholderText("Analysis results will appear here...")
        lv.addWidget(QLabel("Analysis Log"))
        lv.addWidget(self.info_box)

        # ── Right Panel (Plots) ──────────────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right); rv.setSpacing(10)

        self.canvas_time = MplCanvas(rows=1, cols=1, figsize=(10, 2.8))
        self.canvas_freq = MplCanvas(rows=1, cols=1, figsize=(10, 2.8))
        self.canvas_spec = MplCanvas(rows=1, cols=1, figsize=(10, 2.2))

        rv.addWidget(self._section_label("⏱  Time Domain"))
        rv.addWidget(self.canvas_time)
        rv.addWidget(self._section_label("📊  Frequency Spectrum (FFT)"))
        rv.addWidget(self.canvas_freq)
        rv.addWidget(self._section_label("🌡  Spectrogram"))
        rv.addWidget(self.canvas_spec)

        main_layout.addWidget(left)
        main_layout.addWidget(right, stretch=1)

    def _divider(self):
        f = QFrame(); f.setObjectName("divider")
        f.setFrameShape(QFrame.Shape.HLine)
        return f

    def _section_label(self, text):
        lb = QLabel(text)
        lb.setStyleSheet("color:#58a6ff; font-weight:700; font-size:12px; "
                         "padding:4px 0; letter-spacing:0.5px;")
        return lb

    def _generate_signal(self):
        fs = self.fs_spin.value()
        dur = self.dur_spin.value()
        t = np.linspace(0, dur, int(fs * dur), endpoint=False)
        y = np.zeros_like(t)
        sig_type = self.sig_type.currentText()

        for chk, f_spin, a_spin in self.components:
            if chk.isChecked():
                f = f_spin.value()
                a = a_spin.value()
                if sig_type == "Sine" or sig_type == "Mixed Sine":
                    y += a * np.sin(2 * np.pi * f * t)
                elif sig_type == "Square":
                    y += a * sp_signal.square(2 * np.pi * f * t)
                elif sig_type == "Triangle":
                    y += a * sp_signal.sawtooth(2 * np.pi * f * t, width=0.5)
                elif sig_type == "Sawtooth":
                    y += a * sp_signal.sawtooth(2 * np.pi * f * t)

        if self.noise_chk.isChecked():
            snr_db = self.noise_snr.value()
            sig_power = np.mean(y ** 2) if np.any(y) else 1.0
            noise_power = sig_power / (10 ** (snr_db / 10))
            y += np.random.normal(0, np.sqrt(noise_power), len(t))

        return t, y, fs

    def analyze(self):
        t, y, fs = self._generate_signal()
        N = len(y)

        # Window
        win_name = self.window_cb.currentText()
        windows = {
            "Rectangular": np.ones(N),
            "Hanning": np.hanning(N),
            "Hamming": np.hamming(N),
            "Blackman": np.blackman(N),
            "Kaiser": np.kaiser(N, 14),
        }
        win = windows[win_name]
        y_win = y * win

        # FFT
        Y = fft(y_win)
        freqs = fftfreq(N, 1 / fs)
        half = N // 2
        mag = 2 * np.abs(Y[:half]) / N
        phase = np.angle(Y[:half])

        # ── Time Domain ──
        self.canvas_time.clear_all()
        ax = self.canvas_time.ax(0)
        ax.plot(t * 1000, y, color=COLORS[0], linewidth=1.2, alpha=0.9)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Time Domain Signal", color="#e6edf3", fontweight="bold")
        ax.axhline(0, color="#30363d", linewidth=0.8)
        ax.fill_between(t * 1000, y, alpha=0.15, color=COLORS[0])
        self.canvas_time.draw_safe()

        # ── Frequency Domain ──
        self.canvas_freq.clear_all()
        ax2 = self.canvas_freq.ax(0)
        ax2.plot(freqs[:half], mag, color=COLORS[1], linewidth=1.5)
        ax2.fill_between(freqs[:half], mag, alpha=0.2, color=COLORS[1])

        # Annotate peaks
        peaks, props = sp_signal.find_peaks(mag, height=np.max(mag) * 0.05,
                                             distance=max(1, N // 200))
        for pk in peaks[:6]:
            ax2.annotate(f"{freqs[pk]:.0f}Hz",
                          xy=(freqs[pk], mag[pk]),
                          xytext=(0, 10), textcoords="offset points",
                          ha="center", fontsize=9, color="#ffa657",
                          arrowprops=dict(arrowstyle="-", color="#ffa657", lw=0.8))

        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Magnitude")
        ax2.set_title(f"Frequency Spectrum — {win_name} Window",
                      color="#e6edf3", fontweight="bold")
        ax2.set_xlim(0, fs / 2)
        self.canvas_freq.draw_safe()

        # ── Spectrogram ──
        self.canvas_spec.clear_all()
        ax3 = self.canvas_spec.ax(0)
        nperseg = min(256, N // 4)
        f_spec, t_spec, Sxx = sp_signal.spectrogram(y, fs, nperseg=nperseg)
        im = ax3.pcolormesh(t_spec * 1000, f_spec,
                             10 * np.log10(Sxx + 1e-12),
                             shading="gouraud", cmap="inferno")
        self.canvas_spec.fig.colorbar(im, ax=ax3, label="dB")
        ax3.set_xlabel("Time (ms)")
        ax3.set_ylabel("Frequency (Hz)")
        ax3.set_title("Spectrogram", color="#e6edf3", fontweight="bold")
        self.canvas_spec.draw_safe()

        # ── Log ──
        active = [self.components[i] for i in range(3)
                  if self.components[i][0].isChecked()]
        peak_freq = freqs[peaks[np.argmax(mag[peaks])]] if len(peaks) else 0
        rms = np.sqrt(np.mean(y ** 2))
        self.info_box.append(
            f"[OK] Signal: {self.sig_type.currentText()} | "
            f"Fs={fs:.0f}Hz | N={N}\n"
            f"     RMS={rms:.4f} | Peak freq={peak_freq:.1f}Hz | "
            f"Components={len(active)} active\n"
            f"     Window={win_name} | "
            f"{'Noise added' if self.noise_chk.isChecked() else 'No noise'}\n"
        )

    def clear_plots(self):
        for c in [self.canvas_time, self.canvas_freq, self.canvas_spec]:
            c.clear_all(); c.draw()
        self.info_box.clear()


# ─────────────────────────────────────────────────────────────────────
#  TAB 2 — Z-TRANSFORM STABILITY ANALYZER
# ─────────────────────────────────────────────────────────────────────
class ZTransformTab(QWidget):
    """Pole-zero plot, stability, causality analysis."""

    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        main = QHBoxLayout(self)
        main.setContentsMargins(14, 14, 14, 14)
        main.setSpacing(14)

        # ── Left ────────────────────────────────────────────────────
        left = QWidget(); left.setFixedWidth(310)
        lv = QVBoxLayout(left); lv.setSpacing(12)

        h = QLabel("Z-Transform Analyzer"); h.setObjectName("heading")
        sub = QLabel("Pole-Zero Plot & Stability Analysis")
        sub.setObjectName("subheading")
        lv.addWidget(h); lv.addWidget(sub)
        lv.addWidget(self._divider())

        # Transfer Function Input
        tf_group = QGroupBox("Transfer Function H(z)")
        tg = QFormLayout(tf_group)
        self.num_edit = QLineEdit("1 0 -0.5")
        self.den_edit = QLineEdit("1 -1.5 0.7")
        self.num_edit.setPlaceholderText("e.g. 1 0 -0.5  (space-separated)")
        self.den_edit.setPlaceholderText("e.g. 1 -1.5 0.7")
        tg.addRow("Numerator b[]:", self.num_edit)
        tg.addRow("Denominator a[]:", self.den_edit)
        lv.addWidget(tf_group)

        # Presets
        preset_group = QGroupBox("Preset Examples")
        pg = QVBoxLayout(preset_group)
        presets = [
            ("Stable IIR", "1 0.5", "1 -0.9 0.81"),
            ("Unstable", "1", "1 -1.5 0.56"),
            ("Marginally Stable", "1 0", "1 -1 0"),
            ("Low-Pass FIR", "0.25 0.5 0.25", "1"),
            ("Notch Filter", "1 -1.618 1", "1 -1.4 0.81"),
        ]
        for name, num, den in presets:
            btn = QPushButton(name)
            btn.setObjectName("secondary")
            btn.setFixedHeight(30)
            btn.clicked.connect(lambda _, n=num, d=den: self._set_preset(n, d))
            pg.addWidget(btn)
        lv.addWidget(preset_group)

        btn_analyze = QPushButton("▶  Plot Pole-Zero & Analyze")
        btn_analyze.setObjectName("accent")
        btn_analyze.clicked.connect(self.analyze)
        lv.addWidget(btn_analyze)

        # Results
        res_group = QGroupBox("Analysis Results")
        rg = QVBoxLayout(res_group)
        self.stability_lbl = QLabel("—"); self.stability_lbl.setObjectName("badge")
        self.causal_lbl    = QLabel("—"); self.causal_lbl.setObjectName("badge")
        self.poles_lbl  = QLabel("Poles:  —")
        self.zeros_lbl  = QLabel("Zeros:  —")
        self.poles_lbl.setWordWrap(True)
        self.zeros_lbl.setWordWrap(True)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Stability:")); row1.addWidget(self.stability_lbl)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Causality:")); row2.addWidget(self.causal_lbl)
        rg.addLayout(row1); rg.addLayout(row2)
        rg.addWidget(self.poles_lbl); rg.addWidget(self.zeros_lbl)
        lv.addWidget(res_group)

        self.info_box = QTextEdit()
        self.info_box.setMaximumHeight(130)
        self.info_box.setReadOnly(True)
        lv.addWidget(QLabel("System Log"))
        lv.addWidget(self.info_box)
        lv.addStretch()

        # ── Right ────────────────────────────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right); rv.setSpacing(10)

        self.canvas_pz    = MplCanvas(rows=1, cols=1, figsize=(7, 5))
        self.canvas_freq  = MplCanvas(rows=1, cols=2, figsize=(10, 3.2))
        self.canvas_imp   = MplCanvas(rows=1, cols=2, figsize=(10, 2.8))

        rv.addWidget(self._section_label("🎯  Pole-Zero Plot (z-Plane)"))
        rv.addWidget(self.canvas_pz)
        rv.addWidget(self._section_label("📈  Frequency Response (Magnitude & Phase)"))
        rv.addWidget(self.canvas_freq)
        rv.addWidget(self._section_label("⚡  Impulse & Step Response"))
        rv.addWidget(self.canvas_imp)

        main.addWidget(left)
        main.addWidget(right, stretch=1)

    def _divider(self):
        f = QFrame(); f.setObjectName("divider")
        f.setFrameShape(QFrame.Shape.HLine)
        return f

    def _section_label(self, text):
        lb = QLabel(text)
        lb.setStyleSheet("color:#58a6ff; font-weight:700; font-size:12px; "
                         "padding:4px 0;")
        return lb

    def _set_preset(self, num, den):
        self.num_edit.setText(num)
        self.den_edit.setText(den)

    def _parse_coeffs(self, text):
        try:
            return [float(x) for x in text.strip().split()]
        except Exception:
            return None

    def analyze(self):
        b = self._parse_coeffs(self.num_edit.text())
        a = self._parse_coeffs(self.den_edit.text())
        if not b or not a:
            QMessageBox.warning(self, "Input Error",
                                "Invalid coefficients. Use space-separated numbers.")
            return

        zeros = np.roots(b)
        poles = np.roots(a)

        # Stability: all poles strictly inside unit circle
        max_pole_mag = np.max(np.abs(poles)) if len(poles) else 0
        is_stable = max_pole_mag < 1.0
        is_marginal = np.isclose(max_pole_mag, 1.0, atol=0.01)
        is_causal = len(a) > 1

        # Update labels
        if is_stable:
            self.stability_lbl.setText("✓  STABLE")
            self.stability_lbl.setObjectName("badge_green")
        elif is_marginal:
            self.stability_lbl.setText("⚠  MARGINALLY STABLE")
            self.stability_lbl.setObjectName("badge")
        else:
            self.stability_lbl.setText("✗  UNSTABLE")
            self.stability_lbl.setObjectName("badge_red")
        self.stability_lbl.setStyle(self.stability_lbl.style())

        self.causal_lbl.setText("Causal IIR" if is_causal else "Non-Recursive FIR")
        self.causal_lbl.setObjectName("badge")
        self.causal_lbl.setStyle(self.causal_lbl.style())

        poles_str = ", ".join(f"{p:.3f}" for p in poles)
        zeros_str = ", ".join(f"{z:.3f}" for z in zeros)
        self.poles_lbl.setText(f"Poles:  {poles_str if poles_str else '—'}")
        self.zeros_lbl.setText(f"Zeros:  {zeros_str if zeros_str else '—'}")

        # ── Pole-Zero Plot ──
        self.canvas_pz.clear_all()
        ax = self.canvas_pz.ax(0)
        theta = np.linspace(0, 2 * np.pi, 360)
        ax.plot(np.cos(theta), np.sin(theta), "--", color="#30363d",
                linewidth=1.2, label="Unit Circle")
        ax.axhline(0, color="#30363d", linewidth=0.7)
        ax.axvline(0, color="#30363d", linewidth=0.7)
        ax.scatter(poles.real, poles.imag, marker="x", s=120,
                   color="#f85149", linewidths=2.5, zorder=5, label="Poles")
        ax.scatter(zeros.real, zeros.imag, marker="o", s=100,
                   facecolors="none", edgecolors="#58a6ff",
                   linewidths=2.0, zorder=5, label="Zeros")
        # Shade unit disk
        circle = Circle((0, 0), 1, color="#58a6ff", alpha=0.04)
        ax.add_patch(circle)
        ax.set_aspect("equal")
        ax.set_xlabel("Real")
        ax.set_ylabel("Imaginary")
        ax.set_title("Pole-Zero Plot (z-plane)",
                     color="#e6edf3", fontweight="bold")
        ax.legend(facecolor="#21262d", edgecolor="#30363d",
                  labelcolor="#e6edf3", fontsize=9)
        lim = max(1.5, max_pole_mag * 1.3)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        self.canvas_pz.draw_safe()

        # ── Frequency Response ──
        try:
            w, h = sp_signal.freqz(b, a, worN=1024)
            mag_db = 20 * np.log10(np.abs(h) + 1e-12)
            phase_deg = np.unwrap(np.angle(h)) * 180 / np.pi

            self.canvas_freq.clear_all()
            ax1, ax2 = self.canvas_freq.axes
            ax1.plot(w / np.pi, mag_db, color=COLORS[0])
            ax1.set_xlabel("Normalized Freq (×π rad/sample)")
            ax1.set_ylabel("Magnitude (dB)")
            ax1.set_title("Magnitude Response", color="#e6edf3", fontweight="bold")
            ax1.fill_between(w / np.pi, mag_db, mag_db.min(),
                              alpha=0.15, color=COLORS[0])

            ax2.plot(w / np.pi, phase_deg, color=COLORS[3])
            ax2.set_xlabel("Normalized Freq (×π rad/sample)")
            ax2.set_ylabel("Phase (degrees)")
            ax2.set_title("Phase Response", color="#e6edf3", fontweight="bold")
            self.canvas_freq.draw_safe()
        except Exception as e:
            self.info_box.append(f"[WARN] Freq response error: {e}")

        # ── Impulse & Step Response ──
        try:
            N_imp = 50
            imp = np.zeros(N_imp); imp[0] = 1.0
            h_imp = sp_signal.lfilter(b, a, imp)
            h_step = sp_signal.lfilter(b, a, np.ones(N_imp))

            self.canvas_imp.clear_all()
            ax3, ax4 = self.canvas_imp.axes
            n = np.arange(N_imp)
            ax3.stem(n, h_imp, linefmt=f"{COLORS[1]}-",
                     markerfmt="o", basefmt="#30363d")
            ax3.set_xlabel("Sample n")
            ax3.set_ylabel("h[n]")
            ax3.set_title("Impulse Response", color="#e6edf3", fontweight="bold")

            ax4.stem(n, h_step, linefmt=f"{COLORS[2]}-",
                     markerfmt="o", basefmt="#30363d")
            ax4.set_xlabel("Sample n")
            ax4.set_ylabel("y[n]")
            ax4.set_title("Step Response", color="#e6edf3", fontweight="bold")
            self.canvas_imp.draw_safe()
        except Exception as e:
            self.info_box.append(f"[WARN] Response error: {e}")

        self.info_box.append(
            f"[OK] H(z): b={b} / a={a}\n"
            f"     Poles={len(poles)}, Zeros={len(zeros)}\n"
            f"     Max |pole|={max_pole_mag:.4f} → "
            f"{'STABLE' if is_stable else 'UNSTABLE'}\n"
        )


# ─────────────────────────────────────────────────────────────────────
#  TAB 3 — DIGITAL FILTER DESIGNER
# ─────────────────────────────────────────────────────────────────────
class FilterDesignerTab(QWidget):
    """Design LP/HP/BP/BS IIR & FIR filters, visualize & apply."""

    def __init__(self):
        super().__init__()
        self._build_ui()
        self._b = None
        self._a = None

    def _build_ui(self):
        main = QHBoxLayout(self)
        main.setContentsMargins(14, 14, 14, 14)
        main.setSpacing(14)

        # ── Left ────────────────────────────────────────────────────
        left = QWidget(); left.setFixedWidth(310)
        lv = QVBoxLayout(left); lv.setSpacing(12)

        h = QLabel("Digital Filter Designer"); h.setObjectName("heading")
        sub = QLabel("IIR & FIR — LP / HP / BP / BS")
        sub.setObjectName("subheading")
        lv.addWidget(h); lv.addWidget(sub)
        lv.addWidget(self._divider())

        # Filter Spec
        spec_group = QGroupBox("Filter Specification")
        sg = QFormLayout(spec_group)
        self.ftype_cb = QComboBox()
        self.ftype_cb.addItems(["lowpass", "highpass", "bandpass", "bandstop"])
        self.fdesign_cb = QComboBox()
        self.fdesign_cb.addItems(["Butterworth", "Chebyshev Type I",
                                   "Chebyshev Type II", "Elliptic", "Bessel",
                                   "FIR (Window)"])
        self.order_spin = QSpinBox()
        self.order_spin.setRange(1, 20); self.order_spin.setValue(4)
        self.fs_spin = QDoubleSpinBox()
        self.fs_spin.setRange(1000, 100000)
        self.fs_spin.setValue(8000); self.fs_spin.setSuffix(" Hz")
        self.fc1_spin = QDoubleSpinBox()
        self.fc1_spin.setRange(1, 4999); self.fc1_spin.setValue(1000)
        self.fc1_spin.setSuffix(" Hz")
        self.fc2_spin = QDoubleSpinBox()
        self.fc2_spin.setRange(1, 4999); self.fc2_spin.setValue(2000)
        self.fc2_spin.setSuffix(" Hz")
        self.rp_spin = QDoubleSpinBox()
        self.rp_spin.setRange(0.1, 20); self.rp_spin.setValue(1)
        self.rp_spin.setSuffix(" dB")
        sg.addRow("Filter Type:", self.ftype_cb)
        sg.addRow("Design Method:", self.fdesign_cb)
        sg.addRow("Order:", self.order_spin)
        sg.addRow("Sample Rate:", self.fs_spin)
        sg.addRow("Cutoff f₁:", self.fc1_spin)
        sg.addRow("Cutoff f₂ (BP/BS):", self.fc2_spin)
        sg.addRow("Ripple (Cheby/Ellip):", self.rp_spin)
        lv.addWidget(spec_group)

        # Test Signal
        test_group = QGroupBox("Test Signal")
        tg = QFormLayout(test_group)
        self.test_f1 = QDoubleSpinBox()
        self.test_f1.setRange(1, 4000); self.test_f1.setValue(500)
        self.test_f1.setSuffix(" Hz")
        self.test_f2 = QDoubleSpinBox()
        self.test_f2.setRange(1, 4000); self.test_f2.setValue(2500)
        self.test_f2.setSuffix(" Hz")
        tg.addRow("Tone 1:", self.test_f1)
        tg.addRow("Tone 2:", self.test_f2)
        lv.addWidget(test_group)

        btn_design = QPushButton("⚙  Design Filter")
        btn_design.setObjectName("accent")
        btn_design.clicked.connect(self.design_filter)
        btn_apply = QPushButton("▶  Apply to Test Signal")
        btn_apply.setObjectName("secondary")
        btn_apply.clicked.connect(self.apply_filter)
        lv.addWidget(btn_design)
        lv.addWidget(btn_apply)

        self.coeff_box = QTextEdit()
        self.coeff_box.setMaximumHeight(150)
        self.coeff_box.setReadOnly(True)
        lv.addWidget(QLabel("Filter Coefficients (b, a)"))
        lv.addWidget(self.coeff_box)
        lv.addStretch()

        # ── Right ────────────────────────────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right); rv.setSpacing(10)

        self.canvas_resp  = MplCanvas(rows=1, cols=2, figsize=(10, 3.2))
        self.canvas_pz    = MplCanvas(rows=1, cols=1, figsize=(7,  3.5))
        self.canvas_apply = MplCanvas(rows=1, cols=2, figsize=(10, 3.0))

        rv.addWidget(self._section_label("📐  Filter Frequency Response"))
        rv.addWidget(self.canvas_resp)
        rv.addWidget(self._section_label("🎯  Pole-Zero Map"))
        rv.addWidget(self.canvas_pz)
        rv.addWidget(self._section_label("🔈  Before vs After Filtering"))
        rv.addWidget(self.canvas_apply)

        main.addWidget(left)
        main.addWidget(right, stretch=1)

    def _divider(self):
        f = QFrame(); f.setObjectName("divider")
        f.setFrameShape(QFrame.Shape.HLine)
        return f

    def _section_label(self, text):
        lb = QLabel(text)
        lb.setStyleSheet("color:#58a6ff; font-weight:700; font-size:12px; "
                         "padding:4px 0;")
        return lb

    def _build_filter(self):
        fs = self.fs_spin.value()
        order = self.order_spin.value()
        ftype = self.ftype_cb.currentText()
        method = self.fdesign_cb.currentText()
        fc1 = self.fc1_spin.value()
        fc2 = self.fc2_spin.value()
        rp = self.rp_spin.value()
        Wn = ([fc1 / (fs / 2), fc2 / (fs / 2)]
              if ftype in ("bandpass", "bandstop")
              else fc1 / (fs / 2))
        if method == "Butterworth":
            b, a = sp_signal.butter(order, Wn, btype=ftype)
        elif method == "Chebyshev Type I":
            b, a = sp_signal.cheby1(order, rp, Wn, btype=ftype)
        elif method == "Chebyshev Type II":
            b, a = sp_signal.cheby2(order, 40, Wn, btype=ftype)
        elif method == "Elliptic":
            b, a = sp_signal.ellip(order, rp, 40, Wn, btype=ftype)
        elif method == "Bessel":
            b, a = sp_signal.bessel(order, Wn, btype=ftype)
        else:  # FIR
            num_taps = order * 2 + 1
            b = sp_signal.firwin(num_taps, Wn,
                                  pass_zero=(ftype in ("lowpass", "bandstop")))
            a = np.array([1.0])
        return b, a

    def design_filter(self):
        try:
            b, a = self._build_filter()
            self._b, self._a = b, a
        except Exception as e:
            QMessageBox.critical(self, "Design Error", str(e))
            return

        fs = self.fs_spin.value()
        w, h = sp_signal.freqz(b, a, worN=2048, fs=fs)
        mag_db = 20 * np.log10(np.abs(h) + 1e-12)
        phase  = np.unwrap(np.angle(h)) * 180 / np.pi

        # ── Frequency Response ──
        self.canvas_resp.clear_all()
        ax1, ax2 = self.canvas_resp.axes
        ax1.plot(w, mag_db, color=COLORS[0], linewidth=2)
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Magnitude (dB)")
        ax1.set_title(f"{self.fdesign_cb.currentText()} "
                      f"— {self.ftype_cb.currentText().capitalize()}",
                      color="#e6edf3", fontweight="bold")
        ax1.axhline(-3, color=COLORS[2], linestyle="--",
                    linewidth=1, label="-3dB")
        ax1.legend(facecolor="#21262d", edgecolor="#30363d",
                   labelcolor="#e6edf3", fontsize=9)

        ax2.plot(w, phase, color=COLORS[3], linewidth=2)
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Phase (°)")
        ax2.set_title("Phase Response", color="#e6edf3", fontweight="bold")
        self.canvas_resp.draw_safe()

        # ── Pole-Zero ──
        zeros = np.roots(b)
        poles = np.roots(a)
        self.canvas_pz.clear_all()
        ax = self.canvas_pz.ax(0)
        theta = np.linspace(0, 2 * np.pi, 360)
        ax.plot(np.cos(theta), np.sin(theta), "--",
                color="#30363d", linewidth=1.2)
        ax.axhline(0, color="#30363d", linewidth=0.7)
        ax.axvline(0, color="#30363d", linewidth=0.7)
        ax.scatter(poles.real, poles.imag, marker="x",
                   s=100, color="#f85149", linewidths=2.5, label="Poles")
        ax.scatter(zeros.real, zeros.imag, marker="o", s=90,
                   facecolors="none", edgecolors="#58a6ff",
                   linewidths=2.0, label="Zeros")
        ax.set_aspect("equal")
        ax.legend(facecolor="#21262d", edgecolor="#30363d",
                  labelcolor="#e6edf3", fontsize=9)
        ax.set_title("Pole-Zero", color="#e6edf3", fontweight="bold")
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
        self.canvas_pz.draw_safe()

        # Coefficients
        self.coeff_box.setPlainText(
            f"b (numerator):\n{np.array2string(b, precision=6)}\n\n"
            f"a (denominator):\n{np.array2string(a, precision=6)}"
        )

    def apply_filter(self):
        if self._b is None:
            QMessageBox.information(self, "Info", "Design a filter first.")
            return
        fs = self.fs_spin.value()
        t = np.linspace(0, 1, int(fs), endpoint=False)
        f1 = self.test_f1.value()
        f2 = self.test_f2.value()
        x = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)
        y = sp_signal.lfilter(self._b, self._a, x)

        self.canvas_apply.clear_all()
        ax1, ax2 = self.canvas_apply.axes
        t_ms = t[:1000] * 1000
        ax1.plot(t_ms, x[:1000], color=COLORS[0], linewidth=1.2)
        ax1.set_title(f"Input: {f1}Hz + {f2}Hz",
                      color="#e6edf3", fontweight="bold")
        ax1.set_xlabel("Time (ms)"); ax1.set_ylabel("Amplitude")

        ax2.plot(t_ms, y[:1000], color=COLORS[1], linewidth=1.2)
        ax2.set_title("Filtered Output",
                      color="#e6edf3", fontweight="bold")
        ax2.set_xlabel("Time (ms)"); ax2.set_ylabel("Amplitude")
        self.canvas_apply.draw_safe()


# ─────────────────────────────────────────────────────────────────────
#  TAB 4 — NOISE REDUCTION SYSTEM
# ─────────────────────────────────────────────────────────────────────
class NoiseReductionTab(QWidget):
    """Add noise, apply filter, compare before/after in time + freq."""

    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        main = QHBoxLayout(self)
        main.setContentsMargins(14, 14, 14, 14)
        main.setSpacing(14)

        left = QWidget(); left.setFixedWidth(300)
        lv = QVBoxLayout(left); lv.setSpacing(12)

        h = QLabel("Noise Reduction System"); h.setObjectName("heading")
        sub = QLabel("SNR Analysis & Before/After Comparison")
        sub.setObjectName("subheading")
        lv.addWidget(h); lv.addWidget(sub)
        lv.addWidget(self._divider())

        sig_group = QGroupBox("Clean Signal")
        sg = QFormLayout(sig_group)
        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(10, 3000); self.freq_spin.setValue(440)
        self.freq_spin.setSuffix(" Hz")
        self.type_cb = QComboBox()
        self.type_cb.addItems(["Sine", "Square", "Triangle", "Multi-tone"])
        sg.addRow("Frequency:", self.freq_spin)
        sg.addRow("Waveform:", self.type_cb)
        lv.addWidget(sig_group)

        noise_group = QGroupBox("Noise Parameters")
        ng = QFormLayout(noise_group)
        self.noise_type = QComboBox()
        self.noise_type.addItems(["Gaussian White", "Pink Noise",
                                   "Impulsive", "Bandlimited"])
        self.snr_slider = QSlider(Qt.Orientation.Horizontal)
        self.snr_slider.setRange(1, 40); self.snr_slider.setValue(10)
        self.snr_lbl = QLabel("10 dB")
        self.snr_slider.valueChanged.connect(
            lambda v: self.snr_lbl.setText(f"{v} dB"))
        ng.addRow("Noise Type:", self.noise_type)
        ng.addRow("SNR:", self.snr_slider)
        ng.addRow("", self.snr_lbl)
        lv.addWidget(noise_group)

        filter_group = QGroupBox("Denoising Filter")
        fg = QFormLayout(filter_group)
        self.filt_type = QComboBox()
        self.filt_type.addItems(["Butterworth LP", "Moving Average",
                                  "Median Filter", "Wiener Filter",
                                  "Butterworth HP"])
        self.filt_order = QSpinBox()
        self.filt_order.setRange(1, 20); self.filt_order.setValue(4)
        self.cutoff_spin = QDoubleSpinBox()
        self.cutoff_spin.setRange(10, 4000); self.cutoff_spin.setValue(800)
        self.cutoff_spin.setSuffix(" Hz")
        fg.addRow("Filter:", self.filt_type)
        fg.addRow("Order:", self.filt_order)
        fg.addRow("Cutoff:", self.cutoff_spin)
        lv.addWidget(filter_group)

        btn = QPushButton("▶  Run Noise Reduction")
        btn.setObjectName("accent")
        btn.clicked.connect(self.process)
        lv.addWidget(btn)

        self.metrics_box = QTextEdit()
        self.metrics_box.setMaximumHeight(160)
        self.metrics_box.setReadOnly(True)
        lv.addWidget(QLabel("SNR Metrics"))
        lv.addWidget(self.metrics_box)
        lv.addStretch()

        # Right
        right = QWidget()
        rv = QVBoxLayout(right); rv.setSpacing(10)

        self.canvas_time = MplCanvas(rows=2, cols=1, figsize=(10, 4.5))
        self.canvas_freq = MplCanvas(rows=1, cols=3, figsize=(10, 3.2))
        self.canvas_snr  = MplCanvas(rows=1, cols=1, figsize=(10, 2.5))

        rv.addWidget(self._section_label("⏱  Time Domain: Clean | Noisy | Filtered"))
        rv.addWidget(self.canvas_time)
        rv.addWidget(self._section_label("📊  Frequency Spectra Comparison"))
        rv.addWidget(self.canvas_freq)
        rv.addWidget(self._section_label("📉  SNR Improvement Visualization"))
        rv.addWidget(self.canvas_snr)

        main.addWidget(left)
        main.addWidget(right, stretch=1)

    def _divider(self):
        f = QFrame(); f.setObjectName("divider")
        f.setFrameShape(QFrame.Shape.HLine)
        return f

    def _section_label(self, text):
        lb = QLabel(text)
        lb.setStyleSheet("color:#58a6ff; font-weight:700; font-size:12px; "
                         "padding:4px 0;")
        return lb

    def _snr_db(self, clean, noisy):
        noise = noisy - clean
        ps = np.mean(clean ** 2)
        pn = np.mean(noise ** 2)
        return 10 * np.log10(ps / (pn + 1e-15))

    def process(self):
        fs = 8000
        t = np.linspace(0, 1, fs, endpoint=False)
        f0 = self.freq_spin.value()
        wt = self.type_cb.currentText()

        if wt == "Sine":
            clean = np.sin(2 * np.pi * f0 * t)
        elif wt == "Square":
            clean = sp_signal.square(2 * np.pi * f0 * t)
        elif wt == "Triangle":
            clean = sp_signal.sawtooth(2 * np.pi * f0 * t, 0.5)
        else:  # Multi-tone
            clean = (np.sin(2 * np.pi * f0 * t) +
                     0.5 * np.sin(2 * np.pi * f0 * 3 * t) +
                     0.25 * np.sin(2 * np.pi * f0 * 5 * t))
            clean /= np.max(np.abs(clean))

        snr_target = self.snr_slider.value()
        sig_power = np.mean(clean ** 2)
        noise_power = sig_power / (10 ** (snr_target / 10))
        noise_type = self.noise_type.currentText()
        if noise_type == "Gaussian White":
            noise = np.random.normal(0, np.sqrt(noise_power), len(t))
        elif noise_type == "Pink Noise":
            wn = np.random.randn(len(t))
            b_pink, a_pink = sp_signal.butter(1, 0.01 / (fs / 2), "low")
            noise = sp_signal.lfilter(b_pink, a_pink, wn)
            noise = noise / (np.std(noise) + 1e-9) * np.sqrt(noise_power)
        elif noise_type == "Impulsive":
            noise = np.random.normal(0, np.sqrt(noise_power * 0.1), len(t))
            idx = np.random.choice(len(t), size=int(len(t) * 0.01))
            noise[idx] = np.random.normal(0, np.sqrt(noise_power) * 5, len(idx))
        else:  # Bandlimited
            wn = np.random.randn(len(t))
            b_bl, a_bl = sp_signal.butter(2,
                [200 / (fs / 2), 1000 / (fs / 2)], "band")
            noise = sp_signal.lfilter(b_bl, a_bl, wn)
            noise = noise / (np.std(noise) + 1e-9) * np.sqrt(noise_power)

        noisy = clean + noise

        # Denoising
        ft = self.filt_type.currentText()
        order = self.filt_order.value()
        fc = self.cutoff_spin.value()
        Wn = fc / (fs / 2)

        if ft == "Butterworth LP":
            b, a = sp_signal.butter(order, Wn, "low")
            filtered = sp_signal.lfilter(b, a, noisy)
        elif ft == "Butterworth HP":
            b, a = sp_signal.butter(order, Wn, "high")
            filtered = sp_signal.lfilter(b, a, noisy)
        elif ft == "Moving Average":
            k = max(2, int(fs / fc))
            filtered = np.convolve(noisy, np.ones(k) / k, mode="same")
        elif ft == "Median Filter":
            k = max(3, int(fs / fc))
            k = k if k % 2 == 1 else k + 1
            from scipy.ndimage import median_filter
            filtered = median_filter(noisy, size=k)
        else:  # Wiener
            from scipy.signal import wiener
            filtered = wiener(noisy, mysize=max(3, int(fs / fc)))

        snr_in  = self._snr_db(clean, noisy)
        snr_out = self._snr_db(clean, filtered)
        snr_imp = snr_out - snr_in

        # ── Time plots ──
        show = 2000
        self.canvas_time.clear_all()
        ax1, ax2 = self.canvas_time.axes
        ax1.plot(t[:show] * 1000, clean[:show],
                 color=COLORS[1], linewidth=1.2, label="Clean", alpha=0.8)
        ax1.plot(t[:show] * 1000, noisy[:show],
                 color=COLORS[2], linewidth=0.8, label="Noisy", alpha=0.7)
        ax1.legend(facecolor="#21262d", edgecolor="#30363d",
                   labelcolor="#e6edf3", fontsize=9)
        ax1.set_title("Input Signal", color="#e6edf3", fontweight="bold")
        ax1.set_ylabel("Amplitude")

        ax2.plot(t[:show] * 1000, clean[:show],
                 color=COLORS[1], linewidth=1.5, label="Clean", alpha=0.5)
        ax2.plot(t[:show] * 1000, filtered[:show],
                 color=COLORS[0], linewidth=1.2, label="Filtered", alpha=0.9)
        ax2.legend(facecolor="#21262d", edgecolor="#30363d",
                   labelcolor="#e6edf3", fontsize=9)
        ax2.set_title("Filtered Output", color="#e6edf3", fontweight="bold")
        ax2.set_xlabel("Time (ms)"); ax2.set_ylabel("Amplitude")
        self.canvas_time.draw_safe()

        # ── Frequency comparison ──
        N = len(clean)
        half = N // 2
        for i, (sig, color, title) in enumerate([
            (clean,    COLORS[1], "Clean"),
            (noisy,    COLORS[2], "Noisy"),
            (filtered, COLORS[0], "Filtered"),
        ]):
            self.canvas_freq.clear_all()
            break
        self.canvas_freq.clear_all()
        for i, (sig, color, title) in enumerate([
            (clean,    COLORS[1], "Clean"),
            (noisy,    COLORS[2], "Noisy"),
            (filtered, COLORS[0], "Filtered"),
        ]):
            ax = self.canvas_freq.axes[i]
            Y = fft(sig)
            freqs = fftfreq(N, 1 / fs)
            mag = 2 * np.abs(Y[:half]) / N
            ax.plot(freqs[:half], 20 * np.log10(mag + 1e-12),
                    color=color, linewidth=1.2)
            ax.set_title(title, color="#e6edf3", fontweight="bold")
            ax.set_xlabel("Hz"); ax.set_xlim(0, fs / 2)
            if i == 0:
                ax.set_ylabel("dB")
        self.canvas_freq.draw_safe()

        # ── SNR bar ──
        self.canvas_snr.clear_all()
        axs = self.canvas_snr.ax(0)
        bars = axs.bar(["SNR Input", "SNR Output", "Improvement"],
                       [snr_in, snr_out, snr_imp],
                       color=[COLORS[2], COLORS[1], COLORS[0]],
                       width=0.5, edgecolor="#30363d")
        for bar, val in zip(bars, [snr_in, snr_out, snr_imp]):
            axs.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.3,
                     f"{val:.2f} dB", ha="center", va="bottom",
                     color="#e6edf3", fontsize=11, fontweight="bold")
        axs.set_ylabel("SNR (dB)")
        axs.set_title("Signal-to-Noise Ratio Analysis",
                      color="#e6edf3", fontweight="bold")
        self.canvas_snr.draw_safe()

        self.metrics_box.setPlainText(
            f"Input SNR   : {snr_in:.2f} dB\n"
            f"Output SNR  : {snr_out:.2f} dB\n"
            f"Improvement : {snr_imp:+.2f} dB\n"
            f"─────────────────────\n"
            f"Noise Type  : {noise_type}\n"
            f"Filter      : {ft}\n"
            f"Cutoff      : {fc:.0f} Hz\n"
        )


# ─────────────────────────────────────────────────────────────────────
#  MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signal Simulation Suite  —  Fourier & Z-Transform Analyzer")
        self.setMinimumSize(1280, 860)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        cl = QVBoxLayout(central)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)

        # Title bar
        title_bar = QWidget()
        title_bar.setFixedHeight(56)
        title_bar.setStyleSheet(
            "background: qlineargradient("
            "x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #0d1117, stop:1 #161b22);"
            "border-bottom: 1px solid #30363d;")
        tb_layout = QHBoxLayout(title_bar)
        tb_layout.setContentsMargins(16, 0, 16, 0)

        logo = QLabel("◈")
        logo.setStyleSheet("font-size:22px; color:#388bfd;")
        title_lbl = QLabel("Signal Simulation Suite")
        title_lbl.setStyleSheet(
            "font-size:18px; font-weight:800; color:#e6edf3; "
            "letter-spacing:-0.5px;")
        sub_lbl = QLabel("Fourier & Z-Transform  ·  AI-Assisted Engineering")
        sub_lbl.setStyleSheet("font-size:12px; color:#8b949e; margin-left:8px;")

        badge1 = QLabel("Fourier")
        badge2 = QLabel("Z-Transform")
        badge3 = QLabel("DSP")
        for b in [badge1, badge2, badge3]:
            b.setObjectName("badge")

        tb_layout.addWidget(logo)
        tb_layout.addSpacing(8)
        tb_layout.addWidget(title_lbl)
        tb_layout.addWidget(sub_lbl)
        tb_layout.addStretch()
        tb_layout.addWidget(badge1)
        tb_layout.addWidget(badge2)
        tb_layout.addWidget(badge3)
        cl.addWidget(title_bar)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        self.tab_fourier  = FourierTab()
        self.tab_ztrans   = ZTransformTab()
        self.tab_filter   = FilterDesignerTab()
        self.tab_noise    = NoiseReductionTab()

        self.tabs.addTab(self.tab_fourier,  "📊  Fourier Analyzer")
        self.tabs.addTab(self.tab_ztrans,   "🎯  Z-Transform")
        self.tabs.addTab(self.tab_filter,   "⚙  Filter Designer")
        self.tabs.addTab(self.tab_noise,    "🔇  Noise Reduction")

        cl.addWidget(self.tabs)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage(
            "  Ready  ·  Signal Simulation Suite  ·  "
            "Developed with Claude (Anthropic)  ·  "
            "Mata Kuliah: DSP / Linier System")

        self.setStyleSheet(DARK_STYLE)


# ─────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Signal Simulation Suite")
    app.setOrganizationName("DSP Lab")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
