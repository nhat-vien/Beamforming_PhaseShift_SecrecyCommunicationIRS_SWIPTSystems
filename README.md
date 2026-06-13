# Power Sweep GUI for SWIPT-IRS

This project provides a Python/Tkinter GUI to compare four baseline methods
for joint beamforming and IRS phase-shift optimization in a SWIPT secrecy setup.

## Project Context

This codebase is associated with the following research paper Deep learning-driven joint beamforming and phase shift optimization for secrecy communication of IRS-aided SWIPT systems:
- https://www.sciencedirect.com/science/article/abs/pii/S1389128626002926

It is also a research output of the project with code B2004.DNA.19.

## Overview

- Power sweep at fixed transmit powers: 18, 23, 28, 33, 38, 43 dBm.
- Selectable baseline algorithms: AO, PSO, ZF, MRT.
- Real-time plotting in the GUI (4 subplots).
- Progress tracking with live log and status updates.
- Stop control to cancel a running sweep.

## Project Structure

- ao_gui_power_sweep_full.py: main GUI application (simulation + plotting).
- requirements.txt: Python dependencies.

## Requirements

- Python 3.9+ recommended.
- Tkinter available in your Python installation.
- Works on Windows, Linux, and macOS (with Tk support).

## Installation

```bash
pip install -r requirements.txt
```

Notes:
- cvxpy is optional but recommended for AO beamforming optimization.
- If cvxpy is unavailable, the code uses a fallback strategy and still runs.

## Quick Start

```bash
python ao_gui_power_sweep_full.py
```

## How to Use

1. Set simulation parameters in the left panel.
2. Select which algorithms to run (AO/PSO/ZF/MRT).
3. Click Run Power Sweep.
4. Monitor progress in the log and status bar.
5. Inspect updated curves after each power level.
6. Click Stop to terminate early if needed.

## GUI Parameters

User-configurable parameters:
- M (AP antennas), default: 32.
- N (IRS elements), default: 50.
- Samples per power, default: 20.
- AO Trials, default: 3.
- AO Max Iter, default: 10.

Fixed simulation settings in the current script:
- Power levels: [18, 23, 28, 33, 38, 43] dBm.
- rho = 0.5, eta = 0.8, sigma2 = 1e-8.
- K_RICIAN = 10, radius = 1.5.
- Fixed node locations for BS, RIS, IU center, and EU center.

## Output Metrics and Plots

The GUI displays four metrics versus transmit power:
- Secrecy Rate (R_s).
- Energy Harvesting (E_I, dBm).
- Information Rate (R_I).
- Eavesdropper Rate (R_E).

Default plot styles:
- AO: red circle markers.
- PSO: purple square markers.
- ZF: green triangle markers.
- MRT: blue diamond markers.

## Runtime Tips

For quick checks:
- Samples: 10-20.
- AO Trials: 3.
- AO Max Iter: 10.

For more stable averages:
- Increase Samples and AO Trials.
- Enable only required algorithms to reduce runtime.

Total runtime depends on:
- number of enabled algorithms,
- samples per power level,
- and AO/PSO iteration settings.

## Troubleshooting

- Tkinter import errors:
  install a Python distribution with Tcl/Tk support.
- Missing cvxpy:
  install cvxpy, or run without it using fallback mode.
- Slow or unresponsive run:
  reduce Samples, AO trials/iterations, or disable some algorithms.

## Optional: Build an Executable

PyInstaller is included in requirements. Example:

```bash
python -m PyInstaller --onefile --windowed --collect-all cvxpy ao_gui_power_sweep_full.py
```

