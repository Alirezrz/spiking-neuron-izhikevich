# Izhikevich Spiking Neuron Model

A Python implementation and dynamical analysis of the **Izhikevich (2003)** spiking neuron model using **PyMonnTorch** and **PyTorch**. This project reproduces canonical cortical and thalamic firing patterns, characterizes frequency–current ($F\text{–}I$) relationships, and investigates neuronal responses to noisy synaptic-like inputs.

## Model Equations

$$v' = 0.04v^2 + 5v + 140 - u + I$$
$$u' = a(bv - u)$$

with after-spike resetting:

$$\text{if } v \ge 30\text{ mV}, \quad \text{then } \begin{cases} v \leftarrow c \\ & u \leftarrow u + d \end{cases}$$

## Experiments & Features

* **Canonical Firing Patterns:** Simulates and visualizes `RS` (Regular Spiking), `IB` (Intrinsically Bursting), `CH` (Chattering), `FS` (Fast Spiking), `LTS` (Low-Threshold Spiking), `TC` (Thalamo-Cortical tonic & rebound burst), and `RZ` (Resonator, with tuned subthreshold oscillation and bistable pulse switching).
* **Step-Current Response & $F\text{–}I$ Curves:** Compares low, medium, and high step currents for `RS` and runs vectorized current sweeps to compute steady-state firing rates and analytical rheobase thresholds across patterns.
* **Noisy Input Dynamics:** Evaluates `RS` firing under smooth Ornstein–Uhlenbeck (OU) noise ($\tau = 10\text{ ms}$) at both suprathreshold ($I = 10$) and subthreshold ($I = 3$) regimes.
* **Dynamical Analysis:** Includes phase-plane trajectories with $v$- and $u$-nullclines, instantaneous frequency ($1/\text{ISI}$) plots, and spike raster comparisons.

## Project Structure

```text
spiking-neuron-izhikevich/
├── run_izhikevich.ipynb      # Main notebook running all experiments & plots
├── requirements.txt          # Python dependencies
├── src/
│   ├── __init__.py
│   ├── network.py            # Time resolution behavior (dt = 0.1 ms)
│   ├── neuron.py             # Izhikevich neuron dynamics & spike reset
│   ├── current.py            # Constant, Step, Scheduled, and Noisy currents
│   ├── parameters.py         # Parameter presets for all firing patterns
│   ├── step_current.py       # Vectorized current sweep & F–I curve utilities
│   ├── plot_step_compare.py  # Stacked step-response, raster & F–I plots
│   └── analysis_plots.py     # Phase plane, nullclines & ISI visualization
├── figures/                  # Generated simulation plots (patterns, F–I, noise)
└── report.pdf                # ouputs of the model
