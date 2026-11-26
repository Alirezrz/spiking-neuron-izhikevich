import numpy as np
from matplotlib import pyplot as plt

# ----
# Analysis plots for one simulated Izhikevich neuron.
# All functions take the arrays returned by run_pattern: t, v, u, I
# and the parameter dict p of the pattern.
#
# plot_full        : v(t), u(t), I(t) stacked (shared time axis)
# plot_phase_plane : u vs v trajectory + nullclines (for the current I_phase)
# spike_times      : spike times (ms) found from the recorded v
# plot_isi         : instantaneous firing frequency (1/ISI) per spike
# plot_analysis    : all of the above in one figure
# ----

THRESHOLD = 30.0


def spike_times(t, v, threshold=THRESHOLD):
    """Times (ms) of the steps where v is at or above the threshold (raw v, not clipped)."""
    idx = np.where(v >= threshold)[0]
    return t[idx]


def plot_full(name, p, t, v, u, I, clip=True):
    vv = np.minimum(v, THRESHOLD) if clip else v
    fig, axes = plt.subplots(3, 1, figsize=(7, 6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 1]})
    axes[0].plot(t, vv, "k", lw=0.8)
    axes[0].set_ylabel("v (mV)")
    axes[0].set_title(f"{name}: {p['name']}\n"
                      f"a={p['a']}, b={p['b']}, c={p['c']}, d={p['d']}")
    axes[1].plot(t, u, "r", lw=0.8)
    axes[1].set_ylabel("u (recovery)")
    axes[2].plot(t, I, "b")
    axes[2].set_ylabel("I")
    axes[2].set_xlabel("time (ms)")
    fig.tight_layout()
    plt.show()


def plot_phase_plane(name, p, v, u, I_phase, ax=None):
    """
    Trajectory in the (v, u) plane, colored by time, with the nullclines
      v-nullcline : u = 0.04 v^2 + 5 v + 140 + I_phase
      u-nullcline : u = b v
    I_phase is the (constant) current you want the nullclines drawn for,
    e.g. the value of the step.
    """
    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    vv = np.minimum(v, THRESHOLD)
    vs = np.linspace(-90, 35, 400)
    ax.plot(vs, 0.04 * vs**2 + 5 * vs + 140 + I_phase, "g--",
            label=f"v-nullcline (I={I_phase})")
    ax.plot(vs, p["b"] * vs, "m--", label="u-nullcline (u=bv)")
    sc = ax.scatter(vv, u, c=np.arange(len(v)), s=2, cmap="viridis")
    ax.set_xlabel("v (mV)")
    ax.set_ylabel("u")
    ax.set_xlim(-90, 35)
    lo, hi = min(u.min(), (p["b"] * vs).min()), u.max()
    ax.set_ylim(lo - 5, hi + 5)
    ax.set_title(f"{name}: phase plane")
    ax.legend(fontsize=7)
    if show:
        plt.colorbar(sc, label="time step")
        plt.tight_layout()
        plt.show()


def plot_isi(name, t, v, ax=None):
    """Instantaneous frequency 1/ISI (Hz) at each spike: shows adaptation / bursting."""
    st = spike_times(t, v)
    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))
    if len(st) >= 2:
        isi = np.diff(st)                 # ms
        ax.plot(np.arange(1, len(st)), 1000.0 / isi, "o-")
    ax.set_xlabel("spike number")
    ax.set_ylabel("1/ISI (Hz)")
    ax.set_title(f"{name}: instantaneous frequency ({len(st)} spikes)")
    if show:
        plt.tight_layout()
        plt.show()


def plot_analysis(name, p, t, v, u, I, I_phase=None):
    """v, u, I plus phase plane and 1/ISI in one figure."""
    if I_phase is None:
        I_phase = float(np.max(I))
    vv = np.minimum(v, THRESHOLD)
    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.4, 1])

    ax_v = fig.add_subplot(gs[0, 0])
    ax_u = fig.add_subplot(gs[1, 0], sharex=ax_v)
    ax_i = fig.add_subplot(gs[2, 0], sharex=ax_v)
    ax_ph = fig.add_subplot(gs[0:2, 1])
    ax_isi = fig.add_subplot(gs[2, 1])

    ax_v.plot(t, vv, "k", lw=0.8); ax_v.set_ylabel("v (mV)")
    ax_v.set_title(f"{name}: {p['name']}  "
                   f"(a={p['a']}, b={p['b']}, c={p['c']}, d={p['d']})", fontsize=9)
    ax_u.plot(t, u, "r", lw=0.8); ax_u.set_ylabel("u")
    ax_i.plot(t, I, "b"); ax_i.set_ylabel("I"); ax_i.set_xlabel("time (ms)")

    plot_phase_plane(name, p, v, u, I_phase, ax=ax_ph)
    plot_isi(name, t, v, ax=ax_isi)
    fig.tight_layout()
    plt.show()