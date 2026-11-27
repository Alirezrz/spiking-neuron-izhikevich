import numpy as np
from matplotlib import pyplot as plt

try:
    from pymonntorch import *
    import torch
except ImportError:          # lets the plotting code be imported without the framework
    Behavior = object
    torch = None

# ----
# Task 4: constant current + noise (low / medium / high noise level)
#
# OUNoisyCurrent    : I[t] = I_base + n[t], where n is smooth (Ornstein-Uhlenbeck) noise
#                       dn = -n/tau * dt + noise_std * sqrt(2 dt / tau) * N(0, 1)
#                     n has mean 0, standard deviation noise_std and correlation time tau (ms),
#                     so it drifts smoothly instead of jumping at every step.
#                     (NoisyCurrent in current.py is the white-noise version.)
# run_noisy         : simulate one neuron with that current
# plot_noise_levels : v(t) on top, I(t) below for each noise level
#                     (I panel: the base current and the noisy current, two colors)
#
# The same seed is used for all levels, so the noise traces are the same signal
# scaled by noise_std: differences come only from the amplitude.
# ----

THRESHOLD = 30.0
LEVELS_OU = {"low": 1, "medium": 3, "high": 8}        # std of the smooth noise
LEVELS_WHITE = {"low": 5, "medium": 20, "high": 50}   # per-step white noise needs larger values
COLORS = {"low": "#0072B2", "medium": "#E69F00", "high": "#D55E00"}


class OUNoisyCurrent(Behavior):

    def initialize(self, neurons):
        self.I = self.parameter("I", 10.0)
        self.sigma = self.parameter("noise_std", 1.0)
        self.tau = self.parameter("tau", 10.0)       # ms, how long the noise "remembers"
        base = neurons.vector(mode=0.0)
        neurons.noise = self.sigma * torch.randn_like(base)   # start in the stationary distribution
        neurons.I = neurons.vector(mode=self.I) + neurons.noise

    def forward(self, neurons):
        dt = neurons.network.dt
        z = torch.randn_like(neurons.noise)
        neurons.noise = (neurons.noise - dt * neurons.noise / self.tau
                         + self.sigma * (2 * dt / self.tau) ** 0.5 * z)
        neurons.I = neurons.vector(mode=self.I) + neurons.noise


def run_noisy(p, I_base, noise_std, T=500, dt=0.1, seed=0, noise="ou", tau=10.0):
    """Return t, v, u, I (the recorded, noisy current) for one neuron."""
    from pymonntorch import Network, NeuronGroup, Recorder
    from network import TimeResolution
    from neuron import Izhikevich

    if noise == "ou":
        current = OUNoisyCurrent(I=I_base, noise_std=noise_std, tau=tau)
    else:
        from current import NoisyCurrent
        current = NoisyCurrent(I=I_base, noise_std=noise_std)

    torch.manual_seed(seed)          # same noise sequence for every level
    net = Network(behavior={1: TimeResolution(dt=dt)})
    ng = NeuronGroup(
        net=net, size=1,
        behavior={
            10: current,
            20: Izhikevich(a=p["a"], b=p["b"], c=p["c"], d=p["d"], v_init=p["v_init"]),
            600: Recorder(["v", "u", "I"]),
        },
        tag="ng",
    )
    net.initialize()
    net.simulate_iterations(int(T / dt))

    v = torch.as_tensor(ng["v", 0])[:, 0].cpu().numpy()
    u = torch.as_tensor(ng["u", 0])[:, 0].cpu().numpy()
    I = torch.as_tensor(ng["I", 0])[:, 0].cpu().numpy()
    t = np.arange(len(v)) * dt
    return t, v, u, I


def plot_noise_levels(name, p, I_base=10.0, levels=None, noise="ou", tau=10.0,
                      T=500, seed=0, clip=True):
    if levels is None:
        levels = LEVELS_OU if noise == "ou" else LEVELS_WHITE
    n = len(levels)
    fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 5.2), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2]}, squeeze=False)
    ymin, ymax = -80, 35
    i_lo, i_hi = np.inf, -np.inf
    results = {}
    for lvl, sigma in levels.items():
        results[lvl] = run_noisy(p, I_base, sigma, T=T, seed=seed, noise=noise, tau=tau)
        I = results[lvl][3]
        i_lo, i_hi = min(i_lo, I.min(), I_base), max(i_hi, I.max(), I_base)
        ymin = min(ymin, results[lvl][1].min() - 5)   # keep the lowest dips of v visible

    for k, (lvl, sigma) in enumerate(levels.items()):
        t, v, u, I = results[lvl]
        color = COLORS.get(lvl, "k")
        n_sp = int((v >= THRESHOLD).sum())
        ax_v, ax_i = axes[0, k], axes[1, k]
        ax_v.plot(t, np.minimum(v, THRESHOLD) if clip else v, color=color, lw=0.9)
        ax_v.axhline(THRESHOLD, color="gray", ls="--", lw=0.6)
        ax_v.set_ylim(ymin, ymax)
        ax_v.set_title(f"{lvl} noise  (σ = {sigma}),  {n_sp} spikes",
                       fontsize=10, color=color, fontweight="bold")
        ax_i.plot(t, I, color=color, lw=0.9, label="I + noise")
        ax_i.axhline(I_base, color="k", ls="--", lw=1.4, label=f"I (base) = {I_base}")
        pad = 0.1 * (i_hi - i_lo)
        ax_i.set_ylim(i_lo - pad, i_hi + pad)
        ax_i.set_xlabel("time (ms)")
        if k == 0:
            ax_v.set_ylabel("v (mV)")
            ax_i.set_ylabel("I")
        ax_i.legend(fontsize=8, loc="upper right")

    kind = f"smooth (OU, τ = {tau} ms) noise" if noise == "ou" else "white noise"
    fig.suptitle(f"{name}: {p['name']}  (a={p['a']}, b={p['b']}, c={p['c']}, d={p['d']}),  "
                 f"base current I = {I_base},  {kind}")
    fig.tight_layout()
    plt.show()
    return results