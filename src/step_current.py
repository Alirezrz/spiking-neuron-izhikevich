import numpy as np
from matplotlib import pyplot as plt

from pymonntorch import *
import torch


THRESHOLD = 30.0


class StepSweepCurrent(Behavior):

    def initialize(self, neurons):
        self.I_values = torch.tensor(self.parameter("I_values", [10.0]), dtype=torch.float32)
        self.t_on = self.parameter("t_start", 50.0)
        neurons.I = neurons.vector(mode=0.0)

    def forward(self, neurons):
        t = neurons.network.iteration * neurons.network.dt
        if t >= self.t_on:
            neurons.I = self.I_values.clone()
        else:
            neurons.I = neurons.vector(mode=0.0)


def simulate_step(p, I_values, T=500, t_on=50, dt=0.1):
    from pymonntorch import Network, NeuronGroup, Recorder
    from .network import TimeResolution
    from .neuron import Izhikevich

    net = Network(behavior={1: TimeResolution(dt=dt)})
    ng = NeuronGroup(
        net=net, size=len(I_values),
        behavior={
            10: StepSweepCurrent(I_values=list(I_values), t_start=t_on),
            20: Izhikevich(a=p["a"], b=p["b"], c=p["c"], d=p["d"], v_init=p["v_init"]),
            600: Recorder(["v", "u", "I"]),
        },
        tag="ng",
    )
    net.initialize()
    net.simulate_iterations(int(T / dt))

    v = torch.as_tensor(ng["v", 0]).cpu().numpy()
    u = torch.as_tensor(ng["u", 0]).cpu().numpy()
    I = torch.as_tensor(ng["I", 0]).cpu().numpy()
    t = np.arange(v.shape[0]) * dt
    return t, v, u, I


def spike_times(t, v_col, threshold=THRESHOLD):
    return t[np.where(v_col >= threshold)[0]]


def steady_rate(st, k=5):
    if len(st) < 2:
        return 0.0
    
    isi = np.diff(st)
    
    return 1000.0 / np.mean(isi[-k:])


def rheobase(b):
    return (5 - b) ** 2 / 0.16 - 140


def plot_step_responses(name, p, I_values=(5, 10, 20), T=500, t_on=50):
    t, v, u, I = simulate_step(p, I_values, T=T, t_on=t_on)
    n = len(I_values)
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 6), sharex=True,gridspec_kw={"height_ratios": [3, 2, 1]}, squeeze=False)
    
    vmin, vmax = -80, 35
    
    for k, I0 in enumerate(I_values):
        vv = np.minimum(v[:, k], THRESHOLD)
        ax_v, ax_u, ax_i = axes[0, k], axes[1, k], axes[2, k]
        ax_v.plot(t, vv, "k", lw=0.8)
        ax_v.axhline(THRESHOLD, color="gray", ls="--", lw=0.8)
        ax_v.axhline(p["c"], color="orange", ls=":", lw=0.8)
        ax_v.set_ylim(vmin, vmax)
        ax_v.set_title(f"I = {I0}  ({len(spike_times(t, v[:, k]))} spikes)", fontsize=10)
        ax_u.plot(t, u[:, k], "r", lw=0.8)
        ax_i.plot(t, I[:, k], "b")
        ax_i.set_xlabel("time (ms)")
        
        if k == 0:
            ax_v.set_ylabel("v (mV)")
            ax_u.set_ylabel("u")
            ax_i.set_ylabel("I")
    
    fig.suptitle(f"{name}: {p['full']}  (a={p['a']}, b={p['b']}, c={p['c']}, d={p['d']})")
    fig.tight_layout()
    
    plt.show()


def plot_FI(name, p, currents=None, T=1000, t_on=100, mark=(),colors=("#0072B2", "#E69F00", "#D55E00")):

    if currents is None:
        currents = np.arange(0, 20.01, 0.5)
    currents = np.union1d(np.asarray(currents, float), np.asarray(mark, float))
    
    t, v, u, I = simulate_step(p, list(currents), T=T, t_on=t_on)
    rates = np.array([steady_rate(spike_times(t, v[:, k])) for k in range(len(currents))])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(currents, rates, "-o", color="k", ms=3, lw=1.2, zorder=2)
    ax.axvline(rheobase(p["b"]), color="gray", ls="--", lw=0.8,label=f"threshold current ({rheobase(p['b']):.1f})")
    
    for I0, col in zip(mark, colors):
        k = int(np.argmin(np.abs(currents - I0)))
        ax.scatter(currents[k], rates[k], s=70, color=col, zorder=3,label=f"I = {I0}:  {rates[k]:.1f} Hz")
        
    ax.set_xlabel("I (step amplitude)")
    ax.set_ylabel("f = 1/T  (Hz)")
    ax.set_title(f"{name}: frequency-current relation\n"f"a={p['a']}, b={p['b']}, c={p['c']}, d={p['d']}")
    
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    plt.show()
    
    return currents, rates