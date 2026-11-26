import numpy as np
from matplotlib import pyplot as plt

THRESHOLD = 30.0


def plot_step_comparison(step_patterns, run_fn, title=None,
                         colors=("tab:blue", "tab:green", "tab:red"),
                         clip=True):
    fig, (ax_v, ax_i) = plt.subplots(2, 1, figsize=(8, 5), sharex=True,gridspec_kw={"height_ratios": [4, 1]})
    first = None
    
    for color, (name, p) in zip(colors, step_patterns.items()):
        t, v, u, I = run_fn(p)
        if clip:
            v = np.minimum(v, THRESHOLD)        
        I_step = p["schedule"][-1][1]           
        level = name.split("_")[-1]            
        label = f"I = {I_step} ({level})"
        ax_v.plot(t, v, color=color, lw=0.9, label=label)
        ax_i.plot(t, I, color=color, lw=1.5)
        first = p if first is None else first

    ax_v.axhline(THRESHOLD, color="gray", ls="--", lw=0.8)
    ax_v.set_ylabel("v (mV)")
    ax_v.legend(loc="lower right", fontsize=9)
    ax_v.set_title(title or f"{first['name'].split(',')[0]}: response to step currents\n"f"a={first['a']}, b={first['b']}, c={first['c']}, d={first['d']}")
    
    ax_i.set_ylabel("I")
    ax_i.set_xlabel("time (ms)")
    fig.tight_layout()
    plt.show()


def plot_step_stacked(step_patterns, run_fn, title=None,colors=("#0072B2", "#E69F00", "#D55E00"), clip=True):

    items = list(step_patterns.items())
    n = len(items)
    fig, axes = plt.subplots(n + 1, 1, figsize=(8, 2.0 * n + 1.8), sharex=True,gridspec_kw={"height_ratios": [3] * n + [1.3]})
    
    first = items[0][1]
    spikes = []
    
    for k, ((name, p), color) in enumerate(zip(items, colors)):
        t, v, u, I = run_fn(p)
        st = t[np.where(v >= THRESHOLD)[0]]
        spikes.append(st)
        vv = np.minimum(v, THRESHOLD) if clip else v
        ax = axes[k]
        ax.plot(t, vv, color=color, lw=1.0)
        ax.axhline(THRESHOLD, color="gray", ls="--", lw=0.6)
        ax.set_ylim(-80, 35)
        ax.set_ylabel("v (mV)")
        I_step = p["schedule"][-1][1]
        level = name.split("_")[-1]
        ax.set_title(f"I = {I_step} ({level}),  {len(st)} spikes", loc="left",fontsize=10, color=color, fontweight="bold", pad=3)

    ax_r = axes[-1]
    ax_r.eventplot(spikes, colors=list(colors[:n]), lineoffsets=range(n),linelengths=0.8, linewidths=1.2)
    
    ax_r.set_yticks(range(n))
    ax_r.set_yticklabels([f"I={p['schedule'][-1][1]}" for _, p in items], fontsize=8)
    ax_r.set_ylabel("spikes")
    ax_r.set_xlabel("time (ms)")
    axes[0].set_title(title or f"{first['name'].split(',')[0]}: response to step currents\n"f"a={first['a']}, b={first['b']}, c={first['c']}, d={first['d']}")
    
    fig.tight_layout()
    plt.show()


def plot_step_and_FI(step_patterns, run_fn, p, name="RS", currents=None,T=1000, t_on=100,colors=("#0072B2", "#E69F00", "#D55E00"), clip=True):
    
    import step_current as sc

    items = list(step_patterns.items())
    n = len(items)
    marks = [pp["schedule"][-1][1] for _, pp in items]

    fig = plt.figure(figsize=(13, 2.0 * n + 1.8))
    gs = fig.add_gridspec(n + 1, 2, width_ratios=[1.6, 1],height_ratios=[3] * n + [1.3], wspace=0.25)

    axes = [fig.add_subplot(gs[0, 0])]
    for k in range(1, n + 1):
        axes.append(fig.add_subplot(gs[k, 0], sharex=axes[0]))
    spikes = []
    
    for k, ((nm, pp), color) in enumerate(zip(items, colors)):
        t, v, u, I = run_fn(pp)
        st = t[np.where(v >= THRESHOLD)[0]]
        spikes.append(st)
        vv = np.minimum(v, THRESHOLD) if clip else v
        ax = axes[k]
        ax.plot(t, vv, color=color, lw=1.0)
        ax.axhline(THRESHOLD, color="gray", ls="--", lw=0.6)
        ax.set_ylim(-80, 35)
        ax.set_ylabel("v (mV)")
        ax.set_title(f"I = {marks[k]} ({nm.split('_')[-1]}),  {len(st)} spikes",loc="left", fontsize=10, color=color, fontweight="bold", pad=3)
        if k < n:
            plt.setp(ax.get_xticklabels(), visible=False)
    ax_r = axes[-1]
    ax_r.eventplot(spikes, colors=list(colors[:n]), lineoffsets=range(n),linelengths=0.8, linewidths=1.2)
    ax_r.set_yticks(range(n))
    ax_r.set_yticklabels([f"I={m}" for m in marks], fontsize=8)
    ax_r.set_ylabel("spikes")
    ax_r.set_xlabel("time (ms)")
    axes[0].text(0.5, 1.28, f"{name}: {p['name']}   "f"(a={p['a']}, b={p['b']}, c={p['c']}, d={p['d']})",transform=axes[0].transAxes, ha="center", fontsize=11)

    if currents is None:
        currents = np.arange(0, 20.01, 0.5)
    currents = np.union1d(np.asarray(currents, float), np.asarray(marks, float))
    t, v, u, I = sc.simulate_step(p, list(currents), T=T, t_on=t_on)
    rates = np.array([sc.steady_rate(sc.spike_times(t, v[:, k]))for k in range(len(currents))])
    
    ax_fi = fig.add_subplot(gs[:n, 1])
    ax_fi.plot(currents, rates, "-o", color="k", ms=3, lw=1.2, zorder=2)
    ax_fi.axvline(sc.rheobase(p["b"]), color="gray", ls="--", lw=0.8,label=f"threshold current ({sc.rheobase(p['b']):.1f})")
    
    for m, color in zip(marks, colors):
        k = int(np.argmin(np.abs(currents - m)))
        ax_fi.scatter(currents[k], rates[k], s=70, color=color, zorder=3,label=f"I = {m}:  {rates[k]:.1f} Hz")
        
    ax_fi.set_xlabel("I (step amplitude)")
    ax_fi.set_ylabel("f = 1/T  (Hz)")
    ax_fi.set_title("frequency-current relation")
    ax_fi.legend(fontsize=8, loc="upper left")

    plt.show()
    return currents, rates