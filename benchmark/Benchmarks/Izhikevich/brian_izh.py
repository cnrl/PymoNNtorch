import time
import numpy as np
from brian2 import (
    StateMonitor,
    SpikeMonitor,
    defaultclock,
    prefs,
    NeuronGroup,
    run,
    Synapses,
    plot,
    ms,
    mV,
    float32,
)
import matplotlib.pyplot as plt
from globparams import *

defaultclock.dt = 1.0 * ms
prefs.core.default_float_dtype = float32

a, b, c, d = A, B, C, D
offset = OFFSET

ng = NeuronGroup(
    SIZE,
    """dv/dt = (0.04*v**2.0 + 5.0*v + 140.0 - u + I + randn() * NOISE_STD + NOISE_MEAN) / ms : 1
    du/dt = (a * (b * v - u)) / ms  : 1
    dI/dt = (-I + offset) / ms : 1""",
    threshold="v>=THRESHOLD",
    reset="v=c; u+=d",
    method="euler",
)

sg = Synapses(
    ng,
    ng,
    """w : 1
    dApre/dt = -Apre / TRACE_TAU / ms : 1 (event-driven)
    dApost/dt = -Apost / TRACE_TAU / ms : 1 (event-driven)""",
    on_pre="""
    Apre += 1
    I_post += w * DIRAC_STRENGTH
    w = w - Apost * (w - W_MIN) """,
    on_post="""
    Apost += 1
    w = w + Apre * (W_MAX - w) """,
)

# sg.connect(condition='i!=j')
sg.connect()

sg.w = "rand() * W_MAX"
ng.v = "V_STD * randn() + V_MEAN"
ng.u = "U_STD * randn() + U_MEAN"

#print(np.min(ng.u), np.max(ng.u), np.mean(ng.u), np.var(ng.u))
#print(np.sum(ng.u))
#import matplotlib.pyplot as plt
#plt.hist(ng.u, bins=100)
#plt.show()

if PLOT:
    spikemon = SpikeMonitor(ng)

start = time.time()
run(DURATION * ms)
print("simulation time: ", time.time() - start)

if PLOT:
    print(f"Total spikes: {len(spikemon.i)}")
    plt.plot(spikemon.t/ms, spikemon.i, '.k')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    plt.show()

