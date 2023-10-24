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
    set_device,
    device,
)
import matplotlib.pyplot as plt
from globparams import *
set_device('cpp_standalone', clean=True)
prefs.codegen.target = 'cython'

defaultclock.dt = 1 * ms
prefs.core.default_float_dtype = float32


a, b, c, d = A, B, C, D
offset = OFFSET

ng = NeuronGroup(
    SIZE,
    """dv/dt = (0.04*v**2 + 5*v + 140 - u + I + randn() * NOISE_STD + NOISE_MEAN) / ms : 1
    du/dt = (a*(b*v - u)) / ms  : 1
    dI/dt = (-I + OFFSET) / ms : 1""",
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
    I_post += w * DIRAC_STRENGTH
    Apre += 1
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

if PLOT:
    spikemon = SpikeMonitor(ng)

    
run(DURATION * ms, report=REPORT_FUNC)

if PLOT:
    print(f"Total spikes: {len(spikemon.i)}")
    plt.plot(spikemon.t/ms, spikemon.i, '.k')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index');
    plt.show()

