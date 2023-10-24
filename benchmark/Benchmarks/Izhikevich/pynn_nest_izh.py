import pyNN.nest as sim
from pyNN.utility.plotting import Figure, Panel
from pyNN.random import RandomDistribution, NumpyRNG
import time
from globparams import *

sim.setup(timestep=1, min_delay=1, max_delay=1)
rng = NumpyRNG()

cell_type = sim.Izhikevich(a=A, b=B, c=C, d=D, i_offset=OFFSET * 10**(-3))

pop1 = sim.Population(size=SIZE, cellclass=cell_type, label="pop1")

w_min = W_MIN * DIRAC_STRENGTH * 10**(-3)
w_max = W_MAX * DIRAC_STRENGTH * 10**(-3)

stdp_model = sim.STDPMechanism(
    timing_dependence=sim.SpikePairRule(
        tau_plus=TRACE_TAU,
        tau_minus=TRACE_TAU,
        A_plus=A_PLUS,
        A_minus=A_MINUS,
    ),
    weight_dependence=sim.MultiplicativeWeightDependence(w_min=w_min, w_max=w_max),
    voltage_dependence=None,
    dendritic_delay_fraction=1.0,
    weight=RandomDistribution("uniform", (w_min, w_max)),
    delay=None,
)


syn = sim.Projection(
    pop1, pop1, sim.AllToAllConnector(allow_self_connections=True), stdp_model
)

pop1.initialize(
    v=RandomDistribution('normal', mu=V_MEAN, sigma=V_STD), 
    u=RandomDistribution('normal', mu=U_MEAN, sigma=U_STD)
)


noise = sim.standardmodels.electrodes.NoisyCurrentSource(
    mean=NOISE_MEAN * 10**(-3), 
    stdev=NOISE_STD * 10**(-3), 
    dt=1,
)

pop1.inject(noise)

if PLOT:
    pop1.record(["spikes"])

start = time.time()
sim.run(DURATION)
print("simulation time: ", time.time() - start)


if PLOT:
    data = pop1.get_data().segments[0]
    print(data.spiketrains)
    Figure(
        Panel(data.spiketrains, xlabel="Time (ms)", xticks=True)
    ).show()


sim.end()