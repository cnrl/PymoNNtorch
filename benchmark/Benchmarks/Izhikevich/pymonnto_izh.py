from PymoNNto import (
    Network,
    SynapseGroup,
    NeuronGroup,
    Behavior,
    EventRecorder,
    SxD,
    float32,
)
import numpy as np
import time
import matplotlib.pyplot as plt
from globparams import *

settings = {"dtype": float32, "synapse_mode": SxD}


class TimeResolution(Behavior):
    def initialize(self, n):
        n.dt = self.parameter("dt", 1.0)


class Izhikevich(Behavior):
    def initialize(self, n):
        self.a = self.parameter("a")
        self.b = self.parameter("b")
        self.c = self.parameter("c")
        self.d = self.parameter("d")
        self.threshold = self.parameter("threshold")

        n.v = V_STD * n.vector("normal") + V_MEAN
        n.u = U_STD * n.vector("normal") + U_MEAN
        n.spikes = n.vector("bool")

    def iteration(self, n):
        n.spikes = (n.v >= self.threshold) 

        n.v[n.spikes] = self.c
        n.u[n.spikes] += self.d

        dv = (0.04 * n.v**2.0 + 5.0 * n.v + 140.0 - n.u + n.I)
        du = (self.a * (self.b * n.v - n.u))

        n.v += dv * n.network.dt 
        n.u += du * n.network.dt 




class Dendrite(Behavior):
    def initialize(self, n):
        self.offset = self.parameter("offset", None)
        n.I = n.vector(self.offset)

    def iteration(self, n):
        n.I.fill(self.offset)
        n.I += n.vector("normal") * NOISE_STD + NOISE_MEAN

        for s in n.afferent_synapses["GLU"]:
            n.I += s.I


class STDP(Behavior):
    def initialize(self, s):
        self.pre_tau = self.parameter("pre_tau")
        self.post_tau = self.parameter("post_tau")
        self.a_plus = self.parameter("a_plus")
        self.a_minus = self.parameter("a_minus")

        s.src_trace = s.src.vector()
        s.dst_trace = s.dst.vector()

    def iteration(self, s):
        src_spikes = s.src.spikes
        dst_spikes = s.dst.spikes
        s.src_trace += src_spikes - s.src_trace / self.pre_tau * s.network.dt
        s.dst_trace += dst_spikes - s.dst_trace / self.post_tau * s.network.dt
        s.W[src_spikes] -= (
            s.dst_trace[None, ...] * self.a_minus * (s.W[src_spikes] - W_MIN)
        )
        s.W[:, dst_spikes] += (
            s.src_trace[..., None] * self.a_plus * (W_MAX - s.W[:, dst_spikes])
        )
        # s.W = np.clip(s.W, W_MIN, W_MAX)


class DiracInput(Behavior):
    def initialize(self, s):
        self.strength = self.parameter("strength")
        s.I = s.dst.vector()
        s.W = s.matrix("random") * W_MAX + W_MIN
        # np.fill_diagonal(s.W, 0)

    def iteration(self, s):
        s.I = np.sum(s.W[s.src.spikes], axis=0) * self.strength


net = Network(behavior={1: TimeResolution()}, settings=settings)

NeuronGroup(
    net,
    tag="NG",
    size=SIZE,
    behavior={
        1: Dendrite(offset=OFFSET),
        2: Izhikevich(a=A, b=B, c=C, d=D, threshold=THRESHOLD),
    },
)

if PLOT:
    net.NG.add_behavior(9, EventRecorder("spikes"), False)

SynapseGroup(
    net,
    src="NG",
    dst="NG",
    tag="GLU",
    behavior={
        4: DiracInput(strength=DIRAC_STRENGTH),
        5: STDP(a_plus=A_PLUS, a_minus=A_MINUS, pre_tau=TRACE_TAU, post_tau=TRACE_TAU),
    },
)


net.initialize()

start = time.time()
net.simulate_iterations(DURATION, batch_size=DURATION, measure_block_time=True)
print("simulation time: ", time.time() - start)

if PLOT:
    print(f"Total spikes: {len(net['spikes.i', 0])}")
    plt.plot(net["spikes.t", 0], net["spikes.i", 0], ".k")
    plt.show()
