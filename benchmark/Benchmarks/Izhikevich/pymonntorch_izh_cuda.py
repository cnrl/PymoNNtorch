from pymonntorch import Network, SynapseGroup, NeuronGroup, Behavior, EventRecorder
import torch
import time
from globparams import *
import matplotlib.pyplot as plt

settings = {"dtype": torch.float32, "synapse_mode": "SxD", "device": "cuda"}


class TimeResolution(Behavior):
    def initialize(self, n):
        n.dt = self.parameter("dt", 1)


class Izhikevich(Behavior):
    def initialize(self, n):
        self.a = self.parameter("a", None)
        self.b = self.parameter("b", None)
        self.c = self.parameter("c", None)
        self.d = self.parameter("d", None)
        self.threshold = self.parameter("threshold", None)

        n.u = n.vector(f"normal({U_MEAN}, {U_STD})")
        n.v = n.vector(f"normal({V_MEAN}, {V_STD})")
        n.spikes = n.vector(dtype=torch.bool)

    def forward(self, n):
        n.spikes = n.v >= self.threshold

        n.v[n.spikes] = self.c
        n.u[n.spikes] += self.d

        dv = (0.04 * n.v**2 + 5 * n.v + 140 - n.u + n.I)
        du = self.a * (self.b * n.v - n.u)

        n.v += dv * n.network.dt
        n.u += du * n.network.dt


class Dendrite(Behavior):
    def initialize(self, n):
        self.offset = self.parameter("offset", None)
        n.I = n.vector(self.offset)

    def forward(self, n):
        n.I.fill_(self.offset)
        for s in n.afferent_synapses["GLU"]:
            n.I += s.I
        n.I += n.vector(f"normal({NOISE_MEAN}, {NOISE_STD})")


class STDP(Behavior):
    def initialize(self, s):
        self.pre_tau = self.parameter("pre_tau", None)
        self.post_tau = self.parameter("post_tau", None)
        self.a_plus = self.parameter("a_plus", None)
        self.a_minus = self.parameter("a_minus", None)

        s.src_trace = s.src.vector()
        s.dst_trace = s.dst.vector()

    def forward(self, s):
        src_spikes = s.src.spikes
        dst_spikes = s.dst.spikes
        s.src_trace += src_spikes * 1.0 - s.src_trace / self.pre_tau * s.network.dt
        s.dst_trace += dst_spikes * 1.0 - s.dst_trace / self.post_tau * s.network.dt
        s.W[src_spikes] -= (
            s.dst_trace[None, ...] * self.a_minus * (s.W[src_spikes] - W_MIN)
        )
        s.W[:, dst_spikes] += (
            s.src_trace[..., None] * self.a_plus * (W_MAX - s.W[:, dst_spikes])
        )
        # s.W = torch.clip(s.W, W_MIN, W_MAX)


class DiracInput(Behavior):
    def initialize(self, s):
        self.strength = self.parameter("strength", None)
        s.I = s.dst.vector()
        s.W = s.matrix("random") * W_MAX + W_MIN
        # s.W.fill_diagonal_(0)

    def forward(self, s):
        s.I = torch.sum(s.W[s.src.spikes], axis=0) * self.strength


net = Network(behavior={1: TimeResolution()}, **settings)

NeuronGroup(
    net=net,
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
    net=net,
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
net.simulate_iterations(DURATION)
print("simulation time: ", time.time() - start)


if PLOT:
    print(f"Total spikes: {len(net['spikes.i', 0])}")
    plt.plot(net["spikes.t", 0].to("cpu"), net["spikes.i", 0].to("cpu"), ".k")
    plt.show()
