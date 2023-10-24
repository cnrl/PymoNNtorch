from pymonntorch import *
import torch
import time
from matplotlib import pyplot as plt
from globparams import *

settings = {'dtype': torch.float64, 'synapse_mode': "DxS", 'device': 'cuda'}


class SpikeGeneration(Behavior):
    def initialize(self, neurons):
        neurons.spikes = neurons.vector(dtype=torch.bool)
        neurons.spikesOld = neurons.vector(dtype=torch.bool)
        neurons.voltage = neurons.vector()
        self.threshold = self.parameter('threshold', None)
        self.decay = self.parameter('decay', None)

    def forward(self, neurons):
        neurons.spikesOld = neurons.spikes.clone()
        neurons.spikes = neurons.voltage > self.threshold
        #print(np.sum(neurons.spikes)) number of active neurons around 1.5%
        # neurons.voltage.fill(0.0)
        neurons.voltage *= ~neurons.spikes #reset
        neurons.voltage *= self.decay #voltage decay


class Input(Behavior):
    def initialize(self, neurons):
        for s in neurons.synapses('afferent', 'GLU'):
            s.W = s.matrix('random')
            s.W = s.W / SIZE
            # s.W /= torch.sum(s.W, axis=0)##################################

    def forward(self, neurons):
        neurons.voltage += neurons.vector('random')
        for s in neurons.synapses('afferent', 'GLU'):
            input = torch.tensordot(s.W, s.src.spikes.to(neurons.def_dtype), dims=[[1], [0]])
            s.dst.voltage += input


class STDP(Behavior):
    def initialize(self, neurons):
        self.speed = self.parameter('speed', None)

    def forward(self, neurons):
        for s in neurons.synapses('afferent', 'GLU'):
            s.W += s.dst.spikes[:, None] * s.src.spikesOld[None, :] * self.speed
            s.W = torch.clip(s.W, 0.0, 1.0)


#class Norm(Behavior):
#    def iteration(self, neurons):
#        for s in neurons.synapses(afferent, 'GLU'):
#            s.W /= np.sum(s.W, axis=0)


net = Network(**settings)
NeuronGroup(net=net, tag='NG', size=SIZE, behavior={
    1: SpikeGeneration(threshold=VT, decay=DECAY),
    2: Input(),
    3: STDP(speed=STDP_SPEED),
    #4: Norm()
    #5: EventRecorder(variables=['spikes'])
})

if PLOT:
    net.NG.add_behavior(9, EventRecorder('spikes'), False)

SynapseGroup(net=net, src='NG', dst='NG', tag='GLU')
net.initialize()

start = time.time()
net.simulate_iterations(DURATION)
print("simulation time: ", time.time()-start)

if PLOT:
    plt.plot(net['spikes.t', 0].cpu(), net['spikes.i', 0].cpu(), '.k')
    plt.show()
