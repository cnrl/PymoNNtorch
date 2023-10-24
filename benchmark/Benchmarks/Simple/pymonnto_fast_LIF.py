from PymoNNto import *
import time
from globparams import *

settings = {'dtype': float32, 'synapse_mode': SxD}


class SpikeGeneration(Behavior):
    def initialize(self, neurons):
        neurons.spikes = neurons.vector('bool')
        neurons.spikesOld = neurons.vector('bool')
        neurons.voltage = neurons.vector()
        self.threshold = self.parameter('threshold')
        self.decay = self.parameter('decay')

    def iteration(self, neurons):
        neurons.spikesOld = neurons.spikes.copy()
        neurons.spikes = neurons.voltage > self.threshold
        #print(np.sum(neurons.spikes))# number of active neurons around 1.5%
        #neurons.voltage.fill(0.0)
        neurons.voltage *= np.invert(neurons.spikes) #reset VR
        neurons.voltage *= self.decay #voltage decay



class Input(Behavior):
    def initialize(self, neurons):
        for s in neurons.synapses(afferent, 'GLU'):
            s.W = s.matrix('random')
            s.W = s.W / SIZE
            # s.W /= np.sum(s.W, axis=0) #normalize during initialization

    def iteration(self, neurons):
        neurons.voltage += neurons.vector('random')
        for s in neurons.synapses(afferent, 'GLU'):
            input = np.sum(s.W[s.src.spikes], axis=0)
            s.dst.voltage += input


class STDP(Behavior):
    def initialize(self, neurons):
        self.speed = self.parameter('speed')

    def iteration(self, neurons):
        for s in neurons.synapses(afferent, 'GLU'):
            mask = np.ix_(s.src.spikesOld, s.dst.spikes)
            s.W[mask] += self.speed
            s.W[mask] = np.clip(s.W[mask], 0.0, 1.0)


#class Norm(Behavior):
#    def iteration(self, neurons):
#        if neurons.iteration % 10 == 9:
#            for s in neurons.synapses(afferent, 'GLU'):
#                s.W /= np.sum(s.W, axis=0)


net = Network(settings=settings)
NeuronGroup(net, tag='NG', size=SIZE, behavior={
    1: SpikeGeneration(threshold=VT, decay=DECAY),
    2: Input(),
    3: STDP(speed=STDP_SPEED),
    #4: Norm(),
})

if PLOT:
    net.NG.add_behavior(9, EventRecorder('spikes'), False)

SynapseGroup(net, src='NG', dst='NG', tag='GLU')
net.initialize()

start = time.time()
net.simulate_iterations(DURATION)
print("simulation time: ", time.time()-start)

if PLOT:
    plt.plot(net['spikes.t', 0], net['spikes.i', 0], '.k')
    plt.show()
