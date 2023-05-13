"""Main module."""
from pymonntorch import *

net = Network()
pop = NeuronGroup(net = net, tag = "pop0", size =100 ,behavior ={1: Izhikevich_Neuron(), 9: Recorder (['voltage']), 10: EventRecorder (['spike'])})
SynapseGroup(pop, pop, net)
net.initialize()
net.simulate_iterations(1000)
import matplotlib . pyplot as plt

plt.plot(net["voltage", 0][:, :10]), plt.show()
try:
    plt.plot(net["u", 0][:, :10]), plt.show()
except Exception as e:
    print(f"Error >> {e}  << happened")
finally:
    plt.plot(net["spike.t", 0],net["spike.i", 0],'.k')
    plt.show()