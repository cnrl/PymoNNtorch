=====
Usage
=====

You can use the same syntax as `PymoNNto <https://pymonnto.readthedocs.io/en/latest/Introduction/basics2/>`_ to create you network: ::

    from pymonntorch import *

    net = Network(device="cpu")  # To create and simulate the network on GPU, simply change the device.
    ng = NeuronGroup(net=net, tag="my_neuron", size=100)
    SynapseGroup(src=ng, dst=ng, net=net, tag="recurrent_synapse")
    net.initialize()
    net.simulate_iterations(1000)


Similarly, you can write your own `Behavior` Modules with the same logic as PymoNNto; except using `torch` tensors instead of `numpy` ndarrays. ::

    from pymonntorch import *

    class BasicBehavior(Behavior):
        def set_variables(self, neurons):
            neurons.voltage = neurons.get_neuron_vec(mode="zeros")
            self.threshold = 1.0

        def forward(self, neurons):
            firing = neurons.voltage >= self.threshold
            neurons.spike = firing.byte()
            neurons.voltage[firing] = 0.0 # reset
            
            neurons.voltage *= 0.9 # voltage decay
            neurons.voltage += neurons.get_neuron_vec(mode="uniform", density=0.1)

    class InputBehavior(Behavior):
        def set_variables(self, neurons):
            for synapse in neurons.afferent_synapses['GLUTAMATE']:
                synapse.W = synapse.get_synapse_mat('uniform', density=0.1)
                synapse.enabled = synapse.W > 0

        def forward(self, neurons):
            for synapse in neurons.afferent_synapses['GLUTAMATE']:
                neurons.voltage += synapse.W@synapse.src.spike.float() / synapse.src.size * 10

    net = Network()
    ng = NeuronGroup(net=net,
                    size=100, 
                    behavior={
                        1: BasicBehavior(),
                        2: InputBehavior(),
                        9: Recorder(['voltage']),
                        10: EventRecorder(['spike'])
                    })
    SynapseGroup(ng, ng, net, tag='GLUTAMATE')
    net.initialize()
    net.simulate_iterations(1000)

    import matplotlib.pyplot as plt

    plt.plot(net['voltage',0][:, :10])
    plt.show()

    plt.plot(net['spike.t',0], net['spike.i',0], '.k')
    plt.show()
