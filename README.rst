===========
PymoNNtorch
===========

.. image:: https://raw.githubusercontent.com/cnrl/PymoNNtorch/main/docs/_images/pymoNNtorch-logo-t-256.png
    :width: 256
    :alt: pymonntorch logo

|


.. image:: https://img.shields.io/pypi/v/pymonntorch.svg
        :target: https://pypi.python.org/pypi/pymonntorch

.. .. image:: https://img.shields.io/travis/cnrl/pymonntorch.svg
..         :target: https://travis-ci.com/cnrl/pymonntorch

.. image:: https://readthedocs.org/projects/pymonntorch/badge/?version=latest
        :target: https://pymonntorch.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




PymoNNtorch is a *Pytorch*-adapted version of `PymoNNto <https://github.com/trieschlab/PymoNNto>`_.


* Free software: MIT license
* Documentation: https://pymonntorch.readthedocs.io.


Features
--------

* Use ``torch`` tensors and Pytorch-like syntax to create a spiking neural network (SNN).
* Simulate an SNN on CPU or GPU.
* Define dynamics of SNN components as ``Behavior`` modules.
* Control over the order of applying different behaviors in each simulation time step.

Usage
-----

You can use the same syntax as ``PymoNNto`` to create you network:

.. code-block:: python

    from pymonntorch import *

    net = Network()
    ng = NeuronGroup(net=net, tag="my_neuron", size=100, behavior=None)
    SynapseGroup(src=ng, dst=ng, net=net, tag="recurrent_synapse")
    net.initialize()
    net.simulate_iterations(1000)


Similarly, you can write your own ``Behavior`` Modules with the same logic as ``PymoNNto``; except using ``torch`` tensors instead of ``numpy`` ndarrays.

.. code-block:: python

    from pymonntorch import *

    class BasicBehavior(Behavior):
        def initialize(self, neurons):
            super().initialize(neurons)
            neurons.voltage = neurons.vector(mode="zeros")
            self.threshold = 1.0

        def forward(self, neurons):
            firing = neurons.voltage >= self.threshold
            neurons.spike = firing.byte()
            neurons.voltage[firing] = 0.0 # reset
            
            neurons.voltage *= 0.9 # voltage decay
            neurons.voltage += neurons.vector(mode="uniform", density=0.1)

    class InputBehavior(Behavior):
        def initialize(self, neurons):
            super().initialize(neurons)
            for synapse in neurons.afferent_synapses['GLUTAMATE']:
                synapse.W = synapse.matrix('uniform', density=0.1)
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
    SynapseGroup(src=ng, dst=ng, net=net, tag='GLUTAMATE')
    net.initialize()
    net.simulate_iterations(1000)

    import matplotlib.pyplot as plt

    plt.plot(net['voltage',0][:, :10])
    plt.show()

    plt.plot(net['spike.t',0], net['spike.i',0], '.k')
    plt.show()


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.
It changes the codebase of `PymoNNto <https://github.com/trieschlab/PymoNNto>`_ to use ``torch`` rather than ``numpy`` and ``tensorflow numpy``.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
