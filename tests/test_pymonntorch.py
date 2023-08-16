#!/usr/bin/env python

"""Tests for `pymonntorch` package."""

import pytest

# from click.testing import CliRunner

# from pymonntorch import cli
import pymonntorch


def test_network_cpu_annotated():
    from pymonntorch import (
        Network,
        Behavior,
        NeuronGroup,
        SynapseGroup,
        Recorder,
        EventRecorder,
    )

    class BasicBehavior(Behavior):
        def initialize(self, neurons):
            super().initialize(neurons)
            neurons.voltage = neurons.vector(mode="zeros")
            self.threshold = 1.0

        def forward(self, neurons):
            firing = neurons.voltage >= self.threshold
            neurons.spike = firing.byte()
            neurons.voltage[firing] = 0.0  # reset

            neurons.voltage *= 0.9  # voltage decay
            neurons.voltage += neurons.vector(mode="uniform", density=0.1)

    class InputBehavior(Behavior):
        def initialize(self, neurons):
            super().initialize(neurons)
            for synapse in neurons.afferent_synapses["GLUTAMATE"]:
                synapse.W = synapse.matrix("uniform", density=0.1)
                synapse.enabled = synapse.W > 0

        def forward(self, neurons):
            for synapse in neurons.afferent_synapses["GLUTAMATE"]:
                neurons.voltage += (
                    synapse.W @ synapse.src.spike.float() / synapse.src.size * 10
                )

    net = Network()
    ng = NeuronGroup(
        net=net,
        size=100,
        behavior={
            1: BasicBehavior(),
            2: InputBehavior(),
            9: Recorder(["n.voltage", "torch.mean(n.voltage)"], auto_annotate=False),
            10: EventRecorder(["n.spike"]),
        },
    )
    SynapseGroup(ng, ng, net, tag="GLUTAMATE")
    net.initialize()
    net.simulate_iterations(1000)

    assert net.device == "cpu"
    assert ng.device == "cpu"

    for _, b in ng.behavior.items():
        assert b.device == "cpu"


def test_network_cuda_annotated():
    from pymonntorch import (
        Network,
        Behavior,
        NeuronGroup,
        SynapseGroup,
        Recorder,
        EventRecorder,
    )

    class BasicBehavior(Behavior):
        def initialize(self, neurons):
            super().initialize(neurons)
            neurons.voltage = neurons.vector(mode="zeros")
            self.threshold = 1.0

        def forward(self, neurons):
            firing = neurons.voltage >= self.threshold
            neurons.spike = firing.byte()
            neurons.voltage[firing] = 0.0  # reset

            neurons.voltage *= 0.9  # voltage decay
            neurons.voltage += neurons.vector(mode="uniform", density=0.1)

    class InputBehavior(Behavior):
        def initialize(self, neurons):
            super().initialize(neurons)
            for synapse in neurons.afferent_synapses["GLUTAMATE"]:
                synapse.W = synapse.matrix("uniform", density=0.1)
                synapse.enabled = synapse.W > 0

        def forward(self, neurons):
            for synapse in neurons.afferent_synapses["GLUTAMATE"]:
                neurons.voltage += (
                    synapse.W @ synapse.src.spike.float() / synapse.src.size * 10
                )

    net = Network(device="cuda")
    ng = NeuronGroup(
        net=net,
        size=100,
        behavior={
            1: BasicBehavior(),
            2: InputBehavior(),
            9: Recorder(["n.voltage", "torch.mean(n.voltage)"], auto_annotate=False),
            10: EventRecorder(["n.spike"]),
        },
    )
    SynapseGroup(ng, ng, net, tag="GLUTAMATE")
    net.initialize()
    net.simulate_iterations(1000)

    assert net.device == "cuda"
    assert ng.device == "cuda"

    for _, b in ng.behavior.items():
        assert b.device == "cuda"


# def test_command_line_interface():
#     """Test the CLI."""
#     runner = CliRunner()
#     result = runner.invoke(cli.main)
#     assert result.exit_code == 0
#     assert 'pymonntorch.cli.main' in result.output
#     help_result = runner.invoke(cli.main, ['--help'])
#     assert help_result.exit_code == 0
#     assert '--help  Show this message and exit.' in help_result.output
