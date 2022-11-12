#!/usr/bin/env python

"""Tests for `pymonntorch` package."""

import pytest
# from click.testing import CliRunner

# from pymonntorch import cli
import pymonntorch


def test_network_cpu():
    from pymonntorch.NetworkCore.Network import Network
    from pymonntorch.NetworkCore.NeuronGroup import NeuronGroup
    from pymonntorch.NetworkCore.SynapseGroup import SynapseGroup
    from pymonntorch.NetworkBehavior.Structure.Structure import get_squared_dim
    from pymonntorch.NetworkBehavior.Basics.Normalization import SynapticNormalization

    net = Network(device='cuda')
    ng1 = NeuronGroup(size=get_squared_dim(10000), behavior={1: SynapticNormalization()}, net=net)
    syn = SynapseGroup(src=ng1, dst=ng1, net=net, tag='glutamate')
    syn.w = syn.get_synapse_mat('rand')

    net.initialize()
    net.simulate_iterations(1000)

    assert net.device == 'cuda'
    assert ng1.device == 'cuda'
    assert syn.device == 'cuda'

    for _, b in ng1.behavior.items():
        assert b.device == 'cuda'


# def test_command_line_interface():
#     """Test the CLI."""
#     runner = CliRunner()
#     result = runner.invoke(cli.main)
#     assert result.exit_code == 0
#     assert 'pymonntorch.cli.main' in result.output
#     help_result = runner.invoke(cli.main, ['--help'])
#     assert help_result.exit_code == 0
#     assert '--help  Show this message and exit.' in help_result.output
