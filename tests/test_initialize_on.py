import pytest
from pymonntorch import Behavior, Network, NetworkObject, NeuronGroup, SynapseGroup


class Count(Behavior):
    def initialize(self, obj):
        self.calls = getattr(self, "calls", 0) + 1

    def forward(self, obj):
        self.steps = getattr(self, "steps", 0) + 1


@pytest.mark.parametrize("phase", ["normal", "last", "constructor"])
@pytest.mark.parametrize("disabled", ["object", "behavior", "neither"])
def test_initialization_switch(phase, disabled):
    behavior = Count(initialize_on=disabled != "behavior")
    behavior.initialize_on_init = phase == "constructor"
    behavior.initialize_last = phase == "last"
    net = Network()
    ng = NeuronGroup(2, net, behavior={1: behavior}, initialize_on=disabled != "object")
    net.initialize(info=False)
    assert getattr(behavior, "calls", 0) == (1 if disabled == "neither" else 0)
    net.simulate_iteration()
    assert behavior.steps == 1


@pytest.mark.parametrize("kind", [NetworkObject, NeuronGroup, SynapseGroup, Network])
def test_constructor_option_and_add_behavior(kind):
    net = Network()
    if kind is NetworkObject:
        obj = kind(net, initialize_on=False)
    elif kind is NeuronGroup:
        obj = kind(2, net, initialize_on=False)
    elif kind is SynapseGroup:
        obj = kind(net, initialize_on=False)
    else:
        obj = kind(initialize_on=False)
    behavior = obj.add_behavior(1, Count())
    obj.network.initialize(info=False)
    assert not hasattr(behavior, "calls")
    obj.initialize_on = True
    obj.network.initialize(info=False)
    assert behavior.calls == 1
