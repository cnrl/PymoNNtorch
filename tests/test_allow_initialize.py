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
    behavior = Count(allow_initialize=disabled != "behavior")
    behavior.initialize_on_init = phase == "constructor"
    behavior.initialize_last = phase == "last"
    net = Network()
    ng = NeuronGroup(2, net, behavior={1: behavior}, allow_initialize=disabled != "object")
    net.initialize(info=False)
    assert getattr(behavior, "calls", 0) == (1 if disabled == "neither" else 0)
    net.simulate_iteration()
    assert behavior.steps == 1


@pytest.mark.parametrize("kind", [NetworkObject, NeuronGroup, SynapseGroup, Network])
def test_constructor_option_and_add_behavior(kind):
    net = Network()
    if kind is NetworkObject:
        obj = kind(net, allow_initialize=False)
    elif kind is NeuronGroup:
        obj = kind(2, net, allow_initialize=False)
    elif kind is SynapseGroup:
        obj = kind(net, allow_initialize=False)
    else:
        obj = kind(allow_initialize=False)
    behavior = obj.add_behavior(1, Count())
    obj.network.initialize(info=False)
    assert not hasattr(behavior, "calls")
    obj.allow_initialize = True
    obj.network.initialize(info=False)
    assert behavior.calls == 1


def test_behavior_permission_also_applies_to_add_behavior():
    net = Network()
    ng = NeuronGroup(2, net)
    behavior = ng.add_behavior(1, Count(allow_initialize=False))
    net.initialize(info=False)
    assert not hasattr(behavior, "calls")
    net.simulate_iteration()
    assert behavior.steps == 1
    behavior.allow_initialize = True
    net.initialize(info=False)
    assert behavior.calls == 1


def test_constructor_dimension_behavior_retains_its_original_schedule():
    from pymonntorch import NeuronDimension
    net = Network()
    dimension = NeuronDimension(width=3, height=2)
    ng = NeuronGroup(dimension, net)
    assert dimension.initialize_on_init is True
    assert dimension.allow_initialize is True
    assert ng.size == 6
    with pytest.raises(ValueError, match="allow_initialize=False"):
        NeuronGroup(NeuronDimension(width=3), net, allow_initialize=False)
