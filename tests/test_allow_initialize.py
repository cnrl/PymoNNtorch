import pytest
from pymonntorch import Behavior, Network, NetworkObject, NeuronGroup, SynapseGroup


class Count(Behavior):
    def initialize(self, obj):
        self.calls = getattr(self, "calls", 0) + 1

    def forward(self, obj):
        self.steps = getattr(self, "steps", 0) + 1


@pytest.mark.parametrize("phase", ["normal", "last", "constructor"])
@pytest.mark.parametrize("disabled", [True, False])
def test_initialization_switch(phase, disabled):
    behavior = Count(allow_initialize=not disabled)
    behavior.initialize_on_init = phase == "constructor"
    behavior.initialize_last = phase == "last"
    net = Network()
    ng = NeuronGroup(2, net, behavior={1: behavior})
    net.initialize(info=False)
    assert getattr(behavior, "calls", 0) == (0 if disabled else 1)
    net.simulate_iteration()
    assert behavior.steps == 1


@pytest.mark.parametrize("kind", [NetworkObject, NeuronGroup, SynapseGroup, Network])
def test_new_behavior_on_existing_object_initializes(kind):
    net = Network()
    obj = kind() if kind is Network else kind(2, net) if kind is NeuronGroup else kind(net)
    restored = obj.add_behavior(1, Count(allow_initialize=False))
    added = obj.add_behavior(2, Count())
    assert added.calls == 1
    obj.network.initialize(info=False)
    assert added.calls == 2
    assert not hasattr(restored, "calls")
    assert not hasattr(added, "_initialized")
    assert not hasattr(obj, "allow_initialize")


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
    ng = NeuronGroup(dimension, net, behavior={})
    assert dimension.initialize_on_init is True
    assert dimension.allow_initialize is True
    assert ng.size == 6


def test_behavior_owns_its_initialization_marker():
    class OwnMarker(Count):
        def initialize(self, obj):
            super().initialize(obj)
            self._initialized = "behavior-owned"
    net = Network()
    behavior = OwnMarker()
    NeuronGroup(1, net, behavior={1: behavior})
    net.initialize(info=False)
    assert behavior._initialized == "behavior-owned"
