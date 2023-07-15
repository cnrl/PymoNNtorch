from pymonntorch.NetworkCore.Behavior import Behavior
from pymonntorch.NetworkBehavior.EulerEquationModules.Helper import (
    eq_split,
    remove_units,
)


class Variable(Behavior):
    def __init__(self, *args, eq=None, **kwargs):
        super().__init__(*args, eq=eq, **kwargs)

    def initialize(self, neurons):
        super().initialize(neurons)

        n = neurons

        eq_parts = eq_split(self.parameter("eq", None))

        if eq_parts[1] == "=" and len(eq_parts) >= 3:
            self.var_name = eq_parts[0]
        else:
            print("invalid formula")

        self.add_tag("Variable " + self.var_name)

        eq_parts = remove_units(eq_parts, 2)

        self.var_init = "".join(eq_parts[2:])

        setattr(n, self.var_name, neurons.vector() + eval(self.var_init))

        setattr(n, self.var_name + "_new", neurons.vector() + eval(self.var_init))

    def forward(self, n):
        setattr(
            n, self.var_name, getattr(n, self.var_name + "_new")
        )  # apply the new value to variable


class SynapseVariable(Behavior):
    def __init__(self, *args, eq=None, **kwargs):
        super().__init__(*args, eq=eq, **kwargs)

    def initialize(self, synapse):
        s = synapse

        eq_parts = eq_split(self.parameter("eq", None))

        if eq_parts[1] == "=" and len(eq_parts) >= 3:
            self.var_name = eq_parts[0]
        else:
            print("invalid formula")

        self.add_tag("SynapseVariable " + self.var_name)

        eq_parts = remove_units(eq_parts, 2)

        self.var_init = "".join(eq_parts[2:])

        setattr(s, self.var_name, synapse.matrix() + eval(self.var_init))

        setattr(s, self.var_name + "_new", synapse.matrix() + eval(self.var_init))

    def forward(self, s):
        setattr(
            s, self.var_name, getattr(s, self.var_name + "_new")
        )  # apply the new value to variable
