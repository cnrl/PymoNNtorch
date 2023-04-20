import torch

from pymonntorch.NetworkCore.Behavior import Behavior
from pymonntorch.utils import check_is_torch_tensor


class SynapticNormalization(Behavior):
    def initialize(self, neurons):
        super().initialize(neurons)

        self.synapse_type = self.parameter("synapse_type", "glutamate", neurons)

        neurons.require_synapses(self.synapse_type)

        self.norm_factor = check_is_torch_tensor(
            self.parameter("norm_factor", 1.0, neurons),
            device=neurons.device,
            dtype=neurons.def_dtype,
        )

        neurons.sum_w = neurons.vector(kwargs={"dtype": neurons.def_dtype})

    def forward(self, neurons):
        neurons.sum_w.zero_()

        for syn in neurons.afferent_synapses[self.synapse_type]:
            syn.dst.sum_w.add_(syn.w.sum(dim=1))

        neurons.sum_w.div_(self.norm_factor)

        for syn in neurons.afferent_synapses[self.synapse_type]:
            syn.w.T.div_(syn.dst.sum_w)
