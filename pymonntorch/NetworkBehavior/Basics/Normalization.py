import torch

from pymonntorch.NetworkCore.Behavior import Behavior
from pymonntorch.utils import check_is_torch_tensor


class SynapticNormalization(Behavior):
    def set_variables(self, neurons):
        super().set_variables(neurons)

        self.synapse_type = self.get_init_attr("synapse_type", "glutamate", neurons)

        neurons.require_synapses(self.synapse_type)

        self.norm_factor = check_is_torch_tensor(
            self.get_init_attr("norm_factor", 1.0, neurons),
            device=neurons.device,
            dtype=torch.float32,
        )

        neurons.sum_w = neurons.get_neuron_vec(kwargs={"dtype": torch.float32})

    def new_iteration(self, neurons):
        neurons.sum_w.zero_()

        for syn in neurons.afferent_synapses[self.synapse_type]:
            syn.dst.sum_w.add_(syn.w.sum(dim=1))

        neurons.sum_w.div_(self.norm_factor)

        for syn in neurons.afferent_synapses[self.synapse_type]:
            syn.w.T.div_(syn.dst.sum_w)
