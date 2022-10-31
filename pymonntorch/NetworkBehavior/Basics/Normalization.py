import torch

from pymonntorch.NetworkCore.Behavior import Behavior
from pymonntorch.utils import check_is_torch_tensor


class SynapticNormalization(Behavior):
    def set_variables(self, object):
        self.synapse_type = self.get_init_attr('synapse_type', 'glutamate', object=object)

        object.require_synapses(self.synapse_type)

        self.norm_factor = check_is_torch_tensor(
            self.get_init_attr('norm_factor', 1.0, object=object),
            device=object.device,
            dtype=torch.float32
        )

        object.sum_w = object.get_neuron_vec(dtype=torch.float32)

    def new_iteration(self, object):
        object.sum_w.zero_()

        for syn in object.synapses[self.synapse_type]:
            syn.dst.sum_w.add_(syn.w.sum(dim=1))

        object.sum_w.div_(self.norm_factor)

        for syn in object.synapses[self.synapse_type]:
            syn.w.T.div_(syn.dst.sum_w)
