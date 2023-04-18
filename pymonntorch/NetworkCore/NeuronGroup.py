import numpy as np
import torch

from pymonntorch.NetworkCore.Base import *
from pymonntorch.NetworkCore.Behavior import *


class NeuronGroup(NetworkObject):
    """This is the class to construct a neuronal population.

    Attributes:
        size (int): The number of neurons in the population.
        behavior (list or dict): The behaviors of the population.
        net (Network): The network the population belongs to.
        tags (str): The tags of the population.
        BaseNeuronGroup (NeuronGroup): The base `NeuronGroup` the population belongs to.
        afferent_synapses (dict): The afferent synapses of the population.
        efferent_synapses (dict): The efferent synapses of the population.
        mask (bool): Whether to define a mask for the population (Used for nested populations).
        learning (bool): Whether to enable learning for the population.
        recording (bool): Whether to enable recording for the population.
        id (torch.Tensor): The integer id of the population.
    """

    def __init__(self, size, behavior, net, tag=None):
        """Initialize the neuronal population.

        Args:
            size (int or Behavior): The size or dimension of the population.
            behavior (list or dict): The behaviors of the population.
            net (Network): The network the population belongs to.
            tag (str): The tag of the population.
        """
        if tag is None and net is not None:
            tag = "NeuronGroup_" + str(len(net.NeuronGroups) + 1)

        if isinstance(size, Behavior):
            if type(behavior) is dict:
                if 0 in behavior:
                    print(
                        "warning: 0 index behavior will be overwritten by size behavior"
                    )
                behavior[0] = size
            if type(behavior) is list:
                behavior.insert(0, size)
            size = -1  # will be overwritten by size-behavior

        self.size = size

        super().__init__(tag, net, behavior, net.device)
        self.add_tag("ng")

        self.BaseNeuronGroup = self  # used for subgroup reconstruction

        if net is not None:
            net.NeuronGroups.append(self)
            setattr(net, self.tags[0], self)

        self.afferent_synapses = {}  # set by Network
        self.efferent_synapses = {}

        self.mask = True

        self.learning = True
        self.recording = True

        self.id = torch.arange(self.size, device=self.device)

    def require_synapses(self, name, afferent=True, efferent=True, warning=True):
        """Require the existence of synapses.

        Args:
            name (str): The name of the synapse.
            afferent (bool): Whether to require afferent synapses.
            efferent (bool): Whether to require efferent synapses.
            warning (bool): Whether to print a warning if the synapse does not exist.
        """
        if afferent and not name in self.afferent_synapses:
            if warning:
                print("warning: no afferent {} synapses found".format(name))
            self.afferent_synapses[name] = []

        if efferent and not name in self.efferent_synapses:
            if warning:
                print("warning: no efferent {} synapses found".format(name))
            self.efferent_synapses[name] = []

    def get_neuron_vec(
        self, mode="zeros()", scale=None, density=None, plot=False, dtype=None
    ):
        """Get a tensor with population's dimensionality.

        The tensor can be initialized in different modes. List of possible values for mode includes:
        - "random" or "rand" or "rnd" or "uniform": Uniformly distributed random numbers in range [0, 1).
        - "normal": Normally distributed random numbers with zero mean and unit variance.
        - "ones": Tensor filled with ones.
        - "zeros": Tensor filled with zeros.
        - A single number: Tensor filled with that number.
        - You can also use any function from torch package for this purpose. Note that you should **not** use `torch.` prefix.

        Args:
            mode (str): Mode to be used to initialize tensor.
            scale (float): Scale of the tensor. The default is None (i.e. No scaling is applied).
            density (float): Density of the tensor. The default is None (i.e. dense tensor).
            plot (bool): If true, the histogram of the tensor will be plotted. The default is False.
            dtype (str or type): Data type of the tensor. If None, `def_dtype` will be used.

        Returns:
            torch.Tensor: The initialized tensor."""
        return self._get_mat(
            mode=mode,
            dim=(self.size),
            scale=scale,
            density=density,
            plot=plot,
            dtype=dtype,
        )

    def get_neuron_vec_buffer(self, buffer_size, **kwargs):
        """Get a buffer for the population's dimensionality.

        Args:
            buffer_size (int): The size of the buffer.
            **dtype (torch.dtype, optional): The desired data type of returned tensor.

        Returns:
            torch.Tensor: The buffer.
        """
        kwargs.setdefault("dtype", self.def_dtype)
        return self.get_buffer_mat((self.size,), buffer_size, **kwargs)

    def get_combined_synapse_shape(self, Synapse_ID):
        """Get the population size along with the number of afferent synapses.

        Args:
            Synapse_ID (str): The ID of the synapse by which it is registered in list of afferent synapses.

        Returns:
            tuple: The combined shape.
        """
        source_num = 0
        for syn in self.afferent_synapses[Synapse_ID]:
            _, s = syn.get_synapse_mat_dim()
            source_num += s
        return self.size, source_num

    def __repr__(self):
        result = "NeuronGroup" + str(self.tags) + "(" + str(self.size) + "){"
        for k in sorted(list(self.behavior.keys())):
            result += str(k) + ":" + str(self.behavior[k])
        return result + "}"

    def subGroup(self, mask=None):
        """Get a NeuronSubGroup object from the population.

        Args:
            mask (bool): The mask condition indicating which neurons to be included in the subgroup.

        Returns:
            NeuronSubGroup: The subgroup.
        """
        return NeuronSubGroup(self, mask)

    def group_without_subGroup(self):
        """Get the NeuronGroup object itself."""
        return self

    def get_masked_dict(self, dict_name, key):
        """Get value of a key in a specific dictionary attribute of the population.

        Args:
            dict_name (str): Name of the dictionary.
            key (int or str): the key to retrieve from the dictionary.

        Returns:
            any: The value.
        """
        return getattr(self, dict_name)[key]

    def connected_NG_param_list(
        self,
        param_name,
        syn_tag="All",
        afferent_NGs=False,
        efferent_NGs=False,
        same_NG=False,
        search_behaviors=False,
    ):
        """Get a list of parameters of connected neuron groups.

        Args:
            param_name (str): The name of the parameter.
            syn_tag (str): The tag of the synapse. The default is "All".
            afferent_NGs (bool): Whether to include afferent neuron groups. The default is False.
            efferent_NGs (bool): Whether to include efferent neuron groups. The default is False.
            same_NG (bool): Whether to include the connections within the same neuron group. The default is False.
            search_behaviors (bool): Whether to search the dictionary of behaviors for the parameter. The default is False.

        Returns:
            list: The list of parameters.
        """
        result = []

        def search_NG(NG):
            if hasattr(NG, param_name):
                attr = getattr(NG, param_name)
                if callable(attr):
                    result.append(attr(NG))
                else:
                    result.append(attr)
            if search_behaviors:
                for key, behavior in NG.behavior.items():
                    if hasattr(behavior, param_name):
                        attr = getattr(behavior, param_name)
                        if callable(attr):
                            result.append(attr(NG))
                        else:
                            result.append(attr)

        if same_NG:
            search_NG(self)

        if efferent_NGs:
            for syn in self.efferent_synapses[syn_tag]:
                search_NG(syn.dst)

        if afferent_NGs:
            for syn in self.afferent_synapses[syn_tag]:
                search_NG(syn.src)

        return result

    def partition(self, block_size=7):
        """Get a partitioned population.

        Args:
            block_size (int): The size of each block. The default is 7.

        Returns:
            list of NeuronGroup or NeuronSubGroup: The list of partitions.
        """
        w = block_size
        h = block_size
        d = block_size
        split_size = [np.maximum(w, 1), np.maximum(h, 1), np.maximum(d, 1)]
        if split_size[0] < 2 and split_size[1] < 2 and split_size[2] < 2:
            return [self]
        else:
            return self.split_grid_into_sub_group_blocks(split_size)

    def partition_masks(self, steps=[1, 1, 1]):
        """Get a mask tensor for partitioning the population.

        Args:
            steps (list of int): The number of steps in each dimension. The default is [1, 1, 1].

        Returns:
            torch.Tensor: The mask tensor.
        """
        dst_min = [np.min(p) for p in [self.x, self.y, self.z]]
        dst_max = [np.max(p) for p in [self.x, self.y, self.z]]

        def get_start_end(step, dim):
            start = dst_min[dim] + (dst_max[dim] - dst_min[dim]) / steps[dim] * step
            end = dst_min[dim] + (dst_max[dim] - dst_min[dim]) / steps[dim] * (step + 1)
            return start, end

        results = []

        masks = []
        for w_step in range(steps[0]):  # x_steps
            dst_x_start, dst_x_end = get_start_end(w_step, 0)
            for h_step in range(steps[1]):  # y_steps
                dst_y_start, dst_y_end = get_start_end(h_step, 1)
                for d_step in range(steps[2]):  # z_steps
                    dst_z_start, dst_z_end = get_start_end(d_step, 2)

                    sub_group_mask = (
                        (self.x >= dst_x_start)
                        * (self.x <= dst_x_end)
                        * (self.y >= dst_y_start)
                        * (self.y <= dst_y_end)
                        * (self.z >= dst_z_start)
                        * (self.z <= dst_z_end)
                    )

                    # remove redundancies
                    for old_dst_mask in masks:
                        sub_group_mask[old_dst_mask] *= False
                    masks.append(sub_group_mask)

                    results.append(sub_group_mask)

        return torch.tensor(results).to(self.device)

    def split_grid_into_sub_group_blocks(self, steps=[1, 1, 1]):
        """Split the population into partitioned subgroups.

        Returns:
            list of NeuronGroup or NeuronSubGroup: The list of partitions.
        """
        return [self.subGroup(mask) for mask in self.partition_masks(steps)]

    def get_subgroup_receptive_field_mask(self, subgroup, xyz_rf=[1, 1, 1]):
        """Get the receptive field mask of a neuron subgroup.

        Args:
            subgroup (NeuronSubGroup or NeuronGroup): The neuron subgroup.
            xyz_rf (list of int): The receptive field size in each dimension. The default is [1, 1, 1].

        Returns:
            torch.Tensor: The receptive field mask.
        """
        rf_x, rf_y, rf_z = xyz_rf

        src_x_start = np.min(subgroup.x) - rf_x
        src_x_end = np.max(subgroup.x) + rf_x

        src_y_start = np.min(subgroup.y) - rf_y
        src_y_end = np.max(subgroup.y) + rf_y

        src_z_start = np.min(subgroup.z) - rf_z
        src_z_end = np.max(subgroup.z) + rf_z

        mask = (
            (self.x >= src_x_start)
            * (self.x <= src_x_end)
            * (self.y >= src_y_start)
            * (self.y <= src_y_end)
            * (self.z >= src_z_start)
            * (self.z <= src_z_end)
        )

        return mask

    def mask_var(self, var):
        """Mask a variable.

        Args:
            var (torch.Tensor): The variable.

        Returns:
            torch.Tensor: The masked variable.
        """
        return var

    @property
    def def_dtype(self):
        return self.network.def_dtype


class NeuronSubGroup:
    def __init__(self, BaseNeuronGroup, mask):
        self.cache = {}
        self.key_id_cache = {}
        self.BaseNeuronGroup = BaseNeuronGroup
        self.mask = mask
        self.id_mask = torch.from_numpy(np.where(mask)[0]).to(BaseNeuronGroup.device)

    def mask_var(self, var):
        if var.shape[0] != self.mask.shape[0]:
            return var[:, self.mask]
        else:
            return var[self.mask]

    def __getattr__(self, attr_name):
        if attr_name in ["BaseNeuronGroup", "mask", "cache", "key_id_cache", "id_mask"]:
            return super().__getattr__(attr_name)  # setattr

        if attr_name == "size":
            return torch.sum(self.mask)

        attr = getattr(self.BaseNeuronGroup, attr_name)
        if type(attr) == torch.tensor:
            if attr.shape[0] != self.mask.shape[0]:
                return attr[:, self.mask]
            else:
                return attr[self.mask]
        else:
            return attr

    def __setattr__(self, attr_name, value):
        if attr_name in ["BaseNeuronGroup", "mask", "cache", "key_id_cache", "id_mask"]:
            super().__setattr__(attr_name, value)
            return

        attr = getattr(self.BaseNeuronGroup, attr_name)
        if type(attr) == torch.tensor:
            if attr.shape[0] != self.mask.shape[0]:
                attr[:, self.mask] = value
            else:
                attr[self.mask] = value
        else:
            setattr(self.BaseNeuronGroup, attr, value)

    def group_without_subGroup(self):
        return self.BaseNeuronGroup
