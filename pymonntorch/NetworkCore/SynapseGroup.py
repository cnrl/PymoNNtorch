import copy

from pymonntorch.NetworkCore.Base import *


class SynapseGroup(NetworkObject):
    """This is the class to construct synapses between neuronal populations.

    Attributes:
        src (NeuronGroup): The pre-synaptic neuron group.
        dst (NeuronGroup): The post-synaptic neuron group.
        net (Network): The network the synapse group belongs to.
        tags (list): The tags of the synapse group.
        behavior (dict or list): The behaviors of the synapse group.
        enabled (bool): Whether the synapse is enabled for learning or not.
        group_weighting (float): The weighting of the synapse group.
    """

    def __init__(self, src, dst, net, tag=None, behavior=None):
        """This is the constructor of the SynapseGroup class.

        Args:
            src (NeuronGroup): The pre-synaptic neuron group.
            dst (NeuronGroup): The post-synaptic neuron group.
            net (Network): The network the synapse group belongs to.
            tag (str): The tag of the synapse group.
            behavior (dict or list): The behaviors of the synapse group.
        """
        if type(src) is str:
            src = net[src, 0]

        if type(dst) is str:
            dst = net[dst, 0]

        if tag is None and net is not None:
            tag = "SynapseGroup_" + str(len(net.SynapseGroups) + 1)

        super().__init__(tag, net, behavior, net.device)
        self.add_tag("syn")

        if len(src.tags) > 0 and len(dst.tags) > 0:
            self.add_tag(src.tags[0] + " => " + dst.tags[0])

        if net is not None:
            net.SynapseGroups.append(self)
            setattr(net, self.tags[0], self)

        self.recording = True

        self.src = src
        self.dst = dst
        self.enabled = True
        self.group_weighting = 1

        for ng in self.network.NeuronGroups:
            for tag in self.tags + ["All"]:
                if tag not in ng.afferent_synapses:
                    ng.afferent_synapses[tag] = []
                if tag not in ng.efferent_synapses:
                    ng.efferent_synapses[tag] = []

        if (
            self.dst.BaseNeuronGroup == self.dst
        ):  # only add to NeuronGroup not to NeuronSubGroup
            for tag in self.tags + ["All"]:
                self.dst.afferent_synapses[tag].append(self)

        if self.src.BaseNeuronGroup == self.src:
            for tag in self.tags + ["All"]:
                self.src.efferent_synapses[tag].append(self)

    def __repr__(self):
        result = "SynapseGroup" + str(self.tags)
        if self.network.transposed_synapse_matrix_mode:
            result = result + "(S" + str(self.src.size) + "xD" + str(self.dst.size)
        else:
            result = result + "(D" + str(self.dst.size) + "xS" + str(self.src.size)
        result = result + "){"

        for k in sorted(list(self.behavior.keys())):
            result += str(k) + ":" + str(self.behavior[k]) + ","
        return result + "}"

    def set_var(self, key, value):
        """Sets a variable of the synapse group.

        Args:
            key (str): The name of the variable.
            value (any): The value of the variable.

        Returns:
            SynapseGroup: The synapse group itself.
        """
        setattr(self, key, value)
        return self

    @property
    def def_dtype(self):
        return self.network.def_dtype

    @property
    def iteration(self):
        return self.network.iteration

    def matrix_dim(self):
        """Returns the dimension of the synapse matrix.

        For a synapse group between a source population of size n and a destination population of size m, the synapse matrix has the dimension m x n.

        Returns:
            tuple: The dimension of the synapse matrix.
        """
        if self.network.transposed_synapse_matrix_mode:
            return self.src.size, self.dst.size
        return self.dst.size, self.src.size

    def get_random_synapse_mat_fixed(self, min_number_of_synapses=0):
        """Returns a random synapse matrix with a fixed number of synapses per neuron.

        Args:
            min_number_of_synapses (int): The minimum number of synapses per neuron.

        Returns:
            torch.Tensor: The random synapse matrix.
        """
        dim = self.matrix_dim()
        result = torch.zeros(dim, device=self.device)
        if min_number_of_synapses != 0:
            for i in range(dim[0]):
                synapses = torch.randperm(dim[1], device=self.device)[
                    :min_number_of_synapses
                ]
                result[i, synapses] = torch.rand(len(synapses), device=self.device)
        return result

    def matrix(
        self,
        mode="zeros()",
        scale=None,
        density=None,
        only_enabled=True,
        clone_along_first_axis=False,
        plot=False,
        dtype=None,
    ):
        """Get a tensor with synapse group dimensionality.

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
            only_enabled (bool): Whether to only consider enabled synapses or not. The default is True.
            clone_along_first_axis (bool): Whether to clone the tensor along the first axis or not. The default is False.
            plot (bool): If true, the histogram of the tensor will be plotted. The default is False.
            dtype (str or type): Data type of the tensor. If None, `def_dtype` will be used.

        Returns:
            torch.Tensor: The initialized tensor.
        """
        result = self._get_mat(
            mode=mode,
            dim=(self.matrix_dim()),
            scale=scale,
            density=density,
            plot=plot,
            dtype=dtype,
        )

        if clone_along_first_axis:
            result = (
                torch.cat([result[0] for _ in range(self.matrix_dim()[0])])
                .reshape(self.matrix_dim()[0], *result[0].shape)
                .to(self.device)
            )

        if only_enabled:
            result *= self.enabled

        return result

    def get_synapse_group_size_factor(self, synapse_group, synapse_type):
        """Returns the size factor of a synapse group.

        Args:
            synapse_group (SynapseGroup): The synapse group.
            synapse_type (str): The type of the synapse.

        Returns:
            float: The size factor of the synapse group.
        """
        total_weighting = 0
        for s in synapse_group.dst.afferent_synapses[synapse_type]:
            total_weighting += s.group_weighting

        total = 0
        for s in synapse_group.dst.afferent_synapses[synapse_type]:
            total += s.src.size * s.src.group_weighting

        return (
            total_weighting
            / total
            * synapse_group.src.size
            * synapse_group.group_weighting
        )

    def get_distance_mat(self, radius, src_x=None, src_y=None, dst_x=None, dst_y=None):
        """Returns a distance matrix between source and destination neurons.

        Args:
            radius (float): The radius of the distance to be considered.
            src_x (torch.Tensor): The x coordinates of the source neurons. The default is None (i.e. the x coordinates of the source neurons will be used).
            src_y (torch.Tensor): The y coordinates of the source neurons. The default is None (i.e. the y coordinates of the source neurons will be used).
            dst_x (torch.Tensor): The x coordinates of the destination neurons. The default is None (i.e. the x coordinates of the destination neurons will be used).
            dst_y (torch.Tensor): The y coordinates of the destination neurons. The default is None (i.e. the y coordinates of the destination neurons will be used).

        Returns:
            torch.Tensor: The distance matrix.
        """
        if src_x is None:
            src_x = self.src.x
        if src_y is None:
            src_y = self.src.y
        if dst_x is None:
            dst_x = self.dst.x
        if dst_y is None:
            dst_y = self.dst.y

        result_syn_mat = torch.zeros((len(dst_x), len(src_x)), device=self.device)

        for d_n in range(len(dst_x)):
            dx = torch.abs(src_x - dst_x[d_n])
            dy = torch.abs(src_y - dst_y[d_n])

            dist = torch.sqrt(dx * dx + dy * dy)
            inv_dist = torch.clamp(radius - dist, 0.0, None)
            inv_dist /= torch.max(inv_dist)

            result_syn_mat[d_n] = inv_dist

        return result_syn_mat

    def get_ring_mat(
        self, radius, inner_exp, src_x=None, src_y=None, dst_x=None, dst_y=None
    ):
        """Returns a ring-shaped distance matrix between source and destination neurons.

        Args:
            radius (float): The radius of the ring.
            inner_exp (float): The exponent of the inner radius.
            src_x (torch.Tensor): The x coordinates of the source neurons. The default is None (i.e. the x coordinates of the source neurons will be used).
            src_y (torch.Tensor): The y coordinates of the source neurons. The default is None (i.e. the y coordinates of the source neurons will be used).
            dst_x (torch.Tensor): The x coordinates of the destination neurons. The default is None (i.e. the x coordinates of the destination neurons will be used).
            dst_y (torch.Tensor): The y coordinates of the destination neurons. The default is None (i.e. the y coordinates of the destination neurons will be used).

        Returns:
            torch.Tensor: The ring matrix.
        """
        dm = self.get_distance_mat(radius, src_x, src_y, dst_x, dst_y)
        ring = torch.clamp(dm - torch.pow(dm, inner_exp) * 1.5, 0.0, None)
        return ring / torch.max(ring)

    def get_max_receptive_field_size(self):
        """Returns the maximum receptive field size of the synapse group.

        Returns:
            tuple: The maximum receptive field size of the synapse group.
        """
        max_dx = 1
        max_dy = 1
        max_dz = 1

        for i in range(self.dst.size):
            if type(self.enabled) is torch.tensor:
                mask = self.enabled[i]
            else:
                mask = self.enabled

            if torch.sum(mask) > 0:
                x = self.dst.x[i]
                y = self.dst.y[i]
                z = self.dst.z[i]

                sx_v = self.src.x[mask]
                sy_v = self.src.y[mask]
                sz_v = self.src.z[mask]

                max_dx = torch.maximum(torch.max(torch.abs(x - sx_v)), max_dx)
                max_dy = torch.maximum(torch.max(torch.abs(y - sy_v)), max_dy)
                max_dz = torch.maximum(torch.max(torch.abs(z - sz_v)), max_dz)

        return max_dx, max_dy, max_dz

    def get_sub_synapse_group(self, src_mask, dst_mask):
        """Returns a sub synapse group between two neuronal subgroups.

        Args:
            src_mask (torch.Tensor): The mask of the source neurons.
            dst_mask (torch.Tensor): The mask of the destination neurons.

        Returns:
            SynapseGroup: The sub synapse group.
        """
        result = SynapseGroup(
            self.src.subGroup(src_mask),
            self.dst.subGroup(dst_mask),
            net=None,
            behavior={},
        )

        # partition enabled update
        if type(self.enabled) is torch.tensor:
            mat_mask = dst_mask[:, None] * src_mask[None, :]
            result.enabled = self.enabled[mat_mask].copy().reshape(result.matrix_dim())

        # copy al attributes
        sgd = self.__dict__
        for key in sgd:
            if key == "behavior":
                for k in self.behavior:
                    result.behavior[k] = copy.copy(self.behavior[k])
            elif key not in ["src", "dst", "enabled", "_mat_eval_dict"]:
                setattr(result, key, copy.copy(sgd[key]))

        return result

    @property
    def def_dtype(self):
        return self.network.def_dtype
